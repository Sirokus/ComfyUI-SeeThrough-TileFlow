"""Per-part nodes of the PSD-upscale graph (``nodes_psd_min`` drives them).

* ``STLU_LoadCrop``     -- load one crop at its work size as straight RGBA,
  so the model sees the real alpha (plus a MASK key for manual use).
* ``STLU_CutToLayer``   -- cut the model's RGBA crop back to the layer's
  target size and write the upscaled layer PNG.
* ``STLU_PsdAssemble``  -- rebuild the layered PSD from the plan.

Every pixel step lives in ``psd_upscale`` (CPU-only, unit tested); these
nodes only move tensors, JSON and files.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image

if __package__:
    from . import core_io, core_plan, edit_plan, psd_upscale
else:  # standalone import used by the tests and tools
    import core_io
    import core_plan
    import edit_plan
    import psd_upscale

if __package__:
    from .nodes_shared import *  # noqa: F401,F403
    from . import nodes_shared as _shared
else:
    from nodes_shared import *  # noqa: F401,F403
    import nodes_shared as _shared

DEFAULT_PSD_SIGMAS = (
    "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, "
    "0.414568, 0"
)


def _plan_dir_from_plan(plan):
    out_dir = str(plan.get("out_dir") or "").strip()
    if not out_dir:
        raise ValueError("计划文件缺少 out_dir")
    return out_dir


def _rgba_tensor(pixels: np.ndarray):
    # All four channels: the Qwen 2.1 VAE is a transparent VAE and pads a
    # 3-channel image to opaque alpha, so the alpha has to travel with the RGB.
    return torch.from_numpy(
        np.ascontiguousarray(pixels[..., :4]).astype(np.float32) / 255.0)[None]


def _mask_tensor(alpha: np.ndarray):
    return torch.from_numpy(
        (1.0 - np.asarray(alpha, dtype=np.float32) / 255.0)[None])


class STLU_LoadCrop:
    """Load one crop PNG at its work size as straight RGBA; MASK = 1 - alpha."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("STRING", {"default": ""}),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, file, width=0, height=0):
        path = str(file or "").strip()
        if not os.path.isfile(path):
            raise ValueError("裁剪图不存在: %s" % path)
        pixels = core_io.read_rgba(path)
        w = int(width) or pixels.shape[1]
        h = int(height) or pixels.shape[0]
        if pixels.shape[:2] != (h, w):
            rgb = np.array(
                Image.fromarray(pixels[..., :3], "RGB").resize(
                    (w, h), Image.Resampling.LANCZOS), dtype=np.uint8)
            alpha = core_io.resize_alpha(pixels[..., 3], w, h)
            pixels = np.dstack([rgb, alpha])
        return (_rgba_tensor(pixels), _mask_tensor(pixels[..., 3]))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_CutToLayer:
    """Cut the model's RGBA crop back to one target-sized RGBA layer PNG."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "geometry": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                # Only used when the image arrives without an alpha channel.
                "alpha": ("MASK",),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("layer_file", "layer_image")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, image, geometry, alpha=None):
        geo = json.loads(str(geometry or "{}"))
        missing = [k for k in ("crop_size", "crop_box", "box", "size",
                               "target", "out_file") if k not in geo]
        if missing:
            raise ValueError("geometry 缺少字段: %s" % ", ".join(missing))

        edited = np.clip(np.asarray(image[0], dtype=np.float32) * 255.0,
                         0, 255).astype(np.uint8)
        if edited.ndim != 3:
            raise ValueError("编辑结果不是图像")
        if edited.shape[-1] >= 4:
            # The VAE decoded RGBA: the model's own alpha is the silhouette.
            rgba, mode = np.ascontiguousarray(edited[..., :4]), "model"
        else:
            rgb = np.ascontiguousarray(edited[..., :3])
            if alpha is not None:
                mask = np.asarray(alpha[0], dtype=np.float32)
                mask = mask[..., 0] if mask.ndim == 3 else mask
                a = np.clip(np.rint((1.0 - mask) * 255.0), 0, 255).astype(np.uint8)
                if a.shape[:2] != rgb.shape[:2]:
                    a = core_io.resize_alpha(a, rgb.shape[1], rgb.shape[0])
                mode = "mask"
            else:
                a = np.full(rgb.shape[:2], 255, np.uint8)
                mode = "opaque"
            rgba = np.dstack([rgb, a])

        # Premultiplied resample inside restore_crop, so the crop's
        # transparent-area colour never bleeds into the edge.
        layer = edit_plan.restore_crop(
            rgba, geo["crop_size"], geo["crop_box"], geo["box"],
            geo["size"], geo["target"])
        out_file = str(geo["out_file"])
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        core_io.write_rgba(out_file, layer)
        # Cache fingerprint beside the PNG: lets a later run reuse this layer
        # for the final merge even when it is not selected (and drop it when
        # the PSD or the factor changed).
        psd_upscale.record_layer_cache(geo.get("plan_json"), out_file)
        _notify("stlu.psd.layer", {
            "file": out_file,
            "size": [int(layer.shape[1]), int(layer.shape[0])],
        })
        # geometry carries name + crop only on the edit path, so this extra
        # edit progress event never fires during a plain upscale.
        crop_rel = str(geo.get("crop") or "")
        if crop_rel:
            try:
                out_rel = os.path.relpath(os.path.abspath(out_file),
                                          _output_root()).replace("\\", "/")
            except ValueError:
                out_rel = ""
            _notify("stlu.edit.layer", {
                "file": out_file,
                "file_rel": "" if out_rel.startswith("..") else out_rel,
                "crop": crop_rel,
                "name": str(geo.get("name") or ""),
                "size": [int(layer.shape[1]), int(layer.shape[0])],
                "coverage": round(float((layer[..., 3] > 8).mean()), 4),
                "mode": mode,
            })
        return (out_file, _rgba_tensor(layer))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_PsdAssemble:
    """Rebuild the layered upscaled PSD once every layer has been cut."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json": ("STRING", {"default": ""}),
            },
            "optional": {
                "trigger": ("*",),
                "psd_filename": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("psd_path", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, plan_json, trigger=None, psd_filename=""):
        plan = psd_upscale.load_plan(str(plan_json or ""))
        run_abs = _plan_dir_from_plan(plan)
        # safe_name("") returns "_", so guard on the raw string first; fall back
        # to the name ① wrote into the plan before deriving from the source.
        raw_name = str(psd_filename or "").strip()
        if not raw_name:
            raw_name = str((plan.get("settings") or {}).get("psd_name") or "").strip()
        name = core_plan.safe_name(raw_name) if raw_name else ""
        if not name:
            stem = core_plan.safe_name(
                os.path.splitext(os.path.basename(plan.get("psd") or "psd"))[0])
            name = "%s_upscaled" % stem
        out_psd = os.path.join(run_abs, name + ".psd")

        report = psd_upscale.assemble_psd(plan, out_psd)
        preview = psd_upscale.preview_rgba(plan)
        core_io.write_rgba(os.path.join(run_abs, psd_upscale.PREVIEW_NAME), preview)

        out_dir_rel = os.path.relpath(run_abs, _output_root()).replace("\\", "/")
        if out_dir_rel.startswith(".."):
            out_dir_rel = ""
        payload = {
            "psd": out_psd,
            "preview": os.path.join(run_abs, psd_upscale.PREVIEW_NAME),
            # Output-relative folder + file name, so the web panel can build a
            # /stlu/thumb URL without knowing the absolute root.
            "dir": out_dir_rel,
            "name": name,
            "canvas": report["canvas"],
            "upscaled": report["upscaled"],
            "upscaled_count": report["upscaled_count"],
            "cached": report["cached"],
            "cached_count": report["cached_count"],
            "stale": report["stale"],
            "stale_count": report["stale_count"],
            "fallback": report["fallback"],
            "fallback_count": report["fallback_count"],
        }
        _notify("stlu.psd.assembled", payload)
        return (out_psd, json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "STLU_LoadCrop": STLU_LoadCrop,
    "STLU_CutToLayer": STLU_CutToLayer,
    "STLU_PsdAssemble": STLU_PsdAssemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STLU_LoadCrop": "STLU 裁剪读取（裁图→工作尺寸）",
    "STLU_CutToLayer": "STLU 裁回图层（输出放大层）",
    "STLU_PsdAssemble": "STLU 重组 PSD（③ 内部自带）",
}
