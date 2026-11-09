"""PSD-upscale nodes: finished PSD -> per-part crops -> Qwen edit -> PSD.

The flow is four small nodes:

* ``STLU_PsdPrepare``   -- scan the finished PSD, scale the plan, write one
  crop PNG per layer (white context, real alpha kept) and a preview.
* ``STLU_LoadCrop``     -- load one crop at its work size and hand the model
  an IMAGE plus the MASK that carries the real silhouette.
* ``STLU_CutToLayer``   -- cut the edited crop back to the layer's target
  size, un-composite the white and write the upscaled layer PNG.
* ``STLU_PsdAssemble``  -- rebuild the layered PSD from the plan.
* ``STLU_PsdUpscale``   -- batch runner that clones the Qwen edit subgraph
  once per selected layer and chains everything inside one prompt.

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
    from . import core_io, core_plan, psd_upscale
else:  # standalone import used by the tests and tools
    import core_io
    import core_plan
    import psd_upscale

if __package__:
    from .nodes_shared import *  # noqa: F401,F403
    from . import nodes_shared as _shared
else:
    from nodes_shared import *  # noqa: F401,F403
    import nodes_shared as _shared

DEFAULT_PSD_PROMPT = (
    "把这张图放大到更高分辨率，保持原有造型、配色和构图不变，"
    "补充清晰的边缘与纹理细节，不要添加或删除任何元素"
)

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
    return torch.from_numpy(
        np.ascontiguousarray(pixels[..., :3]).astype(np.float32) / 255.0)[None]


def _mask_tensor(alpha: np.ndarray):
    return torch.from_numpy(
        (1.0 - np.asarray(alpha, dtype=np.float32) / 255.0)[None])


class STLU_PsdPrepare:
    """Read a finished PSD and cut one crop per layer for the edit model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "psd_path": ("STRING", {"default": ""}),
                "factor": ("FLOAT", {"default": psd_upscale.DEFAULT_FACTOR,
                                     "min": 0.5, "max": 8.0, "step": 0.25}),
                "max_side": ("INT", {"default": psd_upscale.DEFAULT_MAX_SIDE,
                                     "min": 256, "max": 16384, "step": 64}),
                "select": ("STRING", {"default": "all"}),
            },
            "optional": {
                "out_dir": ("STRING", {"default": "SeeThrough/PsdUpscale"}),
                "margin": ("FLOAT", {"default": psd_upscale.DEFAULT_MARGIN_RATIO,
                                     "min": 0.0, "max": 0.5, "step": 0.01}),
                "reference_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("plan_json", "preview", "info")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, psd_path, factor=psd_upscale.DEFAULT_FACTOR,
                max_side=psd_upscale.DEFAULT_MAX_SIDE, select="all",
                out_dir="SeeThrough/PsdUpscale",
                margin=psd_upscale.DEFAULT_MARGIN_RATIO, reference_path=""):
        raw = str(psd_path or "").strip().strip('"')
        if not raw:
            raise ValueError("请选择成品 PSD 文件")
        path = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isfile(path):
            raise ValueError("PSD 文件不存在: %s" % path)

        out_dir_rel = core_plan.normalize_out_dir(out_dir) or "SeeThrough/PsdUpscale"
        stem = core_plan.safe_name(os.path.splitext(os.path.basename(path))[0])
        run_rel = os.path.join(out_dir_rel, stem)
        run_abs = core_io.out_dir_abs(_output_root(), run_rel)

        plan = psd_upscale.prepare_plan(path, run_abs, factor=float(factor),
                                        max_side=int(max_side), select=select)
        if not plan["count"]:
            raise ValueError("没有选中的图层（select=%r）" % select)
        jobs = psd_upscale.write_crops(plan, margin_ratio=float(margin))
        plan["crops"] = jobs
        ref_path = str(reference_path or "").strip().strip('"')
        if ref_path:
            psd_upscale.write_references(plan, ref_path)
        plan_rel = core_io.out_dir_abs(_output_root(), run_rel)
        psd_upscale.write_plan(run_abs, plan)

        preview = psd_upscale.preview_rgba(plan)
        core_io.write_rgba(os.path.join(run_abs, psd_upscale.PREVIEW_NAME), preview)

        info = {
            "psd": path,
            "out_dir": run_rel,
            "plan": psd_upscale.plan_path(plan_rel),
            "canvas": plan["canvas"],
            "count": plan["count"],
            "capped": psd_upscale.capped_report(plan),
            "reference": plan.get("reference"),
        }
        _notify("stlu.psd.plan", {
            "psd": path,
            "plan": psd_upscale.plan_path(run_abs),
            "canvas": plan["canvas"],
            "count": plan["count"],
            "out_dir": run_rel,
            "capped": info["capped"],
            "reference": plan.get("reference"),
        })
        return (psd_upscale.plan_path(run_abs), _rgba_tensor(preview),
                json.dumps(info, ensure_ascii=False, indent=2))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_LoadCrop:
    """Load one crop PNG at its work size; MASK = 1 - alpha."""

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
    """Cut the edited crop back to one target-sized RGBA layer PNG."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha": ("MASK",),
                "geometry": ("STRING", {"default": "{}", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("layer_file", "layer_image")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, image, alpha, geometry):
        geo = json.loads(str(geometry or "{}"))
        missing = [k for k in ("crop_size", "crop_box", "box", "size",
                               "target", "out_file") if k not in geo]
        if missing:
            raise ValueError("geometry 缺少字段: %s" % ", ".join(missing))

        edited = np.clip(np.asarray(image[0], dtype=np.float32) * 255.0,
                         0, 255).astype(np.uint8)
        # The layer's own PNG alpha is the exact silhouette; the MASK that
        # travelled with the crop went through the work size and softens hard
        # edges.  Prefer the source file, fall back to the MASK.
        src_file = str(geo.get("file") or "")
        if src_file and os.path.isfile(src_file):
            source = core_io.read_rgba(src_file)
            src_alpha = core_io.resize_alpha(
                source[..., 3], int(geo["target"][0]), int(geo["target"][1]))
        else:
            mask = np.asarray(alpha[0], dtype=np.float32)
            src_alpha = np.clip(np.rint((1.0 - mask) * 255.0),
                                0, 255).astype(np.uint8)

        layer = psd_upscale.restore_layer(
            edited, src_alpha,
            geo["crop_size"], geo["crop_box"], geo["box"],
            geo["size"], geo["target"])
        out_file = str(geo["out_file"])
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        core_io.write_rgba(out_file, layer)
        _notify("stlu.psd.layer", {
            "file": out_file,
            "size": [int(layer.shape[1]), int(layer.shape[0])],
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
        name = core_plan.safe_name(str(psd_filename or "").strip())
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
            "fallback": report["fallback"],
        }
        _notify("stlu.psd.assembled", payload)
        return (out_psd, json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_PsdUpscale:
    """Clone the Qwen edit subgraph once per selected PSD layer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json": ("STRING", {"default": ""}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
            "optional": {
                "workflow_json": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": DEFAULT_PSD_PROMPT,
                                      "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                # The whole point of the run is a 2x-4x upscale; a low denoise
                # keeps the edited crop anchored to the source silhouette.
                "denoise": ("FLOAT", {"default": 0.45, "min": 0.05, "max": 1.0,
                                      "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF,
                                 "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0,
                                  "step": 0.5}),
                "scheduler": ("STRING", {"default": "euler"}),
                "sigmas": ("STRING", {"default": DEFAULT_PSD_SIGMAS,
                                      "multiline": True}),
                "psd_filename": ("STRING", {"default": ""}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("psd_path", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, plan_json, model, clip, vae,
                workflow_json="", prompt=DEFAULT_PSD_PROMPT,
                negative_prompt="", denoise=0.45, seed=42, cfg=1.0,
                scheduler="euler", sigmas=DEFAULT_PSD_SIGMAS,
                psd_filename="", dynprompt=None, unique_id=None):
        plan = psd_upscale.load_plan(str(plan_json or ""))
        run_abs = _plan_dir_from_plan(plan)
        jobs = {job["stem"]: job for job in (plan.get("crops") or [])}
        layers = psd_upscale.selected_layers(plan)
        if not layers:
            raise ValueError("计划里没有选中的图层")

        out_dir_rel = os.path.relpath(run_abs, _output_root()).replace("\\", "/")
        if out_dir_rel.startswith(".."):
            out_dir_rel = os.path.basename(run_abs)

        text = str(prompt or "").strip() or DEFAULT_PSD_PROMPT
        parts = []
        for layer in layers:
            job = jobs.get(layer.get("stem"))
            if job is None:
                raise ValueError("图层 %r 缺少裁剪任务，请重新运行 PSD 准备"
                                 % layer.get("name"))
            crop_size = [int(v) for v in job["crop_size"]]
            target = [int(v) for v in layer["target"]]
            size = [int(v) for v in layer["size"]]
            work_w, work_h = psd_upscale.work_size(crop_size, size, target)
            geometry = {
                "crop_size": crop_size,
                "crop_box": [int(v) for v in job["crop_box"]],
                "box": [int(v) for v in layer["box"]],
                "size": size,
                "target": target,
                "out_file": str(layer["out_file"]),
                "file": str(layer.get("file") or ""),
            }
            parts.append({
                "name": str(layer["name"]),
                "prompt": text,
                "crop_file": str(job["crop_file"]),
                "ref_file": str(job.get("ref_file") or ""),
                "width": work_w,
                "height": work_h,
                "resolution": max(work_w, work_h),
                "geometry": geometry,
            })

        if dynprompt is None or unique_id is None:
            raise RuntimeError("STLU_PsdUpscale needs dynprompt/unique_id "
                               "hidden inputs")
        node_inputs = dynprompt.get_node(unique_id).get("inputs", {})

        def _link(name):
            value = node_inputs.get(name)
            if not isinstance(value, list) or len(value) != 2:
                raise RuntimeError("STLU_PsdUpscale input %r is not linked"
                                   % name)
            return [str(value[0]), int(value[1])]

        external = {k: _link(k) for k in ("model", "clip", "vae")}

        wf_path = (workflow_json or "").strip()
        if not wf_path:
            try:
                import edit_parts
                wf_path = edit_parts.default_template_path()
            except Exception:
                wf_path = ""
        if not wf_path or not os.path.isfile(wf_path):
            raise FileNotFoundError(
                "找不到图编辑工作流模板（请在节点里选择 Qwen 编辑工作流）: %s"
                % (wf_path or "<empty>"))

        scaled_sigmas = psd_upscale.scale_sigmas(sigmas, float(denoise))
        if __package__:
            from . import edit_expand
        else:
            import edit_expand
        expand, psd_node, _saves = edit_expand.build_expand(
            workflow_path=wf_path,
            parts=parts,
            external=external,
            seed=int(seed),
            out_dir_rel=out_dir_rel,
            layers_json_path="",
            psd_name=str(psd_filename or ""),
            psd_plan=psd_upscale.plan_path(run_abs),
            crop_mode=True,
            key="green",
            tolerance=12.0,
            softness=6.0,
            protect=True,
            despill=0.65,
            negative_prompt=str(negative_prompt or ""),
            cfg=float(cfg),
            resolution=0,
            scheduler=str(scheduler),
            sigmas=scaled_sigmas,
            width=0,
            height=0,
        )

        run_name = str(psd_filename or "").strip()
        if not run_name:
            stem = core_plan.safe_name(
                os.path.splitext(os.path.basename(plan.get("psd") or "psd"))[0])
            run_name = "%s_upscaled" % stem
        psd_path = os.path.join(run_abs, core_plan.safe_name(run_name) + ".psd")

        _notify("stlu.psd.start", {
            "total": len(parts),
            "parts": [p["name"] for p in parts],
            "out_dir": out_dir_rel,
            "workflow": wf_path,
            "denoise": float(denoise),
        })

        report = {
            "total": len(parts),
            "parts": [p["name"] for p in parts],
            "canvas": plan["canvas"],
            "out_dir": out_dir_rel,
            "psd": psd_path,
            "workflow": wf_path,
            "denoise": float(denoise),
            "sigmas": scaled_sigmas,
            "capped": psd_upscale.capped_report(plan),
            "reference": plan.get("reference"),
        }
        return {
            "result": (psd_path, json.dumps(report, ensure_ascii=False, indent=2)),
            "expand": expand,
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "STLU_PsdPrepare": STLU_PsdPrepare,
    "STLU_LoadCrop": STLU_LoadCrop,
    "STLU_CutToLayer": STLU_CutToLayer,
    "STLU_PsdAssemble": STLU_PsdAssemble,
    "STLU_PsdUpscale": STLU_PsdUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STLU_PsdPrepare": "STLU ① PSD 准备（裁剪每层）",
    "STLU_LoadCrop": "STLU 裁剪读取（裁图→工作尺寸）",
    "STLU_CutToLayer": "STLU 裁回图层（输出放大层）",
    "STLU_PsdAssemble": "STLU ④ 重组 PSD",
    "STLU_PsdUpscale": "STLU ③ 逐层放大（Qwen 编辑）",
}
