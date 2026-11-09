"""Older SeeThrough nodes, kept registered so old workflows still open.

They live under ``SeeThrough/LayerUpscale/Legacy`` in the node menu.  New
workflows should use the ①-⑤ nodes in ``nodes_pipeline`` instead;
nothing in this file is on the guided pipeline's path.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import traceback
import uuid

import numpy as np
import torch
import torch.nn.functional as F

import folder_paths

if __package__:  # ComfyUI loads the folder as a package
    from . import (core_compose, core_geometry, core_io, core_mask4k, core_plan,
                   core_reference, edit_expand, edit_parts, psd_reader,
                   psd_writer, tile_psd, tile_strips)
    from .stlu import json_io, limits as stlu_limits
else:  # standalone import used by the tests and tools
    import core_compose
    import core_geometry
    import core_io
    import core_mask4k
    import core_plan
    import core_reference
    import edit_expand
    import edit_parts
    import psd_reader
    import psd_writer
    import tile_psd
    import tile_strips
    from stlu import json_io, limits as stlu_limits


if __package__:
    from .nodes_shared import *  # noqa: F401,F403
    from . import nodes_shared as _shared
else:
    from nodes_shared import *  # noqa: F401,F403
    import nodes_shared as _shared

CATEGORY = "SeeThrough/LayerUpscale/Legacy"


def _output_root() -> str:
    return _shared._output_root()


def _notify(event: str, payload: dict) -> None:
    return _shared._notify(event, payload)


def _prompt_id():
    return _shared._prompt_id()


def _image_fingerprint(frame):
    return _shared._image_fingerprint(frame)


def _model_identity(model):
    return _shared._model_identity(model)


class STLU_LoadLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "reference": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "name", "width", "height", "target_path")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, plan, index, reference=None):
        try:
            plan_data = _load_plan(plan)
            layers = plan_data["layers"]
            index = int(index)
            if index < 0 or index >= len(layers):
                image, mask = _empty_image()
                return (image, mask, "", 8, 8, "")
            layer = layers[index]
            arr = core_io.read_rgba(layer["file"])
            image = torch.from_numpy(arr.astype(np.float32) / 255.0)[None]
            alpha = arr[..., 3].astype(np.float32) / 255.0
            mask = torch.from_numpy(1.0 - alpha)[None]
            target = layer.get("target") or {}
            width = int(target.get("w") or arr.shape[1])
            height = int(target.get("h") or arr.shape[0])
            # The UI turns this into "正在放大 3/14 · handwear-l"; without it a
            # single 12 GB card spends minutes reloading the text encoder with
            # nothing on screen to show for it.
            _notify("stlu.layer_start", {"index": index, "name": str(layer["name"])})
            return (image, mask, str(layer["name"]), width, height, layer.get("target_file", ""))
        except Exception as exc:  # noqa: BLE001
            print("[STLU] LoadLayer failed for index %r: %s" % (index, exc), flush=True)
            image, mask = _empty_image()
            return (image, mask, "", 8, 8, "")

    @classmethod
    def IS_CHANGED(cls, plan, index, reference=None):
        return core_plan.sha1_text("%s|%s" % (_plan_fingerprint(plan), index))

class STLU_ContextBackdrop:
    """Alpha-composite every source layer into one opaque canvas.

    The batch sends this image (or the user reference) to STLU_FocusCrop so a
    small part is refined inside a window of real surrounding artwork instead of
    a transparent frame the edit model would feel compelled to complete.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("context",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, plan):
        plan_data = _load_plan(plan)
        source = plan_data.get("source") or {}
        width = int(source.get("width") or 1024)
        height = int(source.get("height") or width)
        canvas = np.full((height, width, 3), 255.0, dtype=np.float32)
        for layer in plan_data.get("layers", []):
            path = layer.get("file")
            box = layer.get("box")
            if not path or not box:
                continue
            try:
                rgba = core_io.read_rgba(path)
            except Exception as exc:  # noqa: BLE001
                print("[STLU] backdrop: cannot read %s: %s" % (path, exc), flush=True)
                continue
            _alpha_over(canvas, rgba, box)
        tensor = torch.from_numpy(canvas / 255.0)[None]
        return (tensor,)

    @classmethod
    def IS_CHANGED(cls, plan):
        return core_plan.sha1_text("%s|backdrop" % _plan_fingerprint(plan))

class STLU_FocusCrop:
    """Cut a context window around one layer and return the crop + its geometry.

    The window is the layer's alpha content expanded by ``margin`` (grown to a
    minimum side so the reference encoder can see it) and clamped to the
    canvas.  The layer's own pixels are painted on top of the context so the
    part arrives at source sharpness.  Output is opaque RGB on purpose.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "image": ("IMAGE",),
                "context": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "margin": ("INT", {"default": core_plan.FOCUS_MARGIN, "min": 0, "max": 256, "step": 8}),
            },
            "optional": {
                "alpha_threshold": ("INT", {"default": 8, "min": 0, "max": 255, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("crop", "crop_info")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, plan, index, image, context, width, height, margin,
                alpha_threshold=8):
        plan_data = _load_plan(plan)
        layers = plan_data.get("layers") or []
        index = int(index)
        if index < 0 or index >= len(layers):
            raise ValueError("FocusCrop: layer index %d out of range" % index)
        layer = layers[index]
        source = plan_data.get("source") or {}
        canvas = (int(source.get("width") or 1024), int(source.get("height") or 1024))

        rgba = _numpy_rgba(image)
        box = [int(v) for v in (layer.get("box") or [0, 0, rgba.shape[1], rgba.shape[0]])]
        bw, bh = box[2] - box[0], box[3] - box[1]
        if rgba.shape[1] != bw or rgba.shape[0] != bh:
            # Keep pixel space consistent with the plan's box before measuring.
            rgba = core_io.resize_rgba(rgba, bw, bh)

        name = str(layer.get("name") or "layer")
        target = (max(1, int(width)), max(1, int(height)))
        if core_geometry.is_full_canvas(box, canvas):
            info = core_geometry.bypass_info(name, target)
            rgb = rgba[..., :3]
            return (torch.from_numpy(rgb.astype(np.float32) / 255.0)[None],
                    core_geometry.crop_info_to_text(info))

        bbox = core_geometry.alpha_bbox(rgba[..., 3], int(alpha_threshold))
        content = core_geometry.content_rect(box, bbox, canvas)
        crop = core_geometry.crop_rect(content, canvas, int(margin))

        context_np = _numpy_rgba(context)
        ctx_size = (context_np.shape[1], context_np.shape[0])
        scale = core_geometry.context_scale(ctx_size, canvas)

        if scale > 1.02:
            # The reference is bigger than the source canvas.  Crop it at its
            # own resolution instead of flattening it to 1024 first — that is
            # the whole point of choosing the reference as context.
            crop_px = core_geometry.scale_rect(crop, scale, ctx_size)
            content_px = core_geometry.scale_rect(content, scale, ctx_size)
            layer_w = max(1, int(round(bw * scale)))
            layer_h = max(1, int(round(bh * scale)))
            layer_x = int(round(box[0] * scale))
            layer_y = int(round(box[1] * scale))
            box_px = (layer_x, layer_y, layer_x + layer_w, layer_y + layer_h)
            layer_px = core_io.resize_rgba(rgba, layer_w, layer_h)
            out = core_geometry.extract_crop(context_np, layer_px, box_px, crop_px)
            info = core_geometry.make_crop_info(
                name, canvas, box_px, content_px, crop_px, target, int(margin)
            )
            info["context_scale"] = round(float(scale), 4)
            print("[STLU] focus crop %r: layer %dx%d -> window %dx%d from %dx%d context (%.2fx)"
                  % (name, bw, bh, crop_px[2] - crop_px[0], crop_px[3] - crop_px[1],
                     ctx_size[0], ctx_size[1], scale), flush=True)
        else:
            if ctx_size != (canvas[0], canvas[1]):
                context_np = core_io.resize_rgba(context_np, canvas[0], canvas[1])
            out = core_geometry.extract_crop(context_np, rgba, box, crop)
            info = core_geometry.make_crop_info(
                name, canvas, box, content, crop, target, int(margin)
            )
            print("[STLU] focus crop %r: layer %dx%d -> window %dx%d @ (%d,%d)"
                  % (name, bw, bh, crop[2] - crop[0], crop[3] - crop[1], crop[0], crop[1]),
                  flush=True)
        tensor = torch.from_numpy(out.astype(np.float32) / 255.0)[None]
        return (tensor, core_geometry.crop_info_to_text(info))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_PasteBack:
    """Cut the layer's content out of the generated crop and restore its alpha.

    The generated image may be any size (the encoder snaps the crop to a
    multiple of 32 at ~1 MP), so the content rectangle is mapped by fractions
    of the crop rather than by assuming an output size.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_info": ("STRING", {"forceInput": True}),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, image, crop_info, mask):
        info = core_geometry.crop_info_from_text(crop_info)
        alpha_src = mask
        if alpha_src is not None:
            alpha_src = alpha_src[:1].detach().float().clamp(0.0, 1.0)

        if not info or info.get("bypass"):
            target = (info or {}).get("target")
            if target and len(target) == 2:
                width, height = max(1, int(target[0])), max(1, int(target[1]))
            elif alpha_src is not None:
                width, height = int(alpha_src.shape[2]), int(alpha_src.shape[1])
            else:
                width, height = int(image.shape[2]), int(image.shape[1])
            rgb = image[:1].movedim(-1, 1).float()
            if (rgb.shape[2], rgb.shape[3]) != (height, width):
                rgb = F.interpolate(rgb, size=(height, width), mode="bilinear",
                                    align_corners=False)
            out_mask = _resize_mask(alpha_src, width, height)
            return (rgb.movedim(1, -1).clamp(0.0, 1.0), out_mask)

        fracs = core_geometry.paste_fractions(info)
        target = info.get("target") or [int(image.shape[2]), int(image.shape[1])]
        tw, th = max(1, int(target[0])), max(1, int(target[1]))
        if fracs is None:
            rgb = image[:1].movedim(-1, 1).float()
            rgb = F.interpolate(rgb, size=(th, tw), mode="bilinear", align_corners=False)
            return (rgb.movedim(1, -1).clamp(0.0, 1.0), _resize_mask(alpha_src, tw, th))

        fx0, fy0, fx1, fy1, lx0, ly0, lx1, ly1 = fracs
        rgb = image[:1].movedim(-1, 1).float()
        sampled = _sample_region(rgb[0], fx0, fy0, fx1, fy1, tw, th)  # [3, th, tw]
        if alpha_src is not None:
            alpha = _sample_region(alpha_src[0][None], lx0, ly0, lx1, ly1, tw, th)
        else:
            alpha = torch.zeros((1, th, tw), dtype=torch.float32)
        out_mask = alpha[:1].clamp(0.0, 1.0)
        return (sampled.movedim(0, -1)[None].clamp(0.0, 1.0), out_mask)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_SaveLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "name": ("STRING", {"default": "layer"}),
                # 批量里接 plan 解析好的相对路径；留空时落在 output/UpScale。
                "out_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "keep_input_alpha": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("saved_path", "image")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, images, mask, name, out_dir, keep_input_alpha=False):
        if not str(name or "").strip():
            # LoadLayer signals a missing/broken source by returning an empty
            # name; leave the previous output untouched so PSD assembly can
            # fall back to the original layer art.
            print("[STLU] empty layer name, save skipped", flush=True)
            return ("", images)
        rgba = core_io.tensor_to_rgba(images)
        mask_alpha = core_io.mask_to_alpha(mask)
        merged = core_io.merge_alpha(rgba, mask_alpha, keep_input_alpha=bool(keep_input_alpha))
        filename = core_plan.safe_name(name) + ".png"
        path = os.path.join(core_io.out_dir_abs(_output_root(), out_dir), filename)
        core_io.write_rgba(path, merged)
        print("[STLU] saved layer %r -> %s" % (name, path), flush=True)
        _notify("stlu.progress", {"name": str(name), "path": path})
        return (path, images)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_DetailBatch:
    """Rebuild every layer from a higher-resolution reference (CPU only).

    For character art whose full-body reference is a sharper version of the
    same picture.  Instead of asking an edit model to redraw a 20x20 pupil
    (which only blurs and invents), each layer's alpha is cut out of the split
    and its colour is taken from the reference wherever the two agree.  Visible pixels keep the artist's real
    linework; hidden pixels keep the layer's own colour.

    No model inputs are needed, so running this node loads nothing onto the GPU.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {"default": ""}),
                "reference": ("IMAGE",),
            },
            "optional": {
                "plan_path": ("STRING", {"default": ""}),
                "only_selected": ("BOOLEAN", {"default": False}),
                # Chroma gate: a saturated reference colour this far from the
                # layer's own colour is treated as another part showing
                # through.  Line art and shading pass at the default.
                "color_tolerance": ("FLOAT", {"default": 26.0, "min": 2.0, "max": 120.0, "step": 1.0}),
                # Steepen the silhouette around the 0.5 contour (1.0 = off).
                "alpha_sharpen": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 3.0, "step": 0.1}),
                "alpha_floor": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                # A part always regains its own colour under whatever covered
                # it (that is where a stranger's skin/iris colour used to
                # leak in).  ``hidden_margin`` additionally grows it into the
                # empty area behind an opaque neighbour -- useful, but it
                # invents shape, so it is off by default.
                "hidden_margin": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("last_saved", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, plan, reference, plan_path="", only_selected=False,
                color_tolerance=26.0, alpha_sharpen=1.6, alpha_floor=8,
                hidden_margin=0):
        plan_ref = (plan_path or "").strip() or plan
        plan_data = _load_plan(plan_ref)
        source = plan_data.get("source") or {}
        canvas = (int(source.get("width") or 1024), int(source.get("height") or 1024))
        out_dir_rel = plan_data.get("out_dir") or ""
        out_abs = core_io.out_dir_abs(_output_root(), out_dir_rel)
        ref = core_io.tensor_to_rgba(reference)
        # Once per run: the character's outline is the same for every layer,
        # and filling its interior holes is a full-frame operation.
        silhouette = core_reference.reference_silhouette(ref)

        todo = [layer for layer in plan_data.get("layers", [])
                if not layer.get("empty")
                and (layer.get("selected") or not bool(only_selected))]
        _notify("stlu.start", {
            "total": len(todo),
            "layers": [str(layer.get("name")) for layer in todo],
            "skipped": [],
            "out_dir": out_dir_rel,
            "mode": "detail",
        })

        last_saved = ""
        report = []
        done = []           # (name, path, left, top, right, bottom), z order
        for layer in todo:
            name = str(layer.get("name") or "layer")
            box = layer.get("box") or [0, 0, canvas[0], canvas[1]]
            display = layer.get("display_box") or [
                int(round(box[0] * float(plan_data.get("scale") or 1.0))),
                int(round(box[1] * float(plan_data.get("scale") or 1.0))),
                int(round(box[2] * float(plan_data.get("scale") or 1.0))),
                int(round(box[3] * float(plan_data.get("scale") or 1.0))),
            ]
            dest = (max(1, int(display[2]) - int(display[0])),
                    max(1, int(display[3]) - int(display[1])))
            try:
                rgba = core_io.read_rgba(layer["file"])
                out, stats = core_reference.reconstruct_layer(
                    rgba, box, ref, canvas, dest,
                    color_tolerance=float(color_tolerance),
                    alpha_floor=int(alpha_floor),
                    alpha_sharpen=float(alpha_sharpen),
                    silhouette=silhouette,
                )
            except Exception as exc:  # noqa: BLE001
                print("[STLU] detail layer %r failed: %s" % (name, exc), flush=True)
                report.append({"layer": name, "error": str(exc)})
                continue
            _notify("stlu.layer_start", {"index": int(layer.get("z", 0)), "name": name})
            path = os.path.join(out_abs, core_plan.safe_name(name) + ".png")
            core_io.write_rgba(path, out)
            last_saved = path
            _notify("stlu.progress", {"name": name, "path": path})
            report.append({"layer": name, "size": list(out.shape[1::-1]),
                           "detail": stats})
            done.append((name, path, int(display[0]), int(display[1]),
                         int(display[2]), int(display[3])))
            print("[STLU] detail %r -> %dx%d (%.2fx ref, %.0f%% from ref)"
                  % (name, out.shape[1], out.shape[0],
                     stats.get("scale", 1.0), 100.0 * stats.get("trusted", 0.0)), flush=True)

        filled = self._fill_hidden(done, plan_data, int(hidden_margin))
        if filled:
            print("[STLU] hidden area filled behind opaque parts: %s"
                  % ", ".join("%s %d" % (n, c) for n, c in filled), flush=True)

        return (last_saved, json.dumps({"total": len(todo), "layers": report,
                                        "hidden_filled": dict(filled)},
                                       ensure_ascii=False, indent=2))

    @staticmethod
    def _fill_hidden(done, plan_data, margin):
        """Give every part back the pixels the layer above it was painting.

        Runs top -> bottom so one canvas-sized mask carries "is this pixel
        hidden by something opaque above me" for the layer being visited.
        Because a pixel is only written when an opaque higher layer covers it,
        the assembled PSD looks exactly the same as before -- but each layer is
        now complete under its neighbour, which is what a rig needs.
        """
        if len(done) < 2:
            return []
        size = int(plan_data.get("canvas_size") or 0)
        if size <= 0:
            size = max(max(d[4] for d in done), max(d[5] for d in done))
        cover = np.zeros((size, size), dtype=bool)
        filled = []
        for name, path, left, top, right, bottom in reversed(done):
            try:
                px = core_io.read_rgba(path)
            except Exception:  # noqa: BLE001
                continue
            h, w = px.shape[:2]
            x0, y0 = max(0, left), max(0, top)
            x1, y1 = min(size, left + w), min(size, top + h)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = cover[y0:y1, x0:x1]
            own = px[y0 - top:y1 - top, x0 - left:x1 - left]
            if sub.any():
                out, count = core_reference.fill_hidden(own, sub, margin)
                if count:
                    px[y0 - top:y1 - top, x0 - left:x1 - left] = out
                    core_io.write_rgba(path, px)
                    filled.append((name, count))
            cover[y0:y1, x0:x1] |= px[y0 - top:y1 - top, x0 - left:x1 - left, 3] >= 250
        return filled

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Cheap and CPU-only: always rebuild rather than risk a stale cache
        # when the reference or the split changes size.
        return float("nan")


class STLU_Gate:
    """Pass conditioning/latent through while waiting on a pipeline barrier.

    The batch expansion wires every sampler through one of these.  Each gate
    waits on the tail of the encoder chain, so every TextEncode runs before any
    sampling — the text encoder is loaded once instead of once per layer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
            },
            "optional": {
                "dummy": ("*",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, positive, negative, latent, dummy=None):
        return (positive, negative, latent)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_Barrier:
    """Pass-through chain link: only orders the encoders ahead of sampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": ("*",)},
            "optional": {"prev": ("*",)},
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, value, prev=None):
        return (value,)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_ChromaToAlpha:
    """Key a flat chroma background out of an edit-model result.

    Qwen-Image-Edit can only return RGB.  The extraction prompt asks for the
    part on a solid key colour; this node turns that colour into alpha on the
    CPU, despills the edge, and protects pixels the model left identical to
    the reference (a green eye must survive a green screen).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "key_color": (list(edit_parts.KEY_COLORS.keys()),
                              {"default": "green"}),
                "tolerance": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 60.0,
                                        "step": 1.0}),
                "softness": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 40.0,
                                       "step": 1.0}),
            },
            "optional": {
                "reference": ("IMAGE",),
                "protect": ("BOOLEAN", {"default": True}),
                "despill": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0,
                                      "step": 0.05}),
                "alpha_floor": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba", "mask")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, image, key_color, tolerance, softness, reference=None,
                protect=True, despill=0.65, alpha_floor=8):
        rgba = core_io.tensor_to_rgba(image)
        ref = core_io.tensor_to_rgba(reference) if reference is not None else None
        out = edit_parts.key_out(
            rgba[..., :3], key_color, float(tolerance), float(softness),
            reference_rgb=ref[..., :3] if ref is not None else None,
            protect=bool(protect), despill_strength=float(despill),
            alpha_floor=int(alpha_floor),
        )
        tensor = torch.from_numpy(out.astype(np.float32) / 255.0)[None]
        # ComfyUI MASK must be a torch tensor ([B,H,W]); a bare ndarray here
        # reaches STLU_SaveLayer and blows up on ``mask[0].detach()``.
        mask = torch.from_numpy(
            (1.0 - out[..., 3].astype(np.float32) / 255.0)[None])
        return (tensor, mask)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_PartsPSD:
    """Assemble the extracted part PNGs into a SeeThrough-compatible PSD.

    The layer order is the SeeThrough order (far -> near, bottom -> top);
    empty parts and failed extractions (a full-frame result) are skipped.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers_json": ("STRING", {"default": ""}),
            },
            "optional": {
                # Linked from the last save node so the PSD runs after it.
                "trigger": ("*",),
                "psd_filename": ("STRING", {"default": ""}),
                "alpha_threshold": ("INT", {"default": 8, "min": 0, "max": 255,
                                            "step": 1}),
                "skip_empty": ("BOOLEAN", {"default": True}),
                "max_coverage": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 0.999,
                                           "step": 0.005}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("psd_path", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, layers_json, trigger=None, psd_filename="",
                alpha_threshold=8, skip_empty=True, max_coverage=0.9):
        path = str(layers_json or "").strip()
        if not path or not os.path.isfile(path):
            raise FileNotFoundError("layers.json not found: %r" % path)
        doc = edit_parts.read_layers_json(path)
        base = os.path.dirname(path)
        stem = str(psd_filename or "").strip() or "%s_%s" % (
            doc.get("prefix") or "editparts", doc.get("timestamp") or "")
        psd_path = os.path.join(base, core_plan.safe_name(stem) + ".psd")
        info = edit_parts.assemble_psd(
            path, _output_root(), psd_path,
            alpha_threshold=int(alpha_threshold),
            skip_empty=bool(skip_empty),
            max_coverage=float(max_coverage),
        )
        subfolder, filename = _output_rel(psd_path)
        _notify("stlu.parts.psd", {
            "psd": psd_path,
            "subfolder": subfolder,
            "filename": filename,
            "layers": len(info["layers"]),
            "skipped": len(info["notes"]),
        })
        print("[STLU] parts PSD written: %s (%d layers)" % (
            psd_path, len(info["layers"])), flush=True)
        return (psd_path, json.dumps(info, ensure_ascii=False, indent=2))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_TileGrid:
    """Step ①: cut the reference image and keep the checked tiles.

    Cutting and picking live on one node: ``select`` is an ordinary visible
    widget (``all`` / ``none`` / ``0-2,5``), and the node's panel lets you
    click the grid to fill it in.  Only the checked crops are handed on, so
    unchecked tiles never reach the diffusion chain.  The grid (sizes,
    coordinates) travels in memory as a JSON string on ``grid_doc``; nothing
    is written to disk here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "tile_width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "tile_height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 4096,
                                    "tooltip": "相邻块重叠的像素；合并时用正弦窗平滑接缝"}),
                # Appended last: widget values serialize by position.
                "select": ("STRING", {"default": "all",
                                      "tooltip": "只跑这些块：all / none / 0-2,5；点节点上的格子也会写这里"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("tiles", "indices", "grid_doc", "original")
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, image, tile_width=1024, tile_height=1024, overlap=128,
                select="all"):
        if image is None or not hasattr(image, "shape") or image.ndim < 3:
            raise ValueError("STLU_TileGrid: 需要连接 image（标准 LoadImage 即可）/ image input required")
        batch = image if image.ndim == 4 else image[None]
        frame = batch[0].detach().float().cpu()
        height, width = int(frame.shape[0]), int(frame.shape[1])
        now = datetime.datetime.now()
        doc = tile_psd.grid_document(
            (width, height), (tile_width, tile_height), int(overlap),
            source_image="", name="grid",
            created=now.strftime("%Y%m%d_%H%M%S"),
        )
        coords = doc["coords"]
        raw_select = str(select if select is not None else "all").strip()
        picked = parse_tile_selection(raw_select, len(coords))
        if not picked:
            raise ValueError("STLU_TileGrid: 没有勾选任何块（select=%r）/ no tiles selected" % raw_select)
        encoded = "all" if len(picked) == len(coords) else encode_tile_selection(picked)
        doc["selection"] = encoded
        grid_doc = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))

        tiles = []
        indices = []
        for index in picked:
            x0, y0, x1, y1 = coords[index]
            crop = frame[int(y0):int(y1), int(x0):int(x1), :]
            tiles.append(crop[None].contiguous())
            indices.append(str(index))

        info = {
            # No file: the document goes out through ``grid_doc`` instead.
            "grid": "",
            "grid_doc": grid_doc,
            "dir": "",
            "count": int(doc["count"]),
            "selected": len(picked),
            "indices": indices,
            "rows": int(doc["rows"]),
            "columns": int(doc["columns"]),
            "source": [int(width), int(height)],
            "tile_size": [int(v) for v in doc["tile_size"]],
            "overlap": int(doc["overlap"]),
            "selection": encoded,
        }
        _notify("stlu.grid", info)
        print("[STLU] grid: %d/%d tiles (%dx%d) from %dx%d, kept in memory (%s)" % (
            len(picked), doc["count"], doc["tile_size"][0], doc["tile_size"][1],
            width, height, encoded),
            flush=True)
        return {
            "ui": {"stlu_grid": [info]},
            "result": (tiles, indices, grid_doc, frame[None].contiguous()),
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_SelectTiles:
    """Legacy step ①b: keep only the checked tiles before the expensive chain.

    ``STLU_TileGrid`` now cuts and picks in one node, so new workflows no
    longer need this filter.  It stays for old workflows that already wire
    ``TileGrid -> SelectTiles``: the grid node then emits every tile (its
    ``select`` defaults to ``all``) and this node narrows the list.  ``select``
    is a normal, visible widget: ``all``, ``none`` or ``0-2,5``; unchecked
    tiles never reach SeeThrough, so they cost no diffusion at all.
    """

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
            },
            "optional": {
                "indices": ("STRING", {"forceInput": True}),
                "grid_doc": ("STRING", {"forceInput": True}),
                # Preview only: the reference picture, for the clickable grid.
                "image": ("IMAGE",),
                # Appended last: widget values serialize by position.
                "select": ("STRING", {"default": "all",
                                      "tooltip": "只跑这些块：all / none / 0-2,5；点节点上的格子也会写这里"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("tiles", "indices", "grid_doc")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, tiles, indices=None, grid_doc="", image=None, select="all"):
        tiles_list = list(tiles) if isinstance(tiles, (list, tuple)) else [tiles]
        raw_select = str(_first(select, "all") or "all").strip()
        picked = parse_tile_selection(raw_select, len(tiles_list))
        if not picked:
            raise ValueError("STLU_SelectTiles: 没有勾选任何块（select=%r）/ no tiles selected" % raw_select)

        index_values = [str(v) for v in _list_values(indices)]
        if len(index_values) != len(tiles_list):
            if len(index_values) > len(tiles_list):
                index_values = index_values[:len(tiles_list)]
            else:
                index_values += [str(i) for i in range(len(index_values), len(tiles_list))]

        selected = [tiles_list[i] for i in picked]
        selected_indices = [index_values[i] for i in picked]
        try:
            encoded = encode_tile_selection([int(v) for v in selected_indices])
        except (TypeError, ValueError):
            encoded = encode_tile_selection(picked)
        if encoded == "none":
            encoded = ",".join(selected_indices)

        doc_ref = _first(grid_doc, "")
        doc = None
        if str(doc_ref or "").strip():
            try:
                doc = tile_psd.load_grid_ref(str(doc_ref))
            except Exception as exc:  # noqa: BLE001
                raise ValueError("STLU_SelectTiles: grid_doc 解析失败 / failed to parse grid_doc: %s" % exc)
            doc["selection"] = encoded
            out_doc = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
        else:
            out_doc = ""

        src = (doc or {}).get("source") or {}
        info = {
            "count": len(tiles_list),
            "selected": len(selected),
            "selection": encoded,
            "indices": selected_indices,
            "grid_doc": out_doc,
            "source": [int(src.get("width") or 0), int(src.get("height") or 0)],
            "tile_size": [int(v) for v in ((doc or {}).get("tile_size") or [])],
            "overlap": int((doc or {}).get("overlap") or 0),
            "rows": int((doc or {}).get("rows") or 0),
            "columns": int((doc or {}).get("columns") or 0),
        }
        _notify("stlu.tile_select", info)
        print("[STLU] tile select: %d/%d kept (%s)" % (
            len(selected), len(tiles_list), encoded), flush=True)
        return {
            "ui": {"stlu_tile_select": [info]},
            "result": (selected, selected_indices, out_doc),
        }

class STLU_TilesPSD:
    """Write one layered PSD per tile of a SeeThrough grid-split run.

    ``SplitImageToTileList`` expands every downstream node once per tile, so
    ``parts`` arrives here as a list.  ``SeeThrough_SavePSD`` cannot handle
    that: it dumps layer PNGs and overwrites ``seethrough_psd_info.log`` for
    every tile, so the browser PSD button only ever sees the last one.  This
    node writes real PSDs on the backend instead, into a per-run subfolder.

    ``group_layers`` is fixed on: every tile PSD keeps the exact layers
    SeeThrough produced (hairf/hairb, eyel/eyer, ears-l/ears-r, ...) so step
    four can merge by exact name without collapsing front and back hair.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # Live mode: connect SeeThrough_PostProcess here.
                "parts": ("SEETHROUGH_PARTS",),
                # Optional: the source image is only used to name tiles with
                # their row/column in the original picture.
                "image": ("IMAGE",),
                # "0-2,5" from STLU_TileSelect: keeps original grid positions
                # when only a subset of tiles was generated.
                "tile_indices": ("STRING", {"default": ""}),
                "out_dir": ("STRING", {"default": "SeeThrough"}),
                "filename_prefix": ("STRING", {"default": "tile"}),
                "merge_parts": ("BOOLEAN", {"default": False,
                                            "tooltip": "已停用（保留给旧工作流）：③ 固定保留 eyel/eyer、hairf/hairb 等子层，由 ④ 按原名合并"}),
                "alpha_threshold": ("INT", {"default": 8, "min": 0, "max": 255, "step": 1}),
                "tile_width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "tile_height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 4096}),
                "merged_psd": ("BOOLEAN", {"default": False,
                                           "tooltip": "已停用（保留给旧工作流）：合并统一由 ④ 完成，这里不再自动合并"}),
                "write_depth": ("BOOLEAN", {"default": True,
                                            "tooltip": "固定开启（保留给旧工作流）：每块都存 depth.png，供 ④ 输出独立 depth PSD"}),
                "group_layers": ("BOOLEAN", {"default": True,
                                             "tooltip": "固定开启（保留给旧工作流）：分格 PSD 总是保留 hairf/hairb 等子层，④ 再决定合成还是收进文件夹"}),
                "include_original": ("BOOLEAN", {"default": True,
                                                 "tooltip": "固定开启（保留给旧工作流）：原图随波次保存，供 ④ 的原图底图使用"}),
                # Appended last on purpose: widget values serialize by
                # position, so older workflows keep their saved settings.
                # Offline mode: a folder of layer images (STLU_SaveTiles run,
                # SeeThrough *_layers.json files, or plain PNGs).  Used when
                # ``parts`` is not connected, so an existing run can be turned
                # into PSDs without diffusing again.
                "source": ("STRING", {"default": "",
                                      "tooltip": "已生成的图片目录 / *_layers.json / 单张图片；留空则使用 parts 输入"}),
                "grid": ("STRING", {"default": "",
                                    "tooltip": "接 ① 的 grid_doc（内存里的切格文档，不落盘）；旧版工作流里也可以填 grids/*.json 的路径"}),
                # Appended last for the same serialization reason.  ``group``
                # is a free-form category folder under out_dir (empty = none);
                # ``tiles`` limits an offline rebuild to some tile indices.
                "group": ("STRING", {"default": "",
                                     "tooltip": "分类前缀（可选）：结果会放进 SeeThrough/<前缀>/<本次 id>/"}),
                "tiles": ("STRING", {"default": "",
                                     "tooltip": "只重建这些块，如 0-2,5；留空 = 来源里的全部块"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("psd_dir", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, parts=None, image=None, tile_indices="", out_dir="SeeThrough",
                filename_prefix="tile", merge_parts=True, alpha_threshold=8,
                tile_width=1024, tile_height=1024, overlap=128, merged_psd=True,
                write_depth=True, group_layers=False, include_original=True,
                source="", grid="", group="", tiles=""):
        image = _first(image)
        parts_list = parts if isinstance(parts, list) else [parts]
        parts_list = [p for p in parts_list if isinstance(p, dict) and p.get("tag2pinfo")]
        grid_doc, grid_path = _load_grid(_first(grid, ""))

        out_dir = str(_first(out_dir, "SeeThrough") or "SeeThrough")
        prefix = str(_first(filename_prefix, "tile") or "tile")
        group_name = str(_first(group, "") or "").strip()
        stamp = "%s_%s" % (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            str(uuid.uuid4())[:6],
        )
        # One folder per run, named after the queue prompt id, optionally
        # grouped by a user prefix: SeeThrough/<group?>/<id>/ holds the tile
        # PSDs at the top and the saved layer images in ``images/``.
        wave = tile_psd.wave_id(_prompt_id())
        wave_dir = tile_psd.wave_directory(_output_root(), out_dir, group_name, wave)
        tile_subset = None
        raw_tiles = str(_first(tiles, "") or "").strip()
        if raw_tiles and raw_tiles.lower() not in ("all", "none"):
            tile_subset = _parse_index_list(raw_tiles)

        # An explicit source wins over a connected parts socket: the panel's
        # source picker is the mode switch, so picking existing images must
        # actually skip SeeThrough instead of silently re-diffusing.
        raw_source = str(_first(source, "") or "").strip()
        if raw_source:
            # Offline: rebuild the tiles from images that already exist.
            source_path = _resolve_workspace_path(raw_source)
            if not os.path.exists(source_path):
                raise FileNotFoundError("image source not found: %s" % source_path)
            # The loader knows how many tiles it found, so the index list is
            # passed as-is; it only has to line up with the saved tiles.
            index_map = None
            raw_indices = str(_first(tile_indices, "") or "").strip()
            if raw_indices and raw_indices.lower() not in ("all", "none"):
                index_map = _parse_index_list(raw_indices)
            # Depth and the per-tile sub layers are ALWAYS produced: the depth
            # map is what the merge turns into its own depth PSD, and keeping
            # hairf/hairb separate is what lets step four folder them.  Step
            # four owns the final merge, so this node never merges on its own.
            manifest = tile_psd.write_tiles_from_images(
                source_path,
                _output_root(),
                out_dir=out_dir,
                filename_prefix=prefix,
                merge_parts=bool(_first(merge_parts, True)),
                alpha_threshold=int(_first(alpha_threshold, 8)),
                stamp=stamp,
                merge_after=False,
                grid_doc=grid_doc,
                tile_index_map=index_map,
                write_depth=True,
                merge_group_layers=True,
                include_original=True,
                run_dir_abs=wave_dir,
                tiles=tile_subset,
            )
        elif not parts_list:
            raise ValueError("STLU_TilesPSD: 连接 parts，或在「来源」里选已生成的图片目录 / connect parts, or pick a generated image folder in Source")
        else:
            source_size = None
            original_rgba = None
            if image is not None and hasattr(image, "shape") and image.ndim >= 4:
                source_size = (int(image.shape[2]), int(image.shape[1]))
                try:
                    frame = image[0].cpu().numpy() if hasattr(image, "cpu") else np.asarray(image[0])
                    original_rgba = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
                except Exception:  # noqa: BLE001
                    original_rgba = None

            index_map = _tile_index_map(_first(tile_indices, ""), parts_list, grid_doc)

            # The real tile size comes from the layer documents themselves; the
            # widgets are only a fallback and are kept in sync with the splitter
            # by the front end.
            tile_h = int(_first(tile_height, 1024))
            tile_w = int(_first(tile_width, 1024))
            frame = parts_list[0].get("frame_size")
            if frame and int(frame[0]) > 0 and int(frame[1]) > 0:
                tile_h, tile_w = int(frame[0]), int(frame[1])
            if grid_doc:
                src = grid_doc.get("source") or {}
                if int(src.get("width") or 0) > 0 and int(src.get("height") or 0) > 0:
                    source_size = (int(src["width"]), int(src["height"]))
                tile_w, tile_h = (int(v) for v in grid_doc.get("tile_size") or (tile_w, tile_h))

            # Keep the tile images next to the PSDs: they are the artifact
            # that lets step three be re-run later without diffusing again
            # (the panel lists them, and the merge does not need them).
            images_dir = ""
            images_error = None
            try:
                images_manifest = tile_psd.write_tile_images(
                    parts_list,
                    _output_root(),
                    out_dir=out_dir,
                    filename_prefix=prefix,
                    stamp=stamp,
                    grid_doc=grid_doc,
                    tile_index_map=index_map,
                    source_size=source_size,
                    # One reference per wave: write_tiles below puts it at the
                    # wave root, so images.json points there instead of
                    # keeping a second identical copy in images/.
                    original_rel="../original.png" if original_rgba is not None else None,
                    run_dir_abs=os.path.join(wave_dir, "images"),
                    alpha_threshold=int(_first(alpha_threshold, 8)),
                )
                images_dir = images_manifest["dir"]
            except Exception as exc:  # noqa: BLE001 - PSDs still matter more
                images_error = str(exc)

            manifest = tile_psd.write_tiles(
                parts_list,
                _output_root(),
                out_dir=out_dir,
                filename_prefix=prefix,
                merge_parts=bool(_first(merge_parts, True)),
                alpha_threshold=int(_first(alpha_threshold, 8)),
                source_size=source_size,
                tile_size=(tile_w, tile_h),
                overlap=int(_first(overlap, 128)),
                stamp=stamp,
                merge_after=False,
                tile_index_map=index_map,
                write_depth=True,
                original_rgba=original_rgba,
                merge_group_layers=True,
                include_original=True,
                run_dir_abs=wave_dir,
                generation=(grid_doc or {}).get("generation"),
                tile_coords=(grid_doc or {}).get("coords") or None,
                grid_offset=(grid_doc or {}).get("offset") or (0, 0),
            )
            manifest["images_dir"] = images_dir
            if images_error:
                manifest["images_error"] = images_error
        manifest["grid"] = grid_path
        manifest["group"] = group_name
        manifest["wave"] = wave
        manifest["prompt_id"] = _prompt_id()

        run_dir = os.path.join(_output_root(), manifest["dir"])
        first = manifest["tiles"][0]
        total = 0
        for entry in manifest["tiles"]:
            path = os.path.join(run_dir, entry["file"])
            if os.path.isfile(path):
                total += os.path.getsize(path)

        info = {
            "dir": manifest["dir"],
            "subfolder": manifest["dir"],
            "filename": first["file"],
            "count": manifest["count"],
            "layers": len(first["layers"]),
            "canvas": first["canvas"],
            "size": total,
            "merged": manifest["merge_parts"],
            "merged_psd": None,
            "merge_error": manifest.get("merge_error"),
            "preview": None,
            # Where the per-tile images live, so step three can be re-entered
            # from them later without running SeeThrough again.
            "images_dir": manifest.get("images_dir") or manifest.get("source_dir") or "",
            "images_error": manifest.get("images_error"),
            "group": group_name,
            "wave": wave,
            "error": None,
        }
        merged = manifest.get("merged") or {}
        if merged.get("file"):
            merged_path = os.path.join(run_dir, merged["file"])
            if os.path.isfile(merged_path):
                info["merged_psd"] = {
                    "filename": merged["file"],
                    "subfolder": manifest["dir"],
                    "canvas": merged.get("canvas"),
                    "layers": len(merged.get("layers") or []),
                    "size": os.path.getsize(merged_path),
                }
            if merged.get("preview"):
                preview_path = os.path.join(run_dir, merged["preview"])
                if os.path.isfile(preview_path):
                    info["preview"] = {
                        "filename": merged["preview"],
                        "subfolder": manifest["dir"],
                        "type": "output",
                        "v": int(os.path.getmtime(preview_path) * 1000),
                    }
        if merged.get("depth"):
            depth_path = os.path.join(run_dir, merged["depth"])
            if os.path.isfile(depth_path):
                info["depth"] = {
                    "filename": merged["depth"],
                    "subfolder": manifest["dir"],
                }
        if merged.get("depth_psd"):
            depth_psd_path = os.path.join(run_dir, merged["depth_psd"])
            if os.path.isfile(depth_psd_path):
                info["depth_psd"] = {
                    "filename": merged["depth_psd"],
                    "subfolder": manifest["dir"],
                    "v": int(os.path.getmtime(depth_psd_path) * 1000),
                }
        if info["preview"] is None:
            # No aggregate preview file any more: the manifest names the
            # preview of the richest tile (they used to be identical bytes).
            preview_name = str(manifest.get("preview") or "")
            preview = os.path.join(run_dir, preview_name) if preview_name else ""
            if preview and os.path.isfile(preview):
                info["preview"] = {
                    "filename": preview_name,
                    "subfolder": manifest["dir"],
                    "type": "output",
                    "v": int(os.path.getmtime(preview) * 1000),
                }
        _notify("stlu.tiles", info)
        print("[STLU] %d tile PSDs written: %s" % (manifest["count"], run_dir), flush=True)

        return {
            "ui": {"stlu_tiles": [info]},
            "result": (run_dir, json.dumps(manifest, ensure_ascii=False, indent=2)),
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_Wave:
    """Stage ③: name this queue's output wave.

    The folder is only computed here, never created: a run that produces no
    tiles leaves no stray directory behind.  A fresh wave id per queue also
    guarantees the per-tile writer cannot be skipped by a cache hit.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "out_dir": ("STRING", {"default": "SeeThrough",
                                       "tooltip": "相对 output/ 的根目录"}),
                "group": ("STRING", {"default": "",
                                     "tooltip": "分类前缀（可选）：结果放进 <out_dir>/<前缀>/<波次>/"}),
                "filename_prefix": ("STRING", {"default": "tile",
                                               "tooltip": "每块 PSD 的文件名前缀"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("wave_dir", "wave_doc")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, out_dir="SeeThrough", group="", filename_prefix="tile"):
        wave = tile_psd.wave_id(_prompt_id())
        wave_dir = tile_psd.wave_directory(_output_root(), out_dir, group, wave)
        doc = {
            "dir": wave_dir,
            "rel": os.path.relpath(wave_dir, _output_root()).replace(os.sep, "/"),
            "wave": wave,
            "group": str(group or "").strip(),
            "prefix": str(filename_prefix or "tile"),
            "prompt_id": _prompt_id(),
        }
        return {"result": (wave_dir, json.dumps(doc, ensure_ascii=False))}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_SaveTiles:
    """Stage two's product: save every tile's parts as plain images.

    Writes one folder per tile (``layers.json`` + one PNG per part + depth
    PNGs) plus an ``images.json`` index.  ``STLU_TilesPSD`` can then build the
    per-tile PSDs from this folder without running SeeThrough again -- that is
    what makes the stages independently re-runnable.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parts": ("SEETHROUGH_PARTS",),
            },
            "optional": {
                "grid": ("STRING", {"default": "",
                                    "tooltip": "STLU_TileGrid 输出的 grid 路径（块位置、原图尺寸）"}),
                "tile_indices": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "out_dir": ("STRING", {"default": "SeeThrough/Tiles"}),
                "filename_prefix": ("STRING", {"default": "tile"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("images_dir", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, parts, grid="", tile_indices="", image=None,
                out_dir="SeeThrough/Tiles", filename_prefix="tile"):
        parts_list = parts if isinstance(parts, list) else [parts]
        parts_list = [p for p in parts_list if isinstance(p, dict) and p.get("tag2pinfo")]
        if not parts_list:
            raise ValueError("STLU_SaveTiles received no tile parts")

        grid_doc, grid_path = _load_grid(_first(grid, ""))
        index_map = _tile_index_map(_first(tile_indices, ""), parts_list, grid_doc)

        image = _first(image)
        source_size = None
        original_rgba = None
        if image is not None and hasattr(image, "shape") and image.ndim >= 4:
            source_size = (int(image.shape[2]), int(image.shape[1]))
            try:
                frame = image[0].cpu().numpy() if hasattr(image, "cpu") else np.asarray(image[0])
                original_rgba = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
            except Exception:  # noqa: BLE001
                original_rgba = None

        stamp = "%s_%s" % (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            str(uuid.uuid4())[:6],
        )
        manifest = tile_psd.write_tile_images(
            parts_list,
            _output_root(),
            out_dir=str(_first(out_dir, "SeeThrough/Tiles") or "SeeThrough/Tiles"),
            filename_prefix=str(_first(filename_prefix, "tile") or "tile"),
            stamp=stamp,
            grid_doc=grid_doc,
            tile_index_map=index_map,
            source_size=source_size,
            original_rgba=original_rgba,
        )
        manifest["grid"] = grid_path

        run_dir = os.path.join(_output_root(), manifest["dir"])
        info = {
            "dir": manifest["dir"],
            "subfolder": manifest["dir"],
            "count": manifest["count"],
            "canvas": manifest.get("source"),
            "tile_size": manifest.get("tile_size"),
            "tiles": [int(t["index"]) for t in manifest["tiles"]],
            "has_depth": True,
            "error": None,
        }
        _notify("stlu.tile_images", info)
        print("[STLU] %d tile image folders saved: %s" % (manifest["count"], run_dir), flush=True)
        return {
            "ui": {"stlu_tile_images": [info]},
            "result": (run_dir, json.dumps(manifest, ensure_ascii=False, indent=2)),
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_TileSelect:
    """Filter the tile list so only the checked tiles run the expensive nodes.

    The node sits between ``SplitImageToTileList`` and the SeeThrough chain.
    Its front-end panel draws the source image with the tile grid on top; the
    checked cells are written into the hidden ``select`` widget ("all",
    "none" or "0-2,5").  Downstream nodes are expanded once per surviving
    tile, so unchecked tiles cost no diffusion at all.
    """

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
            },
            "optional": {
                # Only used by the selection grid in the UI.
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "tile_height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 4096}),
                "select": ("STRING", {"default": "all"}),
                # STLU_TileGrid's document: the selection is written back into
                # it so every later stage sees the same checked tiles.
                "grid": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("tiles", "indices")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, tiles, image=None, tile_width=1024, tile_height=1024,
                overlap=128, select="all", grid=""):
        if isinstance(tiles, list):
            tiles_list = list(tiles)
        elif hasattr(tiles, "shape"):
            tiles_list = [tiles[i:i + 1] for i in range(int(tiles.shape[0]))]
        else:
            tiles_list = [tiles]

        indices = parse_tile_selection(_first(select, "all"), len(tiles_list))
        selected = [tiles_list[i] for i in indices]
        encoded = encode_tile_selection(indices)
        # The grid document is the shared, global record: persist the checked
        # tiles so later stages can read the selection without re-deriving it.
        # An inline document (the current stage one) has no file to update.
        grid_ref = str(_first(grid, "") or "").strip()
        grid_path = (_resolve_workspace_path(grid_ref)
                     if grid_ref and not grid_ref.startswith("{") else "")
        if grid_path and os.path.isfile(grid_path):
            tile_psd.update_grid_selection(grid_path, encoded)
        print("[STLU] TileSelect: %d/%d tiles selected (%s)" % (
            len(selected), len(tiles_list), _first(select, "all")), flush=True)
        # The indices string travels with the tiles so the PSD writer can put
        # each tile back at its original grid position instead of renumbering.
        return {"ui": {"stlu_tiles_select": [{
            "total": len(tiles_list),
            "selected": indices,
            "selection": encoded,
            "grid": grid_path,
        }]}, "result": (selected, encoded)}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "STLU_LoadLayer": STLU_LoadLayer,
    "STLU_ContextBackdrop": STLU_ContextBackdrop,
    "STLU_FocusCrop": STLU_FocusCrop,
    "STLU_PasteBack": STLU_PasteBack,
    "STLU_SaveLayer": STLU_SaveLayer,
    "STLU_DetailBatch": STLU_DetailBatch,
    "STLU_Gate": STLU_Gate,
    "STLU_Barrier": STLU_Barrier,
    "STLU_ChromaToAlpha": STLU_ChromaToAlpha,
    "STLU_PartsPSD": STLU_PartsPSD,
    "STLU_TileGrid": STLU_TileGrid,
    "STLU_SelectTiles": STLU_SelectTiles,
    "STLU_TilesPSD": STLU_TilesPSD,
    "STLU_Wave": STLU_Wave,
    "STLU_SaveTiles": STLU_SaveTiles,
    "STLU_TileSelect": STLU_TileSelect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STLU_LoadLayer": "SeeThrough Upscale Load Layer",
    "STLU_ContextBackdrop": "SeeThrough Focus Backdrop",
    "STLU_FocusCrop": "SeeThrough Focus Crop",
    "STLU_PasteBack": "SeeThrough Focus Paste Back",
    "STLU_SaveLayer": "SeeThrough Upscale Save Layer",
    "STLU_DetailBatch": "SeeThrough Detail Rebuild (CPU)",
    "STLU_Gate": "SeeThrough Pipeline Gate",
    "STLU_Barrier": "SeeThrough Pipeline Barrier",
    "STLU_ChromaToAlpha": "Live2D Key Background to Alpha",
    "STLU_PartsPSD": "Live2D Parts to PSD",
    "STLU_TilesPSD": "SeeThrough · 旧版 ③ 图片 → 分格 PSD",
    "STLU_TileSelect": "SeeThrough · 旧版选格",
    "STLU_TileGrid": "SeeThrough ① 切格 + 选块",
    "STLU_SelectTiles": "SeeThrough · 旧版选块（单独节点）",
    "STLU_Wave": "SeeThrough ③ 输出文件夹（每次运行新建）",
    "STLU_SaveTiles": "SeeThrough · 旧版：单独保存分块图片",
}

__all__ = [
    "STLU_LoadLayer",
    "STLU_ContextBackdrop",
    "STLU_FocusCrop",
    "STLU_PasteBack",
    "STLU_SaveLayer",
    "STLU_DetailBatch",
    "STLU_Gate",
    "STLU_Barrier",
    "STLU_ChromaToAlpha",
    "STLU_PartsPSD",
    "STLU_TileGrid",
    "STLU_SelectTiles",
    "STLU_TilesPSD",
    "STLU_Wave",
    "STLU_SaveTiles",
    "STLU_TileSelect",
]
