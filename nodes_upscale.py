"""Plan / edit-parts / 4K-mask nodes (the non-tile workflows).

They share the original image-edit and PSD helpers with the tile pipeline via
``nodes_shared``; none of them run on the guided ①-⑤ path.
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
                   psd_writer, subgraph_clone, tile_psd, tile_strips)
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
    import subgraph_clone
    import tile_psd
    import tile_strips
    from stlu import json_io, limits as stlu_limits


if __package__:
    from .nodes_shared import *  # noqa: F401,F403
    from . import nodes_shared as _shared
else:
    from nodes_shared import *  # noqa: F401,F403
    import nodes_shared as _shared

# ---------------------------------------------------------------------------
# Image-edit part extraction (no SeeThrough weights)
# ---------------------------------------------------------------------------

DEFAULT_EDIT_SIGMAS = (
    "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, "
    "0.414568, 0"
)

# ---------------------------------------------------------------------------
# 4K part masks (CPU): reference-guided outline + geometric iris completion
# ---------------------------------------------------------------------------

MASK4K_MODES = list(core_mask4k.MASK_MODES)

class STLU_Plan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers_json": ("STRING", {"default": DEFAULT_LAYERS_JSON}),
                "select": ("STRING", {"default": "all"}),
                "exclude": ("STRING", {"default": ""}),
                "size_rules": ("STRING", {"default": DEFAULT_SIZE_RULES, "multiline": True}),
                # 留空 = 拆层文件夹下的 UpScale（相对 output，不存在则新建）。
                "out_dir": ("STRING", {"default": ""}),
                "canvas_size": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 8}),
                "skip_empty": ("BOOLEAN", {"default": True}),
                "empty_coverage": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 0.2, "step": 0.0001}),
                # Local crop refinement: generate inside a window of real
                # context instead of the whole (incomplete) part frame.
                "focus_crop": ("BOOLEAN", {"default": True}),
                "focus_margin": ("INT", {"default": core_plan.FOCUS_MARGIN, "min": 0, "max": 256, "step": 8}),
                "focus_context": (["backdrop", "reference"], {"default": "backdrop"}),
                "focus_prompt": ("STRING", {"default": core_plan.DEFAULT_FOCUS_PROMPT, "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("plan", "plan_path", "total", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, layers_json, select, exclude, size_rules, out_dir, canvas_size,
                skip_empty=True, empty_coverage=0.0005,
                focus_crop=True, focus_margin=core_plan.FOCUS_MARGIN,
                focus_context="backdrop", focus_prompt=core_plan.DEFAULT_FOCUS_PROMPT):
        output_root = _output_root()
        plan = core_plan.build_plan(
            layers_json=layers_json,
            output_root=output_root,
            select=select,
            exclude=exclude,
            size_rules_text=size_rules,
            out_dir=out_dir,
            canvas_size=canvas_size,
            skip_empty=bool(skip_empty),
            empty_coverage=float(empty_coverage),
            focus_crop=bool(focus_crop),
            focus_margin=focus_margin,
            focus_context=focus_context,
            focus_prompt=focus_prompt,
        )
        plan_text = core_plan.plan_to_text(plan)
        plan_dir = os.path.join(output_root, plan["out_dir"])
        os.makedirs(plan_dir, exist_ok=True)
        prefix = core_plan.safe_name(plan["source"].get("prefix") or "seethrough")
        plan_path = os.path.join(plan_dir, prefix + "_plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        return (plan_text, plan_path, int(plan["total"]), core_plan.plan_report(plan))

    @classmethod
    def IS_CHANGED(cls, layers_json, select, exclude, size_rules, out_dir, canvas_size,
                   skip_empty=True, empty_coverage=0.0005,
                   focus_crop=True, focus_margin=core_plan.FOCUS_MARGIN,
                   focus_context="backdrop", focus_prompt=core_plan.DEFAULT_FOCUS_PROMPT):
        parts = [layers_json, select, exclude, size_rules, out_dir, str(canvas_size),
                 str(skip_empty), str(empty_coverage), str(focus_crop),
                 str(focus_margin), str(focus_context), str(focus_prompt)]
        fp = _plan_fingerprint(layers_json)
        return core_plan.sha1_text("|".join(parts) + "|" + fp)

class STLU_EditPartsBatch:
    """Extract every Live2D part with an image-edit model, in one run.

    The node clones the Qwen-Image-Edit subgraph once per selected part,
    rewrites the prompt to "isolate this part on a chroma background", keys
    the background out on the CPU, saves SeeThrough-named layer PNGs and
    finally assembles the PSD -- all inside one prompt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                # "all", or names separated by spaces/commas; "-name" removes.
                "select": ("STRING", {"default": "all"}),
                "key_color": (list(edit_parts.KEY_COLORS.keys()),
                              {"default": "green"}),
            },
            "optional": {
                "out_dir": ("STRING", {"default": "SeeThrough/EditParts"}),
                "workflow_json": ("STRING", {"default": ""}),
                # The frontend always inserts a ``control_after_generate``
                # widget right after a widget named "seed", so the saved
                # widgets_values carries one extra entry there ("fixed").
                # The node itself ignores it: each part uses seed + index.
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF,
                                 "step": 1}),
                "tolerance": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 60.0,
                                        "step": 1.0}),
                "softness": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 40.0,
                                       "step": 1.0}),
                "protect": ("BOOLEAN", {"default": True}),
                "despill": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0,
                                      "step": 0.05}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                # The Qwen 2.1 official path runs at cfg 1 (negative unused).
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0,
                                  "step": 0.5}),
                # 0 = follow image_1's own size (the template's default).
                "resolution": ("INT", {"default": 0, "min": 0, "max": 2048,
                                       "step": 32}),
                "scheduler": ("STRING", {"default": "euler"}),
                "sigmas": ("STRING", {"default": DEFAULT_EDIT_SIGMAS,
                                      "multiline": True}),
                "psd_filename": ("STRING", {"default": ""}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("layers_json", "psd_path", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, reference, model, clip, vae, select, key_color,
                out_dir="SeeThrough/EditParts", workflow_json="", seed=42,
                tolerance=12.0, softness=6.0, protect=True, despill=0.65,
                negative_prompt="", cfg=1.0, resolution=0,
                scheduler="euler", sigmas=DEFAULT_EDIT_SIGMAS,
                psd_filename="", dynprompt=None, unique_id=None):
        parts = edit_parts.parse_select(select)
        if not parts:
            raise ValueError("no parts selected")
        width = int(reference.shape[2])
        height = int(reference.shape[1])

        out_dir_rel = core_plan.normalize_out_dir(out_dir) or "SeeThrough/EditParts"
        out_abs = core_io.out_dir_abs(_output_root(), out_dir_rel)

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        timestamp = "%s_%s" % (now, uid)
        prefix = "editparts"

        entries = []
        for name in parts:
            entries.append(edit_parts.layer_entry(
                name, out_dir_rel, core_plan.safe_name(name) + ".png",
                0, 0, width, height,
            ))
        layers_doc = edit_parts.layers_json_doc(
            prefix, timestamp, entries, width, height)
        layers_path = os.path.join(
            out_abs, "%s_%s_layers.json" % (prefix, timestamp))
        edit_parts.write_layers_json(layers_path, layers_doc)

        psd_stem = str(psd_filename or "").strip() or "%s_%s" % (prefix, timestamp)
        psd_path = os.path.join(out_abs, core_plan.safe_name(psd_stem) + ".psd")

        if dynprompt is None or unique_id is None:
            raise RuntimeError("STLU_EditPartsBatch needs dynprompt/unique_id "
                               "hidden inputs")
        node = dynprompt.get_node(unique_id)
        node_inputs = node.get("inputs", {})

        def _link(name):
            value = node_inputs.get(name)
            if not isinstance(value, list) or len(value) != 2:
                raise RuntimeError("STLU_EditPartsBatch input %r is not linked"
                                   % name)
            return [str(value[0]), int(value[1])]

        external = {
            "reference": _link("reference"),
            "model": _link("model"),
            "clip": _link("clip"),
            "vae": _link("vae"),
        }

        wf_path = (workflow_json or "").strip() or edit_parts.default_template_path()
        if not os.path.isfile(wf_path):
            raise FileNotFoundError("edit workflow template not found: %s" % wf_path)

        part_jobs = [{"name": name,
                      "prompt": edit_parts.build_prompt(name, key_color)}
                     for name in parts]

        expand, psd_node, saves = edit_expand.build_expand(
            workflow_path=wf_path,
            parts=part_jobs,
            external=external,
            seed=int(seed),
            out_dir_rel=out_dir_rel,
            layers_json_path=layers_path,
            psd_name=psd_stem,
            key=key_color,
            tolerance=float(tolerance),
            softness=float(softness),
            protect=bool(protect),
            despill=float(despill),
            negative_prompt=str(negative_prompt or ""),
            cfg=float(cfg),
            resolution=int(resolution),
            scheduler=str(scheduler),
            sigmas=str(sigmas),
            width=width,
            height=height,
        )

        _notify("stlu.parts.start", {
            "total": len(parts),
            "parts": parts,
            "out_dir": out_dir_rel,
            "workflow": wf_path,
        })

        report = {
            "total": len(parts),
            "parts": parts,
            "canvas": [width, height],
            "out_dir": out_dir_rel,
            "layers_json": layers_path,
            "psd": psd_path,
            "workflow": wf_path,
            "key_color": key_color,
        }
        return {
            "result": (layers_path, psd_path,
                       json.dumps(report, ensure_ascii=False, indent=2)),
            "expand": expand,
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

def _layer_output_matches(layer: dict, output_root: str) -> bool:
    """True when this layer's upscaled PNG already exists at the planned size.

    A re-run then skips layers that finished last time instead of paying the
    model reload for each of them again.  Delete the PNG to force a redo.
    """
    target_file = str(layer.get("target_file") or "").strip()
    target = layer.get("target") or {}
    tw, th = int(target.get("w") or 0), int(target.get("h") or 0)
    if not target_file or tw <= 0 or th <= 0:
        return False
    path = os.path.join(output_root, target_file)
    if not os.path.isfile(path):
        return False
    try:
        from PIL import Image

        with Image.open(path) as im:
            width, height = im.size
    except Exception:  # noqa: BLE001
        return False
    return max(width, height) == max(tw, th)

class STLU_UpscaleBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {"default": ""}),
                "reference": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
            "optional": {
                "plan_path": ("STRING", {"default": ""}),
                "pe_clip": ("CLIP",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
                "workflow_json": ("STRING", {"default": ""}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("last_saved", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, plan, reference, model, clip, vae, plan_path="", pe_clip=None,
                seed=42, workflow_json="", dynprompt=None, unique_id=None):
        plan_data = _load_plan(plan)
        output_root = _output_root()
        selected, skipped, selected_names = [], [], []
        for layer in plan_data["layers"]:
            if not layer.get("selected"):
                continue
            if _layer_output_matches(layer, output_root):
                skipped.append(layer["name"])
            else:
                selected.append(layer["z"])
                selected_names.append(layer["name"])
        if not selected:
            note = "all selected layers already upscaled at the planned size"
            if skipped:
                note += " (delete their PNGs under UpScale/ to redo)"
            _notify("stlu.start", {"total": 0, "layers": [], "skipped": skipped,
                                   "out_dir": plan_data.get("out_dir")})
            return ("", json.dumps({"total": 0, "skipped": skipped, "note": note},
                                   ensure_ascii=False))

        _notify("stlu.start", {
            "total": len(selected),
            "layers": selected_names,
            "skipped": skipped,
            "out_dir": plan_data.get("out_dir"),
        })

        if dynprompt is None or unique_id is None:
            raise RuntimeError("STLU_UpscaleBatch needs dynprompt/unique_id hidden inputs")

        node = dynprompt.get_node(unique_id)
        node_inputs = node.get("inputs", {})

        def _link(name):
            value = node_inputs.get(name)
            if not isinstance(value, list) or len(value) != 2:
                raise RuntimeError("STLU_UpscaleBatch input %r is not linked" % name)
            return [str(value[0]), int(value[1])]

        reference_link = _link("reference")
        model_link = _link("model")
        clip_link = _link("clip")
        vae_link = _link("vae")
        pe_link = node_inputs.get("pe_clip")
        if isinstance(pe_link, list) and len(pe_link) == 2:
            pe_link = [str(pe_link[0]), int(pe_link[1])]
        else:
            pe_link = clip_link

        external = {
            "reference": reference_link,
            "model": model_link,
            "clip": clip_link,
            "vae": vae_link,
            "pe_clip": pe_link,
        }

        wf_path = (workflow_json or "").strip() or subgraph_clone.default_workflow_path()
        if not os.path.isfile(wf_path):
            raise FileNotFoundError("workflow_json not found: %s" % wf_path)

        plan_ref = (plan_path or "").strip() or plan
        expand, last_save, report = subgraph_clone.build_expand(
            workflow_path=wf_path,
            plan=plan_data,
            plan_ref=plan_ref,
            external=external,
            seed=int(seed),
            layer_indices=selected,
        )
        if last_save is None:
            return ("", json.dumps({"total": 0}, ensure_ascii=False))

        report_text = json.dumps({
            "total": len(selected),
            "skipped": skipped,
            "workflow": wf_path,
            "layers": report,
        }, ensure_ascii=False, indent=2)
        return {
            "result": (last_save.out(0), report_text),
            "expand": expand,
        }

    @classmethod
    def IS_CHANGED(cls, plan, *args, **kwargs):
        wf = kwargs.get("workflow_json") or ""
        wf_fp = core_plan.file_fingerprint(wf) if wf and os.path.isfile(wf) else "default"
        # Pending layers are part of the signature: once a layer's PNG lands,
        # the next run must re-expand for the remaining ones instead of
        # returning the previous (partly skipped) result from the cache.
        try:
            plan_data = _load_plan(plan)
            root = _output_root()
            pending = ",".join(
                str(layer["z"]) for layer in plan_data["layers"]
                if layer.get("selected") and not _layer_output_matches(layer, root)
            )
        except Exception:  # noqa: BLE001
            pending = "?"
        return core_plan.sha1_text("%s|%s|%s|%s" % (
            _plan_fingerprint(plan), wf_fp, kwargs.get("seed"), pending))

def _plan_layer(plan_data, index):
    layers = plan_data.get("layers") or []
    index = int(index)
    if index < 0 or index >= len(layers):
        raise ValueError("layer index %d out of range (%d layers)" % (index, len(layers)))
    return layers[index]


def _layer_alpha(layer):
    path = layer.get("file")
    if not path or not os.path.isfile(path):
        raise ValueError("layer %r has no readable PNG: %r" % (layer.get("name"), path))
    return core_io.read_rgba(path)[..., 3]


def _mask_preview_image(guide, mask, width=384):
    """RGB preview of the guide with the mask tinted; also usable in a sheet."""
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    guide = np.asarray(guide)
    if xs.size:
        m = 8
        h0, w0 = guide.shape[:2]
        x0, y0 = max(0, int(xs.min()) - m), max(0, int(ys.min()) - m)
        x1, y1 = min(w0, int(xs.max()) + m + 1), min(h0, int(ys.max()) + m + 1)
        guide = guide[y0:y1, x0:x1]
        mask = np.asarray(mask, dtype=bool)[y0:y1, x0:x1]
    rgb = core_mask4k.mask_overlay(guide, mask, (0, 255, 0), 0.45)
    h, w = rgb.shape[:2]
    scale = min(1.0, float(width) / max(1, w))
    if scale < 1.0:
        import cv2
        rgb = cv2.resize(rgb, (max(1, int(round(w * scale))),
                               max(1, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)
    return rgb


def _mask_to_tensor(mask):
    return torch.from_numpy(np.asarray(mask, dtype=np.float32))[None]

class STLU_Mask4K:
    """The complete 4K outline of one layer.

    The split alpha is upsampled with the 4K reference as a guide (so the
    contour sits on the artwork's real edges), and the parts the split got
    wrong are rebuilt geometrically: an iris becomes a full circle from its
    visible arc, an eyewhite the whole eyeball opening, a lash/brow the dark
    stroke around the prior.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "reference": ("IMAGE",),
                "mode": (MASK4K_MODES, {"default": "auto"}),
                "radius": ("INT", {"default": core_mask4k.DEFAULT_GUIDE_RADIUS,
                                   "min": 1, "max": 64, "step": 1}),
                "eps": ("FLOAT", {"default": core_mask4k.DEFAULT_GUIDE_EPS,
                                  "min": 1e-6, "max": 1.0, "step": 1e-5}),
                "threshold": ("FLOAT", {"default": core_mask4k.DEFAULT_MASK_THRESHOLD,
                                        "min": 0.01, "max": 0.99, "step": 0.01}),
                "close_px": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                # Move the contour onto the drawn line (0 = keep the split's edge).
                "snap_px": ("INT", {"default": 4, "min": 0, "max": 32, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("part_mask", "preview", "info")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, plan, index, reference, mode="auto",
                radius=core_mask4k.DEFAULT_GUIDE_RADIUS,
                eps=core_mask4k.DEFAULT_GUIDE_EPS,
                threshold=core_mask4k.DEFAULT_MASK_THRESHOLD, close_px=0,
                snap_px=4):
        plan_data = _load_plan(plan)
        layer = _plan_layer(plan_data, index)
        source = plan_data.get("source") or {}
        canvas = (int(source.get("width") or 1024), int(source.get("height") or 1024))
        ref = _numpy_rgba(reference)
        alpha = _layer_alpha(layer)
        result = core_mask4k.mask_for_layer(
            layer.get("name") or "layer", alpha, layer.get("box"), ref, canvas,
            mode=mode, radius=int(radius), eps=float(eps),
            threshold=float(threshold), close_px=int(close_px),
            snap_px=int(snap_px))
        preview = torch.from_numpy(
            _mask_preview_image(result["guide"], result["mask"]).astype(np.float32) / 255.0)[None]
        info = json.dumps({"name": layer.get("name"), **result["stats"]}, ensure_ascii=False)
        print("[STLU] mask4k %r: %s" % (layer.get("name"), info), flush=True)
        return (_mask_to_tensor(result["mask"]), preview, info)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_Mask4KBatch:
    """Write the complete 4K mask of every selected layer (CPU, ~1 min).

    The masks are the constraint for the redraw stage: a model may only change
    pixels inside them, and they already contain the occluded parts the split
    did not cut (the top of an iris, the sclera behind it).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {"default": ""}),
                "reference": ("IMAGE",),
                "mode": (MASK4K_MODES, {"default": "auto"}),
                "radius": ("INT", {"default": core_mask4k.DEFAULT_GUIDE_RADIUS,
                                   "min": 1, "max": 64, "step": 1}),
                "eps": ("FLOAT", {"default": core_mask4k.DEFAULT_GUIDE_EPS,
                                  "min": 1e-6, "max": 1.0, "step": 1e-5}),
                "threshold": ("FLOAT", {"default": core_mask4k.DEFAULT_MASK_THRESHOLD,
                                        "min": 0.01, "max": 0.99, "step": 0.01}),
                "snap_px": ("INT", {"default": 4, "min": 0, "max": 32, "step": 1}),
                "out_dir": ("STRING", {"default": ""}),
                "only_selected": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "sheet_width": ("INT", {"default": 1800, "min": 256, "max": 8192, "step": 64}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("manifest_path", "preview_sheet")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, plan, reference, mode="auto",
                radius=core_mask4k.DEFAULT_GUIDE_RADIUS,
                eps=core_mask4k.DEFAULT_GUIDE_EPS,
                threshold=core_mask4k.DEFAULT_MASK_THRESHOLD,
                snap_px=4, out_dir="", only_selected=True, sheet_width=1800):
        plan_data = _load_plan(plan)
        source = plan_data.get("source") or {}
        canvas = (int(source.get("width") or 1024), int(source.get("height") or 1024))
        ref = _numpy_rgba(reference)
        out_dir = out_dir or plan_data.get("out_dir") or ""
        root = core_io.out_dir_abs(_output_root(), out_dir)
        mask_dir = os.path.join(root, "4K_masks")
        os.makedirs(mask_dir, exist_ok=True)

        entries = []
        previews = []
        layers = [l for l in plan_data.get("layers", [])
                  if l.get("file") and l.get("box") and not l.get("empty")
                  and (l.get("selected") or not bool(only_selected))]
        for i, layer in enumerate(layers):
            name = str(layer.get("name") or "layer")
            alpha = _layer_alpha(layer)
            if int(alpha.max()) <= 8:
                print("[STLU] mask4k %r: empty alpha, skipped" % name, flush=True)
                continue
            result = core_mask4k.mask_for_layer(
                name, alpha, layer.get("box"), ref, canvas,
                mode=mode, radius=int(radius), eps=float(eps),
                threshold=float(threshold), snap_px=int(snap_px))
            mask = result["mask"]
            safe = core_plan.safe_name(name)
            path = os.path.join(mask_dir, safe + ".png")
            value = np.where(mask, 255, 0).astype(np.uint8)
            # Opaque black/white PNG: unambiguous for both the model stage and
            # any image viewer (white = part).
            core_io.write_rgba(path, np.dstack([value, value, value,
                                                np.full_like(value, 255)]))
            entry = {"name": name, "z": layer.get("z"),
                     "mask": path, "mask_rel": os.path.relpath(path, root)}
            entry.update(result["stats"])
            entries.append(entry)
            previews.append(_mask_preview_image(result["guide"], mask))
            _notify("stlu.progress", {"name": name, "path": path,
                                      "index": i + 1, "total": len(layers)})
            print("[STLU] mask4k %r -> %s (%s, %d px)" %
                  (name, path, result["stats"]["kind"], result["stats"]["mask_px"]),
                  flush=True)

        manifest = {
            "version": 1,
            "source": plan_data.get("source"),
            "canvas": [canvas[0], canvas[1]],
            "reference_size": [int(ref.shape[1]), int(ref.shape[0])],
            "out_dir": out_dir,
            "layers": entries,
        }
        manifest_path = os.path.join(mask_dir, "masks.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=1)

        sheet = _contact_sheet(previews, int(sheet_width))
        print("[STLU] mask4k: %d masks -> %s" % (len(entries), manifest_path), flush=True)
        return (manifest_path, torch.from_numpy(sheet.astype(np.float32) / 255.0)[None])

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

def _contact_sheet(images, sheet_width):
    if not images:
        return np.full((64, 64, 3), 32, np.uint8)
    import cv2
    pad = 6
    cols = max(1, int(round(len(images) ** 0.5)))
    tile_w = max(64, (int(sheet_width) - pad * (cols + 1)) // cols)
    tiles = []
    for img in images:
        h, w = img.shape[:2]
        scale = tile_w / float(max(1, w))
        tile = np.asarray(img, np.uint8)
        if abs(scale - 1.0) > 1e-3:
            tile = cv2.resize(tile, (max(1, int(round(w * scale))),
                                     max(1, int(round(h * scale)))),
                              interpolation=cv2.INTER_AREA)
        tiles.append(tile)
    rows = (len(tiles) + cols - 1) // cols
    tw = max(t.shape[1] for t in tiles)
    th = max(t.shape[0] for t in tiles)
    out = np.full((rows * (th + pad) + pad, cols * (tw + pad) + pad, 3), 32, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = pad + r * (th + pad)
        x = pad + c * (tw + pad)
        out[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    return out


NODE_CLASS_MAPPINGS = {
    "STLU_Plan": STLU_Plan,
    "STLU_EditPartsBatch": STLU_EditPartsBatch,
    "STLU_UpscaleBatch": STLU_UpscaleBatch,
    "STLU_Mask4K": STLU_Mask4K,
    "STLU_Mask4KBatch": STLU_Mask4KBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STLU_Plan": "SeeThrough Upscale Plan",
    "STLU_UpscaleBatch": "SeeThrough Upscale Batch",
    "STLU_Mask4K": "SeeThrough 4K Part Mask (CPU)",
    "STLU_Mask4KBatch": "SeeThrough 4K Masks Batch (CPU)",
    "STLU_EditPartsBatch": "Live2D Edit Extract Parts (Batch)",
}

__all__ = [
    "STLU_Plan",
    "STLU_EditPartsBatch",
    "STLU_UpscaleBatch",
    "STLU_Mask4K",
    "STLU_Mask4KBatch",
]
