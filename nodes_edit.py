"""PSD per-layer edit nodes: task list -> Qwen edit -> (key | cut) -> PSD.

The siblings of ``nodes_psd``.  Upscaling keeps the source silhouette; editing
has to allow the model to *change* the part, so a task can pick one of two
tails:

* ``lock`` -- the model may repaint inside the crop but the source alpha is
  pasted back (``STLU_CutToLayer``); the part cannot grow or shrink.
* ``key``  -- the model paints the new part on a flat key background and the
  result is keyed out (``STLU_KeyCut``); the silhouette follows the edit, which
  is what adding a hairpin or changing a sleeve needs.

A task whose name is not in the PSD is a *new layer*: ``STLU_EditPrepare``
writes a plain key canvas at the box the user picked, and the edited result is
inserted at the requested depth when the PSD is rebuilt.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

if __package__:
    from . import core_io, core_plan, edit_parts, edit_plan, psd_upscale
else:  # standalone import used by the tests and tools
    import core_io
    import core_plan
    import edit_parts
    import edit_plan
    import psd_upscale

if __package__:
    from .nodes_shared import *  # noqa: F401,F403
else:
    from nodes_shared import *  # noqa: F401,F403

DEFAULT_EDIT_SIGMAS = (
    "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, "
    "0.414568, 0"
)

_GEOMETRY_KEYS = ("crop_size", "crop_box", "box", "size", "target", "out_file")


def _rgba_tensor(pixels: np.ndarray):
    return torch.from_numpy(
        np.ascontiguousarray(pixels[..., :3]).astype(np.float32) / 255.0)[None]


def _to_rgba8(image) -> np.ndarray:
    return np.clip(np.asarray(image[0], dtype=np.float32) * 255.0,
                   0, 255).astype(np.uint8)


def _to_rgb8(image) -> np.ndarray:
    pixels = _to_rgba8(image)
    return np.ascontiguousarray(pixels[..., :3]) if pixels.ndim == 3 else pixels


class STLU_EditPrepare:
    """Scan the finished PSD, cut one crop per edit task and write the plan."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "psd_path": ("STRING", {"default": ""}),
                "tasks": ("STRING", {"default": "", "multiline": True}),
                "factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 8.0,
                                     "step": 0.25}),
                "max_side": ("INT", {"default": psd_upscale.DEFAULT_MAX_SIDE,
                                     "min": 256, "max": 16384, "step": 64}),
            },
            "optional": {
                "out_dir": ("STRING", {"default": "SeeThrough/PsdEdit"}),
                "margin": ("FLOAT", {"default": psd_upscale.DEFAULT_MARGIN_RATIO,
                                     "min": 0.0, "max": 0.5, "step": 0.01}),
                "reference_path": ("STRING", {"default": ""}),
                "key_color": ("STRING", {"default": edit_plan.DEFAULT_KEY}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("plan_json", "preview", "info")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, psd_path, tasks, factor=1.0,
                max_side=psd_upscale.DEFAULT_MAX_SIDE,
                out_dir="SeeThrough/PsdEdit",
                margin=psd_upscale.DEFAULT_MARGIN_RATIO, reference_path="",
                key_color=edit_plan.DEFAULT_KEY):
        raw = str(psd_path or "").strip().strip('"')
        if not raw:
            raise ValueError("请选择成品 PSD 文件")
        path = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isfile(path):
            raise ValueError("PSD 文件不存在: %s" % path)

        parsed = edit_plan.parse_tasks(tasks)
        if not parsed:
            raise ValueError("没有编辑任务：请至少填写一个图层名和编辑指令")

        out_dir_rel = core_plan.normalize_out_dir(out_dir) or "SeeThrough/PsdEdit"
        stem = core_plan.safe_name(os.path.splitext(os.path.basename(path))[0])
        run_rel = os.path.join(out_dir_rel, stem)
        run_abs = core_io.out_dir_abs(_output_root(), run_rel)

        plan = edit_plan.prepare_edit_plan(
            path, run_abs, parsed, margin_ratio=float(margin),
            reference_path=str(reference_path or ""), factor=float(factor),
            max_side=int(max_side), key=str(key_color or edit_plan.DEFAULT_KEY),
        )

        preview = psd_upscale.preview_rgba(plan)
        core_io.write_rgba(os.path.join(run_abs, psd_upscale.PREVIEW_NAME),
                           preview)

        info = {
            "psd": path,
            "out_dir": run_rel,
            "plan": edit_plan.edit_plan_path(run_abs),
            "canvas": plan["canvas"],
            "count": plan["count"],
            "new_layers": [t["name"] for t in plan.get("tasks", []) if t.get("new")],
            "tasks": [t["name"] for t in plan.get("tasks", [])],
            "capped": psd_upscale.capped_report(plan),
            "reference": plan.get("reference"),
        }
        _notify("stlu.edit.plan", {
            "psd": path,
            "plan": edit_plan.edit_plan_path(run_abs),
            "canvas": plan["canvas"],
            "count": plan["count"],
            "out_dir": run_rel,
            "new_layers": info["new_layers"],
            "capped": info["capped"],
        })
        return (edit_plan.edit_plan_path(run_abs), _rgba_tensor(preview),
                json.dumps(info, ensure_ascii=False, indent=2))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_KeyCut:
    """Key a green-screen edit back into one target-sized layer PNG."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "geometry": ("STRING", {"default": "{}", "multiline": True}),
                "key_color": ("STRING", {"default": edit_plan.DEFAULT_KEY}),
                "tolerance": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0,
                                        "step": 1.0}),
                "softness": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0,
                                       "step": 1.0}),
                "protect": ("BOOLEAN", {"default": True}),
                "despill": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0,
                                      "step": 0.05}),
            },
            "optional": {
                "reference": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("layer_file", "layer_image")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, image, geometry, key_color=edit_plan.DEFAULT_KEY,
                tolerance=12.0, softness=6.0, protect=True, despill=0.65,
                reference=None):
        geo = json.loads(str(geometry or "{}"))
        missing = [k for k in _GEOMETRY_KEYS if k not in geo]
        if missing:
            raise ValueError("geometry 缺少字段: %s" % ", ".join(missing))

        edited = _to_rgb8(image)
        ref_rgb = _to_rgb8(reference) if reference is not None else None
        rgba = edit_parts.key_out(
            edited, key=str(key_color or edit_plan.DEFAULT_KEY),
            tolerance=float(tolerance), softness=float(softness),
            reference_rgb=ref_rgb, protect=bool(protect),
            despill_strength=float(despill),
        )
        layer = edit_plan.restore_keyed(
            rgba, geo["crop_size"], geo["crop_box"], geo["box"],
            geo["size"], geo["target"])
        out_file = str(geo["out_file"])
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        core_io.write_rgba(out_file, layer)
        _notify("stlu.edit.layer", {
            "file": out_file,
            "size": [int(layer.shape[1]), int(layer.shape[0])],
            "coverage": round(float(edit_parts.coverage(layer)), 4),
            "mode": "key",
        })
        return (out_file, _rgba_tensor(layer))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_MattingCut:
    """Turn any matting node's MASK into an RGBA layer (standalone tool)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "out_file": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("layer_file", "layer_image")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, image, mask, out_file=""):
        pixels = _to_rgba8(image)
        alpha = np.asarray(mask[0], dtype=np.float32) if mask is not None else None
        if alpha is None:
            raise ValueError("缺少 mask")
        if alpha.ndim != 2:
            alpha = alpha.reshape(alpha.shape[-2:])
        if alpha.shape != pixels.shape[:2]:
            raise ValueError("mask 尺寸 %r 与 image 尺寸 %r 不一致"
                             % (alpha.shape, pixels.shape[:2]))
        layer = np.dstack([
            pixels[..., :3],
            np.clip(np.rint((1.0 - alpha) * 255.0), 0, 255).astype(np.uint8),
        ])
        out_file = str(out_file or "").strip()
        if out_file:
            os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
            core_io.write_rgba(out_file, layer)
            _notify("stlu.edit.layer", {
                "file": out_file,
                "size": [int(layer.shape[1]), int(layer.shape[0])],
                "coverage": round(float(edit_parts.coverage(layer)), 4),
                "mode": "matting",
            })
        return (out_file, _rgba_tensor(layer))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_LayerEdit:
    """Clone the Qwen edit subgraph once per edit task."""

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
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF,
                                 "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0,
                                  "step": 0.5}),
                "scheduler": ("STRING", {"default": "euler"}),
                "sigmas": ("STRING", {"default": DEFAULT_EDIT_SIGMAS,
                                      "multiline": True}),
                "key_color": ("STRING", {"default": edit_plan.DEFAULT_KEY}),
                "tolerance": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0,
                                        "step": 1.0}),
                "softness": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0,
                                       "step": 1.0}),
                "protect": ("BOOLEAN", {"default": True}),
                "despill": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0,
                                      "step": 0.05}),
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

    def execute(self, plan_json, model, clip, vae, workflow_json="",
                negative_prompt="", seed=42, cfg=1.0, scheduler="euler",
                sigmas=DEFAULT_EDIT_SIGMAS, key_color=edit_plan.DEFAULT_KEY,
                tolerance=12.0, softness=6.0, protect=True, despill=0.65,
                psd_filename="", dynprompt=None, unique_id=None):
        plan = edit_plan.load_edit_plan(str(plan_json or ""))
        run_abs = str(plan.get("out_dir") or "").strip()
        if not run_abs:
            raise ValueError("计划文件缺少 out_dir")
        jobs = {job["stem"]: job for job in (plan.get("crops") or [])}
        tasks = edit_plan.task_by_stem(plan)
        layers = edit_plan.edit_layers(plan)
        if not layers:
            raise ValueError("计划里没有编辑任务")

        out_dir_rel = os.path.relpath(run_abs, _output_root()).replace("\\", "/")
        if out_dir_rel.startswith(".."):
            out_dir_rel = os.path.basename(run_abs)

        key = str(key_color or edit_plan.DEFAULT_KEY)
        parts = []
        for layer in layers:
            stem = str(layer.get("stem"))
            job = jobs.get(stem)
            if job is None:
                raise ValueError("图层 %r 缺少裁剪任务，请重新运行编辑准备"
                                 % layer.get("name"))
            edit = layer.get("edit") or {}
            task_info = tasks.get(stem) or {}
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
                "prompt": str(edit.get("prompt") or task_info.get("prompt") or ""),
                "crop_file": str(job["crop_file"]),
                "ref_file": str(job.get("ref_file") or ""),
                "width": work_w,
                "height": work_h,
                "resolution": max(work_w, work_h),
                "geometry": geometry,
                "silhouette": str(edit.get("silhouette")
                                  or edit_plan.SILHOUETTE_LOCK),
                # Protect the key against the part's own colour only when the
                # crop shows the original part; a new layer's crop is the key.
                "key_reference": bool(job.get("key_reference", True)),
            })

        if dynprompt is None or unique_id is None:
            raise RuntimeError("STLU_LayerEdit needs dynprompt/unique_id "
                               "hidden inputs")
        node_inputs = dynprompt.get_node(unique_id).get("inputs", {})

        def _link(name):
            value = node_inputs.get(name)
            if not isinstance(value, list) or len(value) != 2:
                raise RuntimeError("STLU_LayerEdit input %r is not linked" % name)
            return [str(value[0]), int(value[1])]

        external = {k: _link(k) for k in ("model", "clip", "vae")}

        wf_path = str(workflow_json or "").strip()
        if not wf_path:
            try:
                wf_path = edit_parts.default_template_path()
            except Exception:
                wf_path = ""
        if not wf_path or not os.path.isfile(wf_path):
            raise FileNotFoundError(
                "找不到图编辑工作流模板（请在节点里选择 Qwen 编辑工作流）: %s"
                % (wf_path or "<empty>"))

        if __package__:
            from . import edit_expand
        else:
            import edit_expand
        expand, _psd, _saves = edit_expand.build_expand(
            workflow_path=wf_path,
            parts=parts,
            external=external,
            seed=int(seed),
            out_dir_rel=out_dir_rel,
            layers_json_path="",
            psd_name=str(psd_filename or ""),
            psd_plan=edit_plan.edit_plan_path(run_abs),
            crop_mode=True,
            key=key,
            tolerance=float(tolerance),
            softness=float(softness),
            protect=bool(protect),
            despill=float(despill),
            negative_prompt=str(negative_prompt or ""),
            cfg=float(cfg),
            resolution=0,
            scheduler=str(scheduler),
            sigmas=str(sigmas),
            width=0,
            height=0,
        )

        run_name = str(psd_filename or "").strip()
        if not run_name:
            stem = core_plan.safe_name(
                os.path.splitext(os.path.basename(plan.get("psd") or "psd"))[0])
            run_name = "%s_edited" % stem
        psd_path = os.path.join(run_abs, core_plan.safe_name(run_name) + ".psd")

        _notify("stlu.edit.start", {
            "total": len(parts),
            "parts": [p["name"] for p in parts],
            "modes": [p["silhouette"] for p in parts],
            "out_dir": out_dir_rel,
            "workflow": wf_path,
        })

        report = {
            "total": len(parts),
            "parts": [p["name"] for p in parts],
            "modes": [p["silhouette"] for p in parts],
            "canvas": plan["canvas"],
            "out_dir": out_dir_rel,
            "psd": psd_path,
            "workflow": wf_path,
            "key": key,
            "capped": psd_upscale.capped_report(plan),
        }
        return {
            "result": (psd_path, json.dumps(report, ensure_ascii=False, indent=2)),
            "expand": expand,
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_EditAssemble:
    """Rebuild the layered PSD after every edit task has been cut."""

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
        plan = edit_plan.load_edit_plan(str(plan_json or ""))
        run_abs = str(plan.get("out_dir") or "").strip()
        if not run_abs:
            raise ValueError("计划文件缺少 out_dir")
        name = core_plan.safe_name(str(psd_filename or "").strip())
        if not name:
            stem = core_plan.safe_name(
                os.path.splitext(os.path.basename(plan.get("psd") or "psd"))[0])
            name = "%s_edited" % stem
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
            "dir": out_dir_rel,
            "name": name,
            "canvas": report["canvas"],
            "edited": report["upscaled"],
            "fallback": report["fallback"],
            "new_layers": [t["name"] for t in plan.get("tasks", [])
                           if t.get("new")],
        }
        _notify("stlu.edit.assembled", payload)
        return (out_psd, json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "STLU_EditPrepare": STLU_EditPrepare,
    "STLU_KeyCut": STLU_KeyCut,
    "STLU_MattingCut": STLU_MattingCut,
    "STLU_LayerEdit": STLU_LayerEdit,
    "STLU_EditAssemble": STLU_EditAssemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STLU_EditPrepare": "STLU ① 编辑准备（PSD 切层+任务）",
    "STLU_KeyCut": "STLU 绿幕裁回（重抠→图层）",
    "STLU_MattingCut": "STLU 本地抠图接层（MASK→RGBA）",
    "STLU_LayerEdit": "STLU ② 逐层编辑（Qwen 图编辑）",
    "STLU_EditAssemble": "STLU ③ 重组编辑 PSD",
}
