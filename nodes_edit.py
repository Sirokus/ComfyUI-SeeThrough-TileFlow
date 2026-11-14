"""PSD per-layer edit nodes: task list -> Qwen edit -> PSD.

The siblings of ``nodes_psd``.  Qwen Image 2.1 decodes RGBA, so the tail of the
edit chain is ``STLU_CutToLayer``: the model's own alpha channel is the
silhouette, and there is no chroma key or source lock anywhere.

A task whose name is not in the PSD is a *new layer*: ``STLU_EditPrepare``
writes a transparent canvas at the box the user picked, and the edited result is
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

def _rel_out(path) -> str:
    """Output-relative, forward-slash path for thumbnails ("" if outside)."""
    raw = str(path or "").strip()
    if not raw:
        return ""
    try:
        rel = os.path.relpath(os.path.abspath(raw), _output_root())
    except ValueError:
        return ""
    rel = rel.replace("\\", "/")
    return "" if rel.startswith("..") else rel


def _rgba_tensor(pixels: np.ndarray):
    return torch.from_numpy(
        np.ascontiguousarray(pixels[..., :3]).astype(np.float32) / 255.0)[None]


def _to_rgba8(image) -> np.ndarray:
    return np.clip(np.asarray(image[0], dtype=np.float32) * 255.0,
                   0, 255).astype(np.uint8)


class STLU_EditPrepare:
    """Scan the finished PSD, cut one crop per edit task and write the plan."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "psd_path": ("STRING", {"default": ""}),
                "tasks": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "out_dir": ("STRING", {"default": "SeeThrough/PsdEdit"}),
                "psd_name": ("STRING", {"default": ""}),
                # ① is the single source of the sampling seed; ② reads it from
                # the plan and steps it by one per part -- ② has no own seed
                # either way.
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF,
                                 "step": 1}),
                # Appended last so older workflows keep their widget order.
                "sub_dir": ("STRING", {"default": "",
                    "tooltip": "在输出根目录下的子目录（可空）；成品写到 "
                               "<根>/<子目录>/<PSD 名称>/<PSD 名称>.psd。"}),
                # Global positive prompt: prepended to every task's own
                # instruction, so shared wording (transparency, style) is
                # written once instead of repeated in every row.
                "prompt": ("STRING", {"default": "", "multiline": True}),
                # A negative prompt only changes the result when CFG > 1;
                # the default 1.0 keeps the current single-pass speed.
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "cfg": ("FLOAT", {"default": 1.0,
                                  "min": 0.0, "max": 8.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("plan_json", "preview", "info")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, psd_path, tasks, out_dir="SeeThrough/PsdEdit",
                psd_name="", seed=42, sub_dir="", prompt="",
                negative_prompt="", cfg=1.0):
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
        final_name = (str(psd_name or "").strip()
                      or ("%s_edited" % stem))
        # <根>/<子目录>/<名称>/<名称>.psd -- the run folder is named after the
        # final PSD, so the panel can preview the exact outcome path.
        sub_rel = core_plan.safe_sub_dir(sub_dir)
        run_rel = os.path.join(out_dir_rel, sub_rel,
                               core_plan.safe_name(final_name))
        run_abs = core_io.out_dir_abs(_output_root(), run_rel)

        final_name = (str(psd_name or "").strip()
                      or ("%s_edited" % stem))
        margin = psd_upscale.DEFAULT_MARGIN_RATIO
        settings = {
            "margin": float(margin),
            "psd_name": final_name,
            "seed": int(seed),
            "prompt": str(prompt or "").strip(),
            "negative_prompt": str(negative_prompt or "").strip(),
            "cfg": float(cfg),
            "seed_mode": "step",
        }
        plan = edit_plan.prepare_edit_plan(
            path, run_abs, parsed, margin_ratio=float(margin),
            reference_path="", factor=1.0,
            max_side=psd_upscale.DEFAULT_MAX_SIDE,
            settings=settings,
        )

        preview = psd_upscale.preview_rgba(plan)
        core_io.write_rgba(os.path.join(run_abs, psd_upscale.PREVIEW_NAME),
                           preview)

        info = {
            "psd": path,
            "out_dir": run_rel,
            "plan": edit_plan.edit_plan_path(run_abs),
            "psd_name": final_name,
            "canvas": plan["canvas"],
            "count": plan["count"],
            "new_layers": [t["name"] for t in plan.get("tasks", []) if t.get("new")],
            "tasks": [t["name"] for t in plan.get("tasks", [])],
            "capped": psd_upscale.capped_report(plan),
            "reference": plan.get("reference"),
            "seed_mode": "step",
        }
        # Per-task crop thumbnails: the panel shows each task's source box so
        # the user sees what will be repainted before the GPU starts.
        crop_by_stem = {str(job.get("stem")): job for job in (plan.get("crops") or [])}
        crops = []
        for task in plan.get("tasks", []):
            job = crop_by_stem.get(str(task.get("stem")))
            crops.append({
                "name": str(task.get("name") or ""),
                "crop": _rel_out(job.get("crop_file")) if job else "",
                "new": bool(task.get("new")),
                # The panel marks each row 已生成 / 未生成 from this flag.
                "exists": bool(task.get("exists")),
                "force": bool(task.get("force")),
            })
        _notify("stlu.edit.plan", {
            "psd": path,
            "plan": edit_plan.edit_plan_path(run_abs),
            "canvas": plan["canvas"],
            "count": plan["count"],
            "out_dir": run_rel,
            "psd_name": final_name,
            "new_layers": info["new_layers"],
            "capped": info["capped"],
            "crops": crops,
        })
        return (edit_plan.edit_plan_path(run_abs), _rgba_tensor(preview),
                json.dumps(info, ensure_ascii=False, indent=2))

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
        templates = edit_parts.list_edit_templates()
        return {
            "required": {
                "plan_json": ("STRING", {"default": ""}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
            "optional": {
                # Which edit-model workflow under templates/edit/ to clone.
                # Each template may carry a <name>.adapter.json describing
                # where its loaders / prompt / seed nodes live.
                "edit_template": (templates, {"default": templates[0]}),
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

    def execute(self, plan_json, model, clip, vae, edit_template="",
                dynprompt=None, unique_id=None):
        plan = edit_plan.load_edit_plan(str(plan_json or ""))
        run_abs = str(plan.get("out_dir") or "").strip()
        if not run_abs:
            raise ValueError("计划文件缺少 out_dir")
        jobs = {job["stem"]: job for job in (plan.get("crops") or [])}
        tasks = edit_plan.task_by_stem(plan)
        layers = edit_plan.edit_layers(plan)
        if not layers:
            raise ValueError("计划里没有编辑任务")
        # 已生成的图层不再重跑，除非 ① 的任务上勾了「强制重跑」。
        pending, skipped = edit_plan.split_pending(layers)
        pending_ids = {id(layer) for layer in pending}

        out_dir_rel = os.path.relpath(run_abs, _output_root()).replace("\\", "/")
        if out_dir_rel.startswith(".."):
            out_dir_rel = os.path.basename(run_abs)

        # ① is the single source of the output/sampling settings; the plan JSON
        # carries them, with the defaults below only as a fallback for plans
        # written before the field existed.
        settings = plan.get("settings") or {}
        psd_filename = str(settings.get("psd_name") or "")
        seed = int(settings.get("seed", 42))
        # Fixed behaviour: seed + task index.  Older plans that still carry
        # seed_mode "fixed" are stepped too.
        global_prompt = str(settings.get("prompt") or "").strip()
        negative_prompt = str(settings.get("negative_prompt") or "").strip()
        parts = []
        for layer in layers:
            if id(layer) not in pending_ids:
                continue
            stem = str(layer.get("stem"))
            job = jobs.get(stem)
            if job is None:
                raise ValueError("图层 %r 缺少裁剪任务，请重新运行编辑准备"
                                 % layer.get("name"))
            edit = layer.get("edit") or {}
            task_info = tasks.get(stem) or {}
            # The global positive prompt leads; the task's own instruction
            # follows so per-layer wording still applies.
            task_prompt = str(edit.get("prompt") or task_info.get("prompt") or "")
            if global_prompt:
                task_prompt = (global_prompt + "\n" + task_prompt).strip()
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
                # Carried so the cut nodes can report a per-layer before/after
                # preview (name + crop relative path) back to the panel.
                "name": str(layer["name"]),
                "crop": _rel_out(job["crop_file"]),
            }
            parts.append({
                "name": str(layer["name"]),
                "prompt": task_prompt,
                "crop_file": str(job["crop_file"]),
                "ref_file": str(job.get("ref_file") or ""),
                # Per-task pattern/style references, <image3> onwards when the
                # global reference above owns <image2>.
                "ref_files": [str(p) for p in (job.get("ref_files") or [])
                              if str(p or "").strip()],
                "width": work_w,
                "height": work_h,
                "resolution": max(work_w, work_h),
                "geometry": geometry,
                # edit_expand adds this offset to the panel seed: seed +1 per
                # task, always.
                "seed_index": len(parts),
            })

        # Nothing left to sample: still build the expansion so its PSD
        # assembler runs on the already generated layers, but skip the model
        # links -- there is no subgraph to feed them to.
        external = {}
        if parts:
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

        # The template (chosen above) decides the model stack and its sampling;
        # only the prompt and the reference images vary
        # per part.
        try:
            wf_path = edit_parts.edit_template_path(edit_template)
            adapter = edit_parts.edit_template_adapter(edit_template)
        except Exception:
            wf_path = ""
            adapter = {}
        if not wf_path or not os.path.isfile(wf_path):
            raise FileNotFoundError(
                "找不到图编辑工作流模板: %s" % (wf_path or "<empty>"))

        if __package__:
            from . import edit_expand
        else:
            import edit_expand
        # Sampling knobs belong to the model, so the template declares them;
        # the Qwen 2.1 values are the fallback when an adapter omits them.
        sampling = adapter.get("sampling") or {}
        # ①'s CFG wins over the template's; plans written before the field
        # existed fall back to the adapter value.
        try:
            cfg = float(settings.get("cfg"))
        except (TypeError, ValueError):
            cfg = float(sampling.get("cfg", 1.0))
        expand, _psd, _saves = edit_expand.build_expand(
            workflow_path=wf_path,
            adapter=adapter,
            parts=parts,
            external=external,
            seed=int(seed),
            out_dir_rel=out_dir_rel,
            layers_json_path="",
            psd_name=str(psd_filename or ""),
            psd_plan=edit_plan.edit_plan_path(run_abs),
            crop_mode=True,
            negative_prompt=negative_prompt,
            cfg=cfg,
            resolution=0,
            scheduler=str(sampling.get("scheduler") or "euler"),
            sigmas=str(sampling.get("sigmas") or DEFAULT_EDIT_SIGMAS),
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
            "skipped": [str(layer.get("name") or "") for layer in skipped],
            "items": [{
                "name": p["name"],
                "crop": _rel_out(p["crop_file"]),
            } for p in parts],
            "out_dir": out_dir_rel,
            "workflow": wf_path,
        })

        report = {
            "total": len(parts) + len(skipped),
            "edited": len(parts),
            "parts": [p["name"] for p in parts],
            "skipped": [str(layer.get("name") or "") for layer in skipped],
            "canvas": plan["canvas"],
            "out_dir": out_dir_rel,
            "psd": psd_path,
            "workflow": wf_path,
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
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("psd_path", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, plan_json, trigger=None):
        plan = edit_plan.load_edit_plan(str(plan_json or ""))
        run_abs = str(plan.get("out_dir") or "").strip()
        if not run_abs:
            raise ValueError("计划文件缺少 out_dir")
        # The output name is whatever ① wrote into the plan; deriving one from
        # the source PSD is only a fallback for old plans.
        raw_name = str((plan.get("settings") or {}).get("psd_name") or "").strip()
        name = core_plan.safe_name(raw_name) if raw_name else ""
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
    "STLU_MattingCut": STLU_MattingCut,
    "STLU_LayerEdit": STLU_LayerEdit,
    "STLU_EditAssemble": STLU_EditAssemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STLU_EditPrepare": "STLU ① 编辑准备（PSD 切层+任务）",
    "STLU_MattingCut": "STLU 本地抠图接层（MASK→RGBA）",
    "STLU_LayerEdit": "STLU ② 逐层编辑（Qwen 图编辑）",
    "STLU_EditAssemble": "STLU ③ 重组编辑 PSD",
}
