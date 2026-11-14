"""Minimal PSD upscale: three nodes, one clear split.

Mirrors the psd-edit flow -- user input lives on its own node:

* ``STLU_QwenEditLoader``    -- UNET + CLIP + VAE in one node; ``clip_type``
  picks the text-encoder family and drives the Qwen cache option.
* ``STLU_PsdUpscalePrepare`` -- every user knob (PSD, layer ticks, optional
  reference, factor, sizes, crop margin, seed, output dir/name, prompt).  It
  writes the on-disk plan and hands ``plan_json`` downstream.
* ``STLU_PsdUpscaleMin``     -- model + plan in, layered upscaled PSD out; it
  only reads the plan and expands the edit subgraph.

Sampling details (denoise, sigmas, scheduler, cfg, template) stay pinned to
the values that pipeline used when it worked.
"""

from __future__ import annotations

import os
import sys

if __package__:
    from . import core_io, core_plan, edit_expand, edit_parts, psd_upscale
    from .nodes_psd import DEFAULT_PSD_SIGMAS
    from .nodes_shared import CATEGORY, _notify, _output_root
else:  # standalone import used by the tests and tools
    import core_io
    import core_plan
    import edit_expand
    import edit_parts
    import psd_upscale
    from nodes_psd import DEFAULT_PSD_SIGMAS
    from nodes_shared import CATEGORY, _notify, _output_root

OUT_SUBDIR = "SeeThrough/PsdUpscale"
MARGIN = 0.08
WORK_MAX_SIDE = psd_upscale.DEFAULT_WORK_MAX_SIDE
SEED = 42
# Qwen Image Edit 2.1 is a *reference-latent* full regeneration: the template
# feeds a zero-filled latent (sized to the reference) into the sampler and the
# source image only enters through the conditioning's reference_latents.  So
# the schedule must start at sigma 1.0; scaling it by 0.45 starts the sampler
# half-way through denoising an empty latent, which decodes to the brown noise
# every layer used to come back as.
DENOISE = 1.0
TEMPLATE = "qwen_image_edit_2.1"
# Unlike the full pipeline's prompt this one explicitly *allows* repair: the
# sources are AI-generated, so the job is upscale + clean up, not a verbatim
# resample.  Silhouette and composition stay fixed by the geometry: the crop
# travels as straight RGBA into the transparent VAE, so the alpha the model
# decodes is the outline it was shown.
# The transparency demand is stated twice on purpose (opening + closing): the
# model still drifts towards an opaque card when the wording is loose.
DEFAULT_PROMPT = (
    "输出一张带透明背景、带透明通道的 PNG 图层图片。"
    "把这张图放大到更高分辨率，保持原有造型、配色、构图和轮廓不变；"
    "修复 AI 生成留下的瑕疵：去掉噪点与杂色，理清模糊、锯齿或断裂的边缘，"
    "让发丝根根分明、线条干净连续；补充清晰的纹理细节，不要添加或删除任何元素。"
    "背景必须完全透明，不要白底、边框、阴影或补齐背景，只保留这一层的像素。"
)


def _core_nodes():
    """ComfyUI's root ``nodes`` module (loaders live there, not in this pack)."""
    core = sys.modules.get("nodes")
    if core is None or not hasattr(core, "UNETLoader"):
        raise RuntimeError("找不到 ComfyUI 核心 nodes 模块，加载器节点需在 "
                           "ComfyUI 里运行")
    return core


class STLU_QwenEditLoader:
    """Load the Qwen edit stack (UNET + cache + CLIP + VAE) in one node."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        try:
            # ComfyUI's own text-encoder type list, so the dropdown matches.
            clip_types = _core_nodes().CLIPLoader.INPUT_TYPES()["required"]["type"][0]
        except Exception:  # noqa: BLE001 - registration must not depend on core
            clip_types = ["qwen_image"]
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                # Appended last so older workflows keep their widget order;
                # the Qwen default keeps them behaving exactly as before.
                "clip_type": (list(clip_types), {"default": "qwen_image"}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, unet_name, clip_name, vae_name, clip_type="qwen_image"):
        core = _core_nodes()
        model = core.UNETLoader().load_unet(unet_name, "default")[0]
        if str(clip_type).startswith("qwen_image"):
            model = model.clone()
            model.model_options.setdefault("transformer_options", {})[
                "qwen_image21_cache"] = {"device": "auto", "dtype": "default"}
        clip = core.CLIPLoader().load_clip(clip_name, clip_type, "default")[0]
        vae = core.VAELoader().load_vae(vae_name)[0]
        return (model, clip, vae)


class STLU_PsdUpscalePrepare:
    """User input -> on-disk plan -- the psd-upscale counterpart of EditPrepare."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "psd_path": ("STRING", {"default": ""}),
                # Hidden widget: the panel offers choose/open/clear buttons.
                # Empty means "no reference" -- identical to the old behaviour.
                "reference_path": ("STRING", {"default": ""}),
                "factor": ("FLOAT", {"default": psd_upscale.DEFAULT_FACTOR,
                                     "min": 0.5, "max": 8.0, "step": 0.25}),
                "max_side": ("INT", {"default": psd_upscale.DEFAULT_MAX_SIDE,
                                     "min": 256, "max": 16384, "step": 64}),
                "work_max_side": ("INT", {"default": WORK_MAX_SIDE,
                                          "min": 0, "max": 4096, "step": 64}),
                "margin": ("FLOAT", {"default": MARGIN,
                                     "min": 0.0, "max": 0.5, "step": 0.01}),
                "seed": ("INT", {"default": SEED, "min": 0,
                                 "max": 0xFFFFFFFF, "step": 1}),
                # Hidden widgets: the panel gives them browse/text controls.
                "out_dir": ("STRING", {"default": OUT_SUBDIR}),
                "psd_name": ("STRING", {"default": ""}),
                # Driven by the graphical layer table; ``all`` / comma names /
                # ``-name`` exclusions, same syntax as psd_upscale.parse_selection.
                "select": ("STRING", {"default": "all"}),
                "prompt": ("STRING", {"default": DEFAULT_PROMPT,
                                      "multiline": True}),
                "sub_dir": ("STRING", {"default": "",
                    "tooltip": "在输出根目录下的子目录（可空）；成品写到 "
                               "<根>/<子目录>/<PSD 名称>/<PSD 名称>.psd。"}),
                # Appended last so older workflows keep their widget order.
                # A negative prompt only changes the result when CFG > 1 (the
                # guider subtracts the negative branch); the default 1.0 keeps
                # today's single-pass quality and speed untouched.
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "cfg": ("FLOAT", {"default": 1.0,
                                  "min": 0.0, "max": 8.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("plan_json", "info")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, psd_path, reference_path="", factor=psd_upscale.DEFAULT_FACTOR,
                max_side=psd_upscale.DEFAULT_MAX_SIDE, work_max_side=WORK_MAX_SIDE,
                margin=MARGIN, seed=SEED, out_dir=OUT_SUBDIR, psd_name="",
                select="all", prompt=DEFAULT_PROMPT,
                sub_dir="", negative_prompt="", cfg=1.0):
        raw = str(psd_path or "").strip().strip('"')
        if not raw:
            raise ValueError("请选择成品 PSD 文件")
        path = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isfile(path):
            raise ValueError("PSD 文件不存在: %s" % path)

        work_max_side = int(work_max_side or 0)
        seed = int(seed)
        psd_name = str(psd_name or "").strip()
        root_dir = core_plan.normalize_out_dir(out_dir) or OUT_SUBDIR
        stem = core_plan.safe_name(os.path.splitext(os.path.basename(path))[0])
        # <根>/<子目录>/<源文件名>/ -- the run folder tracks the source PSD so
        # re-runs land together; ``psd_name`` only renames the final .psd
        # (resolved by resolve_psd_name), matching the historical layout.
        sub_rel = core_plan.safe_sub_dir(sub_dir)
        run_rel = os.path.join(root_dir, sub_rel, stem)
        run_abs = core_io.out_dir_abs(_output_root(), run_rel)

        plan = psd_upscale.prepare_plan(path, run_abs, factor=float(factor),
                                        max_side=int(max_side), select=select)
        if not plan["count"]:
            raise ValueError("没有选中的图层（select=%r）" % (select,))
        jobs = psd_upscale.write_crops(plan, margin_ratio=float(margin))
        plan["crops"] = jobs
        ref_path = str(reference_path or "").strip().strip('"')
        if ref_path:
            psd_upscale.write_references(plan, ref_path)
        text = str(prompt or "").strip() or DEFAULT_PROMPT
        settings = plan.setdefault("settings", {})
        settings.update({
            "seed": seed, "work_max_side": work_max_side, "psd_name": psd_name,
            "prompt": text, "select": str(select or "all").strip() or "all",
            "margin": float(margin), "reference_path": ref_path,
            "negative_prompt": str(negative_prompt or "").strip(),
            "cfg": float(cfg),
            "seed_mode": "step",
        })
        plan_file = psd_upscale.write_plan(run_abs, plan)

        _notify("stlu.psd.plan", {
            "psd": path,
            "plan": plan_file,
            "canvas": plan["canvas"],
            "count": plan["count"],
            "out_dir": run_rel,
            "capped": psd_upscale.capped_report(plan),
            "psd_name": psd_upscale.resolve_psd_name(plan, psd_name),
        })

        info = "\n".join([
            "PSD: %s" % path,
            "选中 %d 层，画布 %dx%d" % (plan["count"], plan["canvas"][0],
                                       plan["canvas"][1]),
            "倍率 %g，单层上限 %d，输入上限 %d"
            % (float(factor), int(max_side), work_max_side),
            "裁剪外扩 %g，种子 %d" % (float(margin), seed),
            "种子: %d（每层 +1）" % int(seed),
            "选层: %s" % (str(select or "all").strip() or "all"),
            "负向提示词: %s，CFG %g"
            % ("已填写" if str(negative_prompt or "").strip() else "未设置",
               float(cfg)),
            "参考原图: %s" % (os.path.basename(ref_path) if ref_path else "未设置"),
            psd_upscale.capped_report(plan),
            "计划: %s" % plan_file,
        ])
        return (plan_file, info)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_PsdUpscaleMin:
    """Plan + model in, layered upscaled PSD out -- expand only, no input UI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                # Plan written by STLU_PsdUpscalePrepare (or any plan JSON).
                "plan_json": ("STRING", {"default": ""}),
                # Which edit-model workflow under templates/edit/ to clone.
                # Each template may carry a <name>.adapter.json with its
                # loaders / encoder nodes / sampling knobs.
                "edit_template": (edit_parts.list_edit_templates(),
                                  {"default": TEMPLATE}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("psd_path", "info")
    FUNCTION = "execute"
    # This is the graph's only output node: ComfyUI refuses a prompt without
    # one, and the assembled PSD is the result -- no preview image needed.
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def execute(self, model, clip, vae, plan_json, edit_template="",
                dynprompt=None, unique_id=None):
        ref = str(plan_json or "").strip().strip('"')
        if not ref:
            raise ValueError("请先连接『PSD 放大准备』的 plan_json")
        plan = psd_upscale.load_plan(ref)
        # ``ref`` is the plan JSON path the prepare node returned; tolerate a
        # layer folder too by resolving it to <folder>/<PLAN_NAME>.
        plan_file = psd_upscale.plan_path(ref) if os.path.isdir(ref) else ref
        run_abs = os.path.dirname(os.path.abspath(plan_file))
        if dynprompt is None or unique_id is None:
            raise RuntimeError("STLU_PsdUpscaleMin needs dynprompt/unique_id "
                               "hidden inputs")

        settings = plan.get("settings") or {}
        seed = int(settings.get("seed", SEED))
        work_max_side = int(settings.get("work_max_side", WORK_MAX_SIDE))
        factor = float(plan.get("factor") or 1.0)
        text = str(settings.get("prompt") or "").strip() or DEFAULT_PROMPT
        psd_name = str(settings.get("psd_name") or "")
        negative_prompt = str(settings.get("negative_prompt") or "").strip()
        cfg_override = settings.get("cfg")
        jobs = plan.get("crops") or []
        if not jobs:
            jobs = psd_upscale.write_crops(
                plan, margin_ratio=float(settings.get("margin", MARGIN)))
            plan["crops"] = jobs

        job_by_stem = {job["stem"]: job for job in jobs}
        parts = []
        for index, layer in enumerate(psd_upscale.selected_layers(plan)):
            job = job_by_stem.get(layer.get("stem"))
            if job is None:
                raise ValueError("图层 %r 缺少裁剪任务" % layer.get("name"))
            crop_size = [int(v) for v in job["crop_size"]]
            size = [int(v) for v in layer["size"]]
            target = [int(v) for v in layer["target"]]
            work_w, work_h = psd_upscale.input_work_size(
                crop_size, size, factor, work_max_side)
            parts.append({
                "name": str(layer["name"]),
                "prompt": text,
                "crop_file": str(job["crop_file"]),
                "ref_file": str(job.get("ref_file") or ""),
                "width": work_w,
                "height": work_h,
                "resolution": max(work_w, work_h),
                # edit_expand adds this offset to the panel seed: seed +1 per
                # layer, always.
                "seed_index": index,
                # The decoded RGBA carries the model's own alpha; the cut
                # resamples it straight onto the layer (no separate mask).
                "geometry": {
                    "crop_size": crop_size,
                    "crop_box": [int(v) for v in job["crop_box"]],
                    "box": [int(v) for v in layer["box"]],
                    "size": size,
                    "target": target,
                    "out_file": str(layer["out_file"]),
                    "file": str(layer.get("file") or ""),
                    "plan_json": plan_file,
                },
            })

        node_inputs = dynprompt.get_node(unique_id).get("inputs", {})

        def _link(name):
            value = node_inputs.get(name)
            if not isinstance(value, list) or len(value) != 2:
                raise RuntimeError("STLU_PsdUpscaleMin 输入 %r 没有连线" % name)
            return [str(value[0]), int(value[1])]

        external = {k: _link(k) for k in ("model", "clip", "vae")}

        template = str(edit_template or "").strip() or TEMPLATE
        wf_path = edit_parts.edit_template_path(template)
        adapter = edit_parts.edit_template_adapter(template)
        if not wf_path or not os.path.isfile(wf_path):
            raise FileNotFoundError("找不到图编辑模板: %s" % wf_path)

        out_dir_rel = os.path.relpath(run_abs, _output_root()).replace("\\", "/")
        if out_dir_rel.startswith(".."):
            out_dir_rel = os.path.basename(run_abs)
        run_name = psd_upscale.resolve_psd_name(plan, psd_name)
        psd_out = os.path.join(run_abs, run_name + ".psd")
        # Sampling knobs belong to the model, so the template declares them;
        # the Qwen 2.1 values are the fallback when an adapter omits them.
        sampling = adapter.get("sampling") or {}
        denoise = float(sampling.get("denoise", DENOISE))
        scaled_sigmas = psd_upscale.scale_sigmas(
            str(sampling.get("sigmas") or DEFAULT_PSD_SIGMAS), denoise)
        # The panel's CFG wins over the template's (plans written before the
        # field existed fall back to the adapter value).
        try:
            cfg = float(cfg_override)
        except (TypeError, ValueError):
            cfg = float(sampling.get("cfg", 1.0))

        expand, _psd_node, _saves = edit_expand.build_expand(
            workflow_path=wf_path,
            adapter=adapter,
            parts=parts,
            external=external,
            seed=seed,
            out_dir_rel=out_dir_rel,
            layers_json_path="",
            psd_name=run_name,
            psd_plan=plan_file,
            crop_mode=True,
            negative_prompt=negative_prompt,
            cfg=cfg,
            resolution=0,
            scheduler=str(sampling.get("scheduler") or "euler"),
            sigmas=scaled_sigmas,
            width=0,
            height=0,
        )

        _notify("stlu.psd.start", {
            "total": len(parts),
            "parts": [p["name"] for p in parts],
            "out_dir": out_dir_rel,
            "workflow": wf_path,
            "denoise": denoise,
            "work_max_side": work_max_side,
        })

        ref_path = str(settings.get("reference_path") or "").strip()
        info = "\n".join([
            "PSD: %s" % plan.get("psd", ""),
            "放大 %d 层，画布 %dx%d" % (len(parts), plan["canvas"][0],
                                       plan["canvas"][1]),
            "倍率 %g，单层上限 %d，输入上限 %d"
            % (factor, int(plan.get("max_side") or 0), work_max_side),
            "裁剪外扩 %g，种子 %d"
            % (float(settings.get("margin", MARGIN)), seed),
            "种子: %d（每层 +1）" % int(seed),
            "选层: %s" % (str(settings.get("select") or "all")),
            "参考原图: %s" % (os.path.basename(ref_path) if ref_path else "未设置"),
            psd_upscale.capped_report(plan),
            "输出: %s" % psd_out,
        ])
        return {
            "result": (psd_out, info),
            "expand": expand,
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "STLU_QwenEditLoader": STLU_QwenEditLoader,
    "STLU_PsdUpscalePrepare": STLU_PsdUpscalePrepare,
    "STLU_PsdUpscaleMin": STLU_PsdUpscaleMin,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STLU_QwenEditLoader": "STLU 编辑模型加载器（UNET+CLIP+VAE）",
    "STLU_PsdUpscalePrepare": "STLU PSD 放大准备",
    "STLU_PsdUpscaleMin": "STLU PSD 放大（执行）",
}
