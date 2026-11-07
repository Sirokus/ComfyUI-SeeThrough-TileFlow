"""Stages ①-⑤ of the SeeThrough tile pipeline (the guided workflow).

These nodes are thin ComfyUI adapters: widgets in, ``grid_doc`` and lists out.
The geometry, cache and PSD work lives in ``tile_psd``; shared helpers live in
``nodes_shared``.  Keeping the six active nodes in their own module keeps the
old pipeline (``nodes_legacy``) from growing this file back into a monolith.
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
                   core_reference, edit_expand, edit_parts,
                   psd_writer, region_plan, subgraph_clone, tile_psd, tile_strips)
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
    import psd_writer
    import region_plan
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


def _output_root() -> str:
    return _shared._output_root()


def _relative_to_output(folder: str) -> str:
    """``folder`` relative to the output root (``/``-separated), else None.

    ComfyUI's /view only serves files under output/ (or temp/), so a picture
    written anywhere else cannot be referenced by a ``subfolder``.
    """
    try:
        full = os.path.abspath(folder)
        root = os.path.abspath(_output_root())
        rel = os.path.relpath(full, root)
    except (OSError, ValueError):
        return None
    if os.path.isabs(rel) or rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return None
    return rel.replace(os.sep, "/")


def _notify(event: str, payload: dict) -> None:
    return _shared._notify(event, payload)


def _prompt_id():
    return _shared._prompt_id()


def _image_fingerprint(frame):
    return _shared._image_fingerprint(frame)


def _model_identity(model):
    return _shared._model_identity(model)


def _seethrough_node_class(name: str):
    return _shared._seethrough_node_class(name)


# The stage logic lives in ``pipeline_stages`` so the fine-grained nodes in
# ``nodes_transparent`` reuse exactly the same code.  Re-exported here so the
# old tests keep reaching them through ``nodes`` / ``nodes_pipeline``.
if __package__:  # ComfyUI loads the folder as a package
    from .pipeline_stages import (  # noqa: F401
        GUIDE_EMPTY, GUIDE_ERROR, GUIDE_OK, GUIDED_EMPTY_RETRIES,
        _apply_tile_guide, _box_blur_3x3, _crop_tile_parts,
        _guide_composite, _guide_composite_result, _guide_feather_mask,
        _restore_locked_orphans, _run_seethrough_tile,
        crop_tiles, describe_run, grid_info, guide_state, guide_tile,
        make_plan, merge_run_doc, normalize_settings, pick_tiles, plan_context,
        prepare_run, restore_locks, run_guided_tiles, save_original,
        tile_context, tile_index_of, tile_is_empty, write_manifest,
        write_tile_images, write_tile_psd,
    )
else:  # standalone import used by the tests and tools
    from pipeline_stages import (  # noqa: F401
        GUIDE_EMPTY, GUIDE_ERROR, GUIDE_OK, GUIDED_EMPTY_RETRIES,
        _apply_tile_guide, _box_blur_3x3, _crop_tile_parts,
        _guide_composite, _guide_composite_result, _guide_feather_mask,
        _restore_locked_orphans, _run_seethrough_tile,
        crop_tiles, describe_run, grid_info, guide_state, guide_tile,
        make_plan, merge_run_doc, normalize_settings, pick_tiles, plan_context,
        prepare_run, restore_locks, run_guided_tiles, save_original,
        tile_context, tile_index_of, tile_is_empty, write_manifest,
        write_tile_images, write_tile_psd,
    )


def run_pipeline_input(image, options=None, **overrides):
    """The one implementation behind every ① input node.

    ``options`` holds the node's widget/socket values exactly as ComfyUI
    passes them; ``overrides`` pins settings a simplified node does not
    expose (its panel has no widget for them, so the default would otherwise
    silently be the guided one).  Returns the same ``{"ui", "result"}``
    envelope both nodes emit.
    """
    options = dict(options or {})
    layerdiff_model = options.pop("layerdiff_model", None)
    depth_model = options.pop("depth_model", None)
    options.update(overrides)
    settings = normalize_settings(**options)
    plan = make_plan(image, settings, layerdiff_model, depth_model)
    plan = pick_tiles(plan)
    plan = plan_context(plan)
    plan = prepare_run(plan, image)
    tiles, indices = crop_tiles(image, plan)
    describe_run(plan)
    info = grid_info(plan)
    grid_doc = json.dumps(plan["doc"], ensure_ascii=False, separators=(",", ":"))
    info["grid_doc"] = grid_doc
    batch = image if image.ndim == 4 else image[None]
    frame = batch[0].detach().float().cpu()
    return {
        "ui": {"stlu_grid": [info]},
        "result": (tiles, indices, grid_doc, frame[None].contiguous(),
                   int(plan["resolution"]),
                   int(plan["num_inference_steps"]),
                   int(plan["depth_resolution"]), int(plan["seed"]),
                   bool(settings["tblr_split"]),
                   bool(settings["use_lama"])),
    }


# The simplified ① node's panel has no widgets for these; every other input
# still behaves exactly like STLU_PipelineInput.  Dropping a widget is enough:
# normalize_settings fills the shared DEFAULT_SETTINGS value for anything the
# graph does not pass, so the fingerprint of the simple node is the guided
# node's at its defaults instead of a second, divergent set of numbers.
SIMPLE_INPUT_DROP = (
    "guide_context", "guide_feather", "guide_strength", "guide_tolerance",
    "context_pad", "strip_max_resolution",
)


class STLU_PipelineInput:
    """The single node the user edits: image in, every setting in one place.

    It cuts the reference image, keeps the checked tiles and emits both the
    tile list and the scalar settings the fixed pipeline consumes.  The output
    block it stores inside ``grid_doc`` also names the output folder (the one
    the user picked; no per-run subfolder), so stage ③ can write everything --
    and reuse tiles a previous run left there -- without another node.
    """

    DESCRIPTION = (
        "整条分块流水线唯一需要操作的节点：切块/选块、拆层质量、种子、后处理、"
        "输出位置和合并选项都在它的面板里。它把选中的块和全部参数一起交给后面的"
        "自动节点；②③④ 的参数 widget 会被锁定，只显示进度和结果。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "参考图，用上面的标准 LoadImage 接入；这里只切它，不改原图。"}),
            },
            "optional": {
                "tile_width": ("INT", {"default": 1024, "min": 64, "max": 8192,
                                       "tooltip": "每块裁剪的宽度（像素）。越大细节越多、越吃显存；默认 1024。"}),
                "tile_height": ("INT", {"default": 1024, "min": 64, "max": 8192,
                                        "tooltip": "每块裁剪的高度（像素）。默认 1024，一般与块宽相同。"}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 4096,
                                    "tooltip": "相邻块重叠的像素数；合并时用正弦窗平滑接缝。默认 128，0 = 不重叠。"}),
                "select": ("STRING", {"default": "all",
                                      "tooltip": "只跑这些块：all / none / 0-2,5。点面板里的格子会写这里；未勾选的块不跑 GPU。"}),
                "resolution": ("INT", {"default": 512,
                                       "min": stlu_limits.RES_MIN,
                                       "max": stlu_limits.RES_MAX,
                                       "step": stlu_limits.RES_STEP,
                                       "tooltip": "拆层模型的推理分辨率（512-2048，与上游一致）。512 兼顾速度与细节；显存紧时别调大。"}),
                "num_inference_steps": ("INT", {"default": 30,
                                                "min": stlu_limits.STEPS_MIN,
                                                "max": stlu_limits.STEPS_MAX,
                                                "tooltip": "拆层采样步数。30 是默认折中；越高越慢，细节不一定更好。"}),
                "depth_resolution": ("INT", {"default": -1,
                                             "min": stlu_limits.DEPTH_MIN,
                                             "max": stlu_limits.DEPTH_MAX,
                                             "step": stlu_limits.DEPTH_STEP,
                                             "tooltip": "深度图分辨率；-1 = 跟拆层分辨率一致，也可单独指定（如 768）。"}),
                # Deliberately not named "seed": the frontend inserts a
                # control_after_generate widget right after any input with
                # that exact name, which would shift every later widget value.
                "random_seed": ("INT", {"default": 0,
                                        "min": stlu_limits.SEED_MIN,
                                        "max": stlu_limits.SEED_MAX,
                                        "tooltip": "随机种子，拆层和深度共用（0-2^32-1，上游限制）。同图同块、种子相同则结果可复现。"}),
                "tblr_split": ("BOOLEAN", {"default": True,
                                           "tooltip": "把左右不对称的部件按中线拆成 -l / -r，便于分别修。"}),
                "use_lama": ("BOOLEAN", {"default": True,
                                         "tooltip": "用 LaMa 补全被前层挡住的部分；关闭后遮挡处保持透明。"}),
                "alpha_threshold": ("INT", {"default": 8, "min": 0, "max": 255,
                                            "tooltip": "0-255，低于该 alpha 的像素视为透明。默认 8；调大可去掉半透明脏边。"}),
                "out_dir": ("STRING", {"default": "SeeThrough/我的自设/",
                                       "tooltip": "相对 output/ 的输出根目录，结果直接写进这个文件夹（不再每次新建波次子目录）。"
                                                  "同一个文件夹里已经生成的块会被复用：只勾选几块重跑，未勾选的块仍会参与引导和合并。"}),
                "group": ("STRING", {"default": "",
                                     "tooltip": "可选的二级分类目录，结果写到 <根目录>/<分类>/。"}),
                "filename_prefix": ("STRING", {"default": "tile",
                                               "tooltip": "每块 PSD 的文件名前缀，例如 tile_00_r0_c0.psd。"}),
                "merge_name": ("STRING", {"default": "merged",
                                          "tooltip": "合并后的 PSD 文件名（不含扩展名），默认 merged。"}),
                "group_layers": ("BOOLEAN", {"default": True,
                                             "tooltip": "勾选（不裁剪版）：各块同名部件各自保留完整原件，收进同名文件夹，"
                                                        "不做接缝裁切/幻影清理/完整部件淘汰，便于事后修；"
                                                        "不勾（直出版）：按接缝归属裁切后合成一层。"}),
                "include_depth": ("BOOLEAN", {"default": False,
                                              "tooltip": "额外写一个独立的 <名字>_depth.psd；depth 永远不会进入彩色 PSD。"}),
                "include_original": ("BOOLEAN", {"default": False,
                                                 "tooltip": "在合并 PSD 底部加入原图图层，方便对照。"}),
                "guide_context": ("BOOLEAN", {"default": True,
                                              "tooltip": "顺序生成时，把已经生成好的相邻块在重叠区的内容回填到当前块再送进拆层，"
                                                         "让相邻块看到同一份像素；只有「引导版」工作流会用到。"}),
                "guide_feather": ("INT", {"default": 32, "min": 0, "max": 512,
                                          "tooltip": "回填重叠区的羽化像素。0 = 硬边；越大过渡越软。默认 32。"}),
                "guide_strength": ("INT", {"default": 40, "min": 0, "max": 100,
                                           "tooltip": "回填强度（%）。100 = 重叠区完全采用上一块的生成结果，"
                                                      "连它的补全/幻觉也一起带进来；默认 40 只做轻推。"}),
                "guide_tolerance": ("INT", {"default": 16, "min": 0, "max": 255,
                                            "tooltip": "只回填生成结果与原图平均色差不超过该值的像素（0-255）。"
                                                       "默认 16，越小越保守；0 = 不过滤（危险，会把补全内容带过去）。"}),
                # Last on purpose: widget values serialize by position, so new
                # options must not shift the ones older workflows saved.
                "seam_feather": ("INT", {"default": 32, "min": 0, "max": 512,
                                         "tooltip": "仅影响直出合并（不勾选同名图层单独成组）：落败块的像素在接缝外侧的"
                                                    "最多该宽度内淡出，避免出现颜色边。0 = 全部硬切；默认 32。"}),
                # Optional model sockets: connected in the workflows so the
                # cache check can include the model identity.  Not widgets, so
                # they do not shift the serialized widget values.
                "layerdiff_model": ("SEETHROUGH_LAYERDIFF_MODEL", {
                    "forceInput": True,
                    "tooltip": "可选：接拆层模型加载器。接上后换模型会触发清空重跑。"}),
                "depth_model": ("SEETHROUGH_DEPTH_MODEL", {
                    "forceInput": True,
                    "tooltip": "可选：接深度模型加载器。接上后换模型会触发清空重跑。"}),
                # Grid phase.  Appended after the sockets so older workflows
                # keep their widget order; 0 is exactly the classic layout.
                "grid_offset_x": ("INT", {"default": 0, "min": 0, "max": 65536,
                    "tooltip": "网格横向相位（像素）：平移内部的竖向切分线，让接缝避开人物。"
                               "面板里的「点图对齐主体」会自动算好这个值；0 = 从左边缘开始。"}),
                "grid_offset_y": ("INT", {"default": 0, "min": 0, "max": 65536,
                    "tooltip": "网格纵向相位（像素）：平移内部的横向切分线。0 = 从上边缘开始。"}),
                # Subject strips: one inference sees the whole subject width.
                # Appended after the grid phase so older workflows keep their
                # widget order and continue to open in uniform-grid mode.
                "partition_mode": (["均匀网格", "方形全宽分区", "自定义分区"], {
                    "default": "均匀网格",
                    "tooltip": "均匀网格 = 原来的切块；方形全宽分区 = 自动按主体宽度做方形条，"
                               "每条一次推理看到完整左右，消除竖直穿过主体的接缝；"
                               "自定义分区 = 在面板画布上手动摆放正方形窗口，每块包住一个完整部位。"}),
                "strip_max_side": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 64,
                    "tooltip": "方形分区的目标边长（像素）。主体宽度 + 边距超过它时会自动放大到"
                               "能覆盖主体全宽的方形，不会竖直切开；面板会显示实际边长。"}),
                "strip_max_resolution": ("INT", {"default": 768, "min": 512, "max": 2048, "step": 64,
                    "tooltip": "（已废弃，仅为兼容旧工作流保留）方形分区现在直接使用「拆层质量」卡里的"
                               "「分辨率」，此值不再参与规划。"}),
                "strip_margin": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8,
                    "tooltip": "主体框四周留给模型的背景边距（像素）。主体几乎占满画面时会自动收窄。"}),
                # Written by the ① panel together with a mapped selection.  It
                # tells the node whether `select` already refers to the current
                # region plan; an empty/old id lets the node map the old
                # selection by source area instead of misreading old indices.
                "select_plan": ("STRING", {"default": "",
                    "tooltip": "内部使用：当前 select 所属的分区 plan_id；面板自动维护，不用手填。"}),
                # Kept LAST so workflows saved before it existed keep every
                # widget aligned and simply pick up the default (0 = off).
                "context_pad": ("INT", {"default": 0, "min": 0, "max": 512, "step": 32,
                    "tooltip": "块外上下文（像素）。0 = 关闭；大于 0 时每块向外多裁这么多像素送进模型，"
                               "生成完再裁回原条带。块内因此能看到更多身体，减少「只有腿脚却按上半身补全」"
                               "这类重绘；不改变分块几何。改它会重新生成相关块。"}),
                # Written by the ① panel; the canvas editor is the only sane
                # way to produce it.  Appended LAST so older workflows keep
                # every widget aligned and pick up the empty default.
                "custom_regions": ("STRING", {"default": "[]",
                    "tooltip": "内部使用：自定义正方形窗口 JSON（x/y/size/res/name）。"
                               "由 ① 面板的画布编辑自动维护，不用手填。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "IMAGE",
                    "INT", "INT", "INT", "INT", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("tiles", "indices", "grid_doc", "original",
                    "resolution", "num_inference_steps", "depth_resolution",
                    "seed", "tblr_split", "use_lama")
    OUTPUT_TOOLTIPS = (
        "只含勾选块的裁剪图像列表（按网格顺序），直接接 ② 拆层。",
        "每块在原图网格里的编号列表，与 tiles 一一对应；③ 用它命名 tile_XX。",
        "本次运行的格子 / 选择 / 输出路径 / 合并选项文档（JSON），③ 直接复用。",
        "未裁剪的原图；③ 用它写 original.png。",
        "拆层分辨率，接到 ② GenerateLayers。",
        "拆层采样步数，接到 ② GenerateLayers。",
        "深度图分辨率（-1 = 跟随拆层），接到 ② GenerateDepth。",
        "随机种子，同时接到拆层和深度，保证两者一致。",
        "左右拆分开关，接到 ② PostProcess。",
        "LaMa 补全开关，接到 ② PostProcess。",
    )
    OUTPUT_IS_LIST = (True, True, False, False,
                      False, False, False, False, False, False)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, image, **options):
        return run_pipeline_input(image, options)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_SimpleInput(STLU_PipelineInput):
    """The lightweight ① node: same planning, fewer knobs.

    The input list comes from :class:`STLU_PipelineInput` with
    :data:`SIMPLE_INPUT_DROP` removed, and the whole run is
    :func:`run_pipeline_input` -- so geometry, selection, cache clearing and
    the ``grid_doc`` contract are byte-for-byte the guided version's.  Only
    the guided loop's widgets and the crop-back padding are gone: neither is
    read by the plain four-step graph this node drives, and the shared
    defaults fill their settings back in unchanged.

    The inputs are derived rather than re-listed on purpose: a new shared
    option reaches both nodes, and there is no second copy of the widget
    order that could drift and shift an old workflow's saved values.
    """

    DESCRIPTION = (
        "简化版 ①：只保留四步流水线真正会用到的开关（切块/选块、拆层质量、"
        "分区、后处理、输出与合并），去掉顺序引导与块外上下文。规划、缓存和 "
        "grid_doc 与完整版完全同源。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        types = STLU_PipelineInput.INPUT_TYPES()
        optional = {name: spec for name, spec in types["optional"].items()
                    if name not in SIMPLE_INPUT_DROP}
        return {"required": dict(types["required"]), "optional": optional}

    def execute(self, image, **options):
        return run_pipeline_input(image, options)


class STLU_GuidedLayers:
    """Run the SeeThrough chain tile by tile, feeding finished tiles forward.

    ComfyUI expands a list output per *node*, not per item, so the plain chain
    never lets tile N see tile N-1.  This node owns the loop instead: it runs
    GenerateLayers -> GenerateDepth -> PostProcess for one tile at a time and
    pastes the already-generated neighbours into the overlap before the next
    tile reaches the model.  Outputs are still lists, so the normal ③ nodes
    consume them unchanged.
    """

    DESCRIPTION = (
        "顺序逐块拆层：每块生成前，把已经生成好的相邻块在重叠区的像素回填进来再交给 "
        "SeeThrough，减轻拼接接缝。参数由 ① 用户输入节点提供，节点上的 widget 会被锁定。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE", {"forceInput": True,
                                    "tooltip": "① 用户输入输出的块列表（自动逐块处理）。"}),
                "indices": ("STRING", {"forceInput": True,
                                       "tooltip": "与 tiles 一一对应的原始格号。"}),
                "layerdiff_model": ("SEETHROUGH_LAYERDIFF_MODEL", {}),
                "depth_model": ("SEETHROUGH_DEPTH_MODEL", {}),
            },
            "optional": {
                "grid_doc": ("STRING", {"forceInput": True,
                                        "tooltip": "① 的文档：格子坐标 / 重叠 / 引导参数。"}),
                # Not "seed": the frontend appends a control_after_generate
                # widget after that exact name and shifts every later value.
                "random_seed": ("INT", {"default": 42,
                                        "min": stlu_limits.SEED_MIN,
                                        "max": stlu_limits.SEED_MAX}),
                "resolution": ("INT", {"default": 512,
                                       "min": stlu_limits.RES_MIN,
                                       "max": stlu_limits.RES_MAX,
                                       "step": stlu_limits.RES_STEP}),
                "num_inference_steps": ("INT", {"default": 30,
                                                "min": stlu_limits.STEPS_MIN,
                                                "max": stlu_limits.STEPS_MAX}),
                "depth_resolution": ("INT", {"default": -1,
                                             "min": stlu_limits.DEPTH_MIN,
                                             "max": stlu_limits.DEPTH_MAX,
                                             "step": stlu_limits.DEPTH_STEP}),
                "tblr_split": ("BOOLEAN", {"default": True}),
                "use_lama": ("BOOLEAN", {"default": True}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("SEETHROUGH_PARTS", "STRING")
    RETURN_NAMES = ("parts", "indices")
    OUTPUT_TOOLTIPS = (
        "每块的部件（列表），接 ③ 每块存 PSD。",
        "与 parts 一一对应的格号（列表）。",
    )
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, tiles, indices, layerdiff_model, depth_model, grid_doc="",
                random_seed=42, resolution=512, num_inference_steps=30,
                depth_resolution=-1, tblr_split=True, use_lama=True):
        doc = None
        raw_grid = _first(grid_doc, "")
        if str(raw_grid or "").strip():
            try:
                doc = tile_psd.load_grid_ref(str(raw_grid))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    "STLU ②: 无法读取 ① 的 grid_doc（%s）/ cannot read grid_doc from ①: %s"
                    " / check the ①→② grid_doc link, then re-queue ①." % (exc, exc)) from exc
        state = guide_state({"doc": doc or {}})
        state["doc"] = doc
        run_dir = str(((doc or {}).get("out") or {}).get("dir") or "")
        parts_out, indices_out, _guided, _composites = run_guided_tiles(
            tiles, indices, state, run_dir, layerdiff_model, depth_model,
            random_seed, resolution, num_inference_steps, depth_resolution,
            tblr_split, use_lama)
        return (parts_out, indices_out)

class STLU_TilePSD:
    """Stage ③: write one tile's PSD, preview, depth and layer images.

    The node runs once per tile because the SeeThrough chain above it expands
    the tile list.  It only owns that one tile; ``STLU_WaveIndex`` writes the
    wave-wide ``manifest.json`` / ``images.json`` afterwards.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Live mode: connect SeeThrough_PostProcess here.
                "parts": ("SEETHROUGH_PARTS", {
                    "tooltip": "② PostProcess 输出的部件；列表会自动逐块展开，本节点每块执行一次。"}),
                # One per tile: the original grid index, from the input node.
                "index": ("STRING", {"forceInput": True,
                                     "tooltip": "这块在原图网格里的编号（接 ① 的 indices），用于命名 tile_XX 和回填位置。"}),
            },
            "optional": {
                "grid_doc": ("STRING", {"forceInput": True,
                                        "tooltip": "① 的文档：块大小 / 原图尺寸 / 输出路径 / 合并选项。"}),
                "alpha_threshold": ("INT", {"default": 8, "min": 0, "max": 255, "step": 1,
                                            "tooltip": "低于该 alpha 的像素视为透明；默认 8，一般与 ① 保持一致。"}),
                # Legacy: STLU_Wave's document.  STLU_PipelineInput stores the
                # same values inside grid_doc, so new workflows leave this off.
                "wave_doc": ("STRING", {"forceInput": True,
                                        "tooltip": "旧版 STLU_Wave 的文档；新流程不用接，输出路径已在 grid_doc 里。"}),
            },
        }

    RETURN_TYPES = ("STLU_TILE",)
    RETURN_NAMES = ("tile",)
    OUTPUT_TOOLTIPS = (
        "这一块的记录（PSD / 预览 / depth / images 路径）；接 ③ 写清单。",
    )
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, parts, index, grid_doc="", alpha_threshold=8, wave_doc=""):
        # Legacy facade: parse the JSON documents, then call the same stage
        # functions the fine-grained ③ nodes use.
        doc = None
        raw_grid = _first(grid_doc, "")
        if str(raw_grid or "").strip():
            try:
                doc = tile_psd.load_grid_ref(str(raw_grid))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    "STLU_TilePSD: 无法读取 ① 的 grid_doc（%s）/ cannot read grid_doc from ①: %s"
                    " / check the ①→③ grid_doc link." % (exc, exc)) from exc
        out = dict(_grid_out(doc))
        wave = None
        raw_wave = _first(wave_doc, "")
        if str(raw_wave or "").strip():
            wave = _wave_doc(wave_doc)
        elif out.get("dir"):
            wave = out
        if wave is None:
            raise ValueError(
                "STLU_TilePSD: 需要 grid_doc（接 STLU_PipelineInput）或 wave_doc（旧版 STLU_Wave）"
                " / connect grid_doc (STLU_PipelineInput) or wave_doc (legacy STLU_Wave)")
        if "alpha_threshold" in out:
            alpha_threshold = int(out["alpha_threshold"])
        merged_out = dict(wave or {})
        merged_out.update(out)
        merged_out.setdefault("prefix", str(wave.get("prefix") or "tile"))
        merged_out["alpha_threshold"] = int(alpha_threshold)
        merged_out["dir"] = str(wave["dir"])
        merged_doc = dict(doc or {})
        merged_doc["out"] = merged_out

        plan = {"doc": merged_doc}
        parts = restore_locks(parts, plan, index)
        record = write_tile_psd(parts, plan, index)
        if record.get("status") == "empty":
            return {"ui": {"stlu_tile": [{"index": record["index"],
                                          "status": "empty"}]},
                    "result": ({"index": record["index"], "status": "empty"},)}
        images = write_tile_images(parts, plan, index)
        record["image"] = images.get("entry")
        return {"ui": {"stlu_tile": [{"index": record["index"],
                                      "file": record.get("file")}]},
                "result": (record,)}

class STLU_WaveIndex:
    """Stage ③: write the wave-wide manifest after every tile is saved.

    It collects one record per tile, writes ``original.png`` exactly once and
    builds ``images.json`` + ``manifest.json`` with the same fields the old
    single-node stage three produced, so step four reads them unchanged.

    Because the output folder is fixed, it also folds in the tiles an earlier
    run already finished there (verified against the source fingerprint).  A
    partial re-run therefore writes a manifest that still describes the whole
    picture, and step four merges the new tiles together with the cached ones.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("STLU_TILE", {
                    "tooltip": "③ TilePSD 每块返回的记录（列表）；汇总成 manifest.json。"}),
                "original": ("IMAGE", {
                    "tooltip": "① 的原图；在输出根目录写一份 original.png，images.json 指向它。"}),
                "grid_doc": ("STRING", {"forceInput": True,
                                        "tooltip": "① 的文档：原图尺寸 / 块大小 / 输出路径 / 合并选项。"}),
            },
            "optional": {
                # Legacy: STLU_Wave's document; the pipeline input already
                # stores the same values inside grid_doc.
                "wave_doc": ("STRING", {"forceInput": True,
                                        "tooltip": "旧版 STLU_Wave 的文档；新流程不用接。"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("psd_dir", "report", "merge_doc")
    OUTPUT_TOOLTIPS = (
        "输出文件夹的绝对路径（分格 PSD、manifest、images 都在这里）。",
        "manifest.json 的内容（JSON 字符串），含本次新生成与复用的块。",
        "输出目录 + 合并名字/选项，接 ④ 自动合并。",
    )
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, tiles, original, grid_doc="", wave_doc=""):
        # Legacy facade over save_original + write_manifest.
        doc = None
        raw_grid = _first(grid_doc, "")
        if str(raw_grid or "").strip():
            try:
                doc = tile_psd.load_grid_ref(str(raw_grid))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    "STLU_WaveIndex: 无法读取 ① 的 grid_doc（%s）/ cannot read grid_doc from ①: %s"
                    " / check the ①→③ grid_doc link." % (exc, exc)) from exc
        out = dict(_grid_out(doc))
        wave = None
        if str(_first(wave_doc, "") or "").strip():
            wave = _wave_doc(wave_doc)
        elif out.get("dir"):
            wave = out
        if wave is None:
            raise ValueError(
                "STLU_WaveIndex: 需要 grid_doc（接 STLU_PipelineInput）或 wave_doc（旧版 STLU_Wave）"
                " / connect grid_doc (STLU_PipelineInput) or wave_doc (legacy STLU_Wave)")
        merged_out = dict(wave or {})
        merged_out.update(out)
        merged_out.setdefault("prefix", str(wave.get("prefix") or "tile"))
        merged_out["dir"] = str(wave["dir"])
        merged_doc = dict(doc or {})
        merged_doc["out"] = merged_out
        plan = {"doc": merged_doc}
        original_name = save_original(original, plan)
        # 每块的 images/ 条目由 ③ 的逐块节点带在记录里，旧门面在这里取出来。
        image_records = [{"entry": r["image"]} for r in (tiles or [])
                         if isinstance(r, dict) and isinstance(r.get("image"), dict)]
        result = write_manifest(tiles, image_records, original_name, plan)
        manifest_json = json.dumps(result["manifest"], ensure_ascii=False,
                                   indent=2)
        merge_json = json.dumps(result["merge_doc"], ensure_ascii=False,
                                separators=(",", ":"))
        info = result["info"]
        return {"ui": {"stlu_tiles": [info]},
                "result": (merged_out["dir"], manifest_json, merge_json)}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_RebuildPSD:
    """Stage ③ without the GPU: rebuild tile PSDs from saved layer images.

    Point it at a wave folder (or a ``*_layers.json``) and it writes a fresh
    wave with the same layout.  It is a separate node on purpose: the live
    path never has to carry an offline source switch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("STRING", {"default": "",
                                      "tooltip": "已生成的波次目录 / *_layers.json / 单张图片"}),
                "out_dir": ("STRING", {"default": "SeeThrough"}),
            },
            "optional": {
                "group": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "tile"}),
                "tiles": ("STRING", {"default": "",
                                     "tooltip": "只重建这些块，如 0-2,5；留空 = 来源里的全部块"}),
                "alpha_threshold": ("INT", {"default": 8, "min": 0, "max": 255}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("psd_dir", "report")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, source, out_dir="SeeThrough", group="", filename_prefix="tile",
                tiles="", alpha_threshold=8):
        raw_source = str(source or "").strip()
        if not raw_source:
            raise ValueError("STLU_RebuildPSD: 选择一个已有的波次目录或 *_layers.json / pick an existing run folder or a *_layers.json")
        source_path = _resolve_workspace_path(raw_source)
        if not os.path.exists(source_path):
            raise FileNotFoundError("image source not found: %s" % source_path)

        group_name = str(group or "").strip()
        wave_dir = tile_psd.flat_directory(_output_root(), out_dir, group_name)
        wave = os.path.basename(wave_dir.rstrip("/\\")) or "run"
        stamp = "%s_%s" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                           str(uuid.uuid4())[:6])
        raw_tiles = str(tiles or "").strip()
        tile_subset = None
        if raw_tiles and raw_tiles.lower() not in ("all", "none"):
            tile_subset = _parse_index_list(raw_tiles)

        manifest = tile_psd.write_tiles_from_images(
            source_path,
            _output_root(),
            out_dir=out_dir,
            filename_prefix=str(filename_prefix or "tile"),
            merge_parts=True,
            alpha_threshold=int(alpha_threshold),
            stamp=stamp,
            merge_after=False,
            grid_doc=None,
            tile_index_map=None,
            write_depth=True,
            merge_group_layers=True,
            include_original=True,
            run_dir_abs=wave_dir,
            tiles=tile_subset,
        )
        manifest["group"] = group_name
        manifest["wave"] = wave
        manifest["prompt_id"] = _prompt_id()
        json_io.write_json_atomic(os.path.join(wave_dir, "manifest.json"), manifest)

        info = {
            "dir": manifest.get("dir"),
            "subfolder": manifest.get("dir"),
            "filename": (manifest.get("tiles") or [{}])[0].get("file"),
            "count": manifest.get("count"),
            "canvas": (manifest.get("tiles") or [{}])[0].get("canvas"),
            "images_dir": manifest.get("images_dir") or manifest.get("source_dir") or "",
            "group": group_name,
            "wave": wave,
            "error": None,
        }
        _notify("stlu.tiles", info)
        print("[STLU] rebuilt %s tile PSDs from %s" % (info["count"], source_path),
              flush=True)
        return {
            "ui": {"stlu_tiles": [info]},
            "result": (wave_dir, json.dumps(manifest, ensure_ascii=False, indent=2)),
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_TileMerge:
    """Independent panel for re-merging already-written tile PSDs.

    Carries no data-flow inputs, so running it never re-executes the
    SeeThrough chain.  The panel lists every folder under output/ that holds
    a tile manifest, shows the per-tile previews with checkboxes and calls
    ``/stlu/tiles/merge`` for the selected subset.  A manifest written by a
    partial re-run already includes the folder's cached tiles, so the queued
    merge is complete even when only one tile ran this time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "out_dir": ("STRING", {"default": "SeeThrough/Tiles"}),
                "name": ("STRING", {"default": "merged"}),
                "group_layers": ("BOOLEAN", {"default": True,
                                             "tooltip": "勾选（不裁剪版）：各块同名部件各自保留完整原件，收进同名文件夹，"
                                                        "便于事后修；不勾（直出版）：按接缝归属裁切后合成一层"}),
                "include_depth": ("BOOLEAN", {"default": False,
                                              "tooltip": "导出独立的 <名字>_depth.psd；不勾则不导出。depth 永远不会放进彩色 PSD"}),
                "include_original": ("BOOLEAN", {"default": False,
                                                 "tooltip": "合并 PSD 底部加入原图图层"}),
                # Last on purpose: widget values serialize by position.  The
                # run folder to merge; the panel writes it when a run is
                # picked so queueing the node merges the same folder.
                "run_dir": ("STRING", {"default": "",
                                       "tooltip": "分块 PSD 所在的运行目录（相对 output/，也可以填绝对路径）"}),
                "seam_feather": ("INT", {"default": 32, "min": 0, "max": 512,
                                         "tooltip": "仅直出合并用：落败块在接缝外侧最多该宽度内淡出。"
                                                    "同名图层单独成组时不裁切、不使用该值。"
                                                    "0 = 全部硬切；接上 ③ 时以 ① 的值为准。"}),
                # New pipeline: STLU_WaveIndex names the folder AND the merge
                # options, so the queue merges what this run just wrote.
                "merge_doc": ("STRING", {"forceInput": True,
                                         "tooltip": "③ 写清单输出的文档：输出目录 + 合并名字/选项。接上后本节点自动合并（含目录里已缓存的块），面板只显示结果。"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("psd_path",)
    OUTPUT_TOOLTIPS = (
        "合并后的 PSD 绝对路径（写在分块所在目录里）；接 ⑤ 预览节点即可显示。",
    )
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, out_dir="SeeThrough/Tiles", name="merged", group_layers=True,
                include_depth=False, include_original=False, run_dir="",
                seam_feather=32, merge_doc=""):
        parsed = None
        raw_merge = _first(merge_doc, "")
        if str(raw_merge or "").strip():
            try:
                parsed = json.loads(raw_merge) if isinstance(raw_merge, str) else raw_merge
            except (TypeError, ValueError) as exc:
                raise ValueError("merge_doc is not JSON: %s" % exc)
            if isinstance(parsed, dict) and parsed.get("dir"):
                run_dir = str(parsed.get("dir") or run_dir)
                name = str(parsed.get("name") or name or "merged")
                group_layers = bool(parsed.get("group_layers", group_layers))
                include_depth = bool(parsed.get("include_depth", include_depth))
                include_original = bool(parsed.get("include_original", include_original))
                if parsed.get("seam_feather") is not None:
                    seam_feather = int(parsed.get("seam_feather"))
        rel = str(run_dir or "").strip()
        if not rel:
            # Nothing picked yet: the panel's /stlu/tiles/merge route does the
            # work, this just keeps the node queue-friendly.
            return {"ui": {"stlu_tile_merge": [{
                "dir": str(out_dir or ""),
                "name": str(name or "merged"),
                "group_layers": bool(group_layers),
                "include_depth": bool(include_depth),
                "include_original": bool(include_original),
                "skipped": "没有指定分块目录 / no tile folder specified",
            }]}, "result": ("",)}

        # The merge must use the same transparency cutoff the tiles were
        # written with; ① travels it through merge_doc, older runs only have
        # it in the manifest (merge_run reads it from there).
        raw_alpha = (parsed.get("alpha_threshold")
                     if isinstance(parsed, dict) else None)
        result = merge_run_doc({
            "dir": rel,
            "name": name,
            "group_layers": group_layers,
            "include_depth": include_depth,
            "include_original": include_original,
            "seam_feather": seam_feather,
            "alpha_threshold": raw_alpha,
        })
        return {"ui": {"stlu_tile_merge": [result["payload"]]},
                "result": (result["psd_path"],)}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

_GUIDE_THUMB_MAX = 384
_GUIDE_INDEX = "inputs.json"


def _write_guide_index(dest_dir: str, indices) -> None:
    """Record which tiles have a persisted input picture.

    The panel trusts this list rather than the bare PNGs, so pictures left
    over from an earlier run (a bigger grid) never show up on a row that
    this run did not generate.  Rewritten only when the content changes so
    the file's mtime stays a usable cache stamp.
    """
    wanted = sorted(int(i) for i in indices)
    try:
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, _GUIDE_INDEX)
        data = json.dumps({"indices": wanted})
        try:
            with open(path, "r", encoding="utf-8") as fh:
                if fh.read() == data:
                    _prune_guide_files(dest_dir, wanted)
                    return
        except OSError:
            pass
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(data)
        _prune_guide_files(dest_dir, wanted)
    except OSError as exc:  # noqa: BLE001 - the panel just falls back to dash
        print("[STLU] guide index not written: %s" % exc, flush=True)


def _prune_guide_files(dest_dir: str, keep) -> None:
    kept = set(int(i) for i in keep)
    try:
        names = os.listdir(dest_dir)
    except OSError:
        return
    for name in names:
        if not (name.startswith("stlu_guide_") and name.endswith(".png")):
            continue
        try:
            index = int(name[len("stlu_guide_"):-len(".png")])
        except ValueError:
            continue
        if index in kept:
            continue
        try:
            os.remove(os.path.join(dest_dir, name))
        except OSError:
            pass


def _guide_thumb(tensor, index: int, dest_dir: str = "") -> dict:
    """White-backed thumbnail of one tile's model input.

    This is the debug picture ② used to hand to a PreviewImage: the tile
    after guide backfill and context padding, i.e. exactly what the seam
    models saw.  The ⑤ panel shows it on the tile's own row instead of a
    second preview node flooding the queue with images.

    With ``dest_dir`` the picture is also written there under a stable name
    (so the panel can find it again after a refresh); otherwise only the
    throwaway ``temp/`` copy is kept.
    """
    if tensor is None:
        return {}
    try:
        arr = tensor.detach().cpu().float().numpy()
    except Exception:  # noqa: BLE001 - plain numpy arrays are accepted too
        arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim == 4:
        if not arr.shape[0]:
            return {}
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        return {}
    arr = np.clip(arr, 0.0, 1.0)
    rgb = arr[..., :3]
    if arr.shape[-1] == 4:
        alpha = arr[..., 3:4]
        rgb = rgb * alpha + (1.0 - alpha)
    elif arr.shape[-1] == 1:
        rgb = np.repeat(arr, 3, axis=-1)
    from PIL import Image
    image = Image.fromarray(
        np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8), "RGB")
    if max(image.width, image.height) > _GUIDE_THUMB_MAX:
        scale = _GUIDE_THUMB_MAX / float(max(image.width, image.height))
        image = image.resize(
            (max(1, int(round(image.width * scale))),
             max(1, int(round(image.height * scale)))),
            Image.Resampling.LANCZOS)
    if dest_dir:
        # Persisted next to the run so the ⑤ panel can still show each tile's
        # model input after a browser refresh (the temp file survives a
        # refresh but nothing maps tile -> file once the payload is gone).
        os.makedirs(dest_dir, exist_ok=True)
        name = "stlu_guide_%02d.png" % int(index)
        path = os.path.join(dest_dir, name)
        image.save(path, compress_level=3)
        sub = _relative_to_output(dest_dir)
        if sub is not None:
            return {"filename": name, "subfolder": sub, "type": "output",
                    "v": int(os.path.getmtime(path) * 1000)}
    temp = folder_paths.get_temp_directory()
    os.makedirs(temp, exist_ok=True)
    digest = hashlib.sha1(
        ("%d|%d|%d|" % (index, image.width, image.height)).encode("utf-8")
        + np.asarray(image).tobytes()).hexdigest()[:10]
    name = "stlu_guide_%02d_%s.png" % (int(index), digest)
    path = os.path.join(temp, name)
    if not os.path.isfile(path):
        image.save(path, compress_level=3)
    return {"filename": name, "subfolder": "", "type": "temp",
            "v": int(os.path.getmtime(path) * 1000)}


class STLU_PSDPreview:
    """Show the merged PSD inside the graph through its own panel.

    Stage ④ writes a PSD but a PSD cannot be displayed by ComfyUI itself;
    this node flattens it (layers bottom -> top over white) and returns the
    tensor, while the panel drawn by ``web/stlu.js`` shows a compact overview
    plus the per-tile / per-part groups.  Point it at the ``psd_path`` output
    of ④, or at the run folder (it then picks ``merged.psd``).

    ``guide_images`` is optional: ② can hand over the per-tile pictures it
    actually fed the model, and the panel puts each on its tile's row.
    """

    DESCRIPTION = (
        "显示合并后的 PSD。接 ④ 的 psd_path，或直接填输出文件夹；面板先给全身总览，"
        "再按块 / 按部位分组列出拆分图。老版把 ② 的 guided_inputs / indices "
        "接进来，每块那一行还会多一张「输入」——引导回填后真正送进模型的图。纯 CPU。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "psd_path": ("STRING", {"forceInput": True,
                                        "tooltip": "④ 合并输出的 PSD 路径；也可以接一个输出文件夹。"}),
            },
            "optional": {
                "max_side": ("INT", {"default": 1600, "min": 256, "max": 8192,
                                     "tooltip": "预览的最长边；超过就等比缩小，只影响预览，不改 PSD。"}),
                "show_parts": ("BOOLEAN", {"default": True,
                                           "tooltip": "兼容旧工作流的开关：面板现在始终"
                                                      "按块 / 按部位列出拆分图。"}),
                "guide_images": ("IMAGE", {
                    "forceInput": True,
                    "tooltip": "② 的 guided_inputs：每块回填后真正送进模型的图；"
                               "接上后「按块」每一行多一张「输入」缩略图。"}),
                "guide_indices": ("STRING", {
                    "forceInput": True,
                    "tooltip": "② 的 indices，与 guided_inputs 一一对应。"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    OUTPUT_TOOLTIPS = (
        "合并 PSD 压平后的图片（白底）；可以继续接别的预览/保存节点。",
    )
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, psd_path, max_side=1600, show_parts=True,
                guide_images=None, guide_indices=None):
        raw = str(psd_path or "").strip()
        if not raw:
            raise ValueError("STLU_PSDPreview: 接上 ④ 的 psd_path（或填输出文件夹）/ connect ④'s psd_path (or fill in the output folder)")
        path = _resolve_workspace_path(raw)
        if os.path.isdir(path):
            manifest = tile_psd.read_manifest(path) or {}
            merged = str((manifest.get("merged") or {}).get("file") or "")
            if merged and os.path.isfile(os.path.join(path, merged)):
                path = os.path.join(path, merged)
            else:
                candidates = [
                    name for name in sorted(os.listdir(path))
                    if name.lower().endswith(".psd")
                    and not name.lower().endswith("_depth.psd")
                ]
                if not candidates:
                    raise FileNotFoundError("PSD preview: no .psd in %s" % raw)
                path = os.path.join(path, candidates[0])
        if not os.path.isfile(path):
            raise FileNotFoundError("PSD preview: not found: %s" % path)

        rgba = tile_psd.composite_psd(path)
        if rgba is None:
            raise ValueError("PSD preview: cannot read %s" % path)
        arr = rgba.astype(np.float32) / 255.0
        alpha = arr[..., 3:4]
        rgb = np.clip(np.rint((arr[..., :3] * alpha + (1.0 - alpha)) * 255.0),
                      0, 255).astype(np.uint8)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        limit = max(256, int(max_side or 1600))
        if max(height, width) > limit:
            from PIL import Image
            scale = limit / float(max(height, width))
            size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            rgb = np.array(Image.fromarray(rgb, "RGB").resize(
                size, Image.Resampling.LANCZOS), dtype=np.uint8)

        # No ``images`` key on purpose: ComfyUI would draw the full-canvas
        # image a second time inside the node (and a tall portrait preview at
        # node width makes the node enormously tall).  The panel below owns
        # display: a compact overview plus the tile / part groups.
        print("[STLU] PSD preview: %s (%dx%d -> %dx%d)" % (
            os.path.basename(path), width, height,
            int(rgb.shape[1]), int(rgb.shape[0])), flush=True)
        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.astype(np.float32) / 255.0))[None]
        # Keep this run's model inputs next to the PSD: the panel falls back
        # to these copies when the browser is refreshed and the execution
        # payload (stlu_guide) is gone.
        guide_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "_guide")
        guide = self._guide_payload(guide_images, guide_indices, guide_dir)
        _write_guide_index(guide_dir, [item["index"] for item in guide])

        return {
            "ui": {
                "stlu_parts": [{
                    "file": os.path.basename(path),
                    # The panel remembers this so reopening the workflow can
                    # show the last merge without running the queue again.
                    "path": os.path.abspath(path),
                    "canvas": [width, height],
                }],
                # Per-tile model inputs travel with the panel instead of a
                # separate PreviewImage: one small PNG per tile, keyed by the
                # grid index so the panel pins it to the right row.
                "stlu_guide": guide,
            },
            "result": (tensor,),
        }

    @staticmethod
    def _guide_payload(guide_images, guide_indices, dest_dir=""):
        if guide_images is None:
            return []
        frames = guide_images if isinstance(guide_images, (list, tuple)) \
            else [guide_images]
        raw_indices = (guide_indices if isinstance(guide_indices, (list, tuple))
                       else [guide_indices])
        out = []
        for pos, frame in enumerate(frames):
            raw = raw_indices[pos] if pos < len(raw_indices) else pos
            try:
                index = int(str(raw).strip())
            except (TypeError, ValueError):
                index = pos
            ref = _guide_thumb(frame, index, dest_dir)
            if ref:
                out.append({"index": index, "image": ref})
        return out


    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "STLU_PipelineInput": STLU_PipelineInput,
    "STLU_SimpleInput": STLU_SimpleInput,
    "STLU_GuidedLayers": STLU_GuidedLayers,
    "STLU_TilePSD": STLU_TilePSD,
    "STLU_WaveIndex": STLU_WaveIndex,
    "STLU_RebuildPSD": STLU_RebuildPSD,
    "STLU_TileMerge": STLU_TileMerge,
    "STLU_PSDPreview": STLU_PSDPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STLU_TileMerge": "SeeThrough ④ 合并分格 PSD",
    "STLU_PSDPreview": "SeeThrough ⑤ 预览合并 PSD",
    "STLU_PipelineInput": "SeeThrough ① 用户输入（只改这里）",
    "STLU_SimpleInput": "SeeThrough ① 用户输入（简化版）",
    "STLU_TilePSD": "SeeThrough ③ 每块存 PSD（自动逐块）",
    "STLU_GuidedLayers": "SeeThrough ② 顺序拆层（带重叠引导）",
    "STLU_WaveIndex": "SeeThrough ③ 写清单（全部块完成后）",
    "STLU_RebuildPSD": "SeeThrough ③ 从图片重建（可选，CPU）",
}

__all__ = [
    "STLU_PipelineInput",
    "STLU_SimpleInput",
    "STLU_GuidedLayers",
    "STLU_TilePSD",
    "STLU_WaveIndex",
    "STLU_RebuildPSD",
    "STLU_TileMerge",
    "STLU_PSDPreview",
    "_guide_feather_mask",
    "_box_blur_3x3",
    "GUIDE_OK",
    "GUIDE_EMPTY",
    "GUIDE_ERROR",
    "_guide_composite_result",
    "_guide_composite",
    "GUIDED_EMPTY_RETRIES",
    "_run_seethrough_tile",
    "_apply_tile_guide",
    "_crop_tile_parts",
    "_seethrough_node_class",
    # Stage functions shared with the fine-grained nodes.
    "crop_tiles",
    "describe_run",
    "grid_info",
    "guide_state",
    "guide_tile",
    "make_plan",
    "merge_run_doc",
    "normalize_settings",
    "pick_tiles",
    "plan_context",
    "prepare_run",
    "restore_locks",
    "run_guided_tiles",
    "save_original",
    "tile_context",
    "tile_index_of",
    "tile_is_empty",
    "write_manifest",
    "write_tile_images",
    "write_tile_psd",
]
