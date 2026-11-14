"""Fine-grained nodes for the transparent variant of the guided pipeline.

Every node here is a thin wrapper around one function in
:mod:`pipeline_stages`, which is also what the old monolith nodes call.  The
point is not new behaviour: it is that the workflow graph itself shows the
steps (plan -> pick -> context -> prepare -> crop -> loop -> save -> merge)
instead of hiding them inside one Python node.

Socket payloads are plain dicts of JSON values (``STLU_SETTINGS`` /
``STLU_PLAN``), never tensors, so ComfyUI's cache hash stays stable for the
pure steps and ``IS_CHANGED = NaN`` is only used where a node writes files.
"""

from __future__ import annotations

import folder_paths  # noqa: F401  (same import side effect as nodes_pipeline)

if __package__:  # ComfyUI loads the folder as a package
    from . import tile_psd  # noqa: F401
    from .nodes_pipeline import STLU_PipelineInput
    from .nodes_shared import _first, _notify
    from .pipeline_stages import (
        HEAD_SCOPE_TOP_ROW, crop_back, crop_tiles, guide_state,
        guide_tile, make_plan, merge_run_doc, normalize_settings, pick_tiles,
        plan_context, prepare_run, run_guided_tiles,
        save_original, write_manifest, write_tile_images,
        write_tile_psd,
    )
else:  # standalone import used by the tests and tools
    import tile_psd  # noqa: F401
    from nodes_pipeline import STLU_PipelineInput
    from nodes_shared import _first, _notify
    from pipeline_stages import (
        HEAD_SCOPE_TOP_ROW, crop_back, crop_tiles, guide_state,
        guide_tile, make_plan, merge_run_doc, normalize_settings, pick_tiles,
        plan_context, prepare_run, run_guided_tiles,
        save_original, write_manifest, write_tile_images,
        write_tile_psd,
    )


CATEGORY = "SeeThrough/LayerUpscale"

SETTINGS_TYPE = "STLU_SETTINGS"
PLAN_TYPE = "STLU_PLAN"
TILE_TYPE = "STLU_TILE"
TILE_IMAGE_TYPE = "STLU_TILE_IMAGES"
MANIFEST_TYPE = "STLU_MANIFEST"
MERGE_TYPE = "STLU_MERGE"
GENERATED_TYPE = "STLU_GENERATED"


# ---------------------------------------------------------------------------
# ① plan and crop -- the steps the old STLU_PipelineInput did in one go
# ---------------------------------------------------------------------------

class STLU_PipelineSettings:
    """All ① widgets, and nothing else: it only packs them into STLU_SETTINGS.

    The grid panel lives on this node, so it stays the single place the user
    touches.  Every calculation that used to hide behind it now happens in the
    plan/select/crop nodes below, where the graph can show it.
    """

    DESCRIPTION = (
        "① 用户输入：面板 + 全部参数都在这里，但只负责把参数打包成 STLU_SETTINGS。"
        "切块、选块、外扩、指纹比对、裁剪都在后面的小节点里，节点图上看得见。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        base = STLU_PipelineInput.INPUT_TYPES()
        optional = dict(base.get("optional") or {})
        # The model sockets belong to the planning node, not to a settings
        # packet (a dict of JSON values).
        optional.pop("layerdiff_model", None)
        optional.pop("depth_model", None)
        # The image is not a setting; it is here only so the panel can follow
        # the link back to LoadImage and draw the preview.
        preview_input = {
            "image": ("IMAGE", {
                "forceInput": True,
                "tooltip": "参考图（只给面板画格子用；参数包本身不读它）。"})}
        preview_input.update(optional)
        return {"required": {}, "optional": preview_input}

    RETURN_TYPES = (SETTINGS_TYPE,)
    RETURN_NAMES = ("settings",)
    OUTPUT_TOOLTIPS = (
        "本次运行的全部参数（纯 JSON dict）；接 STLU_PlanGrid。",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, image=None, tile_width=1024, tile_height=1024, overlap=128,
                select="all", resolution=1024, num_inference_steps=30,
                depth_resolution=-1, random_seed=0, tblr_split=True,
                use_lama=True, alpha_threshold=8,
                out_dir="SeeThrough/我的自设/", group="", filename_prefix="tile",
                merge_name="merged", group_layers=True, include_depth=False,
                include_original=False, guide_context=False, guide_feather=32,
                guide_strength=40, guide_tolerance=16, seam_feather=32,
                grid_offset_x=0, grid_offset_y=0, partition_mode="均匀网格",
                strip_max_side=2048, strip_max_resolution=768, strip_margin=32,
                select_plan="", context_pad=0, custom_regions="[]",
                head_scope=HEAD_SCOPE_TOP_ROW, tile_tags="{}",
                depth_random_seed=-1, sub_dir=""):
        settings = normalize_settings(
            tile_width=tile_width, tile_height=tile_height, overlap=overlap,
            select=select, resolution=resolution,
            num_inference_steps=num_inference_steps,
            depth_resolution=depth_resolution, random_seed=random_seed,
            tblr_split=tblr_split, use_lama=use_lama,
            alpha_threshold=alpha_threshold, out_dir=out_dir, group=group,
            filename_prefix=filename_prefix, merge_name=merge_name,
            group_layers=group_layers, include_depth=include_depth,
            include_original=include_original, guide_context=guide_context,
            guide_feather=guide_feather, guide_strength=guide_strength,
            guide_tolerance=guide_tolerance, seam_feather=seam_feather,
            grid_offset_x=grid_offset_x, grid_offset_y=grid_offset_y,
            partition_mode=partition_mode, strip_max_side=strip_max_side,
            strip_max_resolution=strip_max_resolution,
            strip_margin=strip_margin, select_plan=select_plan,
            context_pad=context_pad, custom_regions=custom_regions,
            head_scope=head_scope, tile_tags=tile_tags,
            depth_random_seed=depth_random_seed, sub_dir=sub_dir)
        return (settings,)


class STLU_PlanGrid:
    """Work out the grid / strips / custom windows for this image (pure)."""

    DESCRIPTION = (
        "①-1 规划网格：按参数走「均匀网格 / 方形全宽分区 / 自定义分区」三条分支之一，"
        "只算坐标，不切图、不写盘。纯函数，同样的图 + 参数必得同一份计划。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "参考图；只读它算尺寸和主体位置，不裁剪。"}),
                "settings": (SETTINGS_TYPE, {
                    "forceInput": True,
                    "tooltip": "STLU_PipelineSettings 的参数包。"}),
            },
            "optional": {
                "layerdiff_model": ("SEETHROUGH_LAYERDIFF_MODEL", {
                    "forceInput": True,
                    "tooltip": "可选：接拆层模型，换模型会写进指纹，但不会自动清空目录。"}),
                "depth_model": ("SEETHROUGH_DEPTH_MODEL", {
                    "forceInput": True,
                    "tooltip": "可选：接深度模型，同上。"}),
            },
        }

    RETURN_TYPES = (PLAN_TYPE,)
    RETURN_NAMES = ("plan",)
    OUTPUT_TOOLTIPS = ("网格计划：坐标 / 模式 / 警告（纯 dict）；接下一格。",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, image, settings, layerdiff_model=None, depth_model=None):
        plan = make_plan(image, settings, layerdiff_model, depth_model)
        return (plan,)


class STLU_PickTiles:
    """Resolve which tiles the user checked into concrete indices (pure)."""

    DESCRIPTION = (
        "①-2 选块：把面板里勾选的块（select）换算成本次要跑的格号，"
        "自定义分区按用户排序；一个都没勾就报错，而不是静默什么都不做。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {
                    "forceInput": True,
                    "tooltip": "STLU_PlanGrid 的计划。"}),
            },
        }

    RETURN_TYPES = (PLAN_TYPE,)
    RETURN_NAMES = ("plan",)
    OUTPUT_TOOLTIPS = ("同一份计划，补上 picked / indices / selection。",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, plan):
        return (pick_tiles(plan),)


class STLU_PlanContext:
    """Grow each selected tile's window by context_pad (pure)."""

    DESCRIPTION = (
        "①-3 上下文外扩：context_pad > 0 时，每个勾选块向外多裁这么多像素，"
        "让模型看到更多身体；生成完 STLU_CropBack 会裁回原来的块。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {
                    "forceInput": True,
                    "tooltip": "选块后的计划。"}),
            },
        }

    RETURN_TYPES = (PLAN_TYPE,)
    RETURN_NAMES = ("plan",)
    OUTPUT_TOOLTIPS = ("补上 context_boxes 的计划。",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, plan):
        return (plan_context(plan),)


class STLU_PrepareRun:
    """Fingerprint the run and write generation.json (never deletes)."""

    DESCRIPTION = (
        "①-4 准备运行（会写盘的唯一一步）：给本次参数算指纹，和目录里的旧 manifest "
        "比对；参数变了只提示、绝不清空目录（要清空请手动点「清空此目录」），"
        "写 generation.json 并发出网格通知。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {
                    "forceInput": True,
                    "tooltip": "外扩后的计划。"}),
                "image": ("IMAGE", {
                    "tooltip": "原图；指纹取自它，换了图（哪怕同尺寸）也不会误用旧块。"}),
            },
        }

    RETURN_TYPES = (PLAN_TYPE,)
    RETURN_NAMES = ("plan",)
    OUTPUT_TOOLTIPS = ("带 fingerprint / run / out 目录的计划。",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, plan, image):
        plan = prepare_run(plan, image)
        return (plan,)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_CropTiles:
    """Cut the selected tiles out of the image (pure)."""

    DESCRIPTION = (
        "①-5 裁剪：按计划把勾选的块（含 context_pad 外扩）从原图裁出来，"
        "输出可直接接标准 PreviewImage 查看。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "原图。"}),
                "plan": (PLAN_TYPE, {
                    "forceInput": True,
                    "tooltip": "准备运行后的计划。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("tiles", "indices")
    OUTPUT_TOOLTIPS = (
        "勾选块的裁剪图像列表，接 ② 顺序拆层。",
        "与 tiles 一一对应的原始格号。",
    )
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, image, plan):
        return crop_tiles(image, plan)


# ---------------------------------------------------------------------------
# ② the sequential loop (the one piece that must stay Python)
# ---------------------------------------------------------------------------

class STLU_GuideTile:
    """Paste finished neighbours into one tile's overlap before generation.

    The loop node calls the same function; this node exposes it so a single
    tile can be inspected by hand.
    """

    DESCRIPTION = (
        "②-1 单块回填（纯函数，单独接出来调试用）：把已生成邻居在重叠区的像素"
        "羽化回填进这一块。② 循环内部每一步调用的就是它。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile": ("IMAGE", {"tooltip": "要回填的这一块。"}),
                "index": ("STRING", {"forceInput": True,
                                     "tooltip": "这块的原始格号。"}),
                "plan": (PLAN_TYPE, {"forceInput": True,
                                     "tooltip": "计划（取坐标和引导参数）。"}),
            },
            "optional": {
                "generated": (GENERATED_TYPE, {
                    "forceInput": True,
                    "tooltip": "已经生成好的块（格号 -> 合成图）；不接 = 不回填。"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("guided",)
    OUTPUT_TOOLTIPS = ("回填后的输入图，可直接预览看改了哪里。",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, tile, index, plan, generated=None):
        state = guide_state(plan)
        done = dict(generated or {})
        result = guide_tile(tile, int(str(index).strip() or 0), state, done)
        frame = _first(result)
        return (frame,)


class STLU_SeqTileLoop:
    """The one genuine loop: guide -> SeeThrough chain -> crop back.

    Tile N has to see the pixels tile N-1 produced, which no acyclic graph can
    express, so this stays a single Python node.  Everything it does per tile
    is still a separately callable function (``guide_tile`` /
    ``_run_seethrough_tile`` / ``crop_back``).
    """

    DESCRIPTION = (
        "②-2 顺序拆层循环：逐块「回填 → GenerateLayers → GenerateDepth → "
        "PostProcess → 外扩裁回」。跨块状态（上一块的结果要喂给下一块）"
        "无法用无环节点图表达，所以这一段保留为 Python。"
        "composites 输出可以直接接预览，看每块合成后的结果。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE", {"forceInput": True,
                                    "tooltip": "① 裁好的块列表。"}),
                "indices": ("STRING", {"forceInput": True,
                                       "tooltip": "与 tiles 一一对应的格号。"}),
                "plan": (PLAN_TYPE, {"forceInput": True,
                                     "tooltip": "计划：坐标 / 引导参数 / 分辨率 / 种子。"}),
                "layerdiff_model": ("SEETHROUGH_LAYERDIFF_MODEL", {
                    "tooltip": "拆层模型加载器。"}),
                "depth_model": ("SEETHROUGH_DEPTH_MODEL", {
                    "tooltip": "深度模型加载器。"}),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_PARTS", "STRING", "IMAGE")
    RETURN_NAMES = ("parts", "indices", "composites")
    OUTPUT_TOOLTIPS = (
        "每块的拆层部件，接 ③ 存盘。",
        "与 parts 一一对应的格号。",
        "每块合成后的 RGBA 结果，可直接预览。",
    )
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, tiles, indices, plan, layerdiff_model, depth_model):
        state = guide_state(plan)
        run_dir = str((plan.get("doc") or {}).get("out", {}).get("dir") or "")
        parts, out_indices, _guided, composites = run_guided_tiles(
            tiles, indices, state, run_dir, layerdiff_model, depth_model,
            plan.get("seed", 42), plan.get("resolution", 1024),
            plan.get("num_inference_steps", 30),
            plan.get("depth_resolution", -1),
            bool((plan.get("settings") or {}).get("tblr_split", True)),
            bool((plan.get("settings") or {}).get("use_lama", True)),
            progress=lambda info: _notify("stlu.tile.progress", info),
            depth_seed=plan.get("depth_seed", plan.get("seed", 42)))
        return (parts, out_indices, composites)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_CropBack:
    """Cut a context-padded result back to the strip box (pure)."""

    DESCRIPTION = (
        "②-3 裁回：context_pad 外扩过的块，生成完要把多出来的边裁掉，"
        "否则合并几何会错。② 循环内部已做这一步；单独接出来可用于检查。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parts": ("SEETHROUGH_PARTS", {
                    "tooltip": "模型输出的部件。"}),
                "composite": ("IMAGE", {
                    "tooltip": "模型输出的合成图。"}),
                "index": ("STRING", {"forceInput": True,
                                     "tooltip": "这块的原始格号。"}),
                "plan": (PLAN_TYPE, {"forceInput": True,
                                     "tooltip": "计划（取 context_boxes 和坐标）。"}),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_PARTS", "IMAGE")
    RETURN_NAMES = ("parts", "composite")
    OUTPUT_TOOLTIPS = ("裁回后的部件。", "裁回后的合成图。")
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, parts, composite, index, plan):
        state = guide_state(plan)
        new_parts, new_composite = crop_back(
            parts, composite, int(str(index).strip() or 0), state)
        return (new_parts, new_composite)


# ---------------------------------------------------------------------------
# ③ save to disk -- one visible step per file
# ---------------------------------------------------------------------------

class STLU_WriteTilePSD:
    """Write one tile's layered PSD + preview + depth."""

    DESCRIPTION = (
        "③-1 写单块 PSD：这一块的 <前缀>_XX_rX_cX.psd、预览图、depth。"
        "输入是列表时逐块执行；产出记录接 ③-3 写清单。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parts": ("SEETHROUGH_PARTS", {
                    "tooltip": "逐块展开的部件（② 的输出）。"}),
                "plan": (PLAN_TYPE, {"forceInput": True,
                                     "tooltip": "计划（取目录 / 前缀 / 透明阈值）。"}),
                "index": ("STRING", {"forceInput": True,
                                     "tooltip": "这块的原始格号。"}),
            },
        }

    RETURN_TYPES = (TILE_TYPE,)
    RETURN_NAMES = ("tile",)
    OUTPUT_TOOLTIPS = ("这一块的记录（PSD / 预览 / depth 路径）。",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, parts, plan, index):
        return (write_tile_psd(parts, plan, index),)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_WriteTileImages:
    """Write one tile's per-layer PNGs into ``images/tile_XX_*``."""

    DESCRIPTION = (
        "③-2 写单块图层图：这一块的分层 PNG + layers.json（离线重建 / 二次放大用）。"
        "输入是列表时逐块执行。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parts": ("SEETHROUGH_PARTS", {
                    "tooltip": "逐块展开的部件（② 的输出）。"}),
                "plan": (PLAN_TYPE, {"forceInput": True,
                                     "tooltip": "计划（取目录 / 前缀）。"}),
                "index": ("STRING", {"forceInput": True,
                                     "tooltip": "这块的原始格号。"}),
            },
        }

    RETURN_TYPES = (TILE_IMAGE_TYPE,)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("这一块的 images/ 条目。",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, parts, plan, index):
        return (write_tile_images(parts, plan, index),)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_SaveOriginal:
    """Write the untouched source once per run as ``original.png``."""

    DESCRIPTION = (
        "③-3 存原图：在运行目录写一份 original.png（只一份，images.json 指向它）。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "原图。"}),
                "plan": (PLAN_TYPE, {"forceInput": True,
                                     "tooltip": "计划（取运行目录）。"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("original",)
    OUTPUT_TOOLTIPS = ("original.png 的文件名（没写成时为空串）。",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, image, plan):
        return (save_original(image, plan),)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


class STLU_WriteManifest:
    """Fold this run's tiles into ``manifest.json`` + ``images.json``.

    Also folds in tiles an earlier run finished in the same folder.
    """

    DESCRIPTION = (
        "③-4 写清单：汇总所有块的记录 → manifest.json / images.json；"
        "同时把同一目录里旧一轮已经生成好的块折进来（局部重跑后清单仍描述整张图）。"
        "输出 merge_doc 给 ④ 合并。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": (TILE_TYPE, {
                    "tooltip": "③-1 每块返回的记录（自动聚成列表）。"}),
                "images": (TILE_IMAGE_TYPE, {
                    "tooltip": "③-2 每块返回的 images 条目。"}),
                "original": ("STRING", {
                    "forceInput": True,
                    "tooltip": "③-3 返回的 original.png 文件名。"}),
                "plan": (PLAN_TYPE, {"forceInput": True,
                                     "tooltip": "计划（取目录 / 合并选项 / 指纹）。"}),
            },
        }

    RETURN_TYPES = (MANIFEST_TYPE, MERGE_TYPE, MANIFEST_TYPE)
    RETURN_NAMES = ("manifest", "merge_doc", "info")
    OUTPUT_TOOLTIPS = (
        "manifest.json 的内容（纯 dict）。",
        "④ 合并需要的选项（纯 dict）。",
        "给面板和外部节点看的汇总信息。",
    )
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, tiles, images, original, plan):
        result = write_manifest(tiles, images, _first(original, ""), plan)
        return (result["manifest"], result["merge_doc"], result["info"])

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


# ---------------------------------------------------------------------------
# ④ merge
# ---------------------------------------------------------------------------

class STLU_MergeTiles:
    """Merge one run folder into a single layered PSD.

    Deliberately NOT split further: seam ownership, scaling, whole-part
    exceptions and layer assembly are one indivisible algorithm, and cutting
    it into pieces would only move risk into correctness.  The ``report``
    output makes what it did visible instead.
    """

    DESCRIPTION = (
        "④ 合并：把 manifest 里的所有块按「接缝归属」合成一张分层 PSD。"
        "算法（接缝归属 / 缩放 / 整部件例外 / 分层组装）是一整块，无法再拆成小节点；"
        "但它做了什么全部写在 report 输出里。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merge_doc": (MERGE_TYPE, {
                    "forceInput": True,
                    "tooltip": "③-4 输出的 merge_doc（目录 + 选项）。"}),
            },
        }

    RETURN_TYPES = ("STRING", MERGE_TYPE)
    RETURN_NAMES = ("psd_path", "report")
    OUTPUT_TOOLTIPS = ("合并后的 PSD 完整路径，接 ⑤ 预览。",
                       "合并结果（文件 / 图层 / 组 / 块）。")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, merge_doc):
        result = merge_run_doc(merge_doc)
        return (result["psd_path"], result["payload"])

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "STLU_PipelineSettings": STLU_PipelineSettings,
    "STLU_PlanGrid": STLU_PlanGrid,
    "STLU_PickTiles": STLU_PickTiles,
    "STLU_PlanContext": STLU_PlanContext,
    "STLU_PrepareRun": STLU_PrepareRun,
    "STLU_CropTiles": STLU_CropTiles,
    "STLU_GuideTile": STLU_GuideTile,
    "STLU_SeqTileLoop": STLU_SeqTileLoop,
    "STLU_CropBack": STLU_CropBack,
    "STLU_WriteTilePSD": STLU_WriteTilePSD,
    "STLU_WriteTileImages": STLU_WriteTileImages,
    "STLU_SaveOriginal": STLU_SaveOriginal,
    "STLU_WriteManifest": STLU_WriteManifest,
    "STLU_MergeTiles": STLU_MergeTiles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STLU_PipelineSettings": "①-0 用户输入（参数包）",
    "STLU_PlanGrid": "①-1 规划网格",
    "STLU_PickTiles": "①-2 选块",
    "STLU_PlanContext": "①-3 上下文外扩",
    "STLU_PrepareRun": "①-4 准备运行",
    "STLU_CropTiles": "①-5 裁剪",
    "STLU_GuideTile": "②-1 单块回填（调试）",
    "STLU_SeqTileLoop": "②-2 顺序拆层循环",
    "STLU_CropBack": "②-3 裁回（调试）",
    "STLU_WriteTilePSD": "③-1 写单块 PSD",
    "STLU_WriteTileImages": "③-2 写单块图层图",
    "STLU_SaveOriginal": "③-3 存原图",
    "STLU_WriteManifest": "③-4 写清单",
    "STLU_MergeTiles": "④ 合并 PSD",
}

__all__ = (
    "STLU_PipelineSettings", "STLU_PlanGrid", "STLU_PickTiles",
    "STLU_PlanContext", "STLU_PrepareRun", "STLU_CropTiles", "STLU_GuideTile",
    "STLU_SeqTileLoop", "STLU_CropBack",
    "STLU_WriteTilePSD", "STLU_WriteTileImages", "STLU_SaveOriginal",
    "STLU_WriteManifest", "STLU_MergeTiles",
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
)
