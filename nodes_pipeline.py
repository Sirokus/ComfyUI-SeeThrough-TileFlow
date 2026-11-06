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


def _guide_feather_mask(height: int, width: int, feather: int) -> np.ndarray:
    """1.0 in the middle of the overlap, fading to 0 at its edge."""
    height = max(1, int(height))
    width = max(1, int(width))
    feather = max(0, int(feather))
    if feather <= 0:
        return np.ones((height, width), dtype=np.float32)
    fy = min(float(feather), max(0.0, (height - 1) / 2.0))
    fx = min(float(feather), max(0.0, (width - 1) / 2.0))
    yy = np.arange(height, dtype=np.float32)
    xx = np.arange(width, dtype=np.float32)
    # A 1-2 px span cannot hold a ramp: treat that axis as fully guided.
    wy = (np.ones(height, dtype=np.float32) if fy < 1.0
          else np.clip(np.minimum(yy, height - 1 - yy) / fy, 0.0, 1.0))
    wx = (np.ones(width, dtype=np.float32) if fx < 1.0
          else np.clip(np.minimum(xx, width - 1 - xx) / fx, 0.0, 1.0))
    return (wy[:, None] * wx[None, :]).astype(np.float32)


def _box_blur_3x3(arr: np.ndarray) -> np.ndarray:
    """Edge-padded 3x3 mean, so a per-pixel gate stays spatially coherent."""
    padded = np.pad(arr, 1, mode="edge")
    return (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]) / 9.0


GUIDE_OK = "ok"
GUIDE_EMPTY = "empty"
GUIDE_ERROR = "error"

def _guide_composite_result(parts: dict, alpha_threshold: int = 8) -> dict:
    """Flatten one tile's parts, distinguishing a real empty result from a bug.

    A fully transparent tile is a known model failure and is retried on a new
    seed.  An exception or a malformed ``frame_size`` is not: retrying it only
    burns two more diffusion runs and hides the real error, so it is reported
    as ``error`` with a traceback.
    """
    try:
        layers = tile_psd.build_tile_layers(
            parts, merge_parts=True, alpha_threshold=int(alpha_threshold))
    except Exception as exc:  # noqa: BLE001 - reported to the node layer
        return {
            "status": GUIDE_ERROR,
            "rgba": None,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }
    frame_h, frame_w = (int(v) for v in (parts.get("frame_size") or (0, 0))[:2])
    if frame_h <= 0 or frame_w <= 0:
        return {
            "status": GUIDE_ERROR,
            "rgba": None,
            "error": "parts have an invalid frame_size: %r"
                     % ((parts.get("frame_size"),)),
            "traceback": "",
        }
    if not layers:
        return {"status": GUIDE_EMPTY, "rgba": None, "error": "", "traceback": ""}
    canvas = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    for layer in layers:
        core_compose._alpha_over(canvas, layer["pixels"],
                                 int(layer["left"]), int(layer["top"]))
    if not canvas[..., 3].any():
        return {"status": GUIDE_EMPTY, "rgba": None, "error": "", "traceback": ""}
    return {"status": GUIDE_OK, "rgba": canvas, "error": "", "traceback": ""}


def _guide_composite(parts: dict, alpha_threshold: int = 8):
    """Backwards-compatible: the RGBA canvas, or ``None`` when unusable."""
    return _guide_composite_result(parts, alpha_threshold).get("rgba")

GUIDED_EMPTY_RETRIES = 2


def _run_seethrough_tile(layer_node, depth_node, post_node, input_tile, model,
                         depth_model, seed, resolution, steps, depth_res,
                         tblr, lama, alpha_threshold):
    """Generate -> depth -> post for one tile; return (parts, composite result)."""
    layers_data, _preview = layer_node.generate(
        input_tile, model, seed, resolution, steps)
    # GenerateDepth returns (layers_depth, preview), not a bare object.
    depth_data, _preview_depth = depth_node.generate(
        layers_data, depth_model, seed, depth_res)
    parts, _preview2 = post_node.process(depth_data, tblr, lama)
    return parts, _guide_composite_result(parts, alpha_threshold)

def _apply_tile_guide(tile, index: int, coords, generated: dict, guide: dict,
                      alpha_threshold: int = 8, fixed=None, input_box=None,
                      done=None):
    """Paste already-generated neighbouring tiles into this tile's overlap.

    ``fixed`` holds tiles cached from an earlier run: they are complete, so
    they may guide a tile that comes before them in grid order too.  Tiles
    produced earlier in *this* run are only used forward.

    ``done`` names the tiles finished in this run so far.  The guided node
    passes it because its tiles run in a user-defined order, where "index
    smaller than mine" no longer says which tile came first.  Standalone
    callers that omit it keep the original index-order contract.

    ``coords`` describes the content (strip) boxes the generated tiles cover;
    ``input_box`` is the larger window ① fed to the model when context
    padding is on.  They are the same box when padding is off.
    """
    if not guide.get("enabled", True) or not generated:
        return tile
    strength = min(100.0, max(0.0, float(guide.get("strength", 40)))) / 100.0
    if strength <= 0.0:
        return tile
    if index < 0 or index >= len(coords):
        return tile
    feather = int(guide.get("feather", 32))

    device, dtype = tile.device, tile.dtype
    arr = (tile[0].detach().float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    height, width = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in (input_box or coords[index]))
    tile_w, tile_h = max(1, x1 - x0), max(1, y1 - y0)
    scale_x = width / float(tile_w)
    scale_y = height / float(tile_h)

    changed = False
    fixed = fixed or set()
    done_set = None if done is None else set(done)
    # Earlier tiles only, unless the neighbour is a finished tile cached from
    # a previous run -- then its position in the queue does not matter.
    for other in sorted(generated):
        if other == index:
            # A cached copy of this very tile must not seed itself.
            continue
        if done_set is not None:
            if other not in done_set and other not in fixed:
                continue
        elif other >= index and other not in fixed:
            continue
        source = generated[other]
        ox0, oy0 = max(x0, int(coords[other][0])), max(y0, int(coords[other][1]))
        ox1, oy1 = min(x1, int(coords[other][2])), min(y1, int(coords[other][3]))
        if ox0 >= ox1 or oy0 >= oy1:
            continue

        sx0 = int(coords[other][0])
        sy0 = int(coords[other][1])
        other_w = max(1, int(coords[other][2]) - sx0)
        other_h = max(1, int(coords[other][3]) - sy0)
        rgba = source
        if rgba.shape[0] != other_h or rgba.shape[1] != other_w:
            rgba = core_io.resize_rgba_premultiplied(rgba, other_w, other_h)

        piece = rgba[oy0 - sy0:oy1 - sy0, ox0 - sx0:ox1 - sx0]
        if piece.size == 0:
            continue
        dx0 = int(round((ox0 - x0) * scale_x))
        dy0 = int(round((oy0 - y0) * scale_y))
        dx1 = int(round((ox1 - x0) * scale_x))
        dy1 = int(round((oy1 - y0) * scale_y))
        dst_w, dst_h = dx1 - dx0, dy1 - dy0
        if dst_w <= 0 or dst_h <= 0:
            continue
        if piece.shape[0] != dst_h or piece.shape[1] != dst_w:
            piece = core_io.resize_rgba(piece, dst_w, dst_h)

        weight = _guide_feather_mask(dst_h, dst_w, feather)
        alpha = (piece[..., 3].astype(np.float32) / 255.0) * weight * strength
        # Only carry forward pixels where the earlier tile's *reconstruction*
        # agrees with what this tile's source image really shows.  SeeThrough
        # repaints/completes its input per tile; pasting those invented pixels
        # into the neighbour makes the model split the wrong thing (a half
        # face, stray clothing).  Large disagreements stay on the source.
        tolerance = float(guide.get("tolerance", 16) or 0)
        if tolerance > 0.0:
            source = arr[dy0:dy1, dx0:dx1].astype(np.float32)
            diff = np.abs(piece[..., :3].astype(np.float32) - source).mean(axis=-1)
            gate = np.clip(1.0 - diff / tolerance, 0.0, 1.0).astype(np.float32)
            # Smooth the gate: a raw per-pixel threshold leaves salt-and-pepper
            # holes right at the limit, which the model reads as a broken image.
            alpha = alpha * _box_blur_3x3(gate)
        if not np.any(alpha > 0.0):
            continue
        dst = arr[dy0:dy1, dx0:dx1].astype(np.float32)
        src = piece[..., :3].astype(np.float32)
        a = alpha[..., None]
        arr[dy0:dy1, dx0:dx1] = np.clip(dst * (1.0 - a) + src * a, 0, 255).astype(np.uint8)
        changed = True

    if not changed:
        return tile
    out = torch.from_numpy(arr.astype(np.float32) / 255.0)[None]
    return out.to(device=device, dtype=dtype)


def _crop_tile_parts(parts, composite, content_box, input_box):
    """Crop a context-padded tile's result back to its strip box.

    ① may feed the model a window larger than the strip; PSD size, merge
    geometry and guides all expect the strip box, so every part image/depth
    and the composite RGBA are cut back here.  Parts that do not carry the
    padded frame are left untouched (they were not produced from it).
    """
    if not input_box or not content_box:
        return parts, composite
    cx0, cy0, cx1, cy1 = (int(v) for v in content_box[:4])
    ix0, iy0, ix1, iy1 = (int(v) for v in input_box[:4])
    pad_w, pad_h = ix1 - ix0, iy1 - iy0
    if pad_w <= 0 or pad_h <= 0 or cx1 <= cx0 or cy1 <= cy0:
        return parts, composite

    # The parts are in the MODEL's pixel space, which is not the padded
    # window's pixel size: a 1950 px window comes back at ``resolution``
    # (512).  Map the strip box into that space first.  Cutting with source
    # pixels would slice a 512 px result at 128..1822, and then declaring the
    # frame to be 1694 px puts that sliver in the PSD's top-left corner.
    frame = parts.get("frame_size") or (0, 0)
    frame_h, frame_w = int(frame[0]), int(frame[1])
    if frame_w <= 0 or frame_h <= 0:
        frame_w, frame_h = pad_w, pad_h
    scale_x = frame_w / float(pad_w)
    scale_y = frame_h / float(pad_h)
    dx = max(0, int(round((cx0 - ix0) * scale_x)))
    dy = max(0, int(round((cy0 - iy0) * scale_y)))
    width = min(frame_w, int(round((cx1 - ix0) * scale_x))) - dx
    height = min(frame_h, int(round((cy1 - iy0) * scale_y))) - dy
    if width <= 0 or height <= 0:
        return parts, composite
    if dx == 0 and dy == 0 and (frame_w, frame_h) == (width, height):
        return parts, composite

    out = dict(parts)
    tag2pinfo = {}
    for tag, pinfo in (parts.get("tag2pinfo") or {}).items():
        if not isinstance(pinfo, dict):
            tag2pinfo[tag] = pinfo
            continue
        item = dict(pinfo)
        for key in ("img", "depth"):
            value = item.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if (arr.ndim >= 2
                    and arr.shape[0] == frame_h and arr.shape[1] == frame_w):
                item[key] = np.ascontiguousarray(arr[dy:dy + height, dx:dx + width])
        xyxy = item.get("xyxy")
        if xyxy is not None and len(xyxy) >= 2:
            moved = [min(width, max(0, int(xyxy[0]) - dx)),
                     min(height, max(0, int(xyxy[1]) - dy))]
            if len(xyxy) >= 4:
                # Keep the box ordered: a part that sat fully outside the
                # strip must not come back with its right/top below its left.
                moved += [min(width, max(moved[0], int(xyxy[2]) - dx)),
                          min(height, max(moved[1], int(xyxy[3]) - dy))]
            else:
                moved += list(xyxy[2:])
            item["xyxy"] = moved
        tag2pinfo[tag] = item
    out["tag2pinfo"] = tag2pinfo
    out["frame_size"] = (height, width)

    if isinstance(composite, dict):
        composite = dict(composite)
        rgba = composite.get("rgba")
        if rgba is not None:
            arr = np.asarray(rgba)
            if (arr.ndim == 3
                    and arr.shape[0] == frame_h and arr.shape[1] == frame_w):
                composite["rgba"] = np.ascontiguousarray(
                    arr[dy:dy + height, dx:dx + width])
    return out, composite


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
                    "tooltip": "方形分区的模型分辨率上限。默认 768 更稳；显存允许可调到 896/1024。"
                               "实际值会显示在面板里。"}),
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

    def execute(self, image, tile_width=1024, tile_height=1024, overlap=128,
                select="all", resolution=512, num_inference_steps=30,
                depth_resolution=-1, random_seed=0, tblr_split=True, use_lama=True,
                alpha_threshold=8, out_dir="SeeThrough/我的自设/", group="",
                filename_prefix="tile", merge_name="merged",
                group_layers=True, include_depth=False,
                include_original=False, guide_context=True,
                guide_feather=32, guide_strength=40, guide_tolerance=16,
                seam_feather=32, layerdiff_model=None, depth_model=None,
                grid_offset_x=0, grid_offset_y=0,
                partition_mode="均匀网格", strip_max_side=2048,
                strip_max_resolution=768, strip_margin=32, select_plan="",
                context_pad=0, custom_regions="[]"):
        if image is None or not hasattr(image, "shape") or image.ndim < 3:
            raise ValueError("STLU_PipelineInput: 需要连接 image（标准 LoadImage 即可）")
        try:
            (resolution, num_inference_steps, depth_resolution,
             random_seed) = stlu_limits.check_all(
                resolution, num_inference_steps, depth_resolution, random_seed)
        except ValueError as exc:
            raise ValueError("STLU_PipelineInput: %s" % exc) from exc
        batch = image if image.ndim == 4 else image[None]
        frame = batch[0].detach().float().cpu()
        height, width = int(frame.shape[0]), int(frame.shape[1])
        now = datetime.datetime.now()
        created = now.strftime("%Y%m%d_%H%M%S")
        partition_mode = str(partition_mode or "").strip()
        is_strips = partition_mode == "方形全宽分区"
        is_custom = partition_mode == region_plan.CUSTOM_LABEL
        effective_resolution = int(resolution)
        effective_depth_resolution = int(depth_resolution)
        plan_meta: dict = {}
        plan_warnings: list = []
        fallback = None
        # The run folder is needed before selection mapping (the previous
        # manifest is the authoritative old geometry).
        wave_dir = tile_psd.flat_directory(
            _output_root(), str(out_dir or "SeeThrough"), str(group or ""))

        if is_strips:
            planned = _plan_strips_frame(
                frame, width, height,
                overlap=int(overlap), max_side=int(strip_max_side),
                max_resolution=int(strip_max_resolution), margin=int(strip_margin),
                tile_width=int(tile_width), tile_height=int(tile_height),
                resolution=int(resolution),
                grid_offset_x=int(grid_offset_x), grid_offset_y=int(grid_offset_y),
                created=created)
            doc = planned["doc"]
            coords = planned["coords"]
            effective_resolution = int(planned["resolution"])
            plan_meta = planned["plan_meta"]
            plan_warnings = list(planned["warnings"])
            fallback = planned["fallback"]
            if effective_depth_resolution > effective_resolution:
                effective_depth_resolution = effective_resolution
                plan_warnings.append(
                    "深度分辨率高于分区层分辨率，已跟随层分辨率 %d。"
                    % effective_resolution)
        elif is_custom:
            custom_warnings = []
            regions = region_plan.parse_custom_regions(
                custom_regions, width, height, custom_warnings)
            doc = region_plan.custom_document(
                regions, width, height,
                global_resolution=int(effective_resolution),
                created=created, overlap=int(overlap),
                warnings=custom_warnings)
            coords = doc["coords"]
            plan_meta = dict(doc["partition_plan"])
            plan_warnings = list(custom_warnings)
            fallback = None
        else:
            doc = tile_psd.grid_document(
                (width, height), (tile_width, tile_height), int(overlap),
                source_image="", name="grid", created=created,
                offset_x=int(grid_offset_x), offset_y=int(grid_offset_y),
            )
            coords = doc["coords"]
        raw_select = str(select if select is not None else "all").strip()
        picked = _resolve_picked_selection(
            raw_select, coords, is_strips=is_strips, is_custom=is_custom,
            planned_id=str(plan_meta.get("plan_id") or ""),
            select_plan=str(select_plan or ""), previous_dir=wave_dir,
            width=width, height=height, tile_width=int(tile_width),
            tile_height=int(tile_height), overlap=int(overlap),
            grid_offset_x=int(grid_offset_x), grid_offset_y=int(grid_offset_y))
        if is_custom:
            # Custom windows run in the order the user arranged them, not in
            # index order.  A checked subset keeps that order too, because the
            # guide only flows forward through the run.
            order = list(plan_meta.get("order") or range(len(coords)))
            chosen = set(int(i) for i in picked)
            picked = [int(i) for i in order if int(i) in chosen]
        if not picked:
            mode_hint = "方形分区" if is_strips else "均匀网格"
            raise ValueError(
                "STLU_PipelineInput: 当前%s没有勾选任何块/条（select=%r）。"
                "请在 ① 面板点「全选」或点选要生成的区域，再 Queue。"
                % (mode_hint, raw_select))
        encoded = "all" if len(picked) == len(coords) else encode_tile_selection(picked)
        doc["selection"] = encoded

        # Optional context padding: the model sees a window larger than the
        # strip, so a leg-only strip also carries the torso above it.  The
        # strip geometry (coords) is untouched -- ② crops the result back.
        context_pad = max(0, min(512, int(context_pad or 0)))
        context_boxes = {}
        if context_pad > 0:
            for index in picked:
                x0, y0, x1, y1 = (int(v) for v in coords[index])
                box = [max(0, x0 - context_pad), max(0, y0 - context_pad),
                       min(width, x1 + context_pad), min(height, y1 + context_pad)]
                if box != [x0, y0, x1, y1]:
                    context_boxes[str(index)] = box
            doc["context_boxes"] = context_boxes
        doc["context_pad"] = int(context_pad)

        # The picked folder IS the run folder.  Earlier runs live here too,
        # which is what lets a partial re-run reuse their finished tiles.
        rel = os.path.relpath(wave_dir, _output_root()).replace(os.sep, "/")
        wave = os.path.basename(wave_dir.rstrip("/\\")) or "run"
        fingerprint = _image_fingerprint(frame)
        doc.setdefault("source", {})["fingerprint"] = fingerprint
        # Sequential (guided) runs read this block: neighbouring tiles that
        # already finished are composited back into the overlap before the
        # next tile reaches SeeThrough.
        guide_block = {
            "enabled": bool(guide_context),
            "feather": max(0, int(guide_feather)),
            "strength": min(100, max(0, int(guide_strength))),
            "tolerance": max(0, min(255, int(guide_tolerance))),
        }
        doc["guide"] = guide_block

        # Everything that can change the pixels of a tile.  Merge-only options
        # (group_layers / depth / original / seam_feather) are deliberately
        # absent: changing them only needs ④ to run again, not a re-diffusion.
        generation = {
            "version": 1,
            # Bumped whenever tile pixels can change; joins generation_id so
            # a code update cannot silently reuse tiles made by older logic.
            "algo_version": int(tile_psd.ALGO_VERSION),
            "source": fingerprint,
            "source_size": [int(width), int(height)],
            "tile_size": [int(v) for v in doc["tile_size"]],
            "overlap": int(doc["overlap"]),
            "resolution": int(effective_resolution),
            "num_inference_steps": int(num_inference_steps),
            "depth_resolution": int(effective_depth_resolution),
            "random_seed": int(random_seed),
            "tblr_split": bool(tblr_split),
            "use_lama": bool(use_lama),
            "alpha_threshold": int(alpha_threshold),
            "context_pad": int(context_pad),
            "guide": guide_block,
            "layerdiff_model": _model_identity(layerdiff_model),
            "depth_model": _model_identity(depth_model),
        }
        if is_strips:
            # Only geometry goes into the fingerprint: warnings/notes can
            # change without changing a single pixel, and a warning-only
            # difference must not clear a finished folder.
            generation["partition"] = {
                key: plan_meta.get(key)
                for key in ("mode", "side", "resolution", "density", "overlap",
                            "requested_overlap", "margin", "plan_id", "regions")
                if plan_meta.get(key) is not None
            }
            if plan_meta.get("fallback"):
                generation["partition"]["fallback"] = plan_meta["fallback"]
            # A fallback run really is the uniform grid, including its phase.
            if str(doc.get("mode") or "") == "uniform_fallback" and (
                    int(grid_offset_x) or int(grid_offset_y)):
                generation["offset"] = [int(grid_offset_x), int(grid_offset_y)]
        elif is_custom:
            # The region list is in run order, so moving a window up/down or
            # changing a per-window resolution changes the fingerprint too:
            # both change what the guide sees, and old tiles cannot be reused.
            generation["partition"] = {
                "mode": region_plan.CUSTOM_MODE,
                "plan_id": str(plan_meta.get("plan_id") or ""),
                "regions": [{"box": list(r["box"]),
                             "resolution": int(r["resolution"])}
                            for r in (plan_meta.get("regions") or [])],
            }
        # Only when the grid is actually shifted: keeping the key out at 0
        # means folders written by the previous version stay compatible.
        # Successful subject strips / custom windows never use the grid phase.
        elif int(grid_offset_x) or int(grid_offset_y):
            generation["offset"] = [int(grid_offset_x), int(grid_offset_y)]
        gid = tile_psd.generation_id(generation)
        doc["generation"] = generation
        doc["generation_id"] = gid

        # A changed pixel-affecting parameter means the old tiles cannot be
        # trusted: the folder is cleared before anything runs, so stale tiles
        # from another generation can never be mixed into the merge.  The
        # selection still decides how much work happens -- an explicit subset
        # is the user's instruction and is never widened to the whole grid,
        # it only regenerates the checked tiles into the now-empty folder.
        reset_info = None
        explicit_subset = 0 < len(picked) < len(coords)
        previous = tile_psd.read_manifest(wave_dir)
        reset_keys = []
        reset_reason = ""
        old_generation = None
        if previous is not None:
            old_generation = previous.get("generation")
        else:
            # An interrupted run leaves tiles and generation.json behind but
            # no manifest; that fingerprint still describes those tiles.
            old_generation = (tile_psd.read_generation(wave_dir) or {}).get("generation")
        if previous is not None and (not isinstance(old_generation, dict) or not old_generation):
            reset_reason = "旧结果没有参数指纹"
        elif isinstance(old_generation, dict) and old_generation:
            # Seed / context padding are deliberately excluded: changing them
            # regenerates only the checked tiles and leaves the rest of the
            # folder usable.
            reset_keys = [key for key in generation
                          if key not in tile_psd.LOCAL_RERUN_KEYS
                          and old_generation.get(key) != generation[key]]
            # A key that only the old run has still counts: that is what
            # moving the grid back to offset 0 looks like.
            reset_keys += [key for key in old_generation
                           if key not in tile_psd.LOCAL_RERUN_KEYS
                           and key not in generation]
            if reset_keys:
                reset_reason = "生成参数已变"
        if reset_reason and explicit_subset:
            cleared = tile_psd.clear_run_outputs(wave_dir, previous)
            tile_psd.verify_run_cleared(wave_dir)
            reset_info = {
                "required": True,
                "partial": True,
                "reason": reset_reason,
                "keys": reset_keys,
                "tiles": len(picked),
                "removed": int(cleared.get("removed") or 0),
            }
            _notify("stlu.cache_reset", {
                "dir": wave_dir,
                "rel": rel,
                "reason": reset_reason,
                "keys": reset_keys,
                "tiles": len(picked),
                "partial": True,
                "removed": reset_info["removed"],
            })
            print("[STLU] generation changed (%s) -> cleared %s, re-running only the "
                  "%d selected tiles" % (
                      ", ".join(reset_keys) or reset_reason, rel, len(picked)), flush=True)
        elif reset_reason:
            picked = list(range(len(coords)))
            encoded = "all"
            doc["selection"] = encoded
            cleared = tile_psd.clear_run_outputs(wave_dir, previous)
            tile_psd.verify_run_cleared(wave_dir)
            reset_info = {
                "required": True,
                "reason": reset_reason,
                "keys": reset_keys,
                "tiles": len(picked),
                "removed": int(cleared.get("removed") or 0),
            }
            _notify("stlu.cache_reset", {
                "dir": wave_dir,
                "rel": rel,
                "reason": reset_reason,
                "keys": reset_keys,
                "tiles": len(picked),
                "removed": reset_info["removed"],
            })
            print("[STLU] generation changed -> cleared %s, regenerating all %d tiles (%s)" % (
                rel, len(picked), ", ".join(reset_keys) or reset_reason), flush=True)

        out_info = {
            "dir": wave_dir,
            "rel": rel,
            "wave": wave,
            "flat": True,
            "fingerprint": fingerprint,
            "out_dir": str(out_dir or "SeeThrough"),
            "group": str(group or "").strip(),
            "prefix": str(filename_prefix or "tile"),
            "alpha_threshold": int(alpha_threshold),
            "merge_name": str(merge_name or "merged"),
            "group_layers": bool(group_layers),
            "include_depth": bool(include_depth),
            "include_original": bool(include_original),
            "seam_feather": max(0, int(seam_feather)),
        }
        if reset_info:
            out_info["reset"] = reset_info
        doc["out"] = out_info
        try:
            tile_psd.write_generation(wave_dir, generation, gid)
        except OSError as exc:
            raise RuntimeError(
                "STLU_PipelineInput: 无法写入生成指纹 %s：%s。"
                "请检查输出目录权限后重试。" % (wave_dir, exc)) from exc
        grid_doc = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))

        # Strips are cut from the background-flattened frame when the planner
        # could isolate the subject; otherwise the frame as loaded.
        flat = planned.get("flat") if is_strips else None
        source_frame = frame
        if flat is not None:
            source_frame = torch.from_numpy(
                flat.astype(np.float32) / 255.0).to(frame.device)

        tiles = []
        indices = []
        for index in picked:
            box = context_boxes.get(str(index))
            if box:
                x0, y0, x1, y1 = box
            else:
                x0, y0, x1, y1 = (int(v) for v in coords[index])
            crop = source_frame[int(y0):int(y1), int(x0):int(x1), :]
            tiles.append(crop[None].contiguous())
            indices.append(str(index))

        info = {
            "grid": "",
            "grid_doc": grid_doc,
            "dir": rel,
            "mode": str(doc.get("mode") or "uniform"),
            "partition": plan_meta,
            "warnings": plan_warnings,
            "fallback": fallback,
            "count": int(doc["count"]),
            "selected": len(picked),
            "indices": indices,
            "rows": int(doc["rows"]),
            "columns": int(doc["columns"]),
            "source": [int(width), int(height)],
            "tile_size": [int(v) for v in doc["tile_size"]],
            "overlap": int(doc["overlap"]),
            "resolution": int(effective_resolution),
            "selection": encoded,
            "out": out_info,
        }
        _notify("stlu.grid", info)
        if is_custom:
            print("[STLU] input: %d/%d custom windows (%s) from %dx%d -> %s "
                  "(%s, order %s)" % (
                      len(picked), doc["count"],
                      region_plan.summaries(plan_meta.get("regions") or [],
                                            int(effective_resolution)),
                      width, height, rel, encoded,
                      ",".join(str(i) for i in picked)),
                  flush=True)
        elif is_strips and str(doc.get("mode") or "") == "subject_strips":
            print("[STLU] input: %d/%d strips (side %d, resolution %d) from %dx%d -> %s (%s)" % (
                len(picked), doc["count"], doc["tile_size"][0],
                int(effective_resolution), width, height, rel, encoded),
                flush=True)
        else:
            print("[STLU] input: %d/%d tiles (%dx%d) from %dx%d -> %s (%s)" % (
                len(picked), doc["count"], doc["tile_size"][0], doc["tile_size"][1],
                width, height, rel, encoded),
                flush=True)
        return {
            "ui": {"stlu_grid": [info]},
            "result": (tiles, indices, grid_doc, frame[None].contiguous(),
                       int(effective_resolution), int(num_inference_steps),
                       int(effective_depth_resolution), int(random_seed),
                       bool(tblr_split), bool(use_lama)),
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

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
        import comfy.model_management
        import comfy.utils

        layer_cls = _seethrough_node_class("SeeThrough_GenerateLayers")
        depth_cls = _seethrough_node_class("SeeThrough_GenerateDepth")
        post_cls = _seethrough_node_class("SeeThrough_PostProcess")
        if not (layer_cls and depth_cls and post_cls):
            raise RuntimeError(
                "STLU_GuidedLayers: 找不到 SeeThrough 节点（ComfyUI-See-through 未加载？）")

        layer_node, depth_node, post_node = layer_cls(), depth_cls(), post_cls()

        model = _first(layerdiff_model)
        depth_model = _first(depth_model)
        seed = int(_first(random_seed, 42))
        resolution = int(_first(resolution, 512))
        steps = int(_first(num_inference_steps, 30))
        depth_res = int(_first(depth_resolution, -1))
        try:
            (resolution, steps, depth_res,
             seed) = stlu_limits.check_all(resolution, steps, depth_res, seed)
        except ValueError as exc:
            raise ValueError("STLU ②: %s" % exc) from exc
        tblr = bool(_first(tblr_split, True))
        lama = bool(_first(use_lama, True))

        doc = None
        raw_grid = _first(grid_doc, "")
        if str(raw_grid or "").strip():
            try:
                doc = tile_psd.load_grid_ref(str(raw_grid))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    "STLU ②: 无法读取 ① 的 grid_doc（%s）。"
                    "请检查 ①→② 的 grid_doc 连线，然后重新 Queue ①。" % exc) from exc
        guide = dict((doc or {}).get("guide") or {})
        coords = list((doc or {}).get("coords") or [])
        # User-placed windows may each carry their own model resolution.
        # Uniform grid / strips fall through to the single widget value.
        regions_by_index = {}
        for region in ((doc or {}).get("partition_plan") or {}).get("regions") or []:
            if not isinstance(region, dict):
                continue
            try:
                regions_by_index[int(region["index"])] = int(region["resolution"])
            except (KeyError, TypeError, ValueError):
                continue
        # Keys travel through JSON, so they arrive as strings.
        context_boxes = {}
        for key, box in ((doc or {}).get("context_boxes") or {}).items():
            try:
                context_boxes[int(key)] = [int(v) for v in box[:4]]
            except (TypeError, ValueError, IndexError):
                continue
        if not coords:
            raise ValueError(
                "STLU ②: grid_doc 里没有块坐标，无法做引导。"
                "请确认 ①→② 的 grid_doc 已连接，且 ① 正常完成。")
        alpha_threshold = int(((doc or {}).get("out") or {}).get("alpha_threshold") or 8)

        tiles = _list_values(tiles)
        raw_indices = _list_values(indices)
        total = len(tiles)
        if not total:
            return ([], [])

        index_order = []
        for pos in range(total):
            raw_index = raw_indices[pos] if pos < len(raw_indices) else pos
            try:
                index_order.append(int(str(_first(raw_index, pos)).strip()))
            except (TypeError, ValueError):
                index_order.append(pos)

        # Tiles an earlier run already finished in this same folder are real
        # generated content: merge them as guide now, and step ③ puts them
        # back into the manifest so a one-tile re-run still merges complete.
        cached: dict = {}
        run_dir = str(((doc or {}).get("out") or {}).get("dir") or "")
        if run_dir and guide.get("enabled", True):
            try:
                entries = tile_psd.cached_tile_entries(
                    run_dir, doc, skip=index_order)
            except Exception as exc:  # noqa: BLE001
                entries = []
                print("[STLU] guided: cannot read cached tiles: %s" % exc, flush=True)
            for entry in entries:
                rgba = tile_psd.composite_psd(
                    os.path.join(run_dir, str(entry.get("file") or "")))
                if rgba is not None:
                    cached[int(entry["index"])] = rgba
            if cached:
                print("[STLU] guided: reusing %d finished tiles from %s" % (
                    len(cached), run_dir), flush=True)

        print("[STLU] guided: %d tiles, context=%s feather=%s strength=%s" % (
            total, guide.get("enabled", True), guide.get("feather"),
            guide.get("strength")), flush=True)

        pbar = comfy.utils.ProgressBar(total)
        generated: dict = dict(cached)
        fixed = set(cached)
        parts_out: list = []
        indices_out: list = []

        for pos, tile in enumerate(tiles):
            index = index_order[pos]
            comfy.model_management.throw_exception_if_processing_interrupted()

            # Per-window resolution when the document names one; otherwise the
            # single widget value (uniform grid / subject strips).
            tile_res = int(regions_by_index.get(index, resolution))
            tile_depth = depth_res if depth_res == -1 else min(depth_res, tile_res)
            input_box = context_boxes.get(index)
            input_tile = _apply_tile_guide(
                tile, index, coords, generated, guide, alpha_threshold,
                fixed=fixed, input_box=input_box, done=set(generated))

            try:
                parts, composite = _run_seethrough_tile(
                    layer_node, depth_node, post_node, input_tile, model,
                    depth_model, seed, tile_res, steps, tile_depth, tblr,
                    lama, alpha_threshold)
            except comfy.model_management.InterruptProcessingException:
                raise
            except Exception as exc:  # noqa: BLE001 - add the tile number
                raise RuntimeError(
                    "STLU ②: 第 %d 块生成失败（%s: %s）。\n%s"
                    % (index, type(exc).__name__, exc,
                       traceback.format_exc())) from exc
            if composite["status"] == GUIDE_ERROR:
                raise RuntimeError(
                    "STLU ②: 第 %d 块合成失败（%s）。这是代码/部件数据错误，"
                    "不是模型生成的空块；已停止本次运行。\n%s"
                    % (index, composite.get("error") or "unknown",
                       composite.get("traceback") or ""))

            # An all-transparent result is a model collapse, not a real empty
            # tile: retry this one tile on a new seed before giving up.  The
            # input tile stays the same -- nothing about the guide changes.
            attempt = 0
            while composite["status"] == GUIDE_EMPTY and attempt < GUIDED_EMPTY_RETRIES:
                attempt += 1
                retry_seed = (seed + attempt * 0x9E3779B1) % (2 ** 32)
                print("[STLU] guided: tile %d came back empty with seed %d, "
                      "retrying on seed %d (%d/%d)" % (
                          index, seed, retry_seed, attempt,
                          GUIDED_EMPTY_RETRIES), flush=True)
                comfy.model_management.throw_exception_if_processing_interrupted()
                try:
                    parts, composite = _run_seethrough_tile(
                        layer_node, depth_node, post_node, input_tile, model,
                        depth_model, retry_seed, tile_res, steps, tile_depth,
                        tblr, lama, alpha_threshold)
                except comfy.model_management.InterruptProcessingException:
                    raise
                except Exception as exc:  # noqa: BLE001 - add the tile number
                    raise RuntimeError(
                        "STLU ②: 第 %d 块在重试时生成失败（%s: %s）。\n%s"
                        % (index, type(exc).__name__, exc,
                           traceback.format_exc())) from exc
                if composite["status"] == GUIDE_ERROR:
                    raise RuntimeError(
                        "STLU ②: 第 %d 块在重试时合成失败（%s）。\n%s"
                        % (index, composite.get("error") or "unknown",
                           composite.get("traceback") or ""))
            if composite["status"] == GUIDE_EMPTY:
                print("[STLU] guided: tile %d is still empty after %d "
                      "retries" % (index, GUIDED_EMPTY_RETRIES), flush=True)
            else:
                if input_box:
                    # The model saw the padded window; everything downstream
                    # (merge geometry, PSD size, guides) wants the strip box.
                    parts, composite = _crop_tile_parts(
                        parts, composite, coords[index], input_box)
                generated[index] = composite["rgba"]
            parts_out.append(parts)
            indices_out.append(str(index))
            pbar.update(1)
            print("[STLU] guided: tile %d done (%d/%d)" % (index, pos + 1, total),
                  flush=True)

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
        try:
            tile_index = int(str(_first(index, "0")).strip())
        except (TypeError, ValueError):
            tile_index = 0

        doc = None
        raw_grid = _first(grid_doc, "")
        if str(raw_grid or "").strip():
            try:
                doc = tile_psd.load_grid_ref(str(raw_grid))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    "STLU_TilePSD: 无法读取 ① 的 grid_doc（%s）。"
                    "请检查 ①→③ 的 grid_doc 连线。" % exc) from exc
        out = _grid_out(doc)

        wave = None
        raw_wave = _first(wave_doc, "")
        if str(raw_wave or "").strip():
            wave = _wave_doc(wave_doc)
        elif out.get("dir"):
            wave = out
        if wave is None:
            raise ValueError(
                "STLU_TilePSD: 需要 grid_doc（接 STLU_PipelineInput）或 wave_doc（旧版 STLU_Wave）")
        wave_dir = str(wave["dir"])
        prefix = str(wave.get("prefix") or out.get("prefix") or "tile")
        if "alpha_threshold" in out:
            alpha_threshold = int(out["alpha_threshold"])

        if not isinstance(parts, dict):
            # ``None`` means the upstream node produced nothing at all.  That
            # is an execution error, not the normal "this edge tile is empty"
            # case (which arrives as a dict without ``tag2pinfo``).
            raise ValueError(
                "STLU_TilePSD: tile %d 没有收到部件（上游返回 %r）。"
                "请检查 ② 是否正常完成。" % (tile_index, parts))
        if str(parts.get("status") or "") == "error":
            raise RuntimeError(
                "STLU_TilePSD: tile %d 上游报告错误：%s"
                % (tile_index, parts.get("error") or "unknown"))

        # Locked parts: the per-tile locks set in the ⑤ preview panel.  The
        # ``_locks/`` snapshot survives a folder reset, so it is tried first;
        # the previous tile PSD is only a fallback for locks saved before the
        # store existed.  Either source can revive a tile the new run left
        # empty.
        coords = (doc or {}).get("coords") or []
        src = (doc or {}).get("source") or {}
        source = None
        try:
            if int(src.get("width") or 0) > 0 and int(src.get("height") or 0) > 0:
                source = [int(src["width"]), int(src["height"])]
        except (TypeError, ValueError):
            source = None
        locks = []
        seen_locks = set()
        for name in tile_psd.tile_locks(wave_dir).get(tile_index, []):
            name = str(name).strip()
            if name and name.lower() not in seen_locks:
                seen_locks.add(name.lower())
                locks.append(name)
        if locks:
            parts = dict(parts)
            restored, loaded = tile_psd.restore_locked_parts(
                parts, wave_dir, tile_index, locks, coords=coords, source=source)
            missing = [name for name in locks
                       if name.strip().lower() not in loaded]
            if missing:
                old_path = tile_psd.find_tile_psd(wave_dir, prefix, tile_index)
                if old_path:
                    previous = tile_psd.read_manifest(wave_dir) or {}
                    old_entry = None
                    for entry in previous.get("tiles") or []:
                        try:
                            if int(entry.get("index", -1)) == tile_index:
                                old_entry = entry
                                break
                        except (TypeError, ValueError):
                            continue
                    try:
                        row, col = (tile_psd.tile_rc(coords, tile_index)
                                    if coords else (0, 0))
                        tile_folder = ("tile_%02d_r%d_c%d" % (tile_index, row, col)
                                       if coords else "tile_%02d" % tile_index)
                    except Exception:  # noqa: BLE001 - fall back to no depth reuse
                        tile_folder = "tile_%02d" % tile_index
                    images_dir = os.path.join(wave_dir, "images", tile_folder)
                    restored.extend(tile_psd.replace_locked_parts(
                        parts, old_path, missing, old_entry,
                        images_dir=images_dir))
            if restored:
                print("[STLU] tile %d: kept locked parts: %s" % (
                    tile_index, ", ".join(sorted(set(restored)))), flush=True)

        if not (parts.get("tag2pinfo") or {}):
            # Still nothing to write: skip the tile (an edge tile with no
            # content and no locks to restore).
            return {"ui": {"stlu_tile": [{"index": tile_index, "status": "empty"}]},
                    "result": ({"index": tile_index, "status": "empty"},)}

        frame = (parts.get("frame_size") or (0, 0))
        if (int(frame[0]) <= 0 or int(frame[1]) <= 0) and (doc or {}).get("tile_size"):
            tw, th = (int(v) for v in (doc["tile_size"] or [0, 0])[:2])
            frame = (th, tw)
            parts["frame_size"] = frame
        tile_size = tuple(int(v) for v in ((doc or {}).get("tile_size") or (frame[1], frame[0])))
        overlap = int((doc or {}).get("overlap") or 0)
        src = (doc or {}).get("source") or {}
        source_size = (int(src.get("width") or 0), int(src.get("height") or 0))
        if source_size[0] <= 0 or source_size[1] <= 0:
            source_size = None

        try:
            psd_manifest = tile_psd.write_tiles(
                [parts],
                _output_root(),
                filename_prefix=prefix,
                merge_parts=True,
                alpha_threshold=int(alpha_threshold),
                source_size=source_size,
                tile_size=tile_size,
                overlap=overlap,
                tile_index_map=[tile_index],
                write_depth=True,
                original_rgba=None,
                merge_group_layers=True,
                include_original=True,
                run_dir_abs=wave_dir,
                write_manifest=False,
                # The grid's own boxes, so a phased grid lands where it was cut.
                tile_coords=(doc or {}).get("coords") or None,
                grid_offset=(doc or {}).get("offset") or (0, 0),
            )
        except ValueError as exc:
            # Say which tile came up empty: "every tile ended up empty" alone
            # gives no way to tell this apart from a layout bug.
            raise ValueError("STLU_TilePSD: tile %d: %s" % (tile_index, exc)) from exc
        entry = psd_manifest["tiles"][0]

        images_manifest = tile_psd.write_tile_images(
            [parts],
            _output_root(),
            filename_prefix=prefix,
            grid_doc=doc,
            tile_index_map=[tile_index],
            source_size=source_size,
            original_rel="../original.png",
            run_dir_abs=os.path.join(wave_dir, "images"),
            write_index=False,
        )
        image_entry = images_manifest["tiles"][0]

        # Refresh the lock snapshots from what was actually written.  This
        # migrates locks that came from the old PSD and keeps every locked
        # layer safe for the next time a parameter change clears the folder.
        present = {str(tag).lower() for tag in (parts.get("tag2pinfo") or {})}
        lock_names = [name for name in locks if name.strip().lower() in present]
        if lock_names:
            tile_psd.store_locked_parts(
                wave_dir, tile_index, parts, lock_names,
                tile_box=entry.get("tile"), canvas=entry.get("canvas"),
                source=source)

        record = {
            "index": int(tile_index),
            "status": "ok",
            "entry": entry,
            "image": image_entry,
        }
        print("[STLU] tile %d -> %s" % (tile_index, entry.get("file")), flush=True)
        return {"ui": {"stlu_tile": [{"index": tile_index, "file": entry.get("file")}]},
                "result": (record,)}

def _restore_locked_orphans(wave_dir, doc, prefix, entries, image_entries,
                            alpha_threshold):
    """Rebuild tiles this run did not select from their locked-part snapshots.

    A reset clears the folder down to the tiles this queue click generated, so
    a locked part living in any other tile would disappear with it.  The
    ``_locks/`` store survives the reset, so those tiles are written back with
    just their locked layers and join the manifest (and therefore the merge).
    """
    locks = tile_psd.tile_locks(wave_dir)
    if not locks:
        return [], []
    have = set()
    for entry in entries:
        try:
            have.add(int(entry.get("index", -1)))
        except (TypeError, ValueError):
            continue
    pending = [index for index in sorted(locks) if index not in have]
    if not pending:
        return [], []
    coords = (doc or {}).get("coords") or []
    src = (doc or {}).get("source") or {}
    source = None
    try:
        source = [int(src["width"]), int(src["height"])]
    except (KeyError, TypeError, ValueError):
        source = None
    tile_size = []
    try:
        tile_size = [int(v) for v in ((doc or {}).get("tile_size") or [])
                     if v is not None][:2]
    except (TypeError, ValueError):
        tile_size = []
    made_entries = []
    made_images = []
    for index in pending:
        parts = tile_psd.snapshot_parts(
            wave_dir, index, coords=coords, source=source)
        if not parts:
            print("[STLU] locked tile %d not restored: its crop layout changed"
                  % index, flush=True)
            continue
        frame_h, frame_w = (int(v) for v in parts["frame_size"][:2])
        size_pair = tile_size if len(tile_size) == 2 else [frame_w, frame_h]
        try:
            manifest = tile_psd.write_tiles(
                [parts], _output_root(),
                filename_prefix=prefix,
                merge_parts=True,
                alpha_threshold=int(alpha_threshold),
                source_size=tuple(source) if source else None,
                tile_size=tuple(size_pair),
                overlap=int((doc or {}).get("overlap") or 0),
                tile_index_map=[index],
                write_depth=True,
                original_rgba=None,
                merge_group_layers=True,
                include_original=False,
                run_dir_abs=wave_dir,
                write_manifest=False,
                tile_coords=coords or None,
                grid_offset=(doc or {}).get("offset") or (0, 0),
            )
            image_manifest = tile_psd.write_tile_images(
                [parts], _output_root(),
                filename_prefix=prefix,
                grid_doc=doc,
                tile_index_map=[index],
                source_size=source,
                original_rel="../original.png",
                run_dir_abs=os.path.join(wave_dir, "images"),
                write_index=False,
            )
        except (ValueError, OSError) as exc:
            print("[STLU] could not restore locked tile %d: %s" % (index, exc),
                  flush=True)
            continue
        made_entries.append(manifest["tiles"][0])
        made_images.append(image_manifest["tiles"][0])
        print("[STLU] restored tile %d from locked parts: %s" % (
            index, ", ".join(sorted((parts.get("tag2pinfo") or {}).keys()))),
            flush=True)
    return made_entries, made_images


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
        raw_records = [r for r in _list_values(tiles) if isinstance(r, dict)]
        empty_tiles = [int(r.get("index", -1)) for r in raw_records
                       if r.get("status") == "empty" or r.get("empty")]
        records = [r for r in raw_records
                   if r.get("status") not in ("empty", "error")
                   and not r.get("empty")]

        raw_grid = _first(grid_doc, "")
        doc = None
        if str(raw_grid or "").strip():
            try:
                doc = tile_psd.load_grid_ref(str(raw_grid))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    "STLU_WaveIndex: 无法读取 ① 的 grid_doc（%s）。"
                    "请检查 ①→③ 的 grid_doc 连线。" % exc) from exc
        out = _grid_out(doc)
        raw_seam = out.get("seam_feather")
        if raw_seam is None:
            raw_seam = tile_psd.DEFAULT_SEAM_FEATHER
        # 0 is a legal alpha threshold (nothing counts as transparent), so
        # only a missing value falls back to the default.
        raw_alpha = out.get("alpha_threshold")
        alpha_threshold = (tile_psd.DEFAULT_ALPHA_THRESHOLD if raw_alpha is None
                           else int(raw_alpha))

        wave = None
        if str(_first(wave_doc, "") or "").strip():
            wave = _wave_doc(wave_doc)
        elif out.get("dir"):
            wave = out
        if wave is None:
            raise ValueError(
                "STLU_WaveIndex: 需要 grid_doc（接 STLU_PipelineInput）或 wave_doc（旧版 STLU_Wave）")
        wave_dir = str(wave["dir"])
        rel = str(out.get("rel") or wave.get("rel") or "")
        prefix = str(out.get("prefix") or wave.get("prefix") or "tile")
        group_name = str(out.get("group") or wave.get("group") or "")
        wave_id_value = str(out.get("wave") or wave.get("wave") or "")
        os.makedirs(wave_dir, exist_ok=True)

        # The folder is fixed now, so it can already hold tiles from earlier
        # runs.  Read that manifest BEFORE overwriting it: those tiles join
        # this run's manifest (unless re-generated now) and therefore the
        # automatic merge, which is what makes a partial re-run complete.
        previous = tile_psd.read_manifest(wave_dir)

        # The full pixel-affecting settings travel inside grid_doc.  Keep the
        # document used for cache checks in sync so a folder written by a
        # different resolution / seed / model can never be mixed in.
        generation = dict((doc or {}).get("generation") or {})
        generation_id = str((doc or {}).get("generation_id") or "")
        if not generation:
            stored = tile_psd.read_generation(wave_dir) or {}
            generation = dict(stored.get("generation") or {})
            generation_id = str(stored.get("generation_id") or "")
        cache_doc = dict(doc or {})
        cache_doc["generation"] = generation
        cache_doc["generation_id"] = generation_id

        # One untouched source per wave; images.json points at this copy
        # instead of keeping a second identical PNG in images/.
        original_name = ""
        frame = _first(original)
        fingerprint = str(out.get("fingerprint") or "")
        if not fingerprint and frame is not None and hasattr(frame, "shape") and frame.ndim >= 4:
            fingerprint = _image_fingerprint(frame[0])
        if frame is not None and hasattr(frame, "shape") and frame.ndim >= 4:
            try:
                px = frame[0]
                px = px.detach().cpu().numpy() if hasattr(px, "detach") else np.asarray(px)
                rgba = np.clip(np.rint(np.asarray(px, dtype=np.float32) * 255.0),
                               0, 255).astype(np.uint8)
                if rgba.ndim == 3 and rgba.shape[-1] == 3:
                    rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, np.uint8)])
                from PIL import Image
                Image.fromarray(np.ascontiguousarray(rgba), "RGBA").save(
                    os.path.join(wave_dir, "original.png"), compress_level=3)
                original_name = "original.png"
            except Exception as exc:  # noqa: BLE001 - tiles still matter more
                print("[STLU] original.png failed: %s" % exc, flush=True)

        entries = []
        image_entries = []
        run_indices = set()
        for record in records:
            entry = dict(record.get("entry") or {})
            entries.append(entry)
            run_indices.add(int(entry.get("index", -1)))
            image_entry = dict(record.get("image") or {})
            if image_entry:
                image_entries.append(image_entry)

        # Finished tiles from earlier runs in this same folder.
        try:
            cached = tile_psd.cached_tile_entries(
                wave_dir, cache_doc, skip=run_indices, manifest=previous)
        except Exception as exc:  # noqa: BLE001
            cached = []
            print("[STLU] cannot reuse cached tiles: %s" % exc, flush=True)
        reused_indices = []
        if cached:
            entries.extend(cached)
            reused_indices = [int(e.get("index", -1)) for e in cached]
            prev_images = {}
            images_path = os.path.join(wave_dir, "images", "images.json")
            if os.path.isfile(images_path):
                try:
                    with open(images_path, "r", encoding="utf-8") as handle:
                        prev_doc = json.load(handle)
                    prev_images = {int(t.get("index", -1)): t
                                   for t in prev_doc.get("tiles") or []}
                except Exception:  # noqa: BLE001
                    prev_images = {}
            have_images = {int(e.get("index", -1)) for e in image_entries}
            for entry in cached:
                index = int(entry.get("index", -1))
                image_entry = prev_images.get(index)
                if image_entry and index not in have_images:
                    image_entries.append(dict(image_entry))

        # Tiles the current selection did not touch (after a reset the folder
        # holds only the selected ones).  Locked parts keep their own snapshot,
        # so those tiles are rebuilt instead of silently vanishing.
        restored_entries, restored_images = _restore_locked_orphans(
            wave_dir, doc, prefix, entries, image_entries, alpha_threshold)
        restored_indices = []
        if restored_entries:
            entries.extend(restored_entries)
            image_entries.extend(restored_images)
            restored_indices = [int(e.get("index", -1)) for e in restored_entries]

        entries.sort(key=lambda e: int(e.get("index", -1)))
        for slot, entry in enumerate(entries):
            entry["slot"] = int(slot)
        image_entries.sort(key=lambda e: int(e.get("index", -1)))
        for slot, entry in enumerate(image_entries):
            entry["slot"] = int(slot)

        if not entries:
            raise ValueError("STLU_WaveIndex: every tile ended up empty")
        if reused_indices:
            print("[STLU] reusing %d tiles from an earlier run: %s" % (
                len(reused_indices), reused_indices), flush=True)

        src = (doc or {}).get("source") or {}
        if int(src.get("width") or 0) > 0 and int(src.get("height") or 0) > 0:
            source_size = [int(src["width"]), int(src["height"])]
        else:
            source_size = [
                max(int(e["tile"][2]) for e in entries),
                max(int(e["tile"][3]) for e in entries),
            ]
        tile_size = [int(v) for v in ((doc or {}).get("tile_size") or entries[0]["canvas"])]
        overlap = int((doc or {}).get("overlap") or 0)

        richest = max(entries, key=lambda e: len(e.get("layers") or []))
        preview_name = str(richest.get("preview") or "")
        images_dir = os.path.relpath(os.path.join(wave_dir, "images"),
                                     _output_root()).replace(os.sep, "/")

        manifest = {
            "dir": rel,
            "prefix": prefix,
            "merge_parts": True,
            "source": source_size,
            "tile_size": tile_size,
            "overlap": overlap,
            "offset": tile_psd._int_pair((doc or {}).get("offset")),
            "preview": preview_name,
            "original": original_name,
            "has_depth": any(e.get("depth") for e in entries),
            "group_layers": True,
            "seam_feather": max(0, int(raw_seam)),
            "alpha_threshold": int(alpha_threshold),
            "mode": str((doc or {}).get("mode") or "uniform"),
            "partition_plan": (doc or {}).get("partition_plan") or None,
            "source_fingerprint": fingerprint,
            "tiles": entries,
            "count": len(entries),
            "run_tiles": sorted(run_indices),
            "reused_tiles": sorted(reused_indices),
            "restored_tiles": sorted(restored_indices),
            "generation": generation,
            "generation_id": generation_id,
            "images_dir": images_dir,
            "group": group_name,
            "wave": wave_id_value,
            "prompt_id": _prompt_id(),
        }
        manifest_path = os.path.join(wave_dir, "manifest.json")
        json_io.write_json_atomic(manifest_path, manifest)

        images_doc = {
            "version": 1,
            "kind": tile_psd.IMAGES_KIND,
            "dir": images_dir,
            "prefix": prefix,
            "source": source_size,
            "tile_size": tile_size,
            "overlap": overlap,
            "offset": tile_psd._int_pair((doc or {}).get("offset")),
            "original": "../original.png" if original_name else "",
            "tiles": image_entries,
            "count": len(image_entries),
        }
        images_dir_abs = os.path.join(wave_dir, "images")
        os.makedirs(images_dir_abs, exist_ok=True)
        json_io.write_json_atomic(
            os.path.join(images_dir_abs, "images.json"), images_doc)

        total = 0
        for entry in entries:
            path = os.path.join(wave_dir, str(entry.get("file") or ""))
            if os.path.isfile(path):
                total += os.path.getsize(path)

        preview = None
        preview_path = os.path.join(wave_dir, preview_name) if preview_name else ""
        if preview_path and os.path.isfile(preview_path):
            preview = {
                "filename": preview_name,
                "subfolder": rel,
                "type": "output",
                "v": int(os.path.getmtime(preview_path) * 1000),
            }
        info = {
            "dir": rel,
            "subfolder": rel,
            "filename": entries[0].get("file"),
            "count": len(entries),
            "layers": len(entries[0].get("layers") or []),
            "canvas": entries[0].get("canvas"),
            "size": total,
            "merged": True,
            "merged_psd": None,
            "merge_error": None,
            "preview": preview,
            "images_dir": images_dir,
            "images_error": None,
            "group": group_name,
            "wave": wave_id_value,
            "run_tiles": sorted(run_indices),
            "reused_tiles": sorted(reused_indices),
            "restored_tiles": sorted(restored_indices),
            "empty_tiles": empty_tiles,
            "generation_id": generation_id,
            "algo_version": (generation or {}).get("algo_version"),
            "error": None,
        }
        _notify("stlu.tiles", info)
        print("[STLU] %d tile PSDs written: %s" % (len(entries), wave_dir), flush=True)
        merge_doc = {
            "dir": rel,
            "name": str(out.get("merge_name") or "merged"),
            "group_layers": bool(out.get("group_layers", True)),
            "include_depth": bool(out.get("include_depth", False)),
            "include_original": bool(out.get("include_original", False)),
            "seam_feather": max(0, int(raw_seam)),
            "alpha_threshold": int(alpha_threshold),
            "wave": wave_id_value,
            "prompt_id": _prompt_id(),
        }
        return {
            "ui": {"stlu_tiles": [info]},
            "result": (wave_dir, json.dumps(manifest, ensure_ascii=False, indent=2),
                       json.dumps(merge_doc, ensure_ascii=False, separators=(",", ":"))),
        }

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
            raise ValueError("STLU_RebuildPSD: 选择一个已有的波次目录或 *_layers.json")
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
                "skipped": "没有指定分块目录",
            }]}, "result": ("",)}

        path = _resolve_workspace_path(rel)
        if not os.path.isdir(path):
            raise FileNotFoundError("tile run folder not found: %s" % path)
        if not os.path.isfile(os.path.join(path, "manifest.json")):
            raise FileNotFoundError(
                "manifest.json not found in %s" % path)

        # The merge must use the same transparency cutoff the tiles were
        # written with; ① travels it through merge_doc, older runs only have
        # it in the manifest (merge_run reads it from there).
        raw_alpha = (parsed.get("alpha_threshold")
                     if isinstance(parsed, dict) else None)
        info = tile_psd.merge_run(
            path,
            name=str(name or "merged"),
            alpha_threshold=None if raw_alpha is None else int(raw_alpha),
            group_layers=bool(group_layers),
            # Depth is optional, but when it is exported it is always its own
            # PSD -- merge never puts it inside the colour one.
            include_depth=bool(include_depth),
            include_original=bool(include_original),
            seam_feather=max(0, int(seam_feather)),
        )
        info["dir"] = os.path.relpath(path, _output_root()).replace(os.sep, "/")

        psd_path = os.path.join(path, info.get("file") or "")
        payload = {
            "dir": info["dir"],
            "name": str(name or "merged"),
            "group_layers": bool(group_layers),
            "include_depth": bool(include_depth),
            "include_original": bool(include_original),
            "file": info.get("file"),
            "preview": info.get("preview"),
            "depth": info.get("depth"),
            "depth_psd": info.get("depth_psd"),
            "original": info.get("original"),
            "canvas": info.get("canvas"),
            "layers": info.get("layers"),
            "groups": info.get("groups"),
            "tiles": info.get("tiles"),
            "notes": info.get("notes"),
        }
        _notify("stlu.tile_merge", payload)
        print("[STLU] merged %d tiles -> %s" % (len(info.get("tiles") or []), psd_path), flush=True)
        return {"ui": {"stlu_tile_merge": [payload]}, "result": (psd_path,)}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

class STLU_PSDPreview:
    """Show the merged PSD inside the graph through its own panel.

    Stage ④ writes a PSD but a PSD cannot be displayed by ComfyUI itself;
    this node flattens it (layers bottom -> top over white) and returns the
    tensor, while the panel drawn by ``web/stlu.js`` shows a compact overview
    plus the per-tile / per-part groups.  Point it at the ``psd_path`` output
    of ④, or at the run folder (it then picks ``merged.psd``).
    """

    DESCRIPTION = (
        "显示合并后的 PSD。接 ④ 的 psd_path，或直接填输出文件夹；面板先给全身总览，"
        "再按块 / 按部位分组列出拆分图。纯 CPU。"
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

    def execute(self, psd_path, max_side=1600, show_parts=True):
        raw = str(psd_path or "").strip()
        if not raw:
            raise ValueError("STLU_PSDPreview: 接上 ④ 的 psd_path（或填输出文件夹）")
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
                    raise FileNotFoundError("PSD 预览：%s 里没有 .psd" % raw)
                path = os.path.join(path, candidates[0])
        if not os.path.isfile(path):
            raise FileNotFoundError("PSD 预览：找不到 %s" % path)

        rgba = tile_psd.composite_psd(path)
        if rgba is None:
            raise ValueError("PSD 预览：读不出 %s" % path)
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
        return {
            "ui": {
                "stlu_parts": [{
                    "file": os.path.basename(path),
                    # The panel remembers this so reopening the workflow can
                    # show the last merge without running the queue again.
                    "path": os.path.abspath(path),
                    "canvas": [width, height],
                }],
            },
            "result": (tensor,),
        }


    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "STLU_PipelineInput": STLU_PipelineInput,
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
    "STLU_TilePSD": "SeeThrough ③ 每块存 PSD（自动逐块）",
    "STLU_GuidedLayers": "SeeThrough ② 顺序拆层（带重叠引导）",
    "STLU_WaveIndex": "SeeThrough ③ 写清单（全部块完成后）",
    "STLU_RebuildPSD": "SeeThrough ③ 从图片重建（可选，CPU）",
}

__all__ = [
    "STLU_PipelineInput",
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
]
