"""分块取遮罩 → 原图着色 → 分层 PSD（新的两层结构）。

设计约束，以及为什么这样写：

* **原图是唯一的颜色来源。** SeeThrough 只贡献"每个部件在哪里"（alpha 遮罩），
  PSD 里每个像素的 RGB 都直接取自原图。于是无论图层顺序、重叠、透明度如何，
  合成结果在数学上等于原图：不会有幻觉颜色、色差或接缝。
* **覆盖有保底。** 所有部件遮罩并集之外、原图不透明的像素自动进最底层
  ``其余``，所以不会出现空块截断、漏掉的躯干或背景洞。
* **遮罩拼接按归属权重取胜者。** 每块内部权重为到"相邻块边界"的距离（贴着
  图片外边缘的方向不衰减），每个部件逐像素只保留权重最大的那一块，再把
  小于阈值的碎块丢掉。并集/取最大会把每块各自幻觉出来的边界叠在一起，
  这是旧版"块之间有重影、错位"的主因。
* **不跑深度模型。** 层序由 tag 的固定顺序决定；跨块的深度中位数本来就
  不可比，而深度模型占掉一半时间。
* **模型在 512 分辨率下 1:1。** 块边长 = 拆层分辨率时不发生缩放，遮罩边缘
  就是模型自己的边缘，最锐利。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:  # cv2 是 ComfyUI 自带依赖；缺了只是少一层碎块清理
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

import comfy.utils

if __package__:  # ComfyUI loads the folder as a package
    from . import core_io, nodes_shared as _shared, psd_writer, tile_psd
else:  # standalone import used by the tests and tools
    import core_io
    import nodes_shared as _shared
    import psd_writer
    import tile_psd

CATEGORY = "SeeThrough/LayerUpscale"

#: 由后到前的层序。SeeThrough 的 tag 就是这些英文名，未知 tag 落在
#: "objects" 之上（属于前景杂物，压在被遮的东西上面）。
TAG_ORDER: Tuple[str, ...] = (
    "back hair", "hairb", "wings", "tail", "objects",
    "legwear", "footwear", "bottomwear", "neck", "topwear", "neckwear",
    "handwear", "ears", "head", "face", "eyewhite", "irides", "eyelash",
    "eyebrow", "nose", "mouth", "earwear", "front hair", "hairf",
    "eyewear", "headwear",
)
_TAG_RANK = {tag: i for i, tag in enumerate(TAG_ORDER)}
_UNKNOWN_RANK = _TAG_RANK["objects"] + 0.5

#: 成对部件（左右眼、左右手……）。拆成两块后接缝归属按像素就近判断，
#: 不会出现"一只眼睛从另一块借了一块"。
BILATERAL_TAGS: Tuple[str, ...] = (
    "eyewhite", "irides", "eyelash", "eyebrow", "ears", "earwear", "handwear",
)

HEAD_TAGS = frozenset((
    "head", "face", "eyewhite", "irides", "eyelash", "eyebrow", "nose",
    "mouth", "ears", "earwear", "eyewear", "headwear", "front hair",
    "back hair", "hairf", "hairb",
))

MAX_PREVIEW_SIDE = 1600


# --------------------------------------------------------------------------
# 纯函数：几何、权重、拼接
# --------------------------------------------------------------------------
def base_tag(tag: str) -> str:
    """``handwear-l`` / ``handwear-r`` 等拆开的部件回到原始 tag。"""
    text = str(tag)
    if text.endswith(("-l", "-r")) or text.endswith(("_l", "_r")):
        return text[:-2]
    return text


def tag_rank(tag: str) -> float:
    return _TAG_RANK.get(base_tag(tag), _UNKNOWN_RANK)


def plan_boxes(width: int, height: int, tile_w: int, tile_h: int,
               overlap: int) -> List[Tuple[int, int, int, int]]:
    """与旧流水线同一套网格算法，保证坐标含义一致。"""
    return tile_psd.grid_coords(width, height, tile_w, tile_h, overlap)


def parse_tiles(text: str, count: int) -> List[int]:
    picked = _shared.parse_tile_selection(str(text or "all"), count)
    return picked or list(range(count))


def ownership_weight(box: Sequence[int], width: int, height: int,
                     feather: int) -> np.ndarray:
    """块内每个像素的归属权重（0..255）。

    只有"旁边还有另一块"的方向才衰减：贴着图片外边缘的一侧保持满权重，
    否则最边上那块会把自己并不可信的外缘让给邻居，导致图片边缘缺一条。
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = x1 - x0, y1 - y0
    big = np.float32(max(w, h) + 1)
    cols = np.arange(w, dtype=np.float32) + 1.0
    rows = np.arange(h, dtype=np.float32) + 1.0
    dx = np.full(w, big, np.float32)
    dy = np.full(h, big, np.float32)
    if x0 > 0:
        dx = np.minimum(dx, cols)
    if x1 < width:
        dx = np.minimum(dx, w - np.arange(w, dtype=np.float32))
    if y0 > 0:
        dy = np.minimum(dy, rows)
    if y1 < height:
        dy = np.minimum(dy, h - np.arange(h, dtype=np.float32))
    dist = np.minimum(dy[:, None], dx[None, :])
    ramp = np.clip(dist / float(max(1, int(feather))), 0.0, 1.0)
    return np.rint(ramp * 255.0).astype(np.uint8)


def frame_masks(layer_dict: Dict[str, Any], layers_data, crop_w: int,
                crop_h: int) -> Dict[str, np.ndarray]:
    """模型帧（resolution×resolution）里的遮罩裁回本块并缩放到块尺寸。"""
    scale = float(getattr(layers_data, "scale", 1.0) or 1.0)
    pad_x, pad_y = getattr(layers_data, "pad_pos", (0, 0))
    x0 = int(round(float(pad_x) / scale))
    y0 = int(round(float(pad_y) / scale))
    x1 = int(round((float(pad_x) + crop_w) / scale))
    y1 = int(round((float(pad_y) + crop_h) / scale))
    out: Dict[str, np.ndarray] = {}
    for tag, rgba in (layer_dict or {}).items():
        if rgba is None:
            continue
        arr = np.asarray(rgba)
        alpha = arr[..., 3] if arr.ndim == 3 and arr.shape[-1] >= 4 else arr
        sub = alpha[y0:y1, x0:x1]
        if sub.size == 0:
            continue
        sub = sub.astype(np.uint8)
        if sub.shape != (crop_h, crop_w):
            sub = core_io.resize_alpha(sub, crop_w, crop_h)
        if int(sub.max()) == 0:
            continue
        out[str(tag)] = np.ascontiguousarray(sub)
    return out


def _drop_specks(alpha: np.ndarray, min_area: int) -> np.ndarray:
    """丢掉孤立碎块：模型在块边缘最容易凭空生成小零件。"""
    if cv2 is None or min_area <= 0:
        return alpha
    binary = (alpha > 8).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 2:
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    floor = max(int(min_area), int(areas.max() * 0.02))
    keep = [i + 1 for i, area in enumerate(areas) if area >= floor]
    if len(keep) == count - 1:
        return alpha
    if not keep:
        return np.zeros_like(alpha)
    mask = np.isin(labels, keep)
    return np.where(mask, alpha, 0).astype(np.uint8)


def tighten_alpha(alpha: np.ndarray, min_area: int) -> Optional[Tuple[np.ndarray, int, int]]:
    """清理碎块后裁到紧包围盒，返回 ``(alpha, left, top)``；全空返回 ``None``。"""
    alpha = _drop_specks(alpha, min_area)
    ys, xs = np.nonzero(alpha)
    if ys.size == 0:
        return None
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    return np.ascontiguousarray(alpha[top:bottom, left:right]), left, top


def split_lr(alpha: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """把左右成对的部件按中间的空隙切开，返回 ``[(列偏移, alpha), ...]``。

    切不开（中间没有明显空隙、或一半是空的）就原样返回一个。
    """
    height, width = alpha.shape[:2]
    if width < 8:
        return [(0, alpha)]
    col_mass = (alpha > 8).sum(axis=0).astype(np.int64)
    if col_mass.sum() == 0:
        return [(0, alpha)]
    lo = int(width * 0.2)
    hi = int(width * 0.8)
    # 中段最长的空列段：两侧各有一团内容，中间是空隙
    best_len, best_mid, run_start = 0, -1, None
    for x in range(lo, hi):
        if col_mass[x] == 0:
            if run_start is None:
                run_start = x
            if x - run_start + 1 > best_len:
                best_len = x - run_start + 1
                best_mid = (run_start + x + 1) // 2
        else:
            run_start = None
    if best_len < max(3, int(width * 0.02)):
        best_mid = int(np.argmin(col_mass[lo:hi])) + lo
    left = alpha[:, :best_mid]
    right = alpha[:, best_mid:]
    if not left.any() or not right.any():
        return [(0, alpha)]
    return [(0, left), (best_mid, right)]


def merge_tile_masks(tiles: List[Tuple[Sequence[int], Dict[str, np.ndarray]]],
                     width: int, height: int, feather: int,
                     source_alpha: np.ndarray,
                     min_area: int,
                     split_bilateral: bool = True
                     ) -> Dict[str, Dict[str, Any]]:
    """把逐块遮罩拼成整图部件遮罩。

    返回 ``{tag: {"alpha": uint8 紧裁图, "left", "top", "area"}}``，
    ``alpha`` 已乘上原图不透明度，所以合成永远不会超过原图的 alpha。
    """
    weights = [ownership_weight(box, width, height, feather) for box, _ in tiles]
    order: List[str] = []
    seen = set()
    for _, masks in tiles:
        for tag in masks:
            if tag not in seen:
                seen.add(tag)
                order.append(tag)

    parts: Dict[str, Dict[str, Any]] = {}
    for tag in order:
        conf = np.zeros((height, width), np.uint8)
        acc = np.zeros((height, width), np.uint8)
        for (box, masks), weight in zip(tiles, weights):
            mask = masks.get(tag)
            if mask is None:
                continue
            x0, y0, x1, y1 = (int(v) for v in box)
            view_conf = conf[y0:y1, x0:x1]
            view_acc = acc[y0:y1, x0:x1]
            if view_acc.shape != mask.shape or view_conf.shape != weight.shape:
                continue
            better = weight > view_conf
            view_acc[better] = mask[better]
            view_conf[better] = weight[better]
        if not acc.any():
            continue
        acc = (acc.astype(np.uint16) * source_alpha.astype(np.uint16) // 255).astype(np.uint8)
        # 成对部件在这里切成左右两块；(x 偏移, y 偏移, alpha)
        pieces: List[Tuple[int, int, str, np.ndarray]] = [(0, 0, tag, acc)]
        if split_bilateral and base_tag(tag) in BILATERAL_TAGS:
            tight = tighten_alpha(acc, min_area)
            if tight is not None:
                body, left, top = tight
                halves = split_lr(body)
                if len(halves) == 2:
                    pieces = [(left + x, top, f"{tag}-{'lr'[i]}", arr)
                              for i, (x, arr) in enumerate(halves)]
        for off_x, off_y, name, piece in pieces:
            tightened = tighten_alpha(piece, min_area)
            if tightened is None:
                continue
            box, left, top = tightened
            parts[name] = {
                "alpha": box,
                "left": int(off_x + left),
                "top": int(off_y + top),
                "area": int(np.count_nonzero(box > 8)),
            }
    return parts


def build_layers(parts: Dict[str, Dict[str, Any]], source: np.ndarray,
                 fallback_name: str = "其余",
                 group_head: bool = True) -> List[Dict[str, Any]]:
    """用原图 RGB + 部件 alpha 组装 PSD 图层（由底到顶）。"""
    height, width = source.shape[:2]
    ordered = sorted(parts.items(),
                     key=lambda item: (tag_rank(item[0]), -item[1]["area"]))
    # PSD 文件夹只能包住连续的一段记录，而头部部件在堆叠顺序里并不相邻；
    # 把头部部件整体挪到「第一个头部部件出现的位置」，内部相对顺序不变。
    # 每层颜色都取原图、alpha 递增补齐与原图无关，换序不影响合成结果。
    if group_head:
        head = [item for item in ordered if base_tag(item[0]) in HEAD_TAGS]
        if head:
            first = next(i for i, item in enumerate(ordered)
                         if base_tag(item[0]) in HEAD_TAGS)
            rest = [item for item in ordered if base_tag(item[0]) not in HEAD_TAGS]
            before = sum(1 for item in ordered[:first]
                         if base_tag(item[0]) not in HEAD_TAGS)
            ordered = rest[:before] + head + rest[before:]

    def make_layer(name: str, alpha: np.ndarray, left: int, top: int) -> Dict[str, Any]:
        h, w = alpha.shape[:2]
        rgb = source[top:top + h, left:left + w, :3]
        pixels = np.dstack([rgb, alpha])
        return {"name": name, "left": int(left), "top": int(top),
                "pixels": np.ascontiguousarray(pixels)}

    layers: List[Dict[str, Any]] = []
    head_open = head_closed = False

    def trim(name: str, alpha: np.ndarray, left: int, top: int) -> Optional[Dict[str, Any]]:
        ys, xs = np.nonzero(alpha)
        if not ys.size:
            return None
        t, l = int(ys.min()), int(xs.min())
        layer = make_layer(name, alpha[t:int(ys.max()) + 1, l:int(xs.max()) + 1],
                           left + l, top + t)
        layers.append(layer)
        return layer

    # PSD 文件夹在记录（由底到顶）里是两个分隔：收尾记录在下、头部记录在上，
    # 子层夹在中间。顺序写反的话 Photoshop 只把相邻的一层收进组。
    def open_head():
        layers.append({"type": psd_writer.GROUP_END})

    def close_head():
        layers.append({"type": psd_writer.GROUP_START, "name": "头部"})

    # 半透明像素上两块部件重叠时，叠出来的不透明度会超过原图 alpha。
    # 按自下而上的顺序把每块压到「补足到原图 alpha 所需的增量」，这样 over
    # 合成后的总 alpha 精确等于原图，任何重叠都不会溢出。
    target = source[..., 3].astype(np.float32) / 255.0
    cum = np.zeros((height, width), np.float32)
    # 压 alpha 之后可能只剩几十个像素的碎渣（模型误判的零星像素）。这种层
    # 不单独成层：不累进 cum，alpha 全部落到底部残差层，合成仍逐像素精确。
    speck = max(64, (width * height) // 100000)
    dropped: List[str] = []
    for name, part in ordered:
        left, top = part["left"], part["top"]
        h, w = part["alpha"].shape[:2]
        view = cum[top:top + h, left:left + w]
        cap = (target[top:top + h, left:left + w] - view) / np.maximum(1.0 - view, 1e-6)
        alpha = np.rint(np.minimum(part["alpha"].astype(np.float32) / 255.0,
                                   np.clip(cap, 0.0, 1.0)) * 255.0)
        alpha = alpha.astype(np.uint8)
        if np.count_nonzero(alpha) < speck:
            nz = int(np.count_nonzero(alpha))
            dropped.append(f"{name}({nz}px)")
            continue
        view += (1.0 - view) * (alpha.astype(np.float32) / 255.0)
        if group_head:
            is_head = base_tag(name) in HEAD_TAGS
            if is_head and not head_open:
                open_head()
                head_open = True
            elif head_open and not is_head and not head_closed:
                close_head()
                head_closed = True
        trim(name, alpha, left, top)

    residual = (target - cum) / np.maximum(1.0 - cum, 1e-6)
    residual = np.rint(np.clip(residual, 0.0, 1.0) * 255.0).astype(np.uint8)
    fallback = trim(fallback_name, residual, 0, 0)
    if fallback is not None:
        layers.insert(0, layers.pop(next(i for i, item in enumerate(layers)
                                         if item is fallback)))

    if head_open and not head_closed:
        close_head()
    if dropped:
        print(f"[STLU] 碎渣层并入「{fallback_name}」（<{speck}px）："
              + "、".join(dropped), flush=True)
    return layers


def composite_rgba(layers: Sequence[Dict[str, Any]], width: int,
                   height: int) -> np.ndarray:
    """直通 alpha 合成（straight alpha），用来核对 PSD 是否还原原图。"""
    out = np.zeros((height, width, 4), np.float32)
    for layer in layers:
        if "pixels" not in layer:
            continue
        pixels = layer["pixels"].astype(np.float32) / 255.0
        left, top = int(layer["left"]), int(layer["top"])
        h, w = pixels.shape[:2]
        y0, x0 = max(0, top), max(0, left)
        y1, x1 = min(height, top + h), min(width, left + w)
        if y1 <= y0 or x1 <= x0:
            continue
        src = pixels[y0 - top:y1 - top, x0 - left:x1 - left]
        dst = out[y0:y1, x0:x1]
        a = src[..., 3:4]
        base_a = dst[..., 3:4].copy()
        base_rgb = dst[..., :3].copy()
        new_a = a + base_a * (1.0 - a)
        num = src[..., :3] * a + base_rgb * base_a * (1.0 - a)
        dst[..., :3] = np.where(new_a > 1e-6, num / np.maximum(new_a, 1e-6), 0.0)
        dst[..., 3:4] = new_a
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def preview_tensor(rgba: np.ndarray, max_side: int = MAX_PREVIEW_SIDE) -> torch.Tensor:
    """把 RGBA 叠到白底上，缩到能放进面板的尺寸。"""
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgb * alpha + (1.0 - alpha)
    height, width = rgb.shape[:2]
    scale = min(1.0, float(max_side) / max(1, max(width, height)))
    if scale < 1.0:
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)[None]
        rgb = torch.nn.functional.interpolate(
            rgb, size=(max(1, int(round(height * scale))),
                       max(1, int(round(width * scale)))),
            mode="bilinear", align_corners=False)
        return rgb[0].permute(1, 2, 0)[None].contiguous()
    return torch.from_numpy(rgb)[None]


def _tile_tensor(crop: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(crop.astype(np.float32) / 255.0)[None]


def _apply_mask_alpha(source: np.ndarray, mask) -> None:
    """把 ComfyUI 的 MASK（1=透明）换回 alpha 写进 RGBA 源图。"""
    if mask is None:
        return
    arr = mask
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[:2] != source.shape[:2]:
        raise RuntimeError(
            f"遮罩尺寸 {arr.shape[1]}x{arr.shape[0]} 与原图 "
            f"{source.shape[1]}x{source.shape[0]} 不一致")
    source[..., 3] = np.clip(np.rint((1.0 - arr) * 255.0), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# ① 分块拆遮罩
# --------------------------------------------------------------------------
class STLU_TileMasks:
    """逐块跑 SeeThrough，只保留每块各部件的 alpha，拼成整图遮罩。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "layerdiff_model": ("SEETHROUGH_LAYERDIFF_MODEL",),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048,
                                      "step": 64,
                                      "tooltip": "块边长；与拆层分辨率相同即 1:1"}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8,
                                    "tooltip": "相邻块重叠像素，接缝归属按距离平分"}),
                "resolution": ("INT", {"default": 512, "min": 512, "max": 2048,
                                       "step": 64,
                                       "tooltip": "拆层分辨率；上游最小 512"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2 ** 32 - 1}),
                "tiles": ("STRING", {"default": "all",
                                     "tooltip": "all / 0-5 / 1,3,7：只跑部分块"}),
            },
            # ComfyUI 的 IMAGE 张量不带 alpha，LoadImage 把透明度单独放在 MASK
            # 输出里；不接这一路的话整张图会被当成不透明。
            "optional": {
                "mask": ("MASK", {"tooltip": "接 LoadImage 的「遮罩」输出，"
                                             "保留原图透明度"}),
            },
        }

    RETURN_TYPES = ("STLU_TILE_MASKS",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, layerdiff_model, tile_size=512, overlap=64,
            resolution=512, steps=30, seed=42, tiles="all", mask=None):
        node_cls = _shared._seethrough_node_class("SeeThrough_GenerateLayers")
        if node_cls is None:
            raise RuntimeError("找不到 SeeThrough_GenerateLayers，请先安装并更新 ComfyUI-See-through")
        layer_node = node_cls()

        source = core_io.tensor_to_rgba(image)
        _apply_mask_alpha(source, mask)
        height, width = source.shape[:2]
        boxes = plan_boxes(width, height, int(tile_size), int(tile_size), int(overlap))
        picked = parse_tiles(tiles, len(boxes))
        source_alpha = source[..., 3]

        started = time.time()
        collected: List[Tuple[Sequence[int], Dict[str, np.ndarray]]] = []
        pbar = comfy.utils.ProgressBar(len(picked))
        empty = 0
        for index in picked:
            box = boxes[index]
            x0, y0, x1, y1 = box
            crop = source[y0:y1, x0:x1]
            if int(crop[..., 3].max()) <= 8:
                empty += 1
                pbar.update(1)
                continue
            layers_data, _preview = layer_node.generate(
                _tile_tensor(crop), layerdiff_model,
                seed=int(seed), resolution=int(resolution),
                num_inference_steps=int(steps))
            masks = frame_masks(layers_data.layer_dict, layers_data,
                                crop.shape[1], crop.shape[0])
            collected.append((box, masks))
            print(f"[STLU] 块 {index}/{len(boxes)} 完成：{len(masks)} 个部件", flush=True)
            pbar.update(1)

        if not collected:
            raise RuntimeError("所有块都是空的，没有可拆的内容")

        parts = merge_tile_masks(collected, width, height, int(overlap),
                                 source_alpha, min_area=_min_area(width, height))
        report = {
            "width": int(width), "height": int(height),
            "tile_size": [int(tile_size), int(tile_size)],
            "overlap": int(overlap), "resolution": int(resolution),
            "steps": int(steps), "seed": int(seed),
            "tiles_total": len(boxes), "tiles_run": len(picked),
            "tiles_empty": int(empty),
            "boxes": [list(map(int, b)) for b in boxes],
            "parts": {name: {"area": part["area"],
                             "box": [part["left"], part["top"],
                                     part["left"] + part["alpha"].shape[1],
                                     part["top"] + part["alpha"].shape[0]]}
                      for name, part in parts.items()},
            "seconds": round(time.time() - started, 2),
        }
        print(f"[STLU] 遮罩完成：{len(parts)} 个部件，{report['seconds']}s", flush=True)
        return ({"kind": "STLU_TILE_MASKS", "version": 1, "parts": parts,
                 "width": int(width), "height": int(height),
                 "alpha": source_alpha, "report": report},)


def _min_area(width: int, height: int) -> int:
    """碎块门限：整图面积的 0.02%，再兜底 64 像素。"""
    return max(64, int(width * height * 0.0002))


# --------------------------------------------------------------------------
# ② 原图着色 + 分层 PSD
# --------------------------------------------------------------------------
class STLU_LayerPSD:
    """用原图给每个部件上色，补一层``其余``兜底，写出分层 PSD 并自检。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "masks": ("STLU_TILE_MASKS",),
                "out_dir": ("STRING", {"default": "SeeThrough/MaskLayers",
                                       "tooltip": "相对 ComfyUI output/ 的目录"}),
                "name": ("STRING", {"default": "layers.psd"}),
                "group_head": ("BOOLEAN", {"default": True,
                                           "label_on": "头部单独成组",
                                           "label_off": "平铺"}),
                "fallback_name": ("STRING", {"default": "其余"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "一般不用接：① 已把透明度传过来。"
                                             "换图或旧缓存时再接"}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("psd_path", "preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, image, masks, out_dir="SeeThrough/MaskLayers",
            name="layers.psd", group_head=True, fallback_name="其余", mask=None):
        source = core_io.tensor_to_rgba(image)
        doc = masks or {}
        if mask is not None:
            _apply_mask_alpha(source, mask)
        else:
            alpha = doc.get("alpha")
            if alpha is not None and getattr(alpha, "shape", None) == source.shape[:2]:
                source[..., 3] = np.asarray(alpha, dtype=np.uint8)
            elif alpha is None:
                print("[STLU] 警告：masks 里没有原图透明度，先重跑 ①"
                      "（否则 PSD 背景会是不透明）", flush=True)
        height, width = source.shape[:2]
        if int(doc.get("width", 0)) != width or int(doc.get("height", 0)) != height:
            raise RuntimeError(
                f"遮罩尺寸 {doc.get('width')}x{doc.get('height')} 与原图 "
                f"{width}x{height} 不一致，请重新运行 ①")

        parts = doc.get("parts") or {}
        if not parts:
            raise RuntimeError("遮罩为空")

        layers = build_layers(parts, source, fallback_name=fallback_name,
                              group_head=bool(group_head))
        flat = composite_rgba([l for l in layers if "pixels" in l], width, height)
        visible = source[..., 3] > 0
        diff = 0
        if visible.any():
            diff = int(np.abs(flat[visible][:, :3].astype(np.int16)
                              - source[visible][:, :3].astype(np.int16)).max())
        alpha_diff = int(np.abs(flat[..., 3].astype(np.int16)
                                - source[..., 3].astype(np.int16)).max())
        if diff > 2 or alpha_diff > 2:
            print(f"[STLU] 警告：合成与原图不一致（颜色差 {diff}，alpha 差 "
                  f"{alpha_diff}）", flush=True)

        path = self._resolve_path(out_dir, name)
        psd_writer.write_psd(path, layers, width, height, compression="rle")
        print(f"[STLU] 写出 {path}：{len([l for l in layers if 'pixels' in l])} 层，"
              f"合成差 {diff}", flush=True)
        info = {"psd": path,
                "layers": len([l for l in layers if "pixels" in l]),
                "parts": [l["name"] for l in layers
                          if "pixels" in l and l["name"] != fallback_name],
                "composite_max_diff": int(max(diff, alpha_diff)),
                "warn": bool(diff > 2 or alpha_diff > 2)}
        return {"ui": {"stlu_psd": [info]},
                "result": (path, preview_tensor(flat))}

    @staticmethod
    def _resolve_path(out_dir: str, name: str) -> str:
        root = _shared._output_root()
        rel = str(out_dir or "").strip().replace("\\", "/").strip("/")
        folder = os.path.join(root, *rel.split("/")) if rel else root
        os.makedirs(folder, exist_ok=True)
        filename = os.path.basename(str(name or "").strip() or "layers.psd")
        if not filename.lower().endswith(".psd"):
            filename += ".psd"
        return os.path.join(folder, filename)


NODE_CLASS_MAPPINGS = {
    "STLU_TileMasks": STLU_TileMasks,
    "STLU_LayerPSD": STLU_LayerPSD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STLU_TileMasks": "① 分块拆遮罩 (Tile Masks)",
    "STLU_LayerPSD": "② 分层 PSD (Layer PSD)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
    "build_layers", "composite_rgba", "frame_masks", "merge_tile_masks",
    "ownership_weight", "plan_boxes", "split_lr", "tag_rank", "tighten_alpha",
]
