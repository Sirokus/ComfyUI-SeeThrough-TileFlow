"""Crop / paste-back geometry for refining small layers in a local window.

A 17x15 pupil sent to Qwen-Image-Edit as a whole edit target is not "a part to
refine" but "a broken frame to complete": the model fills in whatever its
training says belongs around a pupil.  The fix is geometric, not verbal — cut a
window of real context around the part, generate only inside that window, and
paste the content region back with the original alpha.

Everything here is pure numpy/PIL so it can be tested without ComfyUI or torch.
Rects are ``(x0, y0, x1, y1)`` with an exclusive right/bottom edge, in whatever
coordinate space the caller documents.
"""

from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

import numpy as np

Rect = Tuple[int, int, int, int]

# A layer whose box covers this much of the canvas already *is* the frame;
# cropping it would only shrink the context it already has.
FULL_CANVAS_RATIO = 0.95

# Below this the reference encoder cannot see anything (a 17 px pupil used to
# reach the model as a 32x32 thumbnail), so the window is grown to this size.
MIN_CROP_SIDE = 96

CROP_INFO_VERSION = 1


def alpha_bbox(alpha: np.ndarray, threshold: int = 8) -> Optional[Rect]:
    """Tight bbox of ``alpha > threshold``; ``None`` when fully transparent."""
    if alpha is None or alpha.size == 0:
        return None
    ys, xs = np.nonzero(np.asarray(alpha) > int(threshold))
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def content_rect(box: Rect, bbox: Optional[Rect], canvas: Tuple[int, int]) -> Rect:
    """Map a layer-local alpha bbox into canvas coordinates, clamped."""
    width, height = int(canvas[0]), int(canvas[1])
    left, top, right, bottom = (int(v) for v in box)
    if bbox is None:
        rect = (left, top, right, bottom)
    else:
        rect = (left + int(bbox[0]), top + int(bbox[1]),
                left + int(bbox[2]), top + int(bbox[3]))
    return _clamp_rect(rect, (0, 0, width, height), min_side=1)


def crop_rect(content: Rect, canvas: Tuple[int, int], margin: int,
              min_side: int = MIN_CROP_SIDE) -> Rect:
    """Expand ``content`` by ``margin``, grow short sides, clamp to the canvas."""
    width, height = int(canvas[0]), int(canvas[1])
    x0, y0, x1, y1 = (int(v) for v in content)
    m = max(0, int(margin))
    x0 -= m
    y0 -= m
    x1 += m
    y1 += m

    # Grow the short side symmetrically so the window is never a sliver: the
    # encoder rounds to multiples of 32 and would otherwise collapse it.
    min_side = max(1, int(min_side))
    if x1 - x0 < min_side:
        need = min_side - (x1 - x0)
        x0 -= need // 2
        x1 += need - need // 2
    if y1 - y0 < min_side:
        need = min_side - (y1 - y0)
        y0 -= need // 2
        y1 += need - need // 2

    return _clamp_rect((x0, y0, x1, y1), (0, 0, width, height), min_side)


def _clamp_rect(rect: Rect, outer: Rect, min_side: int = 1) -> Rect:
    """Shift then shrink ``rect`` so it fits inside ``outer``."""
    ox0, oy0, ox1, oy1 = outer
    x0, y0, x1, y1 = (int(v) for v in rect)

    # Shift back inside when possible, so the window keeps its size.
    if x1 - x0 >= min_side:
        if x0 < ox0:
            x1 += ox0 - x0
            x0 = ox0
        if x1 > ox1:
            x0 -= x1 - ox1
            x1 = ox1
    if y1 - y0 >= min_side:
        if y0 < oy0:
            y1 += oy0 - y0
            y0 = oy0
        if y1 > oy1:
            y0 -= y1 - oy1
            y1 = oy1

    x0 = max(ox0, min(x0, ox1 - min_side))
    y0 = max(oy0, min(y0, oy1 - min_side))
    x1 = max(x0 + min_side, min(x1, ox1))
    y1 = max(y0 + min_side, min(y1, oy1))
    return (x0, y0, x1, y1)


def context_scale(context_size: Tuple[int, int], canvas: Tuple[int, int],
                  tolerance: float = 0.02) -> float:
    """How many context-image pixels map to one source-canvas pixel.

    Returns ``1.0`` when the context is the canvas itself or when its aspect
    ratio does not match (a differently framed image cannot be mapped by a
    single scale factor, so the caller must fit it to the canvas instead).
    """
    cw, ch = int(context_size[0]), int(context_size[1])
    width, height = int(canvas[0]), int(canvas[1])
    if cw <= 0 or ch <= 0 or width <= 0 or height <= 0:
        return 1.0
    kx = cw / float(width)
    ky = ch / float(height)
    if abs(kx - ky) > float(tolerance) * max(kx, ky):
        return 1.0
    return (kx + ky) / 2.0


def scale_rect(rect: Rect, k: float, outer: Optional[Tuple[int, int]] = None) -> Rect:
    """Scale a rect by ``k`` (rounding), optionally clamped to an image size."""
    x0, y0, x1, y1 = (int(round(float(v) * float(k))) for v in rect)
    x1 = max(x0 + 1, x1)
    y1 = max(y0 + 1, y1)
    if outer is not None:
        width, height = int(outer[0]), int(outer[1])
        x0 = max(0, min(x0, max(0, width - 1)))
        y0 = max(0, min(y0, max(0, height - 1)))
        x1 = min(width, max(x0 + 1, x1))
        y1 = min(height, max(y0 + 1, y1))
    return (x0, y0, x1, y1)


def is_full_canvas(box: Rect, canvas: Tuple[int, int],
                   ratio: float = FULL_CANVAS_RATIO) -> bool:
    """True when the layer box already spans the frame (no crop to be had)."""
    bw = max(0, int(box[2]) - int(box[0]))
    bh = max(0, int(box[3]) - int(box[1]))
    area = max(1, int(canvas[0]) * int(canvas[1]))
    return (bw * bh) >= float(ratio) * area


def bypass_info(name: str, target: Optional[Tuple[int, int]] = None) -> Dict:
    """Marker for layers that must keep the original full-frame path."""
    info = {"v": CROP_INFO_VERSION, "name": str(name), "bypass": True}
    if target is not None:
        info["target"] = [int(target[0]), int(target[1])]
    return info


def make_crop_info(name: str, canvas: Tuple[int, int], box: Rect, content: Rect,
                   crop: Rect, target: Tuple[int, int], margin: int) -> Dict:
    return {
        "v": CROP_INFO_VERSION,
        "name": str(name),
        "canvas": [int(canvas[0]), int(canvas[1])],
        "box": [int(v) for v in box],
        "content": [int(v) for v in content],
        "crop": [int(v) for v in crop],
        "target": [int(target[0]), int(target[1])],
        "margin": int(margin),
    }


def crop_info_to_text(info: Dict) -> str:
    return json.dumps(info, ensure_ascii=False)


def crop_info_from_text(text: str) -> Optional[Dict]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        info = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(info, dict) or info.get("v") != CROP_INFO_VERSION:
        return None
    if info.get("bypass"):
        return info
    expected = {"canvas": 2, "target": 2, "box": 4, "content": 4, "crop": 4}
    for key, length in expected.items():
        value = info.get(key)
        if not isinstance(value, list) or len(value) != length:
            return None
    return info


def paste_fractions(info: Dict) -> Optional[Tuple[float, float, float, float, float, float, float, float]]:
    """Where the content sits inside the crop and inside the layer box.

    Returns ``(fx0, fy0, fx1, fy1, lx0, ly0, lx1, ly1)``: the first four are
    fractions of the generated crop, the last four fractions of the layer's
    own pixels (used to resample the original alpha).
    """
    try:
        cx0, cy0, cx1, cy1 = (float(v) for v in info["crop"])
        nx0, ny0, nx1, ny1 = (float(v) for v in info["content"])
        bx0, by0, bx1, by1 = (float(v) for v in info["box"])
    except (KeyError, TypeError, ValueError):
        return None
    cw, ch = cx1 - cx0, cy1 - cy0
    bw, bh = bx1 - bx0, by1 - by0
    if cw <= 0 or ch <= 0 or bw <= 0 or bh <= 0:
        return None

    def _unit(value: float) -> float:
        # Rects are rounded independently when scaled to a reference image, so
        # clamp away the sub-pixel overshoot before it reaches grid_sample.
        return max(0.0, min(1.0, value))

    return (
        _unit((nx0 - cx0) / cw), _unit((ny0 - cy0) / ch),
        _unit((nx1 - cx0) / cw), _unit((ny1 - cy0) / ch),
        _unit((nx0 - bx0) / bw), _unit((ny0 - by0) / bh),
        _unit((nx1 - bx0) / bw), _unit((ny1 - by0) / bh),
    )


def extract_crop(context_rgb: np.ndarray, layer_rgba: np.ndarray, box: Rect,
                 crop: Rect) -> np.ndarray:
    """Cut ``crop`` out of the opaque context and blend the layer on top.

    The layer's own pixels are painted at their original place, so the part the
    user asked to refine is present at full source sharpness (the context image
    may be a lower-quality composite).  The result is opaque RGB: the model
    must never see a transparent hole, or it will treat the frame as incomplete
    and fill it in.
    """
    cx0, cy0, cx1, cy1 = (int(v) for v in crop)
    ctx = np.asarray(context_rgb)
    window = ctx[cy0:cy1, cx0:cx1]
    if window.ndim == 3 and window.shape[-1] >= 4:
        # A reference image is usually RGBA; its transparent area would arrive
        # as raw RGB (normally black) and be read as shadow.  Flatten over white
        # exactly like Qwen's vision tower does.
        alpha = window[..., 3:4].astype(np.float32) / 255.0
        out = window[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    else:
        out = np.array(window[..., :3], dtype=np.float32, copy=True)

    layer = np.asarray(layer_rgba)
    l, t, r, b = (int(v) for v in box)
    lw, lh = r - l, b - t
    if layer.ndim != 3 or lw <= 0 or lh <= 0 or out.size == 0:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    # Overlap of the layer box and the crop window, in canvas coordinates.
    ix0, iy0 = max(cx0, l), max(cy0, t)
    ix1, iy1 = min(cx1, r), min(cy1, b)
    if ix1 <= ix0 or iy1 <= iy0:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    src = layer[iy0 - t:iy1 - t, ix0 - l:ix1 - l]
    if src.shape[0] != (iy1 - iy0) or src.shape[1] != (ix1 - ix0):
        # Layer PNG does not match its declared box; skip the overlay rather
        # than misaligning pixels.
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    src_rgb = src[..., :3].astype(np.float32)
    alpha = (src[..., 3:4].astype(np.float32) / 255.0) if src.shape[-1] >= 4 else 1.0
    dst = out[iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0]
    out[iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0] = src_rgb * alpha + dst * (1.0 - alpha)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
