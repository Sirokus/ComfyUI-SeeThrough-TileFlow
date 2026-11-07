"""Square, full-subject-width strips for SeeThrough.

The uniform grid is a grid of independent samples.  Every vertical seam that
crosses the character makes the two sides invent their own completion, so a
face, a pair of ears or a piece of hair can disagree even after seam feathering.
This module plans the alternative: a vertical stack of square windows, each as
wide as the whole subject.  One window is one inference, so left and right are
generated together; only horizontal seams remain, and those can be placed at
low-information rows.

The planner is CPU-only and deterministic.  It returns the same document shape
as :func:`tile_psd.grid_document`, so the existing tile writer, guide and merge
consume it unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

if __package__:  # ComfyUI loads the folder as a package
    from . import tile_psd
else:  # standalone tests / scripts
    import tile_psd

STRIP_MODE = "uniform"
SUBJECT_MODE = "subject_strips"
DENSITY = 2.0                 # source pixels per mask pixel (1024/512)
MIN_RESOLUTION = 512          # SeeThrough's own minimum
MODEL_MAX_RESOLUTION = 2048

# A detached accessory smaller than this fraction of the largest subject
# component is treated as a speck (the user's Test image has a 152 px speck in
# the bottom-right corner).  0.5 % is conservative: a real detached part is
# almost always larger.
MIN_COMPONENT_FRACTION = 0.005
MIN_COMPONENT_PIXELS = 64


def _ceil64(value: float) -> int:
    return int(math.ceil(max(0.0, float(value)) / 64.0) * 64)


def _clamp(value: int, low: int, high: int) -> int:
    return max(int(low), min(int(high), int(value)))


def _subject_meta(subject: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe subject metadata (never the full foreground ndarray)."""
    if not isinstance(subject, dict):
        return {}
    return {key: value for key, value in subject.items()
            if key not in ("foreground", "components")}


def _union_box(boxes: Sequence[Sequence[int]]) -> Optional[Tuple[int, int, int, int]]:
    boxes = [b for b in boxes if b and len(b) >= 4]
    if not boxes:
        return None
    x0 = min(int(b[0]) for b in boxes)
    y0 = min(int(b[1]) for b in boxes)
    x1 = max(int(b[2]) for b in boxes)
    y1 = max(int(b[3]) for b in boxes)
    return (x0, y0, x1, y1)


def resolve_source_rgb(path: str, max_side: int = 0) -> np.ndarray:
    """Load an image as RGB uint8 for the CPU planner/route."""
    from PIL import Image
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if max_side and max(rgb.size) > max_side:
            scale = float(max_side) / float(max(rgb.size))
            rgb = rgb.resize(
                (max(1, int(round(rgb.width * scale))),
                 max(1, int(round(rgb.height * scale)))),
                Image.Resampling.LANCZOS)
        return np.asarray(rgb, dtype=np.uint8)


def detect_subject(rgb: np.ndarray, alpha: Optional[np.ndarray] = None,
                   alpha_threshold: int = 8) -> Dict[str, Any]:
    """Find the character bbox on a flat background (or from alpha).

    The flat-background detector is the same one the merge uses for phantom
    cleanup.  If it cannot make a confident background it falls back to the
    full frame; that keeps the strip mode usable on photographs, at the cost of
    a wider window and a visible warning.  The planner must never silently
    decide that a busy background is the subject.
    """
    array = np.asarray(rgb)
    if array.ndim == 2:
        array = np.dstack([array, array, array])
    if array.shape[-1] == 4:
        if alpha is None:
            alpha = array[..., 3]
        array = array[..., :3]
    rgb8 = np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))
    height, width = rgb8.shape[:2]
    warnings: List[str] = []
    source = "background"
    foreground: Optional[np.ndarray] = None

    # An explicit alpha channel is the most reliable subject mask when the
    # source really has transparency.  A fully opaque alpha is ignored.
    if alpha is not None:
        a = np.asarray(alpha)
        if a.ndim == 3:
            a = a[..., 0]
        a = np.clip(a.astype(np.float32), 0.0, 255.0)
        if a.shape[:2] == (height, width):
            transparent = float(np.count_nonzero(a <= float(alpha_threshold)))
            if transparent > max(16, int(height * width * 0.005)):
                foreground = a > float(alpha_threshold)
                source = "alpha"

    if foreground is None:
        background = tile_psd._border_background_mask(rgb8)
        if background is not None:
            foreground = ~background
            source = "background"
        else:
            foreground = np.ones((height, width), dtype=bool)
            source = "full"
            warnings.append(
                "没有检测到稳定的平坦背景，主体框退回整幅图；分区仍会按整幅图宽度进行。"
                " / no stable flat background was detected; the subject box"
                " falls back to the whole image and partitions still use the"
                " full image width.")

    fg = np.ascontiguousarray(foreground.astype(np.uint8))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(fg, 8)
    components = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        components.append({"area": area, "box": (x, y, x + w, y + h)})
    if not components:
        # An empty image: use the full frame and let the node/UI show it.
        components = [{"area": int(height * width),
                       "box": (0, 0, int(width), int(height))}]
        source = "full"
        warnings.append("前景检测为空，主体框退回整幅图 / foreground detection empty; subject box falls back to the whole image")

    largest = max(components, key=lambda item: int(item["area"]))
    largest_area = max(1, int(largest["area"]))
    threshold = max(MIN_COMPONENT_PIXELS,
                    int(round(largest_area * MIN_COMPONENT_FRACTION)))
    kept = [largest]
    ignored_pixels = 0
    ignored_count = 0
    for item in components:
        if item is largest:
            continue
        if int(item["area"]) >= threshold:
            kept.append(item)
        else:
            ignored_pixels += int(item["area"])
            ignored_count += 1
    if ignored_count:
        warnings.append(
            "忽略 %d 个小于主体 0.5%% 的碎点（共 %d px）。"
            " / ignored %d specks smaller than 0.5%% of the subject"
            " (%d px total)."
            % (ignored_count, ignored_pixels, ignored_count, ignored_pixels))

    bbox = _union_box([item["box"] for item in kept])
    if bbox is None:
        bbox = (0, 0, int(width), int(height))
        source = "full"
        warnings.append("主体框为空，退回整幅图 / empty subject box; falling back to the whole image")
    x0, y0, x1, y1 = bbox
    return {
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "width": int(x1 - x0),
        "height": int(y1 - y0),
        "source": source,
        "background_fraction": float(1.0 - np.count_nonzero(foreground) / float(max(1, height * width))),
        "components": kept,
        "ignored_pixels": int(ignored_pixels),
        "warnings": warnings,
        "foreground": foreground,
        "shape": [int(width), int(height)],
    }


def plan_subject_strips(width: int, height: int, subject: Dict[str, Any],
                        *, overlap: int = 256, max_side: int = 2048,
                        max_resolution: int = 768, margin: int = 32,
                        resolution: Optional[int] = None) -> Dict[str, Any]:
    """Plan a deterministic top-to-bottom stack of square subject-wide windows.

    ``max_side`` is a preferred cap, not a hard one.  When the subject needs a
    wider square to keep its full width, the planner grows to that width (up to
    the canvas' short side) and reports it.  A square that cannot fit the
    subject at all (a very wide subject on a short canvas) still falls back to
    the uniform grid; the planner never cuts the subject vertically.

    ``resolution`` — when given, it is the model resolution directly (the
    same value as the pipeline's quality setting), clamped to the model's
    supported range.  ``max_resolution`` is only used when ``resolution`` is
    omitted (legacy callers/tests): the square is derived from ``side`` and
    capped by it.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    warnings: List[str] = list(subject.get("warnings") or [])
    bbox = subject.get("bbox") or [0, 0, width, height]
    x0, y0, x1, y1 = (int(v) for v in bbox[:4])
    x0, x1 = _clamp(x0, 0, width), _clamp(x1, 0, width)
    y0, y1 = _clamp(y0, 0, height), _clamp(y1, 0, height)
    sw, sh = max(0, x1 - x0), max(0, y1 - y0)
    if sw <= 0 or sh <= 0:
        return {
            "ok": False, "fallback": "uniform",
            "error": "主体框为空，无法规划方形分区。"
                     " / the subject box is empty; cannot plan square partitions.",
            "warnings": warnings, "subject": _subject_meta(subject),
        }

    max_side = max(64, int(max_side))
    # A square window can never be larger than the canvas' short side.  The
    # configured cap may be smaller than the subject; in that case the only
    # way to keep the subject's full width is to grow the square (the model
    # resolution has its own separate cap, so this does not add VRAM).
    hard_square = min(width, height)
    margin = max(0, int(margin))
    effective_margin = min(margin, max(0, (hard_square - sw) // 2))
    if effective_margin < margin:
        warnings.append(
            "主体太宽，边距从 %d px 收窄到 %d px；方形窗口仍覆盖主体全宽。"
            " / the subject is too wide, so the margin shrank from %d px to"
            " %d px; the square window still covers the full subject width."
            % (margin, effective_margin, margin, effective_margin))
    required_side = sw + 2 * effective_margin
    if required_side > hard_square:
        return {
            "ok": False, "fallback": "uniform",
            "error": ("主体宽度 %d px 加上边距后超过画布短边 %d px，"
                      "方形窗口无法完整覆盖主体；不会竖直切开主体。"
                      " / the subject width of %d px plus margins exceeds the"
                      " canvas' short side of %d px, so a square window cannot"
                      " cover it; the subject will not be split vertically."
                      % (required_side, hard_square, required_side, hard_square)),
            "warnings": warnings, "subject": _subject_meta(subject),
        }
    max_square = min(hard_square, max(max_side, required_side))
    if max_side < required_side:
        warnings.append(
            "区域最大边 %d px 小于覆盖主体所需的 %d px，已自动放大到 %d px；"
            "不会竖直切开主体。" % (max_side, required_side, max_square)
            + " / the area cap of %d px is below the %d px needed to cover the"
              " subject, so it grew to %d px; the subject will not be split"
              " vertically." % (max_side, required_side, max_square))

    y_top = max(0, y0 - effective_margin)
    y_bottom = min(height, y1 + effective_margin)
    subject_h = max(1, y_bottom - y_top)
    side_floor = sw + 2 * effective_margin
    overlap = max(0, min(int(overlap), max_square - 1))

    if subject_h <= max_square:
        side = max(side_floor, subject_h)
        side = min(side, max_square)
        top = _clamp(y_top - (side - subject_h) // 2, 0, height - side)
        starts = [int(top)]
        n = 1
    else:
        if max_square <= overlap:
            n = max(1, int(math.ceil(subject_h / float(max_square))))
            overlap_used = 0
        else:
            overlap_used = overlap
            n = max(1, int(math.ceil(
                (subject_h - overlap_used) / float(max_square - overlap_used))))
        # Grow n until the width-constrained side also covers the subject
        # height with the requested overlap.
        while True:
            side = max(side_floor, int(math.ceil(
                subject_h / float(n) + overlap_used * (n - 1) / float(n))))
            if side <= max_square:
                break
            n += 1
            if n > subject_h + 1:
                return {
                    "ok": False, "fallback": "uniform",
                    "error": "无法在方形上限内覆盖主体高度，请调大 strip_max_side。"
                             " / cannot cover the subject height within the"
                             " square cap; raise strip_max_side.",
                    "warnings": warnings, "subject": _subject_meta(subject),
                }
        side = min(side, max_square)
        last = y_bottom - side
        if last < y_top:
            last = y_top
        distance = (last - y_top) / float(max(1, n - 1))
        starts = []
        for index in range(n):
            value = int(round(y_top + index * distance))
            starts.append(_clamp(value, 0, height - side))
        starts[0] = _clamp(y_top, 0, height - side)
        starts[-1] = _clamp(last, 0, height - side)
        starts = sorted(set(starts))
        # Rounding can merge two starts; the resulting region count is the
        # truth and the overlap is recomputed below.
        n = len(starts)

    x_center = (x0 + x1) // 2
    left = _clamp(x_center - side // 2, 0, width - side)
    coords = [[int(left), int(top), int(left + side), int(top + side)]
              for top in starts]
    actual_overlap = 0
    if len(coords) > 1:
        actual_overlap = min(
            int(coords[index][3] - coords[index + 1][1])
            for index in range(len(coords) - 1))
    if resolution is None:
        resolution = max(MIN_RESOLUTION,
                         min(int(max_resolution), _ceil64(side / DENSITY)))
        resolution = min(MODEL_MAX_RESOLUTION, resolution)
        capped = resolution < _ceil64(side / DENSITY)
        if capped:
            warnings.append(
                "分区边长 %d px，模型分辨率被上限截到 %d；遮罩密度约 %.2f 源像素/遮罩像素。"
                " / the partition side is %d px, so the model resolution was"
                " capped at %d; mask density is about %.2f source px per mask px."
                % (side, resolution, side / float(max(1, resolution)),
                   side, resolution, side / float(max(1, resolution))))
    else:
        resolution = max(MIN_RESOLUTION,
                         min(MODEL_MAX_RESOLUTION, int(resolution)))
    density = side / float(max(1, resolution))
    plan_id = hashlib.sha1(json.dumps({
        "coords": coords,
        "side": int(side),
        "resolution": int(resolution),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    return {
        "ok": True,
        "mode": SUBJECT_MODE,
        "coords": coords,
        "side": int(side),
        "resolution": int(resolution),
        "density": float(density),
        "overlap": int(actual_overlap),
        "requested_overlap": int(overlap),
        "margin": int(effective_margin),
        "plan_id": plan_id,
        "subject": _subject_meta(subject),
        "warnings": warnings,
    }


def plan_from_subject(width: int, height: int, subject: Dict[str, Any], *,
                      overlap: int = 256, max_side: int = 2048,
                      max_resolution: int = 768,
                      margin: int = 32,
                      resolution: Optional[int] = None) -> Dict[str, Any]:
    """Plan from an already detected subject (no full-size pixel work).

    Split out from :func:`plan_from_rgb` so a caller that caches subject
    detection can change the plan parameters without re-scanning a 4K image.
    ``subject`` must keep ``bbox`` and ``warnings``; ``foreground`` is not
    needed.
    """
    plan = plan_subject_strips(
        width, height, subject, overlap=overlap, max_side=max_side,
        max_resolution=max_resolution, margin=margin, resolution=resolution)
    plan["subject_detection"] = {
        "source": subject.get("source"),
        "bbox": list(subject.get("bbox") or []),
        "background_fraction": subject.get("background_fraction"),
        "ignored_pixels": subject.get("ignored_pixels"),
    }
    return plan


def plan_from_rgb(rgb: np.ndarray, *, overlap: int = 256,
                  max_side: int = 2048, max_resolution: int = 768,
                  margin: int = 32,
                  resolution: Optional[int] = None) -> Dict[str, Any]:
    """Convenience entry point used by the node and standalone tests."""
    height, width = np.asarray(rgb).shape[:2]
    subject = detect_subject(rgb)
    return plan_from_subject(
        width, height, subject, overlap=overlap, max_side=max_side,
        max_resolution=max_resolution, margin=margin, resolution=resolution)


def strips_document(plan: Dict[str, Any], width: int, height: int,
                    *, source_image: str = "", name: str = "grid",
                    created: str = "") -> Dict[str, Any]:
    """Build the same grid document shape the rest of the pipeline reads."""
    coords = [[int(v) for v in box[:4]] for box in plan["coords"]]
    tiles: List[Dict[str, Any]] = []
    for index, box in enumerate(coords):
        row, col = tile_psd.tile_rc(coords, index)
        tiles.append({
            "index": int(index),
            "row": int(row),
            "col": int(col),
            "box": [int(v) for v in box],
        })
    side = int(plan["side"])
    partition = {
        "mode": SUBJECT_MODE,
        "side": side,
        "resolution": int(plan["resolution"]),
        "density": float(plan.get("density") or 0.0),
        "overlap": int(plan.get("overlap") or 0),
        "requested_overlap": int(plan.get("requested_overlap") or 0),
        "margin": int(plan.get("margin") or 0),
        "plan_id": str(plan.get("plan_id") or ""),
        "subject": plan.get("subject") or {},
        "subject_detection": plan.get("subject_detection") or {},
        "warnings": list(plan.get("warnings") or []),
        "regions": [{
            "index": int(index),
            "box": [int(v) for v in box],
            "resolution": int(plan["resolution"]),
        } for index, box in enumerate(coords)],
    }
    return {
        "version": 1,
        "kind": tile_psd.GRID_KIND,
        "name": str(name or "grid"),
        "created": str(created or ""),
        "mode": SUBJECT_MODE,
        "source": {
            "image": str(source_image or ""),
            "width": int(width),
            "height": int(height),
        },
        "tile_size": [side, side],
        "overlap": int(plan.get("overlap") or 0),
        "offset": [0, 0],
        "rows": len(tiles),
        "columns": 1,
        "count": len(tiles),
        "coords": coords,
        "tiles": tiles,
        "selection": "all",
        "partition_plan": partition,
    }


def public_plan(plan: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    """A compact plan for the HTTP preview route (no foreground arrays)."""
    if not plan.get("ok"):
        return {
            "ok": False,
            "fallback": plan.get("fallback"),
            "error": plan.get("error") or "cannot plan strips",
            "warnings": list(plan.get("warnings") or []),
            "subject": plan.get("subject") or {},
            "source_size": [int(width), int(height)],
        }
    return {
        "ok": True,
        "mode": SUBJECT_MODE,
        "regions": [[int(v) for v in box[:4]] for box in plan["coords"]],
        "side": int(plan["side"]),
        "resolution": int(plan["resolution"]),
        "density": float(plan.get("density") or 0.0),
        "overlap": int(plan.get("overlap") or 0),
        "requested_overlap": int(plan.get("requested_overlap") or 0),
        "margin": int(plan.get("margin") or 0),
        "plan_id": str(plan.get("plan_id") or ""),
        "subject": plan.get("subject") or {},
        "subject_detection": plan.get("subject_detection") or {},
        "warnings": list(plan.get("warnings") or []),
        "source_size": [int(width), int(height)],
    }
