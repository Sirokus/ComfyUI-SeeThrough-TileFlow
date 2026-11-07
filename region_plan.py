"""User-placed square windows for the four-step SeeThrough pipeline.

The uniform grid and the automatic subject strips both derive their boxes.
This module is the third option: the user drags a handful of squares, each
one wrapping a whole subject part (head, torso, legs, a hand).  One square is
one inference, so the model sees that part complete instead of cut by a seam.

The planner is CPU-only and deterministic.  Its document has the same shape
as :func:`tile_strips.strips_document`, so the tile writer, guide and merge
consume it unchanged.  The only extra information downstream needs is
``partition_plan.regions[].resolution`` (a per-window model size) and the
array order, which is the generation order.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

CUSTOM_MODE = "custom_regions"
CUSTOM_LABEL = "自定义分区"
# Mirrors tile_psd.GRID_KIND.  Kept as a literal so this module stays free of
# the image/PSD imports and can be unit tested (and imported by the browser-
# facing helpers) without ComfyUI loaded.
GRID_KIND = "stlu_tile_grid"

SIZE_MIN = 64
SIZE_STEP = 64
RES_MIN = 512
RES_MAX = 2048
RES_STEP = 64
NAME_MAX = 24

# Characters that cannot survive a round trip through a workflow widget
# without making the JSON harder to read; names are display-only anyway.
_NAME_STRIP = re.compile(r"[\r\n\t]+")


def _round_step(value: Any, step: int = SIZE_STEP) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if number != number:  # NaN
        return 0
    return int(round(number / float(step))) * int(step)


def _clamp(value: int, low: int, high: int) -> int:
    return max(int(low), min(int(high), int(value)))


def _side_limit(width: int, height: int) -> int:
    return min(int(width), int(height))


def parse_custom_regions(
    text: str,
    width: int,
    height: int,
    warnings: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Parse the ``custom_regions`` widget into normalised windows.

    Each window is ``{x, y, size, res, name}``; ``res`` is 0 for "follow the
    global resolution".  Invalid entries are skipped with a Chinese warning so
    one broken row cannot cost the user the whole layout.  An empty/invalid
    list is an error: unlike the automatic planner there is no sensible
    fallback when the user asked for a specific geometry.
    """
    width, height = int(width), int(height)
    limit = _side_limit(width, height)
    if limit < SIZE_MIN:
        raise ValueError(
            "自定义分区需要至少 %d px 的画布，当前 %dx%d 太小。"
            " / custom partitions need a canvas of at least %d px;"
            " the current %dx%d is too small."
            % (SIZE_MIN, width, height, SIZE_MIN, width, height))
    warn = warnings if warnings is not None else []

    raw = (text or "").strip()
    if not raw:
        raw = "[]"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("custom_regions 不是有效 JSON / custom_regions is not valid JSON: %s" % exc) from exc
    if not isinstance(data, list):
        raise ValueError("custom_regions 必须是数组 / custom_regions must be an array (one square window per element)")

    regions: List[Dict[str, Any]] = []
    seen = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            warn.append("第 %d 个窗口格式不对，已跳过 / window %d skipped: bad format"
                        % (index + 1, index + 1))
            continue
        try:
            raw_size = float(item.get("size", 0))
            if raw_size != raw_size:  # NaN
                raise ValueError
            size = int(round(raw_size / SIZE_STEP)) * SIZE_STEP
            x = int(round(float(item.get("x", 0))))
            y = int(round(float(item.get("y", 0))))
            res = int(round(float(item.get("res", 0))))
        except (TypeError, ValueError):
            warn.append("第 %d 个窗口的坐标/边长不是数字，已跳过 / window %d skipped: x/size is not numeric"
                        % (index + 1, index + 1))
            continue
        if size < SIZE_MIN:
            warn.append("第 %d 个窗口边长太小，已放大到 %d px / window %d size too small, scaled up to %d px"
                        % (index + 1, SIZE_MIN, index + 1, SIZE_MIN))
            size = SIZE_MIN
        if size > limit:
            warn.append("第 %d 个窗口边长超过画布短边，已收到 %d px / window %d exceeds the canvas short side, clamped to %d px"
                        % (index + 1, limit, index + 1, limit))
            size = limit
        x = _clamp(x, 0, width - size)
        y = _clamp(y, 0, height - size)
        if res < 0:
            res = 0
        if res > 0:
            rounded = _round_step(res)
            if rounded < RES_MIN:
                if rounded != res:
                    warn.append("第 %d 个窗口分辨率已调整到 %d / window %d resolution adjusted to %d"
                                % (index + 1, RES_MIN, index + 1, RES_MIN))
                rounded = RES_MIN
            if rounded > RES_MAX:
                warn.append("第 %d 个窗口分辨率超过上限，已收到 %d / window %d resolution above the limit, clamped to %d"
                            % (index + 1, RES_MAX, index + 1, RES_MAX))
                rounded = RES_MAX
            res = rounded
        key = (x, y, size)
        if key in seen:
            warn.append("第 %d 个窗口与前面的窗口完全重复，已跳过 / window %d duplicates an earlier window, skipped"
                        % (index + 1, index + 1))
            continue
        seen.add(key)
        name = _NAME_STRIP.sub(" ", str(item.get("name") or "")).strip()
        regions.append({
            "x": int(x),
            "y": int(y),
            "size": int(size),
            "res": int(res),
            "name": name[:NAME_MAX],
        })

    if not regions:
        raise ValueError(
            "自定义分区没有有效的正方形窗口。请在 ① 面板点「＋ 新窗口」，"
            "拖动和缩放后再 Queue。"
            " / custom partitions contain no valid square window; click"
            " \"+ New window\" in the ① panel, drag and resize it, then queue.")
    return regions


def region_resolution(region: Dict[str, Any], global_resolution: int = RES_MIN) -> int:
    """Effective model resolution for one window (0 means follow global)."""
    try:
        override = int(region.get("res") or 0)
    except (TypeError, ValueError):
        override = 0
    if override > 0:
        return _clamp(override, RES_MIN, RES_MAX)
    try:
        fallback = int(global_resolution)
    except (TypeError, ValueError):
        fallback = RES_MIN
    return _clamp(fallback, RES_MIN, RES_MAX)


def plan_key(regions: Sequence[Dict[str, Any]], width: int, height: int) -> str:
    """Deterministic geometry key shared with the browser panel.

    The frontend builds the exact same string, so ``select_plan`` can be
    compared without implementing SHA1 in the browser.  Names are excluded:
    renaming a window must not clear a finished output folder.
    """
    parts = []
    for region in regions:
        parts.append("%d,%d,%d,%d" % (
            int(region.get("x", 0)), int(region.get("y", 0)),
            int(region.get("size", 0)), int(region.get("res", 0))))
    return "custom:%s:%dx%d" % (";".join(parts), int(width), int(height))


def custom_document(
    regions: Sequence[Dict[str, Any]],
    width: int,
    height: int,
    *,
    global_resolution: int = RES_MIN,
    created: str = "",
    overlap: int = 0,
    name: str = "grid",
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a grid document the rest of the pipeline already understands."""
    width, height = int(width), int(height)
    regions = list(regions)
    if not regions:
        raise ValueError("自定义分区没有窗口 / custom partitions contain no windows")

    coords: List[List[int]] = []
    plan_regions: List[Dict[str, Any]] = []
    plan_tiles: List[Dict[str, Any]] = []
    sides: List[int] = []
    for index, region in enumerate(regions):
        x = _clamp(int(region["x"]), 0, max(0, width - int(region["size"])))
        y = _clamp(int(region["y"]), 0, max(0, height - int(region["size"])))
        size = int(region["size"])
        box = [x, y, x + size, y + size]
        resolution = region_resolution(region, global_resolution)
        sides.append(size)
        coords.append(box)
        plan_regions.append({
            "index": int(index),
            "box": list(box),
            "side": int(size),
            "resolution": int(resolution),
            "res_follow": not bool(int(region.get("res") or 0)),
            "name": str(region.get("name") or ""),
        })
        plan_tiles.append({
            "index": int(index),
            "box": list(box),
            "side": int(size),
            "resolution": int(resolution),
        })

    side = max(sides) if sides else 0
    key = plan_key(regions, width, height)
    partition = {
        "mode": CUSTOM_MODE,
        "side": int(side),
        "resolution": int(_clamp(global_resolution, RES_MIN, RES_MAX)),
        "plan_id": key,
        "order": list(range(len(plan_regions))),
        "regions": plan_regions,
        "tiles": plan_tiles,
        "warnings": list(warnings or []),
    }
    tiles = [{
        "index": int(index),
        "row": 0,
        "col": int(index),
        "box": [int(v) for v in box],
    } for index, box in enumerate(coords)]

    return {
        "version": 1,
        "kind": GRID_KIND,
        "name": str(name or "grid"),
        "created": str(created or ""),
        "mode": CUSTOM_MODE,
        "source": {
            "image": "",
            "width": width,
            "height": height,
        },
        # Compatibility fields for readers that still expect a single size.
        # Every consumer that matters uses coords / partition_plan.regions.
        "tile_size": [int(side), int(side)],
        "overlap": int(max(0, overlap)),
        "offset": [0, 0],
        "rows": 1,
        "columns": 1,
        "count": len(coords),
        "coords": coords,
        "tiles": tiles,
        "selection": "all",
        "partition_plan": partition,
    }


def uniform_windows(
    coords: Sequence[Sequence[int]],
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    resolution: int = 0,
) -> List[Dict[str, Any]]:
    """Turn grid / strip boxes into editable square windows.

    A uniform grid with square tiles maps one-to-one.  A non-square box (an
    edge tile, a wide strip) keeps its largest inscribed square, centred, so
    the import never invents an area the user did not have.  Duplicates are
    dropped because two identical squares would be the same inference twice.
    """
    width = int(width) if width is not None else 0
    height = int(height) if height is not None else 0
    windows: List[Dict[str, Any]] = []
    seen = set()
    for box in coords or []:
        if box is None or len(box) < 4:
            continue
        x0, y0, x1, y1 = (int(v) for v in box[:4])
        bw, bh = max(0, x1 - x0), max(0, y1 - y0)
        if bw <= 0 or bh <= 0:
            continue
        size = _round_step(min(bw, bh))
        limit = _side_limit(width, height) if width and height else size
        size = _clamp(size, SIZE_MIN, max(SIZE_MIN, limit))
        if size > bw or size > bh:
            # A box smaller than SIZE_MIN cannot become a valid window.
            continue
        x = x0 + (bw - size) // 2
        y = y0 + (bh - size) // 2
        if width:
            x = _clamp(x, 0, max(0, width - size))
        if height:
            y = _clamp(y, 0, max(0, height - size))
        key = (x, y, size)
        if key in seen:
            continue
        seen.add(key)
        windows.append({
            "x": int(x),
            "y": int(y),
            "size": int(size),
            "res": int(resolution or 0),
            "name": "",
        })
    return windows


def order_from_document(doc: Optional[Dict[str, Any]], count: int) -> List[int]:
    """Generation order stored in a document, validated against ``count``."""
    order = []
    if isinstance(doc, dict):
        plan = doc.get("partition_plan") or {}
        raw = plan.get("order") if isinstance(plan, dict) else None
        if isinstance(raw, (list, tuple)):
            for value in raw:
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < count and index not in order:
                    order.append(index)
    for index in range(int(count)):
        if index not in order:
            order.append(index)
    return order


def summaries(regions: Sequence[Dict[str, Any]], global_resolution: int = RES_MIN) -> str:
    """One-line log summary, e.g. ``0:1536@512 1:1024@1024``.

    Accepts both the editor's windows (``size``) and the document's plan
    regions (``side``).
    """
    bits = []
    for index, region in enumerate(regions or []):
        size = int(region.get("size", region.get("side", 0)) or 0)
        bits.append("%d:%d@%d" % (
            index, size, region_resolution(region, global_resolution)))
    return " ".join(bits)
