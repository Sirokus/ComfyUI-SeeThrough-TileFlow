"""Tile-wise PSD assembly for SeeThrough grid-split runs.

``SplitImageToTileList`` turns one illustration into a list of tiles; every
downstream SeeThrough node runs once per tile, so ``SeeThrough_PostProcess``
returns one ``SEETHROUGH_PARTS`` document per tile.  This module turns each of
those documents into a layered PSD without going through the browser-only
"Download PSD" button (which can only read the last tile, because every run
overwrites ``seethrough_psd_info.log``).

Optionally re-merges layers that SeeThrough's ``tblr_split`` style splitting
created from a single part: ``eyel``/``eyer`` -> ``eyes``, ``hairf``/
``hairb`` -> ``hair``, ``ears-l``/``ears-r`` -> ``ears`` and so on.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import psd_reader, psd_writer
    from .core_compose import _alpha_over, alpha_bbox
    from .stlu import json_io
except ImportError:  # pragma: no cover - standalone import
    import psd_reader
    import psd_writer
    from core_compose import _alpha_over, alpha_bbox
    from stlu import json_io

DEFAULT_ALPHA_THRESHOLD = 8
# Flat-mode only: how far past the seam the losing tile fades out.  Folder mode
# keeps every copy whole instead.  Averaging two independently re-composed
# crops over the whole overlap turned their disagreement into a wide ghost
# band; cutting the loser at the ownership line keeps each edge on its side.
DEFAULT_SEAM_FEATHER = 32

# Bump when a change alters the pixels a tile is made from (guide compositing,
# tile cutting, layer building).  It joins the generation fingerprint, so a
# code update invalidates cached tiles instead of silently reusing old ones.
# 2: context_pad tiles are cut back in the model's pixel space, not the source
# window's.  Runs from 1 with padding wrote 512 px parts into a 1694 px PSD,
# so those tile PSDs must never be reused.
ALGO_VERSION = 2


# ---------------------------------------------------------------------------
# Wave folders: one folder per generation run
# ---------------------------------------------------------------------------

def wave_id(prompt_id: Optional[str] = None) -> str:
    """Folder name for one generation wave.

    The ComfyUI prompt id is the natural identity: every queue click gets a
    fresh one, so two runs can never overwrite each other.  It is shortened to
    eight characters so the folder stays readable.  Browser-side runs (the
    "build from images" route) have no prompt id and fall back to a timestamp
    plus a random suffix.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = re.sub(r"[^0-9A-Za-z]", "", str(prompt_id or ""))[:8]
    return "%s_%s" % (stamp, short or uuid.uuid4().hex[:4])


def flat_directory(output_root: str, out_dir: str, group: str = "") -> str:
    """``<output>/<out_dir>/<group?>`` -- the folder the user picked.

    Tiles from earlier runs stay in this exact folder, so a later run can
    re-generate only the checked tiles and still merge the cached neighbours.
    """
    base = os.path.join(output_root, (out_dir or "SeeThrough").strip("/\\"))
    group = str(group or "").strip().strip("/\\")
    if group:
        try:
            from .core_plan import safe_name as _safe
        except ImportError:  # pragma: no cover - standalone import
            from core_plan import safe_name as _safe
        base = os.path.join(base, _safe(group))
    return base


def wave_directory(output_root: str, out_dir: str, group: str, wave: str) -> str:
    """``<output>/<out_dir>/<group?>/<wave>`` (legacy one-folder-per-run)."""
    return os.path.join(flat_directory(output_root, out_dir, group),
                        str(wave or wave_id()))


# Names produced by SeeThrough_PostProcess' left/right and hair splitting.
# Everything else is merged by stripping the "-l"/"-r" suffix, which is enough
# for handwear, eyewhite, irides, eyelash, eyebrow and ears.
MERGE_ALIASES: Dict[str, str] = {
    "eyel": "eyes",
    "eyer": "eyes",
    "browl": "eyebrow",
    "browr": "eyebrow",
    "hairf": "hair",
    "hairb": "hair",
    # v3 tag names, in case the pipeline is switched to the newer model.
    "front hair": "hair",
    "back hair": "hair",
}


def merge_group_name(tag: str) -> str:
    if tag in MERGE_ALIASES:
        return MERGE_ALIASES[tag]
    for suffix in ("-l", "-r"):
        if tag.endswith(suffix) and len(tag) > len(suffix):
            return tag[: -len(suffix)]
    return tag


def _depth(pinfo: Dict[str, Any]) -> float:
    try:
        return float(pinfo.get("depth_median", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Cheap 4-neighbour dilation (no scipy) used to keep soft edges."""
    out = np.array(mask, dtype=bool, copy=True)
    for _ in range(max(0, int(radius))):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


SOLID_ALPHA = 224
HALO_RADIUS = 12
SPECK_MIN_AREA = 512


def _drop_specks(mask: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Remove isolated low-alpha blobs that no real part could be.

    SeeThrough drops a few dozen pixels of alpha 24-36 around every tile's
    rim.  They pass any plain threshold, but they are tiny islands detached
    from the part -- keeping them stretches the layer bbox back out to the
    whole tile (the "huge blank margins" on headwear/face/nose/mouth).  A real
    part is either one big blob or contains a solid pixel (a faint eye line
    still has an opaque core), so keep a connected blob when it spans
    ``SPECK_MIN_AREA`` pixels or holds any pixel at ``SOLID_ALPHA``.
    """
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover - scipy ships with ComfyUI
        return mask
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count <= 1:
        return mask
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    keep = areas >= SPECK_MIN_AREA
    keep[0] = False
    kept = np.unique(labels[alpha >= SOLID_ALPHA])
    keep[kept] = True
    return keep[labels]


def content_mask(alpha: np.ndarray, threshold: int = DEFAULT_ALPHA_THRESHOLD) -> np.ndarray:
    """Separate a part's real pixels from SeeThrough's alpha "dust" and "wash".

    The model leaves two kinds of junk on the otherwise empty background of
    every layer:

    * scattered alpha speckles (values well under 64) -- "dust";
    * on layers it is unsure about, a broad translucent field (alpha 64-192)
      covering most of the tile -- "wash".  ``earwear`` on some tiles is
      nothing but wash: 61% coverage, not one fully opaque pixel.

    Neither is visible against white on its own, but a layered PSD turns every
    bbox into the whole tile and the wash tints every flattened preview
    lavender/pink.  A real part always has a solid core (``SOLID_ALPHA`` and
    up).  So: drop the dust, and when a layer has no core at all, or its faint
    field swamps the core, keep only the core plus an anti-aliased rim.
    """
    alpha = np.asarray(alpha)
    weak = alpha > int(threshold)
    if not weak.any():
        return np.zeros(alpha.shape, dtype=bool)
    weak = _drop_specks(weak, alpha)
    if not weak.any():
        return np.zeros(alpha.shape, dtype=bool)
    solid = alpha >= SOLID_ALPHA
    if not solid.any():
        # No solid pixel at all.  A broad translucent field is a wash; a
        # small faint part can still be real (a handful of speckles is grain).
        if float(weak.mean()) > 0.05:
            return np.zeros(alpha.shape, dtype=bool)
        return weak if int(weak.sum()) >= 512 else np.zeros(alpha.shape, dtype=bool)
    coverage = float(weak.mean())
    core = float(solid.mean())
    if coverage > max(2.0 * core, core + 0.10):
        # Faint field swamps the core -- keep the core and its soft rim only.
        return weak & _dilate(solid, HALO_RADIUS)
    return weak


def _clean_part(
    pinfo: Dict[str, Any],
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Return (dust-free RGBA crop, bbox inside the part image)."""
    img = pinfo.get("img")
    if img is None:
        return None
    arr = np.array(img, dtype=np.uint8, copy=True)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        return None
    mask = content_mask(arr[..., 3], int(alpha_threshold))
    if not mask.any():
        return None
    dead = ~mask
    # Straight-alpha RGBA keeps colour under alpha 0; resampling (thumbnails,
    # LANCZOS, the web preview) bleeds that hidden colour into visible pixels
    # as pink fringes.  Whitening the dead pixels keeps every flatten white.
    arr[..., 3][dead] = 0
    arr[..., :3][dead] = 255
    ys, xs = np.where(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return arr, bbox


def _compose_group(
    items: Sequence[Dict[str, Any]],
    frame_h: int,
    frame_w: int,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> Optional[Tuple[np.ndarray, int, int]]:
    """Alpha-composite one merge group onto a tile-sized canvas."""
    canvas = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    placed = False
    for pinfo in items:  # far -> near
        cleaned = _clean_part(pinfo, int(alpha_threshold))
        if cleaned is None:
            continue
        arr, (bx0, by0, bx1, by1) = cleaned
        x1, y1 = pinfo.get("xyxy", [0, 0, frame_w, frame_h])[:2]
        _alpha_over(canvas, arr[by0:by1, bx0:bx1], int(x1) + bx0, int(y1) + by0)
        placed = True
    if not placed:
        return None
    bbox = alpha_bbox(content_mask(canvas[..., 3], int(alpha_threshold)).astype(np.uint8) * 255, 1)
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return canvas[top:bottom, left:right].copy(), int(left), int(top)


def build_tile_layers(
    parts: Dict[str, Any],
    merge_parts: bool = True,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Return the PSD layers for one tile, bottom -> top order."""
    tag2pinfo = parts.get("tag2pinfo") or {}
    frame_h, frame_w = (int(v) for v in (parts.get("frame_size") or (0, 0))[:2])
    if frame_h <= 0 or frame_w <= 0:
        raise ValueError("parts document has no frame_size")

    groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for tag, pinfo in tag2pinfo.items():
        if not isinstance(pinfo, dict) or pinfo.get("img") is None:
            continue
        key = merge_group_name(tag) if merge_parts else tag
        groups.setdefault(key, []).append((tag, pinfo))

    layers: List[Dict[str, Any]] = []
    for name, items in groups.items():
        # Far first, so the nearest item ends up on top when compositing.
        items.sort(key=lambda item: _depth(item[1]), reverse=True)

        if merge_parts and len(items) > 1:
            composed = _compose_group([p for _, p in items], frame_h, frame_w, int(alpha_threshold))
            if composed is None:
                continue
            pixels, left, top = composed
            depth = min(_depth(p) for _, p in items)
            sources = [tag for tag, _ in items]
        else:
            tag, pinfo = items[0]
            cleaned = _clean_part(pinfo, int(alpha_threshold))
            if cleaned is None:
                continue
            arr, (l, t, r, b) = cleaned
            pixels = arr[t:b, l:r].copy()
            left = int(pinfo.get("xyxy", [0, 0])[0]) + int(l)
            top = int(pinfo.get("xyxy", [0, 0])[1]) + int(t)
            depth = _depth(pinfo)
            sources = [tag]

        layers.append({
            "name": name,
            "pixels": np.ascontiguousarray(pixels),
            "left": int(left),
            "top": int(top),
            "depth_median": float(depth),
            "sources": sources,
        })

    # PSD bottom -> top is far -> near.
    layers.sort(key=lambda layer: layer["depth_median"], reverse=True)
    return layers


def build_tile_depth(
    parts: Dict[str, Any],
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> Optional[np.ndarray]:
    """One tile-sized depth map (uint8, 0 = no surface).

    Every part carries Marigold depth cropped to the same box as its RGBA.
    Painting far -> near means the nearest surface wins per pixel, which is
    what a flat depth map should show.  0 is kept as "nothing here" so a
    merged map can tell uncovered pixels from a legitimately near surface
    (the real values are shifted into 1..255).
    """
    tag2pinfo = parts.get("tag2pinfo") or {}
    frame_h, frame_w = (int(v) for v in (parts.get("frame_size") or (0, 0))[:2])
    if frame_h <= 0 or frame_w <= 0:
        return None

    items = []
    for tag, pinfo in tag2pinfo.items():
        if not isinstance(pinfo, dict) or pinfo.get("img") is None:
            continue
        depth = pinfo.get("depth")
        if depth is None:
            continue
        items.append((tag, pinfo, np.asarray(depth)))

    if not items:
        return None
    items.sort(key=lambda item: _depth(item[1]), reverse=True)  # far -> near

    canvas = np.zeros((frame_h, frame_w), dtype=np.float32)
    covered = np.zeros((frame_h, frame_w), dtype=bool)
    for _tag, pinfo, depth in items:
        cleaned = _clean_part(pinfo, int(alpha_threshold))
        if cleaned is None or depth.ndim != 2:
            continue
        arr, (bx0, by0, bx1, by1) = cleaned
        if depth.shape[:2] != arr.shape[:2]:
            continue
        mask = arr[..., 3] > 0
        x, y = int(pinfo.get("xyxy", [0, 0])[0]), int(pinfo.get("xyxy", [0, 0])[1])
        sub_mask = mask[by0:by1, bx0:bx1]
        sub_depth = depth[by0:by1, bx0:bx1].astype(np.float32) / 255.0
        x0, y0 = max(0, x + bx0), max(0, y + by0)
        x1 = min(frame_w, x + bx1)
        y1 = min(frame_h, y + by1)
        if x1 <= x0 or y1 <= y0:
            continue
        mx0, my0 = x0 - (x + bx0), y0 - (y + by0)
        m = sub_mask[my0:my0 + (y1 - y0), mx0:mx0 + (x1 - x0)]
        d = sub_depth[my0:my0 + (y1 - y0), mx0:mx0 + (x1 - x0)]
        target = canvas[y0:y1, x0:x1]
        crop = covered[y0:y1, x0:x1]
        target[m] = d[m]
        crop[m] = True

    if not covered.any():
        return None
    out = np.zeros((frame_h, frame_w), dtype=np.uint8)
    out[covered] = np.clip(np.rint(1.0 + canvas[covered] * 254.0), 1, 255).astype(np.uint8)
    return out


def _axis_starts(length: int, tile: int, stride: int, phase: int) -> List[int]:
    """Tile starts along one axis, with an optional lattice phase.

    The first and last tiles always touch the canvas edges; only the interior
    split lines move.  That keeps every tile at the nominal ``tile`` size --
    clipping an edge tile would hand SeeThrough a non-square crop, which it
    letterboxes and the merge would then have to stretch back.  ``phase`` is
    taken modulo the stride, so 0 reproduces the classic grid exactly.
    """
    length = max(1, int(length))
    tile = max(1, int(tile))
    if tile >= length:
        return [0]
    edge = length - tile
    starts = {0, edge}
    stride = max(1, int(stride))
    phase = int(phase) % stride
    start = phase
    while start < edge:
        if start > 0:
            starts.add(start)
        start += stride
    return sorted(starts)


def grid_coords(width: int, height: int, tile_w: int, tile_h: int, overlap: int,
                offset_x: int = 0, offset_y: int = 0) -> List[Tuple[int, int, int, int]]:
    """Mirror of comfy_extras SplitImageToTileList.get_grid_coords.

    ``offset_x`` / ``offset_y`` shift the interior split lines (the "phase" of
    the grid) without adding blank margins: 0 is the classic layout, and the
    usual reason to change it is to stop a seam from cutting the subject in
    half.  The value wraps modulo the stride, so any pixel value is valid.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    tile_w = max(1, int(tile_w))
    tile_h = max(1, int(tile_h))
    stride_x = round(max(tile_w * 0.25, tile_w - overlap))
    stride_y = round(max(tile_h * 0.25, tile_h - overlap))
    xs = _axis_starts(width, tile_w, stride_x, offset_x)
    ys = _axis_starts(height, tile_h, stride_y, offset_y)
    coords: List[Tuple[int, int, int, int]] = []
    for y in ys:
        y_end = min(y + tile_h, height)
        for x in xs:
            x_end = min(x + tile_w, width)
            coords.append((int(x), int(y), int(x_end), int(y_end)))
    return coords


def tile_rc(coords: Sequence[Sequence[int]], index: int) -> Tuple[int, int]:
    """Row/column of tile ``index`` inside the grid (top-left based)."""
    if index < 0 or index >= len(coords):
        return 0, 0
    x0, y0 = int(coords[index][0]), int(coords[index][1])
    row = len({c[1] for c in coords if c[1] < y0})
    col = len({c[0] for c in coords if c[1] == y0 and c[0] < x0})
    return row, col


def _flatten_preview(layers: Sequence[Dict[str, Any]], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    for layer in layers:
        _alpha_over(canvas, layer["pixels"], layer["left"], layer["top"])
    arr = canvas.astype(np.float32) / 255.0
    alpha = arr[..., 3:4]
    rgb = arr[..., :3] * alpha + (1.0 - alpha)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def write_tiles(
    parts_list: Sequence[Dict[str, Any]],
    output_root: str,
    out_dir: str = "SeeThrough/Tiles",
    filename_prefix: str = "tile",
    merge_parts: bool = True,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    source_size: Optional[Tuple[int, int]] = None,
    tile_size: Tuple[int, int] = (1024, 1024),
    overlap: int = 128,
    stamp: Optional[str] = None,
    merge_after: bool = False,
    merged_name: str = "merged",
    tile_index_map: Optional[Sequence[int]] = None,
    write_depth: bool = True,
    original_rgba: Optional[np.ndarray] = None,
    merge_group_layers: bool = False,
    include_original: bool = True,
    run_dir_abs: Optional[str] = None,
    write_manifest: bool = True,
    generation: Optional[Dict[str, Any]] = None,
    generation_id: str = "",
    tile_coords: Optional[Sequence[Sequence[int]]] = None,
    grid_offset: Sequence[int] = (0, 0),
) -> Dict[str, Any]:
    """Write one PSD per tile.  Returns a manifest dict for the UI.

    ``run_dir_abs`` writes into an exact folder instead of the historical
    ``<out_dir>/<prefix>_<stamp>`` layout -- that is how a wave keeps its PSDs
    and its saved layer images in the same folder.

    ``write_manifest=False`` skips ``manifest.json``: the per-tile node writes
    one tile at a time, and the collector node owns the final index.
    """
    if not parts_list:
        raise ValueError("no tiles to write")

    if run_dir_abs:
        run_dir = str(run_dir_abs)
    else:
        base = os.path.join(output_root, out_dir.strip("/\\"))
        run_dir = os.path.join(base, "%s_%s" % (filename_prefix, stamp or "run"))
    os.makedirs(run_dir, exist_ok=True)

    coords: List[Tuple[int, int, int, int]] = []
    try:
        ox, oy = int(grid_offset[0]), int(grid_offset[1])
    except (TypeError, ValueError, IndexError):
        ox, oy = 0, 0
    if tile_coords:
        for box in tile_coords:
            if box is not None and len(box) >= 4:
                coords.append(tuple(int(v) for v in box[:4]))
    if not coords and source_size and source_size[0] > 0 and source_size[1] > 0:
        coords = grid_coords(int(source_size[0]), int(source_size[1]),
                             int(tile_size[0]), int(tile_size[1]), int(overlap),
                             ox, oy)

    # The untouched source is worth keeping next to the tiles: it is the
    # reference for the mask work and the merge puts it under the parts.
    original_name = ""
    if original_rgba is not None:
        try:
            from PIL import Image
            arr = np.asarray(original_rgba)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
            Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8), "RGBA").save(
                os.path.join(run_dir, "original.png"), compress_level=3)
            original_name = "original.png"
        except Exception:  # noqa: BLE001
            original_name = ""

    # Folders need the left/right and front/back variants to survive as
    # separate layers, so the per-tile merge has to be skipped in that mode.
    tile_merge_parts = bool(merge_parts) and not bool(merge_group_layers)

    entries: List[Dict[str, Any]] = []
    for slot, parts in enumerate(parts_list):
        # ``tile_index_map`` carries the original grid index of each tile when
        # STLU_TileSelect skipped some of them before the expensive chain ran.
        index = int(tile_index_map[slot]) if tile_index_map and slot < len(tile_index_map) else slot
        frame_h, frame_w = (int(v) for v in (parts.get("frame_size") or (0, 0))[:2])
        if frame_h <= 0 or frame_w <= 0:
            continue
        layers = build_tile_layers(
            parts,
            merge_parts=tile_merge_parts,
            alpha_threshold=int(alpha_threshold),
        )
        if not layers:
            continue

        if index < len(coords):
            x0, y0, x1, y1 = coords[index]
            # Row/column derived from the tile's top-left corner so the name
            # says where the tile sits in the source image.
            row, col = tile_rc(coords, index)
            name = "%s_%02d_r%d_c%d.psd" % (filename_prefix, index, row, col)
        else:
            x0 = y0 = 0
            x1, y1 = frame_w, frame_h
            name = "%s_%02d.psd" % (filename_prefix, index)

        psd_path = os.path.join(run_dir, name)
        psd_layers = [{
            "name": layer["name"],
            "pixels": layer["pixels"],
            "left": layer["left"],
            "top": layer["top"],
        } for layer in layers]
        psd_writer.write_psd(psd_path, psd_layers, frame_w, frame_h, "rle")

        # One small preview per tile for the selection wall in the UI.
        preview_name = name[:-4] + "_preview.png" if name.lower().endswith(".psd") else name + "_preview.png"
        try:
            from PIL import Image
            Image.fromarray(_flatten_preview(layers, frame_w, frame_h)).save(
                os.path.join(run_dir, preview_name), compress_level=3)
        except Exception:  # noqa: BLE001
            preview_name = ""

        # The depth map stays a plain grayscale image next to the PSD: it is
        # merged on its own so it never tints the colour layers.
        depth_name = ""
        if write_depth:
            try:
                from PIL import Image
                depth_map = build_tile_depth(parts, int(alpha_threshold))
                if depth_map is not None:
                    stem = name[:-4] if name.lower().endswith(".psd") else name
                    depth_name = stem + "_depth.png"
                    Image.fromarray(depth_map, "L").save(
                        os.path.join(run_dir, depth_name), compress_level=6)
            except Exception:  # noqa: BLE001
                depth_name = ""

        entries.append({
            "index": int(index),
            "slot": int(slot),
            "file": name,
            "preview": preview_name,
            "depth": depth_name,
            "canvas": [frame_w, frame_h],
            "tile": [int(x0), int(y0), int(x1), int(y1)],
            "layers": [{
                "name": layer["name"],
                "left": int(layer["left"]),
                "top": int(layer["top"]),
                "right": int(layer["left"] + layer["pixels"].shape[1]),
                "bottom": int(layer["top"] + layer["pixels"].shape[0]),
                "sources": layer["sources"],
                # Kept so the cross-tile merge can order parts by the model's
                # own depth instead of guessing from per-tile layer indices.
                "depth_median": float(layer["depth_median"]),
            } for layer in layers],
        })

    if not entries:
        raise ValueError("every tile ended up empty")

    # Card preview: reuse the preview of the tile with the most layers (a
    # corner tile is often empty).  A separate ``_preview.png`` was the exact
    # same bytes as that tile preview, so it is simply the same file now.
    richest = max(entries, key=lambda entry: len(entry["layers"]))
    card_preview = str(richest.get("preview") or "")

    src = None
    if source_size and source_size[0] > 0 and source_size[1] > 0:
        src = [int(source_size[0]), int(source_size[1])]
    else:
        src = [max(e["tile"][2] for e in entries), max(e["tile"][3] for e in entries)]

    manifest = {
        "dir": os.path.relpath(run_dir, output_root).replace(os.sep, "/"),
        "prefix": filename_prefix,
        "merge_parts": bool(merge_parts),
        "source": src,
        "tile_size": [int(tile_size[0]), int(tile_size[1])],
        "overlap": int(overlap),
        "offset": [int(ox), int(oy)],
        "preview": card_preview,
        "original": original_name,
        "has_depth": any(entry.get("depth") for entry in entries),
        "group_layers": bool(merge_group_layers),
        "tiles": entries,
        "count": len(entries),
        "generation": generation or {},
        "generation_id": str(generation_id or ""),
    }
    if merge_after:
        try:
            manifest["merged"] = merge_tiles(
                run_dir, manifest,
                indices=None,
                out_name=merged_name,
                merge_parts=bool(merge_parts),
                alpha_threshold=int(alpha_threshold),
                group_layers=bool(merge_group_layers),
                include_depth=bool(write_depth),
                include_original=bool(include_original),
            )
        except Exception as exc:  # noqa: BLE001
            manifest["merge_error"] = str(exc)

    if write_manifest:
        manifest_path = os.path.join(run_dir, "manifest.json")
        json_io.write_json_atomic(manifest_path, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Cross-tile merge
# ---------------------------------------------------------------------------

# Anything at or below this alpha counts as "this tile has no pixel here".
# Weighting only the tiles that actually draw the part is what keeps a part
# that appears in one tile (and not in its neighbour, which covers the same
# seam) at full opacity instead of the average of one and zero.
ALPHA_EPS = 4.0 / 255.0


def _accumulate_part(
    canvas_w: int,
    canvas_h: int,
    pieces: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    """Combine one part's pixels across tiles (straight alpha).

    ``weight`` is the ownership ramp from :func:`_apply_seam_weights`: one
    tile owns the pixel, the neighbour only contributes inside the narrow
    handover.  Colour is weighted by alpha (premultiplied) and alpha itself is
    normalised by the weight of the pieces that actually cover the pixel, so a
    part drawn by a single tile keeps full opacity instead of averaging with
    its neighbour's empty pixels.
    """
    rgb_acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    a_acc = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    covered_w = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    notes: List[str] = []

    for piece in pieces:
        px = piece["pixels"]
        weight = piece["weight"]
        x, y = int(piece["x"]), int(piece["y"])
        lh, lw = px.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(canvas_w, x + lw), min(canvas_h, y + lh)
        if x1 <= x0 or y1 <= y0:
            notes.append("layer piece outside canvas")
            continue
        sx0, sy0 = x0 - x, y0 - y
        src = px[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)].astype(np.float32) / 255.0
        wgt = weight[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
        covered = src[..., 3] > ALPHA_EPS
        contrib = src[..., 3] * wgt
        rgb_acc[y0:y1, x0:x1] += src[..., :3] * contrib[..., None]
        a_acc[y0:y1, x0:x1] += contrib
        covered_w[y0:y1, x0:x1] += np.where(covered, wgt, 0.0)

    # Divide by the coverage, but never by less than one full owner: a lone
    # tile keeps full opacity, while a pixel that only the losing tile drew
    # fades out with that tile's ownership weight instead of being renormalised
    # back to 255 -- which is what used to keep the invented pixels alive.
    alpha = np.divide(a_acc, np.maximum(covered_w, 1.0),
                      out=np.zeros_like(a_acc), where=covered_w > 1e-9)
    rgb = np.divide(rgb_acc, np.maximum(a_acc, 1e-6)[..., None],
                    out=np.zeros_like(rgb_acc), where=(a_acc > 1e-6)[..., None])
    rgba = np.dstack([
        np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8),
        np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8),
    ])
    return rgba, notes


def _accumulate_depth(
    canvas_w: int,
    canvas_h: int,
    pieces: Sequence[Dict[str, Any]],
) -> np.ndarray:
    """Blend the per-tile depth maps into one full-canvas map (0 = no data).

    Pieces from a mixed merge carry a ``priority``: where a higher-priority
    tile covers a pixel, lower-priority tiles are ignored outright -- the same
    rule the colour parts use, so "head from wave 2" also means "depth of the
    head from wave 2".  Tiles sharing the top priority still blend across
    their overlap.
    """
    acc = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    w_acc = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    best = np.full((canvas_h, canvas_w), -1e9, dtype=np.float32)
    for piece in pieces:
        d = piece["depth"].astype(np.float32) / 255.0
        valid = piece["depth"] > 0
        x, y = int(piece["x"]), int(piece["y"])
        lh, lw = d.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(canvas_w, x + lw), min(canvas_h, y + lh)
        if x1 <= x0 or y1 <= y0:
            continue
        sx0, sy0 = x0 - x, y0 - y
        sub = d[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
        wgt = piece["weight"][sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
        wgt = wgt * valid[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
        prio = float(piece.get("priority", 0))
        best_slice = best[y0:y1, x0:x1]
        acc_slice = acc[y0:y1, x0:x1]
        w_slice = w_acc[y0:y1, x0:x1]
        higher = (prio > best_slice) & (wgt > 0)
        if higher.any():
            acc_slice[higher] = 0.0
            w_slice[higher] = 0.0
            best_slice[higher] = prio
        take = (prio >= best_slice) & (wgt > 0)
        acc_slice[take] += sub[take] * wgt[take]
        w_slice[take] += wgt[take]
    return np.divide(acc, w_acc, out=np.zeros_like(acc), where=w_acc > 1e-6)


def _read_rgba_png(path: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return np.array(im.convert("RGBA"), dtype=np.uint8)
    except Exception:  # noqa: BLE001
        return None


def _read_depth_png(path: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return np.array(im.convert("L"), dtype=np.uint8)
    except Exception:  # noqa: BLE001
        return None


def _resize_rgba(px: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale one layer without dragging the transparent black into the edge.

    Resizing straight alpha RGBA pulls the fully transparent border pixels
    into every edge, which shows up as a dark rim around the part.  Multiplying
    by alpha first and dividing it out afterwards keeps the colour intact.
    """
    from PIL import Image
    src = px.astype(np.float32) / 255.0
    alpha = src[..., 3:4]
    prem = np.concatenate([src[..., :3] * alpha, alpha], axis=2)
    prem8 = np.clip(np.rint(prem * 255.0), 0, 255).astype(np.uint8)
    resized = np.array(
        Image.fromarray(np.ascontiguousarray(prem8), "RGBA").resize(
            (max(1, int(width)), max(1, int(height))), Image.Resampling.LANCZOS),
        dtype=np.float32) / 255.0
    out_a = resized[..., 3:4]
    rgb = np.divide(resized[..., :3], np.maximum(out_a, 1e-6))
    out = np.concatenate([np.clip(rgb, 0.0, 1.0), out_a], axis=2)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def _resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image
    return np.array(
        Image.fromarray(np.ascontiguousarray(depth, dtype=np.uint8), "L").resize(
            (max(1, int(width)), max(1, int(height))), Image.Resampling.BILINEAR),
        dtype=np.uint8)


# ---------------------------------------------------------------------------
# Reusing the tiles a previous run already left in the same folder
# ---------------------------------------------------------------------------

# ``tile_00_r0_c0.psd`` / ``tile_7.psd`` -> (prefix, index).  The optional
# ``_rR_cC`` suffix is accepted but not required.
TILE_FILE_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<index>\d+)(?:_r\d+_c\d+)?\.psd$", re.IGNORECASE)


def read_manifest(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(str(run_dir or ""), "manifest.json")
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except Exception:  # noqa: BLE001
        return None
    return doc if isinstance(doc, dict) and isinstance(doc.get("tiles"), list) else None


# ---------------------------------------------------------------------------
# Generation fingerprint: everything that can change the pixels of a tile
# ---------------------------------------------------------------------------

GENERATION_FILE = "generation.json"

# Settings that only affect the checked tiles: geometry and every other
# setting stay identical, so finished tiles remain valid and a change to any
# of these regenerates the selection without clearing the folder.  Keeping
# them out of the identity is what makes a seed / context change a targeted
# re-run instead of a full reset.
SEED_KEY = "random_seed"
LOCAL_RERUN_KEYS = (SEED_KEY, "context_pad")


def generation_identity(generation: Dict[str, Any]) -> Dict[str, Any]:
    """The part of ``generation`` that decides tile compatibility."""
    return {key: value for key, value in generation.items()
            if key not in LOCAL_RERUN_KEYS}


def generation_id(generation: Dict[str, Any]) -> str:
    """Stable hash of the generation parameters (dict order does not matter)."""
    try:
        payload = json.dumps(generation_identity(generation), sort_keys=True,
                             ensure_ascii=False, separators=(",", ":"),
                             default=str)
    except (TypeError, ValueError):
        payload = str(generation_identity(generation))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def generation_matches(old: Optional[Dict[str, Any]],
                       new: Dict[str, Any],
                       keys: Optional[Sequence[str]] = None) -> bool:
    """True when every compared key matches (nested values compared as data).

    ``keys`` defaults to every key of ``new`` except the seed; extra keys in
    ``old`` (model identities, for a caller that does not load models) are
    ignored.
    """
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    compared = (list(keys) if keys is not None
                else [key for key in new if key not in LOCAL_RERUN_KEYS])
    for key in compared:
        if key not in old or old[key] != new[key]:
            return False
    return True


def read_generation(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(str(run_dir or ""), GENERATION_FILE)
    if not run_dir or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except Exception:  # noqa: BLE001
        return None
    return doc if isinstance(doc, dict) else None


def write_generation(run_dir: str, generation: Dict[str, Any],
                     gid: str = "") -> str:
    """Persist the full generation block next to the tiles (for inspection)."""
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "version": 1,
        "generation": generation,
        "generation_id": str(gid or generation_id(generation)),
    }
    path = os.path.join(run_dir, GENERATION_FILE)
    json_io.write_json_atomic(path, payload)
    return path


class ClearFailed(RuntimeError):
    """The output folder could not be reset completely.

    Continuing after a partial clear is unsafe: the surviving PSDs belong to
    the previous generation but would look like ordinary files in the folder,
    so they could be reused as guide/merge input under the new fingerprint.
    """

    def __init__(self, run_dir: str, failures: Sequence[Tuple[str, str]]):
        self.run_dir = str(run_dir)
        self.failures = [(str(path), str(reason)) for path, reason in failures]
        names = [os.path.basename(path) for path, _reason in self.failures[:3]]
        extra = len(self.failures) - 3
        more = "" if extra <= 0 else "（另有 %d 个）" % extra
        more_en = "" if extra <= 0 else " (%d more)" % extra
        super().__init__(
            "输出目录里有文件无法删除（可能被 Photoshop / 资源管理器占用）：%s%s。"
            "请关闭占用程序后重新 Queue。目录：%s"
            " / some files in the output folder cannot be deleted (they may be"
            " locked by Photoshop or Explorer): %s%s; close the program using"
            " them and queue again. Folder: %s"
            % (", ".join(names) or "?", more, self.run_dir,
               ", ".join(names) or "?", more_en, self.run_dir))


def verify_run_cleared(run_dir: str) -> None:
    """Raise if a managed state file survived :func:`clear_run_outputs`."""
    run_dir = os.path.abspath(str(run_dir or ""))
    survivors = []
    for name in ("manifest.json", GENERATION_FILE):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            survivors.append((path, "still exists after clear"))
    images_dir = os.path.join(run_dir, "images")
    if os.path.isdir(images_dir):
        survivors.append((images_dir, "still exists after clear"))
    if survivors:
        raise ClearFailed(run_dir, survivors)


def clear_run_outputs(run_dir: str,
                      manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Delete the files this pipeline wrote into ``run_dir``.

    Called only when a result-affecting parameter changed.  Every PSD,
    ``*_preview.png`` / ``*_depth.png``, the wave documents and ``images/``
    are removed; unrelated files the user put in the folder are left alone.
    The ``_locks/`` snapshot store is deliberately kept: that is what lets a
    user-locked part survive the clear and come back on the next run.

    Any file that cannot be removed raises :class:`ClearFailed` instead of
    printing and continuing: a partial reset must never let old tiles look
    like current output.
    """
    run_dir = os.path.abspath(str(run_dir or ""))
    removed: List[str] = []
    failures: List[Tuple[str, str]] = []
    if not os.path.isdir(run_dir):
        return {"dir": run_dir, "removed": 0, "files": removed, "failed": 0}

    def drop(name: str) -> None:
        name = os.path.basename(str(name or ""))
        if not name:
            return
        path = os.path.join(run_dir, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed.append(name)
        except OSError as exc:  # noqa: BLE001
            failures.append((path, str(exc)))
            print("[STLU] cannot remove %s: %s" % (path, exc), flush=True)

    # Manifest-listed outputs first (also covers names in subpaths, if any).
    doc = manifest if isinstance(manifest, dict) else (read_manifest(run_dir) or {})
    for entry in doc.get("tiles") or []:
        if isinstance(entry, dict):
            for key in ("file", "preview", "depth"):
                drop(entry.get(key))
    merged = doc.get("merged") or {}
    if isinstance(merged, dict):
        for key in ("file", "preview", "depth", "depth_psd"):
            drop(merged.get(key))

    # Sweep the managed naming patterns, so leftovers from older manifests
    # or interrupted runs cannot survive a reset either.
    try:
        for name in os.listdir(run_dir):
            low = name.lower()
            if (low.endswith(".psd") or low.endswith("_preview.png")
                    or low.endswith("_depth.png")
                    or low in ("manifest.json", GENERATION_FILE, "original.png")):
                drop(name)
    except OSError as exc:
        failures.append((run_dir, "cannot list directory: %s" % exc))

    images_dir = os.path.join(run_dir, "images")
    if os.path.isdir(images_dir):
        try:
            shutil.rmtree(images_dir)
            removed.append("images/")
        except OSError as exc:  # noqa: BLE001
            failures.append((images_dir, str(exc)))
            print("[STLU] cannot remove %s: %s" % (images_dir, exc), flush=True)

    if failures:
        raise ClearFailed(run_dir, failures)
    return {"dir": run_dir, "removed": len(removed), "files": removed, "failed": 0}


def composite_psd(path: str) -> Optional[np.ndarray]:
    """Flatten a PSD into one straight-alpha RGBA canvas (or None).

    Hidden layers and everything inside a hidden folder are skipped, so the
    repair-only source copies never leak into a preview or a guide.
    """
    try:
        data = psd_reader.read_psd(path)
    except Exception:  # noqa: BLE001
        return None
    canvas = np.zeros((int(data["height"]), int(data["width"]), 4), np.uint8)
    suppressed = psd_writer.hidden_record_mask(data["layers"])
    for index, layer in enumerate(data["layers"]):
        kind = str(layer.get("kind") or "layer")
        if kind != "layer":
            continue
        if layer.get("hidden") or suppressed[index]:
            continue
        if (int(layer["right"]) <= int(layer["left"])
                or int(layer["bottom"]) <= int(layer["top"])):
            continue
        _alpha_over(canvas, layer["pixels"], int(layer["left"]), int(layer["top"]))
    return canvas


def _int_pair(value: Any) -> List[int]:
    """Two-int pair from a loosely typed document field."""
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError, IndexError, KeyError):
        return [0, 0]


def _coords_from_manifest(manifest: Dict[str, Any]) -> List[List[int]]:
    # A custom / subject plan stores the real boxes; the uniform-grid rebuild
    # below would invent geometry the run never used.  Both layouts name the
    # boxes under ``tiles`` (custom) or ``regions`` (strips).  Only a complete
    # 0..n-1 index range is usable here: callers address the result by tile
    # index.
    plan = manifest.get("partition_plan")
    if isinstance(plan, dict):
        for key in ("tiles", "regions"):
            boxes: Dict[int, List[int]] = {}
            for position, item in enumerate(plan.get(key) or []):
                if not isinstance(item, dict):
                    continue
                box = item.get("box")
                if box is None or len(box) < 4:
                    continue
                try:
                    index = int(item.get("index", position))
                    boxes[index] = [int(v) for v in box[:4]]
                except (TypeError, ValueError):
                    continue
            if boxes and sorted(boxes) == list(range(len(boxes))):
                return [boxes[index] for index in range(len(boxes))]
    source = manifest.get("source") or []
    tile_size = manifest.get("tile_size") or []
    try:
        width, height = int(source[0]), int(source[1])
        tile_w, tile_h = int(tile_size[0]), int(tile_size[1])
    except (IndexError, TypeError, ValueError):
        return []
    if min(width, height, tile_w, tile_h) <= 0:
        return []
    offset = manifest.get("offset") or [0, 0]
    try:
        offset_x, offset_y = int(offset[0]), int(offset[1])
    except (TypeError, ValueError, IndexError):
        offset_x, offset_y = 0, 0
    return [[int(v) for v in box] for box in grid_coords(
        width, height, tile_w, tile_h,
        int(manifest.get("overlap") or 0), offset_x, offset_y)]


def scan_tile_entries(
    run_dir: str,
    grid_doc: Optional[Dict[str, Any]] = None,
    manifest: Optional[Dict[str, Any]] = None,
    allow_unlisted: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Every tile PSD currently sitting in ``run_dir``, as manifest entries.

    Manifest entries are kept verbatim (they carry the model's depth order).
    ``allow_unlisted`` controls whether a PSD the manifest does not mention is
    rebuilt from its layers and adopted:

    * ``False`` -- the manifest is the authoritative list.  Needed after a
      generation change: a locked/leftover PSD from the old run must not look
      like current output.
    * ``True``  -- adopt unlisted PSDs, for a genuinely interrupted run whose
      ``generation.json`` already matched.
    * ``None`` (default) -- adopt only when there is no manifest with tiles.
    """
    directory = str(run_dir or "")
    if not directory or not os.path.isdir(directory):
        return []
    manifest = manifest if isinstance(manifest, dict) else (read_manifest(directory) or {})
    if allow_unlisted is None:
        allow_unlisted = not (manifest.get("tiles") or [])

    excluded = set()
    merged = manifest.get("merged") or {}
    for key in ("file", "depth_psd"):
        if merged.get(key):
            excluded.add(os.path.basename(str(merged[key])))

    entries: Dict[int, Dict[str, Any]] = {}
    for tile in manifest.get("tiles") or []:
        try:
            index = int(tile.get("index"))
        except (TypeError, ValueError):
            continue
        name = os.path.basename(str(tile.get("file") or ""))
        if not name or name in excluded or not os.path.isfile(os.path.join(directory, name)):
            continue
        entry = dict(tile)
        entry["file"] = name
        entries[index] = entry

    coords = [[int(v) for v in c] for c in ((grid_doc or {}).get("coords") or [])]
    if not coords:
        coords = _coords_from_manifest(manifest)

    try:
        # When the manifest is authoritative, unlisted PSDs are leftovers from
        # another generation (or files copied in by hand); adopting them would
        # silently mix old pixels into the guide/merge.
        names = sorted(os.listdir(directory)) if allow_unlisted else []
    except OSError:
        names = []
    for name in names:
        lower = name.lower()
        if not lower.endswith(".psd") or lower.endswith("_depth.psd"):
            continue
        if name in excluded:
            continue
        match = TILE_FILE_RE.match(name)
        if not match:
            continue
        index = int(match.group("index"))
        if index in entries:
            continue
        path = os.path.join(directory, name)
        try:
            data = psd_reader.read_psd(path)
        except Exception:  # noqa: BLE001
            continue
        real_layers = [layer for layer in data["layers"]
                       if str(layer.get("kind") or "layer") == "layer"
                       and not layer.get("hidden")]
        span = max(1, len(real_layers) - 1)
        layers = []
        for layer_index, layer in enumerate(real_layers):
            left, top = int(layer["left"]), int(layer["top"])
            right, bottom = int(layer["right"]), int(layer["bottom"])
            if right <= left or bottom <= top:
                continue
            layers.append({
                "name": str(layer["name"]),
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "sources": [],
                # Older files carry no depth_median; the PSD's own bottom ->
                # top order is the same far -> near order the merge expects.
                "depth_median": 1.0 - layer_index / float(span),
            })
        stem = name[:-4]
        box = coords[index] if index < len(coords) else [0, 0, int(data["width"]), int(data["height"])]
        preview = stem + "_preview.png"
        depth = stem + "_depth.png"
        entries[index] = {
            "index": int(index),
            "file": name,
            "preview": preview if os.path.isfile(os.path.join(directory, preview)) else "",
            "depth": depth if os.path.isfile(os.path.join(directory, depth)) else "",
            "canvas": [int(data["width"]), int(data["height"])],
            "tile": [int(v) for v in box],
            "layers": layers,
        }

    ordered = [entries[index] for index in sorted(entries)]
    for slot, entry in enumerate(ordered):
        entry["slot"] = int(slot)
    return ordered


def cache_compatible(manifest: Optional[Dict[str, Any]],
                     grid_doc: Optional[Dict[str, Any]]) -> bool:
    """Can this folder's old tiles join a run with ``grid_doc``?"""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tiles"), list):
        return False
    if not grid_doc:
        return True
    source = grid_doc.get("source") or {}
    want = [int(source.get("width") or 0), int(source.get("height") or 0)]
    have = [int(v) for v in (manifest.get("source") or [0, 0])[:2]]
    if want[0] > 0 and want != have:
        return False
    want_tiles = [int(v) for v in (grid_doc.get("tile_size") or [])]
    have_tiles = [int(v) for v in (manifest.get("tile_size") or [])]
    if want_tiles and have_tiles and want_tiles != have_tiles:
        return False
    if int(grid_doc.get("overlap") or 0) != int(manifest.get("overlap") or 0):
        return False
    # Same size/overlap but a different phase means the old tiles sit in boxes
    # this run no longer uses; their pixels would land in the wrong place.
    try:
        old_offset = [int(v) for v in (manifest.get("offset") or [0, 0])[:2]]
        new_offset = [int(v) for v in (grid_doc.get("offset") or [0, 0])[:2]]
    except (TypeError, ValueError):
        old_offset, new_offset = [0, 0], [0, 0]
    if old_offset != new_offset:
        return False
    old_print = str(manifest.get("source_fingerprint") or "")
    new_print = str(source.get("fingerprint") or "")
    if old_print and new_print and old_print != new_print:
        return False
    # Generation parameters (resolution / steps / seed / models / ...) are
    # stricter than geometry: a match here means the cached pixels were made
    # by the same settings.  A missing id on either side only happens for
    # documents written before the fingerprint existed.  On a mismatch
    # PipelineInput clears the folder first, so nothing stale can join.
    old_gen = str(manifest.get("generation_id") or "")
    new_gen = str((grid_doc or {}).get("generation_id") or "")
    if old_gen and new_gen and old_gen != new_gen:
        return False
    return True


def cached_tile_entries(
    run_dir: str,
    grid_doc: Optional[Dict[str, Any]] = None,
    skip: Sequence[int] = (),
    manifest: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Finished tiles from an earlier run in the same folder.

    ``skip`` holds the indices this run is re-generating: their old copies
    must not be merged (or used as guide), only their fresh output.
    """
    if not run_dir or not os.path.isdir(run_dir):
        return []
    manifest = manifest if manifest is not None else read_manifest(run_dir)
    if cache_compatible(manifest, grid_doc):
        # The manifest describes the current generation: it is the complete
        # list of tiles that may join, so never go hunting for unlisted PSDs.
        allow_unlisted = False
    else:
        # An interrupted run leaves its PSDs behind but never writes
        # manifest.json.  The folder's generation.json still records the
        # settings that made them, so those tiles can join as long as the
        # current run uses the same ones.
        interrupted = not isinstance(manifest, dict) or not manifest.get("tiles")
        stored = (read_generation(run_dir) or {}).get("generation") or {}
        wanted = (grid_doc or {}).get("generation") or {}
        if (not interrupted or not stored or not wanted
                or not generation_matches(stored, wanted)):
            return []
        manifest = {}
        allow_unlisted = True
    skip_set = {int(i) for i in skip}
    return [entry for entry in scan_tile_entries(
                run_dir, grid_doc, manifest, allow_unlisted=allow_unlisted)
            if int(entry.get("index", -1)) not in skip_set]


# ---------------------------------------------------------------------------
# Locked parts: keep the finished copy of a part on a targeted re-run
# ---------------------------------------------------------------------------

def find_tile_psd(run_dir: str, prefix: str, index: int) -> Optional[str]:
    """The tile PSD an earlier run left for ``index`` (newest file wins)."""
    directory = str(run_dir or "")
    if not directory or not os.path.isdir(directory):
        return None
    exact = []
    loose = []
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    for name in names:
        if not name.lower().endswith(".psd") or name.lower().endswith("_depth.psd"):
            continue
        match = TILE_FILE_RE.match(name)
        if not match:
            continue
        try:
            if int(match.group("index")) != int(index):
                continue
        except (TypeError, ValueError):
            continue
        path = os.path.join(directory, name)
        loose.append(path)
        if str(match.group("prefix") or "").lower() == str(prefix or "").lower():
            exact.append(path)
    candidates = exact or loose
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p) if os.path.isfile(p) else 0.0)


def _load_part_depth(images_dir: str, name: str) -> Optional[np.ndarray]:
    """The per-part depth PNG an earlier run saved for ``name`` (if any)."""
    if not images_dir:
        return None
    path = os.path.join(str(images_dir), _safe_stem(name) + "_depth.png")
    if not os.path.isfile(path):
        return None
    try:
        from PIL import Image
        with Image.open(path) as image:
            return np.array(image.convert("L"), dtype=np.uint8)
    except Exception:  # noqa: BLE001 - depth is a bonus, never a failure
        return None


def replace_locked_parts(
    parts: Dict[str, Any],
    old_psd_path: str,
    locks: Sequence[str],
    old_entry: Optional[Dict[str, Any]] = None,
    images_dir: str = "",
) -> List[str]:
    """Swap freshly generated parts for the copies saved in an earlier PSD.

    ``parts`` is mutated in place (callers pass a shallow copy): every locked
    tag is removed from ``tag2pinfo`` and the earlier layer with the same name
    takes its place.  Because both the tile PSD and the per-part PNGs are
    written from ``tag2pinfo``, the preserved pixels stay consistent.
    """
    wanted = {str(n).strip().lower() for n in (locks or []) if str(n).strip()}
    if not wanted or not os.path.isfile(old_psd_path):
        return []
    try:
        data = psd_reader.read_psd(old_psd_path)
    except Exception as exc:  # noqa: BLE001 - a broken old PSD is not fatal
        print("[STLU] locked parts: cannot read %s: %s" % (old_psd_path, exc), flush=True)
        return []

    old_layers = []
    for layer in data.get("layers") or []:
        if str(layer.get("kind") or "layer") != "layer" or layer.get("hidden"):
            continue
        if (int(layer.get("right", 0)) <= int(layer.get("left", 0))
                or int(layer.get("bottom", 0)) <= int(layer.get("top", 0))):
            continue
        old_layers.append(layer)

    depths: Dict[str, float] = {}
    for entry in (old_entry or {}).get("layers") or []:
        try:
            depths[str(entry.get("name") or "").lower()] = float(entry.get("depth_median", 1.0))
        except (TypeError, ValueError):
            continue

    tag2pinfo = dict(parts.get("tag2pinfo") or {})
    parts["tag2pinfo"] = tag2pinfo
    kept: List[str] = []
    order_total = max(1, len(old_layers) - 1)
    seen_locked = set()
    for order, layer in enumerate(old_layers):
        name = str(layer.get("name") or "")
        key = name.lower()
        if key not in wanted:
            continue
        seen_locked.add(key)
        for tag in [t for t in tag2pinfo if str(t).lower() == key]:
            tag2pinfo.pop(tag, None)
        px = np.ascontiguousarray(layer.get("pixels"))
        if px.ndim != 3 or px.shape[-1] != 4:
            continue
        height, width = int(px.shape[0]), int(px.shape[1])
        left, top = int(layer.get("left", 0)), int(layer.get("top", 0))
        depth = depths.get(key)
        if depth is None:
            # PSD records run bottom -> top, which is far -> near.
            depth = 1.0 - (order / float(order_total))
        pinfo = {
            "img": px,
            "xyxy": [left, top, left + width, top + height],
            "depth_median": float(depth),
        }
        saved_depth = _load_part_depth(images_dir, name)
        if saved_depth is not None and saved_depth.shape[:2] == px.shape[:2]:
            pinfo["depth"] = saved_depth
        tag2pinfo[name] = pinfo
        kept.append(name)

    # A locked part the earlier tile did not have still means "keep the old
    # state": drop the freshly generated copy instead of reviving it.
    for key in sorted(wanted - seen_locked):
        for tag in [t for t in tag2pinfo if str(t).lower() == key]:
            tag2pinfo.pop(tag, None)
        kept.append(key)
    return kept


# ---------------------------------------------------------------------------
# Lock store: per-tile part snapshots that survive a run-folder clear
# ---------------------------------------------------------------------------
#
# ``clear_run_outputs`` removes every PSD in the run folder, so a lock that
# only pointed at "the layer inside the old PSD" disappeared together with it.
# The store below keeps a copy of each locked layer in ``<run>/_locks/`` -- a
# directory the clear deliberately leaves alone.  A later run (even after a
# parameter change wiped the folder) restores the pixels from this copy, and a
# tile that is not re-generated at all can be rebuilt from its locked parts.

LOCK_DIR = "_locks"
LOCK_INDEX = "index.json"


def lock_dir(run_dir: str) -> str:
    return os.path.join(str(run_dir or ""), LOCK_DIR)


def lock_index_path(run_dir: str) -> str:
    return os.path.join(lock_dir(run_dir), LOCK_INDEX)


def read_lock_index(run_dir: str) -> Dict[str, Any]:
    """The lock store document (empty when nothing was ever locked)."""
    empty = {"version": 1, "source": [], "tiles": {}}
    path = lock_index_path(run_dir)
    if not run_dir or not os.path.isfile(path):
        return empty
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - a broken store is not fatal
        print("[STLU] locks: cannot read %s: %s" % (path, exc), flush=True)
        return empty
    if not isinstance(doc, dict):
        return empty
    if not isinstance(doc.get("tiles"), dict):
        doc["tiles"] = {}
    doc.setdefault("version", 1)
    doc.setdefault("source", [])
    return doc


def write_lock_index(run_dir: str, doc: Dict[str, Any]) -> str:
    os.makedirs(lock_dir(run_dir), exist_ok=True)
    doc["version"] = 1
    if not isinstance(doc.get("tiles"), dict):
        doc["tiles"] = {}
    path = lock_index_path(run_dir)
    json_io.write_json_atomic(path, doc)
    return path


def tile_locks(run_dir: str) -> Dict[int, List[str]]:
    """``{tile index: [locked layer names]}`` for the panel and the nodes."""
    doc = read_lock_index(run_dir)
    out: Dict[int, List[str]] = {}
    for raw, tile in (doc.get("tiles") or {}).items():
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        names = []
        for key, record in ((tile or {}).get("parts") or {}).items():
            names.append(str((record or {}).get("name") or key))
        if names:
            out[index] = names
    return out


def lock_summary(run_dir: str) -> Dict[str, List[str]]:
    """JSON-friendly ``{"3": ["face", "hair"]}`` for the preview panel."""
    return {str(index): names for index, names in tile_locks(run_dir).items()}


def _lock_geometry_matches(doc: Dict[str, Any], tile: Dict[str, Any],
                           coords: Optional[Sequence[Sequence[int]]],
                           tile_index: int,
                           source: Optional[Sequence[int]],
                           frame_size: Optional[Sequence[int]]) -> bool:
    """True when the snapshot still sits on the same crop of the same picture."""
    stored_source = doc.get("source") or []
    if stored_source and source:
        try:
            if ([int(stored_source[0]), int(stored_source[1])]
                    != [int(source[0]), int(source[1])]):
                return False
        except (TypeError, ValueError, IndexError):
            pass
    stored_box = tile.get("tile")
    if stored_box and coords:
        try:
            current = coords[int(tile_index)]
            if [int(v) for v in stored_box[:4]] != [int(v) for v in current[:4]]:
                return False
        except (TypeError, ValueError, IndexError):
            pass
    canvas = tile.get("canvas")
    if canvas and frame_size:
        try:
            if [int(canvas[0]), int(canvas[1])] != [int(frame_size[1]), int(frame_size[0])]:
                return False
        except (TypeError, ValueError, IndexError):
            pass
    return True


def _lock_pixels(run_dir: str, record: Dict[str, Any]
                 ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """``(rgba, depth)`` from the store; either side may be None."""
    rel = str(record.get("file") or "")
    path = os.path.join(lock_dir(run_dir), rel)
    if not rel or not os.path.isfile(path):
        return None, None
    try:
        from PIL import Image
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGBA"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        print("[STLU] locks: cannot read %s: %s" % (path, exc), flush=True)
        return None, None
    depth = None
    depth_rel = str(record.get("depth") or "")
    if depth_rel:
        depth_path = os.path.join(lock_dir(run_dir), depth_rel)
        if os.path.isfile(depth_path):
            try:
                from PIL import Image
                with Image.open(depth_path) as image:
                    candidate = np.array(image.convert("L"), dtype=np.uint8)
                if candidate.shape[:2] == pixels.shape[:2]:
                    depth = candidate
            except Exception:  # noqa: BLE001 - depth is a bonus
                depth = None
    return np.ascontiguousarray(pixels), depth


def _write_lock_record(run_dir: str, tile_index: int, name: str,
                       pixels: np.ndarray, left: int, top: int,
                       depth: Optional[np.ndarray],
                       depth_median: float) -> Dict[str, Any]:
    """Write one RGBA (and optional depth) file and return its record."""
    px = np.asarray(pixels)
    if px.ndim != 3 or px.shape[-1] != 4 or px.size == 0:
        raise ValueError("locked layer %r has no RGBA pixels" % name)
    index = int(tile_index)
    dest = os.path.join(lock_dir(run_dir), "tile_%02d" % index)
    os.makedirs(dest, exist_ok=True)
    stem = _safe_stem(str(name)) or "part"
    filename = stem + ".png"
    if not _save_png(os.path.join(dest, filename), px, "RGBA"):
        raise OSError("cannot write locked layer %r" % name)
    record: Dict[str, Any] = {
        "name": str(name),
        "file": "tile_%02d/%s" % (index, filename),
        "left": int(left),
        "top": int(top),
        "size": [int(px.shape[1]), int(px.shape[0])],
        "depth_median": float(depth_median),
    }
    depth_arr = None if depth is None else np.asarray(depth)
    if (depth_arr is not None and depth_arr.ndim == 2
            and depth_arr.shape[:2] == px.shape[:2]):
        depth_name = stem + "_depth.png"
        if _save_png(os.path.join(dest, depth_name), depth_arr, "L"):
            record["depth"] = "tile_%02d/%s" % (index, depth_name)
    return record


def store_locked_layer(run_dir: str, tile_index: int, name: str,
                       pixels: np.ndarray, left: int, top: int,
                       tile_box: Optional[Sequence[int]] = None,
                       canvas: Optional[Sequence[int]] = None,
                       source: Optional[Sequence[int]] = None,
                       depth: Optional[np.ndarray] = None,
                       depth_median: float = 1.0) -> Dict[str, Any]:
    """Snapshot one part from the preview panel (lock button)."""
    index = int(tile_index)
    record = _write_lock_record(
        run_dir, index, name, pixels, left, top, depth, depth_median)
    doc = read_lock_index(run_dir)
    tile = doc["tiles"].setdefault(str(index), {})
    try:
        if tile_box is not None:
            tile["tile"] = [int(v) for v in tile_box[:4]]
        if canvas is not None:
            tile["canvas"] = [int(canvas[0]), int(canvas[1])]
        if source is not None:
            doc["source"] = [int(source[0]), int(source[1])]
    except (TypeError, ValueError, IndexError):
        pass
    tile.setdefault("parts", {})[str(name).strip().lower()] = record
    write_lock_index(run_dir, doc)
    return record


def store_locked_parts(run_dir: str, tile_index: int, parts: Dict[str, Any],
                       names: Sequence[str], tile_box: Optional[Sequence[int]] = None,
                       canvas: Optional[Sequence[int]] = None,
                       source: Optional[Sequence[int]] = None) -> List[str]:
    """Snapshot the locked layers of a freshly written tile.

    Keeping the store current means the next generation change cannot lose a
    part the user locked before the folder is cleared.
    """
    wanted = {str(n).strip().lower() for n in (names or []) if str(n).strip()}
    if not wanted or not isinstance(parts, dict):
        return []
    index = int(tile_index)
    doc = read_lock_index(run_dir)
    tile = doc["tiles"].setdefault(str(index), {})
    try:
        if tile_box is not None:
            tile["tile"] = [int(v) for v in tile_box[:4]]
        if canvas is not None:
            tile["canvas"] = [int(canvas[0]), int(canvas[1])]
        if source is not None:
            doc["source"] = [int(source[0]), int(source[1])]
    except (TypeError, ValueError, IndexError):
        pass
    parts_doc = tile.setdefault("parts", {})
    stored: List[str] = []
    for tag, pinfo in (parts.get("tag2pinfo") or {}).items():
        key = str(tag).strip().lower()
        if key not in wanted or not isinstance(pinfo, dict):
            continue
        img = pinfo.get("img")
        if img is None:
            continue
        xyxy = pinfo.get("xyxy") or [0, 0, 0, 0]
        try:
            left, top = int(xyxy[0]), int(xyxy[1])
            depth_median = float(pinfo.get("depth_median", 1.0))
        except (TypeError, ValueError, IndexError):
            continue
        try:
            record = _write_lock_record(
                run_dir, index, str(tag), np.asarray(img), left, top,
                pinfo.get("depth"), depth_median)
        except (ValueError, OSError) as exc:
            print("[STLU] locks: cannot snapshot %s (tile %d): %s"
                  % (tag, index, exc), flush=True)
            continue
        parts_doc[key] = record
        stored.append(str(tag))
    if stored:
        write_lock_index(run_dir, doc)
    return stored


def remove_locked_layer(run_dir: str, tile_index: int, name: str) -> bool:
    """Drop one lock (and its files).  False when it was not locked."""
    doc = read_lock_index(run_dir)
    try:
        index = int(tile_index)
    except (TypeError, ValueError):
        return False
    tile = (doc.get("tiles") or {}).get(str(index))
    if not isinstance(tile, dict):
        return False
    parts = tile.get("parts") or {}
    record = parts.pop(str(name).strip().lower(), None)
    if record is None:
        return False
    for rel in (record.get("file"), record.get("depth")):
        if rel:
            try:
                os.remove(os.path.join(lock_dir(run_dir), str(rel)))
            except OSError:
                pass
    if not parts:
        doc["tiles"].pop(str(index), None)
    write_lock_index(run_dir, doc)
    return True


def snapshot_parts(run_dir: str, tile_index: int,
                   names: Optional[Sequence[str]] = None,
                   coords: Optional[Sequence[Sequence[int]]] = None,
                   source: Optional[Sequence[int]] = None
                   ) -> Optional[Dict[str, Any]]:
    """A parts document rebuilt from the lock store, or None.

    ``names=None`` restores every locked layer of the tile: that is how a tile
    the new run did not select at all is rebuilt from its locks.
    """
    doc = read_lock_index(run_dir)
    try:
        index = int(tile_index)
    except (TypeError, ValueError):
        return None
    tile = (doc.get("tiles") or {}).get(str(index))
    if not isinstance(tile, dict):
        return None
    canvas = tile.get("canvas") or []
    try:
        frame_w, frame_h = int(canvas[0]), int(canvas[1])
    except (TypeError, ValueError, IndexError):
        return None
    if frame_w <= 0 or frame_h <= 0:
        return None
    if not _lock_geometry_matches(
            doc, tile, coords, index, source, (frame_h, frame_w)):
        return None
    wanted = (None if names is None
              else {str(n).strip().lower() for n in names if str(n).strip()})
    tag2pinfo: Dict[str, Any] = {}
    for key, record in (tile.get("parts") or {}).items():
        if not isinstance(record, dict):
            continue
        if wanted is not None and str(key).lower() not in wanted:
            continue
        px, depth = _lock_pixels(run_dir, record)
        if px is None:
            continue
        try:
            left = int(record.get("left") or 0)
            top = int(record.get("top") or 0)
            depth_median = float(record.get("depth_median", 1.0))
        except (TypeError, ValueError):
            left = top = 0
            depth_median = 1.0
        pinfo = {
            "img": px,
            "xyxy": [left, top, left + int(px.shape[1]), top + int(px.shape[0])],
            "depth_median": depth_median,
        }
        if depth is not None:
            pinfo["depth"] = depth
        tag2pinfo[str(record.get("name") or key)] = pinfo
    if not tag2pinfo:
        return None
    return {"tag2pinfo": tag2pinfo, "frame_size": (frame_h, frame_w),
            "status": "ok", "_locked_tile": index}


def restore_locked_parts(parts: Dict[str, Any], run_dir: str, tile_index: int,
                         names: Sequence[str],
                         coords: Optional[Sequence[Sequence[int]]] = None,
                         source: Optional[Sequence[int]] = None
                         ) -> Tuple[List[str], set]:
    """Replace fresh parts with the snapshots kept in ``<run>/_locks``.

    Returns ``(kept names, loaded keys)``.  ``kept`` also lists wanted locks
    the store has no copy of -- their fresh copy is dropped, so "keep the old
    state" still means absent.  ``loaded keys`` lets the caller fall back to
    the old PSD for exactly the names the store could not provide.
    """
    wanted = {str(n).strip().lower() for n in (names or []) if str(n).strip()}
    if not wanted or not isinstance(parts, dict):
        return [], set()
    try:
        index = int(tile_index)
    except (TypeError, ValueError):
        return [], set()
    doc = read_lock_index(run_dir)
    tile = (doc.get("tiles") or {}).get(str(index))
    if not isinstance(tile, dict):
        return [], set()
    if not _lock_geometry_matches(
            doc, tile, coords, index, source, parts.get("frame_size")):
        return [], set()
    tag2pinfo = parts.setdefault("tag2pinfo", {})
    parts_doc = tile.get("parts") or {}
    kept: List[str] = []
    loaded = set()
    for key in sorted(wanted):
        record = None
        for raw, candidate in parts_doc.items():
            if str(raw).lower() == key:
                record = candidate
                break
        for tag in [t for t in tag2pinfo if str(t).lower() == key]:
            tag2pinfo.pop(tag, None)
        px = depth = None
        if isinstance(record, dict):
            px, depth = _lock_pixels(run_dir, record)
        if px is None:
            kept.append(key)
            continue
        try:
            left = int(record.get("left") or 0)
            top = int(record.get("top") or 0)
            depth_median = float(record.get("depth_median", 1.0))
        except (TypeError, ValueError):
            left = top = 0
            depth_median = 1.0
        pinfo = {
            "img": px,
            "xyxy": [left, top, left + int(px.shape[1]), top + int(px.shape[0])],
            "depth_median": depth_median,
        }
        if depth is not None:
            pinfo["depth"] = depth
        name = str(record.get("name") or key)
        tag2pinfo[name] = pinfo
        loaded.add(key)
        kept.append(name)
    return kept, loaded


def _crop_piece(rgba: np.ndarray, threshold: int) -> Optional[Tuple[np.ndarray, int, int]]:
    """Trim one piece to its real pixels (and zero the dust) or drop it."""
    mask = content_mask(rgba[..., 3], int(threshold))
    if not mask.any():
        return None
    rgba = rgba.copy()
    rgba[..., 3][~mask] = 0
    ys, xs = np.where(mask)
    l, t = int(xs.min()), int(ys.min())
    r, b = int(xs.max()) + 1, int(ys.max()) + 1
    return np.ascontiguousarray(rgba[t:b, l:r]), l, t


def _unique_layer_name(base: str, used: set) -> str:
    name = str(base or "layer")
    if name not in used:
        used.add(name)
        return name
    index = 2
    while "%s_%d" % (name, index) in used:
        index += 1
    name = "%s_%d" % (name, index)
    used.add(name)
    return name


def _shared_sides(
    box: Tuple[int, int, int, int],
    others: Sequence[Tuple[int, int, int, int]],
) -> Tuple[bool, bool, bool, bool]:
    """Which of a tile's four borders it actually shares with a neighbour.

    A canvas edge is not a seam: counting it made a tile near the top of a
    row "close to its border" on both tiles at once, so ownership tied and
    both copies survived there.  Only a border another tile of the same wave
    overlaps can decide who owns the pixels beside it.
    """
    bx0, by0, bx1, by1 = box
    left = right = top = bottom = False
    for ox0, oy0, ox1, oy1 in others:
        if min(by1, oy1) > max(by0, oy0):
            if ox0 < bx1 < ox1:
                right = True
            if ox0 < bx0 < ox1:
                left = True
        if min(bx1, ox1) > max(bx0, ox0):
            if oy0 < by1 < oy1:
                bottom = True
            if oy0 < by0 < oy1:
                top = True
    return left, right, top, bottom


def _tile_border_score(
    box: Tuple[int, int, int, int],
    sides: Tuple[bool, bool, bool, bool],
) -> np.ndarray:
    """Distance from every pixel of one tile box to its nearest shared border."""
    bx0, by0, bx1, by1 = box
    left, right, top, bottom = sides
    xs = np.arange(bx0, bx1, dtype=np.float32)
    ys = np.arange(by0, by1, dtype=np.float32)
    score = np.full((len(ys), len(xs)), 1e9, dtype=np.float32)
    if left:
        np.minimum(score, (xs - bx0)[None, :], out=score)
    if right:
        np.minimum(score, (bx1 - 1 - xs)[None, :], out=score)
    if top:
        np.minimum(score, (ys - by0)[:, None], out=score)
    if bottom:
        np.minimum(score, (by1 - 1 - ys)[:, None], out=score)
    # A tile with no neighbour gets no seam: everything in it is its own.
    score[score >= 1e9] = 0.0
    return score


def _content_bbox(piece: Dict[str, Any], threshold: int) -> Optional[Tuple[int, int, int, int]]:
    """Content-only box of a piece in source coordinates (alpha, not layer rect)."""
    px = piece.get("pixels")
    if px is None:
        return None
    alpha = px[..., 3] > max(0, int(threshold))
    rows = np.where(alpha.any(axis=1))[0]
    cols = np.where(alpha.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return None
    x, y = int(piece["x"]), int(piece["y"])
    return (x + int(cols[0]), y + int(rows[0]),
            x + int(cols[-1]) + 1, y + int(rows[-1]) + 1)


def _box_area(box: Optional[Tuple[int, int, int, int]]) -> int:
    if not box:
        return 0
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def _box_intersection(a: Tuple[int, int, int, int],
                      b: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    x0, y0 = max(int(a[0]), int(b[0])), max(int(a[1]), int(b[1]))
    x1, y1 = min(int(a[2]), int(b[2])), min(int(a[3]), int(b[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _touches_shared_border(tile_box: Sequence[int],
                           sides: Tuple[bool, bool, bool, bool],
                           content: Tuple[int, int, int, int],
                           tol: int = 2) -> bool:
    """Does a part's content reach a border this tile shares with a neighbour?"""
    bx0, by0, bx1, by1 = (int(v) for v in tile_box)
    l, t, r, b = content
    if sides[0] and l <= bx0 + tol:
        return True
    if sides[1] and r >= bx1 - tol:
        return True
    if sides[2] and t <= by0 + tol:
        return True
    if sides[3] and b >= by1 - tol:
        return True
    return False


def _border_margin(tile_box: Sequence[int],
                   sides: Tuple[bool, bool, bool, bool],
                   content: Tuple[int, int, int, int]) -> float:
    """How far the content stays from the shared borders (larger = more context)."""
    bx0, by0, bx1, by1 = (int(v) for v in tile_box)
    l, t, r, b = content
    margins: List[float] = []
    if sides[0]:
        margins.append(float(l - bx0))
    if sides[1]:
        margins.append(float(bx1 - r))
    if sides[2]:
        margins.append(float(t - by0))
    if sides[3]:
        margins.append(float(by1 - b))
    return min(margins) if margins else 1e9


def _prefer_complete_parts(
    pieces_by_name: Dict[str, List[Dict[str, Any]]],
    tile_groups: Dict[Any, Sequence[Tuple[int, Tuple[int, int, int, int]]]],
    threshold: int,
) -> Tuple[set, set]:
    """Use the tile that saw a part whole instead of cutting it at a seam.

    SeeThrough re-composes every crop, so a face cut by a seam is completed
    differently on the two sides; the geometric ownership cut then joins two
    halves that were never meant to match.  When one tile's copy of a part
    stays clear of every shared border, that tile saw the part whole, and its
    copy is used as-is instead of being weighted down at the bisector.

    A fragment guard refuses to prefer a complete copy that does not cover
    most of everything the part was seen as (a strand of hair inside one tile
    must not replace the hair another tile rendered in full).  A copy cut by
    a crop edge can only invent content *inside* that crop, so when one tile
    really saw the part whole its box swallows the other copies' boxes.

    Returns ``(piece_ids_to_keep_whole, names_that_were_narrowed)``.
    """
    sides: Dict[Tuple[int, str, int], Tuple[Any, Tuple[bool, bool, bool, bool]]] = {}
    for key, tiles in tile_groups.items():
        boxes = [(sid, box) for sid, box in tiles]
        for sid, box in boxes:
            others = [ob for osid, ob in boxes if osid != sid]
            sides[(int(key[0]), str(key[1]), int(sid))] = (box, _shared_sides(box, others))

    whole: set = set()
    narrowed: set = set()
    for name, pieces in pieces_by_name.items():
        if len(pieces) < 2:
            continue
        items = []
        for piece in pieces:
            info = sides.get((int(piece["priority"]), str(piece["wave"]), int(piece["sid"])))
            content = _content_bbox(piece, threshold)
            if info is None or content is None:
                continue
            tile_box, shared = info
            items.append((piece, content, tile_box,
                          _touches_shared_border(tile_box, shared, content),
                          _border_margin(tile_box, shared, content)))
        if len(items) < 2:
            continue
        complete = [item for item in items if not item[3]]
        if not complete:
            continue
        best = max(complete, key=lambda item: (_box_area(item[1]), item[4]))
        # The complete copy must contain what the other tiles saw.  A piece
        # that clears its borders but only covers one corner is a fragment, and
        # replacing a larger part with it would delete real content.  A copy
        # cut by a crop edge can only invent content inside that crop, so when
        # one tile really saw the part whole, every other box sits inside it.
        covered = True
        for item in items:
            if item[0] is best[0]:
                continue
            other_area = _box_area(item[1])
            inter = _box_intersection(best[1], item[1])
            if _box_area(inter) < other_area * 0.85:
                covered = False
                break
        if not covered:
            continue
        pieces_by_name[name] = [best[0]]
        whole.add(id(best[0]))
        narrowed.add(name)
    return whole, narrowed


def _build_ownership_fields(
    tile_groups: Dict[Any, Sequence[Tuple[int, Tuple[int, int, int, int]]]],
    canvas_w: int,
    canvas_h: int,
    feather: int,
    hard: bool = False,
) -> Dict[Any, Dict[int, Tuple[Tuple[int, int, int, int], np.ndarray]]]:
    """Decide, from tile geometry alone, which tile owns each pixel.

    Ownership cannot be decided from a part's pixels: SeeThrough re-composes
    every crop, so on the tile that saw only half of a headphone the model
    often returns a much larger amodal shape than the neighbour does.  Letting
    "the tile that draws something here" compete kept exactly those invented
    shapes, so the loser's copy must be cut even where its neighbour is empty.

    Every tile in a wave competes for every pixel it covers.  ``score`` is the
    distance to the tile's own crop border, so the tile with the most context
    around that pixel wins (the seam lands on the bisector of the two borders,
    ~150 px from each crop edge, where both renderings still agree).  The
    losing tile is cut: ``hard`` (depth map) leaves it at exactly 0 past the
    seam, otherwise it fades out over at most ``feather`` px so a flat
    composite does not show a hard colour edge.
    """
    feather = max(0, int(feather))
    if hard:
        feather = 0
    fields: Dict[Any, Dict[int, Tuple[Tuple[int, int, int, int], np.ndarray]]] = {}
    for key, tiles in tile_groups.items():
        boxes: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for sid, box in tiles:
            bx0, by0 = max(0, int(box[0])), max(0, int(box[1]))
            bx1, by1 = min(int(canvas_w), int(box[2])), min(int(canvas_h), int(box[3]))
            if bx1 > bx0 and by1 > by0:
                boxes.append((int(sid), (bx0, by0, bx1, by1)))
        if not boxes:
            continue
        scores = {}
        for sid, box in boxes:
            others = [ob for osid, ob in boxes if osid != sid]
            scores[sid] = _tile_border_score(box, _shared_sides(box, others))
        bucket: Dict[int, Tuple[Tuple[int, int, int, int], np.ndarray]] = {}
        for sid, (bx0, by0, bx1, by1) in boxes:
            score = scores[sid]
            # The best OTHER tile at each pixel: the signed margin decides who
            # owns the pixel, independent of whether that part is drawn there.
            other = np.full(score.shape, -1e9, dtype=np.float32)
            for osid, (ox0, oy0, ox1, oy1) in boxes:
                if osid == sid:
                    continue
                ix0, iy0 = max(bx0, ox0), max(by0, oy0)
                ix1, iy1 = min(bx1, ox1), min(by1, oy1)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                sub = other[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
                oscore = scores[osid][iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0]
                np.maximum(sub, oscore, out=sub)
            margin = score - other
            if feather > 0:
                # Owner == 1; the loser is at most half opacity right past
                # the seam and reaches 0 one feather band later.
                weight = np.where(
                    margin >= 0.0, 1.0,
                    np.clip((margin + float(feather)) / (2.0 * float(feather)),
                            0.0, 1.0))
            else:
                weight = (margin >= 0.0).astype(np.float32)
            bucket[sid] = ((bx0, by0, bx1, by1), weight.astype(np.float32))
        fields[key] = bucket
    return fields


def _apply_seam_weights(
    pieces: Sequence[Dict[str, Any]],
    fields: Dict[Any, Dict[int, Tuple[Tuple[int, int, int, int], np.ndarray]]],
) -> None:
    """Slice each tile's ownership map onto the pieces that came from it."""
    for piece in pieces:
        px = piece.get("pixels")
        if px is None:
            px = piece["depth"]
        h, w = int(px.shape[0]), int(px.shape[1])
        weight = np.ones((h, w), dtype=np.float32)
        bucket = fields.get((int(piece.get("priority", 0)), str(piece.get("wave", ""))))
        entry = bucket.get(int(piece.get("sid", -1))) if bucket else None
        if entry is not None:
            box, field = entry
            x, y = int(piece["x"]), int(piece["y"])
            x0, y0 = max(x, box[0]), max(y, box[1])
            x1, y1 = min(x + w, box[2]), min(y + h, box[3])
            if x1 > x0 and y1 > y0:
                weight[y0 - y:y1 - y, x0 - x:x1 - x] = field[
                    y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]]
        piece["weight"] = weight


# A SeeThrough crop that is mostly empty background often invents skin/hair in
# that empty area ("only the left half was here, so the rest must be a neck").
# Those pixels are not in the source image at all.  At merge time the source is
# available, so a component that sits almost entirely inside the detected
# background AND differs from the source is dropped.  Only whole connected
# components are removed, so a real part's anti-aliased edge is never shaved.
PHANTOM_COLOR_DIFF = 24.0
PHANTOM_BG_FRACTION = 0.85
PHANTOM_MIN_AREA = 12
# A phantom rarely floats fully inside the background: SeeThrough grows it
# around the subject outline, so 60-75% of it sits on the background and the
# rest overlaps the character.  Whole-component deletion misses those; the
# pixels that really are on the background can still be cut one by one.
PHANTOM_PARTIAL_MIN = 64
PHANTOM_PARTIAL_DIFF = 32.0
BACKGROUND_COLOR_TOL = 22
BACKGROUND_BORDER = 2

# SeeThrough splits these symmetrically, but only when both sides are visible
# in the same crop.  A tile that sees one ear keeps the unsuffixed name, and
# two such tiles then merge their single ears into one "ears" layer.  At merge
# time a stable face centre is available, so the side can be assigned anyway.
SYMMETRIC_TAGS = {"ears", "handwear", "eyewhite", "irides", "eyelash",
                  "eyebrow", "eyes"}
# Paired parts that are separated by the face centre: cutting a blob that
# spans it into two sides is safe.  A hand crossing the centre is not.
SPLIT_ACROSS_FACE = {"ears", "eyewhite", "irides", "eyelash", "eyebrow",
                     "eyes"}


def _load_source_rgb(run_dir: str, width: int, height: int) -> Optional[np.ndarray]:
    """The untouched source next to the tiles, when stage ③ wrote it."""
    path = os.path.join(run_dir, "original.png")
    if not os.path.isfile(path):
        return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    except Exception:  # noqa: BLE001
        return None
    if rgb.ndim != 3 or rgb.shape[0] != int(height) or rgb.shape[1] != int(width):
        return None
    return rgb


def _border_background_mask(rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Flat background connected to the frame, holes filled.

    Character art on a flat backdrop (white, magenta, a two-tone poster) has
    border colours that flood-fill inward.  An invented blob inside a white
    side shape is enclosed by that background, so filling the background's
    holes marks it too.  If the frame is not made of a handful of colours the
    function returns None and merge behaves exactly as before.
    """
    if rgb is None:
        return None
    import cv2

    height, width = rgb.shape[:2]
    if height < 8 or width < 8:
        return None
    border = max(1, min(int(BACKGROUND_BORDER), height // 2, width // 2))
    frame = np.concatenate([
        rgb[:border, :, :].reshape(-1, 3),
        rgb[-border:, :, :].reshape(-1, 3),
        rgb[border:-border, :border, :].reshape(-1, 3),
        rgb[border:-border, -border:, :].reshape(-1, 3),
    ], axis=0)
    if frame.size == 0:
        return None
    quant = (frame // 16).astype(np.int32)
    keys = quant[:, 0] * 256 + quant[:, 1] * 16 + quant[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    order = np.argsort(counts)[::-1]
    minimum = max(4, int(frame.shape[0] * 0.02))
    centres = []
    covered = 0
    for pos in order[:8]:
        if counts[pos] < minimum:
            continue
        key = int(values[pos])
        centres.append((((key >> 8) & 15) * 16 + 8,
                        ((key >> 4) & 15) * 16 + 8,
                        (key & 15) * 16 + 8))
        covered += int(counts[pos])
    # A photograph/gradient frame is not a flat backdrop; leave those images
    # alone instead of classifying their scenery as background.
    if not centres or covered < int(frame.shape[0] * 0.75):
        return None

    best = np.full((height, width), 255, dtype=np.int16)
    src = rgb.astype(np.int16)
    for centre in centres:
        diff = np.abs(src - np.array(centre, dtype=np.int16)).max(axis=2)
        np.minimum(best, diff, out=best)
    candidate = best <= int(BACKGROUND_COLOR_TOL)

    count, labels, _stats, _cent = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8)
    touching = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    touching.discard(0)
    if not touching:
        return None
    # A label lookup is one gather; np.isin would compare every pixel against
    # the label list and is markedly slower on a 4K frame.
    keep = np.zeros(int(count), dtype=bool)
    keep[list(touching)] = True
    background = keep[labels]

    # Fill enclosed holes, but only small ones: the character itself is also a
    # hole in the background, and filling that would mark the whole subject as
    # background.  An invented blob is a small pocket; a person is not.
    inverse = (~background).astype(np.uint8)
    count, labels, stats, _cent = cv2.connectedComponentsWithStats(
        inverse, connectivity=4)
    touching = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    touching.discard(0)
    max_hole = max(4096, int(height * width * 0.01))
    fill = np.zeros(int(count), dtype=bool)
    for label in range(1, int(count)):
        if label in touching:
            continue
        if int(stats[label, cv2.CC_STAT_AREA]) <= max_hole:
            fill[label] = True
    if fill.any():
        background |= fill[labels]

    fraction = float(background.mean())
    if fraction < 0.02 or fraction > 0.98:
        return None
    return background


def _trim_phantom_background(
    px: np.ndarray,
    gx: int,
    gy: int,
    source_rgb: Optional[np.ndarray],
    background: Optional[np.ndarray],
    alpha_threshold: int,
) -> Tuple[Optional[np.ndarray], int, int]:
    """Drop layer pixels that are invented inside the source background.

    Returns ``(pixels_or_None, removed_pixels, removed_components)``.  A
    component that lies at least 85% in the detected background and differs
    from the source is dropped whole.  A component that straddles the subject
    outline (the usual hallucinated sheet/cloud) keeps the pixels that sit on
    the subject, but its background pixels that disagree with the source are
    cut individually.  Real art matches the source, so it survives either way.
    """
    if source_rgb is None or background is None:
        return px, 0, 0
    import cv2

    height, width = px.shape[:2]
    x0, y0 = max(0, int(gx)), max(0, int(gy))
    x1 = min(source_rgb.shape[1], int(gx) + width)
    y1 = min(source_rgb.shape[0], int(gy) + height)
    if x1 <= x0 or y1 <= y0:
        return px, 0, 0
    sx0, sy0 = x0 - int(gx), y0 - int(gy)
    sub_alpha = px[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0), 3]
    if not np.any(sub_alpha > int(alpha_threshold)):
        return px, 0, 0
    bg = background[y0:y1, x0:x1]
    if not bg.any():
        return px, 0, 0
    source = source_rgb[y0:y1, x0:x1].astype(np.int16)
    layer = px[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0), :3].astype(np.int16)
    diff = np.abs(layer - source).mean(axis=2)

    count, labels, stats, _cent = cv2.connectedComponentsWithStats(
        (sub_alpha > int(alpha_threshold)).astype(np.uint8), connectivity=8)
    remove = np.zeros_like(sub_alpha, dtype=bool)
    removed_components = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < PHANTOM_MIN_AREA:
            continue
        comp = labels == label
        mean_diff = float(diff[comp].mean()) if np.any(comp) else 0.0
        if mean_diff < PHANTOM_COLOR_DIFF:
            continue
        inside = float(np.count_nonzero(comp & bg)) / float(area)
        if inside >= PHANTOM_BG_FRACTION:
            remove |= comp
            removed_components += 1
            continue
        # Straddles the outline: cut only the pixels that actually sit on the
        # background and disagree with it.  The part drawn over the subject is
        # kept, so a real layer crossing the silhouette is never ripped up.
        cut = comp & bg & (diff >= PHANTOM_PARTIAL_DIFF)
        if int(np.count_nonzero(cut)) < PHANTOM_PARTIAL_MIN:
            continue
        remove |= cut
        removed_components += 1
    removed = int(np.count_nonzero(remove))
    if removed == 0:
        return px, 0, 0
    out = px.copy()
    out[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0), 3][remove] = 0
    if int(np.count_nonzero(out[..., 3] > int(alpha_threshold))) == 0:
        return None, removed, removed_components
    return out, removed, removed_components


def _entry_center_x(entry: Dict[str, Any], layer: Dict[str, Any]) -> Optional[float]:
    box = entry.get("tile") or [0, 0, 0, 0]
    try:
        box_w = float(box[2]) - float(box[0])
        canvas = entry.get("canvas") or [box_w, box_w]
        canvas_w = float(canvas[0]) or box_w
        scale = box_w / canvas_w if canvas_w else 1.0
        return float(box[0]) + (float(layer.get("left", 0))
                                + float(layer.get("right", 0))) * 0.5 * scale
    except (TypeError, ValueError, IndexError):
        return None


def _face_center_x(manifest: Dict[str, Any]) -> Optional[float]:
    """Stable horizontal centre of the face, in source pixels.

    Nose and mouth are small, central and always tagged; their bboxes are far
    less polluted than ``face`` (a re-composed crop can stretch its face layer
    across the whole tile).  Both irides are the fallback.  When none exist,
    side assignment is skipped rather than guessed from the canvas centre.
    """
    entries = manifest.get("tiles") or []
    for tag in ("nose", "mouth"):
        centres = []
        for entry in entries:
            for layer in entry.get("layers") or []:
                if str(layer.get("name")) != tag:
                    continue
                centre = _entry_center_x(entry, layer)
                if centre is not None:
                    centres.append(centre)
        if centres:
            return float(np.median(centres))
    left, right = [], []
    for entry in entries:
        for layer in entry.get("layers") or []:
            name = str(layer.get("name"))
            if name not in ("irides-l", "irides-r"):
                continue
            centre = _entry_center_x(entry, layer)
            if centre is None:
                continue
            (left if name.endswith("-l") else right).append(centre)
    if left and right:
        return (float(np.median(left)) + float(np.median(right))) * 0.5
    return None


def _crop_alpha(px: np.ndarray, mask: np.ndarray, gx: int, gy: int):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    sub = px[y0:y1, x0:x1].copy()
    sub[..., 3] = np.where(mask[y0:y1, x0:x1], sub[..., 3], 0)
    return sub, int(gx) + x0, int(gy) + y0


def _split_symmetric_piece(
    px: np.ndarray,
    gx: int,
    gy: int,
    name: str,
    face_x: Optional[float],
    alpha_threshold: int,
):
    """Name a one-sided symmetric part by the face centre; split paired blobs.

    ``-r`` is the smaller-x (screen left) copy, ``-l`` the larger-x one, the
    same convention as SeeThrough's own split.  Returns a list of
    ``(name, pixels, gx, gy)``; unchanged layers come back as one entry.
    """
    if face_x is None or name not in SYMMETRIC_TAGS:
        return [(name, px, gx, gy)]
    import cv2

    mask = px[..., 3] > int(alpha_threshold)
    if not np.any(mask):
        return []
    count, labels, stats, _cent = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return [(name, px, gx, gy)]

    columns = np.arange(px.shape[1], dtype=np.float32)[None, :]
    sides = {0: np.zeros_like(mask), 1: np.zeros_like(mask)}
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) <= 0:
            continue
        comp = labels == label
        xs = np.where(comp.any(axis=0))[0]
        if len(xs) == 0:
            continue
        left_x = int(gx) + int(xs[0])
        right_x = int(gx) + int(xs[-1]) + 1
        if left_x >= face_x:
            sides[1] |= comp
        elif right_x <= face_x:
            sides[0] |= comp
        elif name not in SPLIT_ACROSS_FACE:
            # A single hand/blob straddling the centre is ambiguous; leave the
            # original name so no real content is cut in half.
            return [(name, px, gx, gy)]
        else:
            line = int(round(face_x - gx))
            line = max(0, min(px.shape[1], line))
            sides[0] |= comp & (columns < line)
            sides[1] |= comp & (columns >= line)

    out = []
    for side, side_mask in sides.items():
        if not np.any(side_mask):
            continue
        cropped = _crop_alpha(px, side_mask, gx, gy)
        if cropped is None:
            continue
        sub, sx, sy = cropped
        out.append((name + ("-l" if side else "-r"), sub, sx, sy))
    return out or [(name, px, gx, gy)]


def _wave_label(wave: Any) -> str:
    """Short tag for a tile's source run (mixed merges only)."""
    text = str(wave or "").replace("\\", "/").rstrip("/")
    label = os.path.basename(text) if text else ""
    return label[-8:] if label else ""


def _record_tile_copy(snapshot: Dict[str, List[Dict[str, Any]]],
                      raw_name: str, tile_index: int, rank: float,
                      priority: int, left: int, top: int, px: np.ndarray,
                      wave: Any = "") -> None:
    """Keep a tile's copy of a part exactly as the tile wrote it.

    Taken after dust cleanup/scaling/clamping but before every lossy merge
    step, so folder mode can hand back the full, uncropped copy.
    """
    snapshot[str(raw_name)].append({
        "name": str(raw_name),
        "tile": int(tile_index),
        "rank": float(rank),
        "priority": int(priority),
        "left": int(left),
        "top": int(top),
        "wave": str(wave or ""),
        "pixels": np.array(px, dtype=np.uint8, copy=True, order="C"),
    })


def _build_uncropped_groups(snapshot: Dict[str, List[Dict[str, Any]]],
                            alpha_threshold: int
                            ) -> Tuple[List[Dict[str, Any]], List[str], int]:
    """Same-named copies collected under one folder, with no seam cut.

    Folder mode is the "keep everything" output: every tile's copy of a part
    stays whole (only its empty border is trimmed), so a bad generation can be
    repaired in the PSD.  A part with a single copy stays a plain layer; only
    a part that two tiles both drew becomes a folder.

    Returns ``(layers, folder_names, layer_count)``.
    """
    grouped = []
    for name, copies in snapshot.items():
        members = [copy for copy in copies
                   if content_mask(copy["pixels"][..., 3],
                                   int(alpha_threshold)).any()]
        if members:
            members.sort(key=lambda m: (-float(m["rank"]), int(m["tile"])))
            grouped.append((str(name), members))
    if not grouped:
        return [], [], 0
    # Far -> near, matching the finished part order (PSD list is bottom->top).
    grouped.sort(key=lambda item: -float(np.mean([m["rank"] for m in item[1]])))

    layers: List[Dict[str, Any]] = []
    folder_names: List[str] = []
    layer_count = 0
    used: set = set()
    for name, members in grouped:
        # A mixed merge can leave copies of one part from several runs; the
        # run tag is then what tells the same tile number apart.
        waves = {_wave_label(member.get("wave")) for member in members}
        tag_waves = len(waves - {""}) > 1
        children: List[Dict[str, Any]] = []
        child_names: set = set()
        for member in members:
            cropped = _crop_piece(member["pixels"], int(alpha_threshold))
            if cropped is None:
                continue
            px, l, t = cropped
            base = name if len(members) == 1 else "%s_%d" % (name, int(member["tile"]))
            wave = _wave_label(member.get("wave")) if tag_waves else ""
            if wave:
                base += "_" + wave
            children.append({
                "name": _unique_layer_name(base, child_names),
                "pixels": px,
                "left": int(member["left"]) + int(l),
                "top": int(member["top"]) + int(t),
            })
        if not children:
            continue
        if len(children) > 1:
            folder = _unique_layer_name(name, used)
            layers.append({"type": "group", "name": folder, "children": children})
            folder_names.append(folder)
        else:
            child = children[0]
            child["name"] = _unique_layer_name(child["name"], used)
            layers.append(child)
        layer_count += len(children)
    return layers, folder_names, layer_count


def merge_tiles(
    run_dir: str,
    manifest: Dict[str, Any],
    indices: Optional[Sequence[int]] = None,
    out_name: str = "merged",
    merge_parts: bool = True,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    group_layers: bool = False,
    include_depth: bool = True,
    include_original: bool = True,
    tile_sources: Optional[Sequence[Dict[str, Any]]] = None,
    seam_feather: int = DEFAULT_SEAM_FEATHER,
) -> Dict[str, Any]:
    """Merge the selected tile PSDs into one full-size layered PSD.

    In an overlap each pixel is owned by exactly one tile -- the one farther
    from its own crop border -- decided from the tile boxes alone, so the
    losing tile is cut even where it invented extra part pixels.  Flat mode
    fades the loser out over at most ``seam_feather`` pixels so its single
    composite layer does not show a colour edge.

    Exception: a part one tile saw whole (its content never reaches a shared
    border) is taken from that tile as-is, with no ownership cut.  A face that
    happens to sit inside one crop must not be stitched from two independently
    completed halves.

    A tile PSD whose canvas is smaller than its grid box (a 1024 split rendered
    at 512) is scaled back up to the box before it is placed: the layers and
    the depth map keep the source coordinates instead of landing half size in
    the top-left corner.

    ``group_layers`` picks the output style.  Checked, every tile's copy of a
    part is written WHOLE -- no ownership cut, no phantom trim, no complete-
    part elimination -- and copies with the same exact name are collected in
    one PSD folder (``hairf`` and ``hairb`` stay separate because they sit at
    different depths).  This is the keep-everything version used to repair a
    bad generation.  Unchecked, the copies are cut at the ownership seam,
    blended into one layer per part (aliases such as hairf+hairb -> hair apply
    here) and written as the clean, directly merged artwork.

    ``include_depth`` writes the merged depth as its OWN ``<name>_depth.psd``
    next to the colour PSD, never as a layer inside it, and
    ``include_original`` puts the untouched source under the colour layers.

    ``tile_sources`` mixes tiles from several runs: each entry is
    ``{"dir", "tile", "priority", "wave"}``.  A part that shows up in more than
    one run is taken ONLY from the highest-priority run -- two redrawn copies of
    a face must never be blended into a ghost.  Within one run the overlap
    blend still applies, so neighbouring tiles stay seamless.
    """
    if not out_name.lower().endswith(".psd"):
        out_name += ".psd"

    tiles = manifest.get("tiles") or []
    wanted = set(int(i) for i in indices) if indices is not None else None
    if tile_sources is not None:
        sources = [dict(s) for s in tile_sources]
        selected = [s["tile"] for s in sources]
    else:
        selected = [t for t in tiles if wanted is None or int(t["index"]) in wanted]
        sources = [{"dir": run_dir, "tile": t, "priority": 0, "wave": run_dir}
                   for t in selected]
    if not selected:
        raise ValueError("no tiles selected")

    source = manifest.get("source") or [0, 0]
    canvas_w, canvas_h = int(source[0]), int(source[1])
    if canvas_w <= 0 or canvas_h <= 0:
        canvas_w = max(e["tile"][2] for e in selected)
        canvas_h = max(e["tile"][3] for e in selected)

    pieces_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rank_by_name: Dict[str, List[float]] = defaultdict(list)
    # Folder mode writes these untouched copies, bucketed by exact part name.
    folder_copies: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    depth_pieces: List[Dict[str, Any]] = []
    scale_notes: List[str] = []
    cleanup_notes: List[str] = []
    side_notes: List[str] = []
    source_cache: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
    phantom_px = 0
    phantom_pieces = 0
    # A face centre from the selected tiles' own nose/mouth copies.  The raw
    # manifest may include tiles this merge does not use, so build a view of
    # exactly what is being merged.
    face_x = _face_center_x({"tiles": [s["tile"] for s in sources]})

    # Newer manifests store the model's depth_median per layer, which orders
    # parts correctly even when a part is missing from some tiles.  Older runs
    # fall back to the layer's normalised position inside its own tile.
    depth_lookup: Dict[Tuple[int, str], float] = {}
    for source in sources:
        for entry in source["tile"].get("layers") or []:
            if "depth_median" in entry:
                depth_lookup[(int(source["tile"]["index"]), str(entry.get("name")))] = float(entry["depth_median"])

    tile_groups: Dict[Any, List[Tuple[int, Tuple[int, int, int, int]]]] = defaultdict(list)
    for sid, source in enumerate(sources):
        tile = source["tile"]
        tile_used = False
        path = os.path.join(source["dir"], tile["file"])
        data = psd_reader.read_psd(path)
        ox, oy = int(tile["tile"][0]), int(tile["tile"][1])
        # The split render does not have to come back at the grid's tile size:
        # a 1024 grid fed through a 512 pipeline writes a 512 PSD for a 1024
        # box.  Scale every layer (and the depth map) up to the box before
        # placing it, or the pieces land half size in the top-left corner.
        box_w = max(1, int(tile["tile"][2]) - ox)
        box_h = max(1, int(tile["tile"][3]) - oy)
        sx = box_w / float(max(1, data["width"]))
        sy = box_h / float(max(1, data["height"]))
        scaled = abs(sx - 1.0) > 1e-4 or abs(sy - 1.0) > 1e-4
        if scaled:
            scale_notes.append("tile %d: %dx%d render scaled into a %dx%d box"
                               % (int(tile["index"]), data["width"], data["height"],
                                  box_w, box_h))
        box_px_w = max(1, int(round(data["width"] * sx)))
        box_px_h = max(1, int(round(data["height"] * sy)))
        tile_box = (ox, oy, ox + box_w, oy + box_h)
        # Folder markers and hidden layers are not artwork; a tile PSD
        # normally has none, but a hand-edited one might.
        real_layers = [layer for layer in data["layers"]
                       if str(layer.get("kind") or "layer") == "layer"
                       and not layer.get("hidden")]
        layer_span = max(1, len(real_layers) - 1)
        tile_index = int(tile["index"])
        for layer_index, layer in enumerate(real_layers):
            raw_name = layer["name"]
            # Merge by the EXACT part name, never by alias: hairf (front hair)
            # and hairb (back hair) sit at different depths and must stay two
            # layers/folders, same for eyel/eyer and ears-l/ears-r.  What the
            # merge combines is the same part coming from two tiles.
            name = raw_name
            left, top = int(layer["left"]), int(layer["top"])
            px = layer["pixels"]
            lh, lw = px.shape[:2]
            if left < 0 or top < 0 or left + lw > data["width"] or top + lh > data["height"]:
                continue
            # Ordering uses every tile's copy of the layer, empty or not: a
            # part that only exists in one tile must still sort against the
            # parts around it.  1.0 == far/back (bottom), same as depth.
            rank = float(depth_lookup.get((tile_index, str(raw_name)),
                                          1.0 - layer_index / layer_span))
            # Runs written before the dust filter existed still carry speckles;
            # drop them here so an old run merges as cleanly as a new one.
            fresh = content_mask(px[..., 3], int(alpha_threshold))
            if not fresh.any():
                continue
            if not fresh.all():
                px = px.copy()
                px[..., 3][~fresh] = 0
            if scaled:
                nw = max(1, int(round(lw * sx)))
                nh = max(1, int(round(lh * sy)))
                px = _resize_rgba(px, nw, nh)
                left = int(round(left * sx))
                top = int(round(top * sy))
                lw, lh = nw, nh
            # Rounding can leave the box a pixel or two past the tile window.
            lw = min(lw, box_px_w - left)
            lh = min(lh, box_px_h - top)
            if lw <= 0 or lh <= 0:
                continue
            if lw != px.shape[1] or lh != px.shape[0]:
                px = np.ascontiguousarray(px[:lh, :lw])

            # SeeThrough re-composes a crop with whatever it believes is
            # missing, so a mostly empty background often gets invented skin,
            # hair or a whole "headwear" sheet.  The untouched source is right
            # here; drop the pixels that sit on its flat background and
            # disagree with it.  Folder mode runs this too: a per-tile copy is
            # only useful when it does not carry the invented backdrop.
            source_dir = str(source.get("dir") or run_dir)
            if source_dir not in source_cache:
                source_rgb = _load_source_rgb(source_dir, canvas_w, canvas_h)
                source_cache[source_dir] = (source_rgb,
                                            _border_background_mask(source_rgb))
            source_rgb, source_background = source_cache[source_dir]
            if source_background is not None:
                trimmed, removed, _components = _trim_phantom_background(
                    px, ox + left, oy + top, source_rgb, source_background,
                    int(alpha_threshold))
                if removed:
                    phantom_px += removed
                    phantom_pieces += 1
                    if trimmed is None:
                        cleanup_notes.append(
                            "removed %s as background phantom (tile %d)"
                            % (name, tile_index))
                        continue
                    px = trimmed
                    cleanup_notes.append(
                        "trimmed %d px from %s (tile %d)"
                        % (removed, name, tile_index))

            # Folder mode keeps one whole copy per tile: no ownership cut, no
            # symmetric split, no complete-part pick.  The phantom trim above
            # still applies, because a copy that carries the invented backdrop
            # is not a useful reference.
            if group_layers:
                _record_tile_copy(
                    folder_copies, raw_name, tile_index, rank,
                    int(source.get("priority", 0)),
                    ox + left, oy + top, px, wave=source.get("wave", ""))
                tile_used = True
                continue

            # A crop that only saw one ear keeps the unsuffixed tag, and two
            # such tiles would merge both ears into one layer.  The face centre
            # is stable across tiles, so name/split by it before grouping.
            subpieces = _split_symmetric_piece(
                px, ox + left, oy + top, name, face_x, int(alpha_threshold))
            if not subpieces:
                continue
            if len(subpieces) != 1 or subpieces[0][0] != name:
                side_notes.append("%s -> %s (tile %d)" % (
                    name, ", ".join(sorted({p[0] for p in subpieces})),
                    tile_index))
            for subname, subpx, subx, suby in subpieces:
                rank_by_name[subname].append(rank)
                pieces_by_name[subname].append({
                    "x": int(subx),
                    "y": int(suby),
                    "pixels": subpx,
                    "box": tile_box,
                    "priority": int(source.get("priority", 0)),
                    "wave": str(source.get("wave", "")),
                    "sid": sid,
                    "tile": tile_index,
                    "rank": rank,
                })
                tile_used = True

        depth_file = tile.get("depth") or ""
        if include_depth and depth_file:
            depth_px = _read_depth_png(os.path.join(source["dir"], depth_file))
            if depth_px is not None and depth_px.shape[:2] == (data["height"], data["width"]):
                if depth_px.shape[:2] != (box_px_h, box_px_w):
                    depth_px = _resize_depth(depth_px, box_px_w, box_px_h)
                depth_pieces.append({
                    "x": ox,
                    "y": oy,
                    "depth": depth_px,
                    "box": tile_box,
                    "priority": int(source.get("priority", 0)),
                    "wave": str(source.get("wave", "")),
                    "sid": sid,
                })
        # A tile only competes for ownership if it actually produced content;
        # a failed/empty render must not cut its neighbour's pixels.
        if tile_used:
            tile_groups[(int(source.get("priority", 0)),
                         str(source.get("wave", "")))].append((sid, tile_box))

    # A part that exists in several runs (an old full-body bake plus a fresh
    # head crop) must come from ONE of them: two redrawn copies of a face
    # would ghost.  Keep only the highest-priority run for BOTH output styles;
    # inside that run folder mode still keeps every tile's whole copy.
    if tile_sources is not None:
        for bucket in (pieces_by_name, folder_copies):
            for name, members in list(bucket.items()):
                top = max(int(m["priority"]) for m in members)
                chosen_wave = next(str(m["wave"]) for m in members
                                   if int(m["priority"]) == top)
                bucket[name] = [m for m in members
                                if int(m["priority"]) == top
                                and str(m["wave"]) == chosen_wave]

    if group_layers:
        if not folder_copies:
            raise ValueError("selected tiles contain no layers")
    else:
        if not pieces_by_name:
            raise ValueError("selected tiles contain no layers")

        # A part that one tile saw whole is taken from that tile as-is.  This
        # is what keeps a face split by a seam from being merged out of two
        # independently completed halves.
        whole_pieces, narrowed = _prefer_complete_parts(
            pieces_by_name, tile_groups, alpha_threshold)
        for name in narrowed:
            rank_by_name[name] = [float(p.get("rank", 0.0))
                                  for p in pieces_by_name[name]]

        # Ownership comes from the tile boxes, not from which part happens to
        # have pixels there: SeeThrough's re-composed crops disagree about the
        # extent of amodal parts (and about which part a pixel belongs to at
        # all), so the losing tile must be cut even where the winner draws
        # nothing.
        fields = _build_ownership_fields(tile_groups, canvas_w, canvas_h,
                                         seam_feather, hard=False)
        for pieces in pieces_by_name.values():
            _apply_seam_weights(pieces, fields)
        if whole_pieces:
            # The chosen copy owns its part outright: no seam fade, no cut.
            for pieces in pieces_by_name.values():
                for piece in pieces:
                    if id(piece) in whole_pieces:
                        h, w = piece["pixels"].shape[:2]
                        piece["weight"] = np.ones((h, w), dtype=np.float32)
    if include_depth and depth_pieces:
        depth_fields = _build_ownership_fields(tile_groups, canvas_w, canvas_h, 0,
                                               hard=True)
        _apply_seam_weights(depth_pieces, depth_fields)

    if phantom_pieces:
        print("[STLU] merge: removed %d invented background px from %d tile layers"
              % (phantom_px, phantom_pieces), flush=True)
    if side_notes:
        print("[STLU] merge: resolved symmetric sides: %s" % "; ".join(side_notes),
              flush=True)

    notes: List[str] = list(scale_notes) + cleanup_notes + side_notes
    colour_layers: List[Dict[str, Any]] = []
    group_names: List[str] = []

    if group_layers:
        # Keep-everything output: same-named copies become a folder, each copy
        # stays whole.  A part only one tile drew stays a plain layer.
        colour_layers, group_names, copy_count = _build_uncropped_groups(
            folder_copies, int(alpha_threshold))
        if not colour_layers:
            raise ValueError("merge produced no non-empty layers")
        notes.append("同名图层单独成组：%d 个部位 / %d 层（未裁剪）"
                     " / same-named layers grouped: %d parts, %d layers"
                     " (uncropped)"
                     % (len(colour_layers), copy_count,
                        len(colour_layers), copy_count))
    else:
        # Far -> near (bottom -> top); larger depth == farther away.
        order = sorted(pieces_by_name,
                       key=lambda n: float(np.mean(rank_by_name[n])), reverse=True)
        built: Dict[str, Dict[str, Any]] = {}
        for name in order:  # far -> near == PSD bottom -> top
            rgba, part_notes = _accumulate_part(canvas_w, canvas_h, pieces_by_name[name])
            notes.extend("%s: %s" % (name, n) for n in part_notes)
            cropped = _crop_piece(rgba, int(alpha_threshold))
            if cropped is None:
                notes.append("%s: empty after merge" % name)
                continue
            px, l, t = cropped
            built[name] = {
                "name": name,
                "pixels": px,
                "left": int(l),
                "top": int(t),
            }
        if not built:
            raise ValueError("merge produced no non-empty layers")
        colour_layers = [built[name] for name in order if name in built]

    # Depth never lives in the colour PSD.  It is its own file (plus a PNG
    # copy for thumbnails), so opening the colour PSD shows the artwork and
    # nothing else -- a depth layer tinting or hiding parts is impossible.
    depth_name = ""
    depth_psd_name = ""
    if include_depth and depth_pieces:
        dmap = _accumulate_depth(canvas_w, canvas_h, depth_pieces)
        valid = dmap > 0
        if valid.any():
            try:
                from PIL import Image
                depth_name = out_name[:-4] + "_depth.png"
                Image.fromarray(
                    np.clip(np.rint(dmap * 255.0), 0, 255).astype(np.uint8), "L"
                ).save(os.path.join(run_dir, depth_name), compress_level=6)
            except Exception:  # noqa: BLE001
                depth_name = ""
            ys, xs = np.where(valid)
            l, t, r, b = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            grey = np.clip(np.rint(dmap[t:b, l:r] * 255.0), 0, 255).astype(np.uint8)
            depth_rgba = np.dstack([
                grey, grey, grey,
                np.where(valid[t:b, l:r], 255, 0).astype(np.uint8),
            ])
            depth_psd_name = out_name[:-4] + "_depth.psd"
            try:
                psd_writer.write_psd(
                    os.path.join(run_dir, depth_psd_name),
                    [{
                        "name": "depth",
                        "pixels": np.ascontiguousarray(depth_rgba),
                        "left": int(l),
                        "top": int(t),
                    }],
                    canvas_w, canvas_h, "rle")
            except Exception as exc:  # noqa: BLE001
                notes.append("depth psd failed: %s" % exc)
                depth_psd_name = ""
            notes.append("depth: %d px blended (own PSD)" % int(valid.sum()))

    original_name = ""
    original_layers: List[Dict[str, Any]] = []
    if include_original:
        original_file = manifest.get("original") or ""
        if original_file:
            orig = _read_rgba_png(os.path.join(run_dir, original_file))
            if orig is not None:
                from PIL import Image
                original_name = original_file
                if orig.shape[0] != canvas_h or orig.shape[1] != canvas_w:
                    orig = np.array(Image.fromarray(orig, "RGBA").resize(
                        (canvas_w, canvas_h), Image.Resampling.LANCZOS), dtype=np.uint8)
                original_layers.append({
                    "name": "original",
                    "pixels": np.ascontiguousarray(orig),
                    "left": 0,
                    "top": 0,
                })

    # Bottom -> top: original (optional), then the colour parts/folders.  No
    # depth layer here: it has its own PSD file.
    psd_layers = original_layers + colour_layers

    psd_path = os.path.join(run_dir, out_name)
    psd_writer.write_psd(psd_path, psd_layers, canvas_w, canvas_h, "rle")

    preview_name = out_name[:-4] + "_preview.png"
    preview_path = os.path.join(run_dir, preview_name)
    try:
        from PIL import Image
        preview_layers: List[Dict[str, Any]] = []
        for layer in colour_layers:
            if layer.get("type") == "group":
                preview_layers.extend(layer["children"])
            else:
                preview_layers.append(layer)
        Image.fromarray(_flatten_preview(preview_layers, canvas_w, canvas_h)).save(
            preview_path, compress_level=3)
    except Exception:  # noqa: BLE001
        preview_name = ""

    layer_names: List[str] = []
    for layer in original_layers + colour_layers:
        if layer.get("type") == "group":
            layer_names.append(layer["name"] + "/")
            layer_names.extend(child["name"] for child in layer["children"])
        else:
            layer_names.append(layer["name"])

    return {
        "file": out_name,
        "preview": preview_name,
        "depth": depth_name,
        "depth_psd": depth_psd_name,
        "original": original_name,
        "canvas": [canvas_w, canvas_h],
        "layers": layer_names,
        "groups": group_names,
        "tiles": [int(t["index"]) for t in selected],
        "notes": notes[:40],
    }


def merge_run(
    run_dir: str,
    *,
    manifest: Optional[Dict[str, Any]] = None,
    indices: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
    merge_parts: Optional[bool] = None,
    alpha_threshold: Optional[int] = None,
    group_layers: Optional[bool] = None,
    include_depth: Optional[bool] = None,
    include_original: Optional[bool] = None,
    seam_feather: Optional[int] = None,
) -> Dict[str, Any]:
    """Merge one run folder and record the result in its manifest.

    Shared by the ④ node and ``/stlu/tiles/merge`` so both paths apply the
    same defaults (read from the manifest) and write the same ``merged``
    block.  Any argument left as ``None`` falls back to the manifest value,
    which is what the earlier run was actually written with.
    """
    run_dir = os.path.abspath(run_dir)
    if manifest is None:
        manifest = read_manifest(run_dir)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tiles"), list):
        raise ValueError("manifest has no tiles: %s"
                         % os.path.join(run_dir, "manifest.json"))

    info = merge_tiles(
        run_dir,
        manifest,
        indices=indices,
        out_name=str(name or "merged"),
        merge_parts=bool(manifest.get("merge_parts", True)
                         if merge_parts is None else merge_parts),
        alpha_threshold=int(
            manifest.get("alpha_threshold", DEFAULT_ALPHA_THRESHOLD)
            if alpha_threshold is None else alpha_threshold),
        group_layers=bool(manifest.get("group_layers", True)
                          if group_layers is None else group_layers),
        include_depth=bool(False if include_depth is None else include_depth),
        include_original=bool(False if include_original is None else include_original),
        seam_feather=max(0, int(
            manifest.get("seam_feather", DEFAULT_SEAM_FEATHER)
            if seam_feather is None else seam_feather)),
    )
    manifest["merged"] = {
        "file": info.get("file"),
        "preview": info.get("preview"),
        "depth": info.get("depth"),
        "depth_psd": info.get("depth_psd"),
        "canvas": info.get("canvas"),
        "layers": len(info.get("layers") or []),
        "groups": info.get("groups"),
        "tiles": info.get("tiles"),
    }
    try:
        json_io.write_json_atomic(
            os.path.join(run_dir, "manifest.json"), manifest)
    except OSError as exc:  # the PSD is written; keep the merge result
        print("[STLU] warning: cannot record merge result: %s" % exc, flush=True)
    return info


def merge_mixed(
    entries: Sequence[Dict[str, Any]],
    output_root: str,
    out_name: str = "merged",
    merge_parts: bool = True,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    group_layers: bool = False,
    include_depth: bool = True,
    include_original: bool = True,
    seam_feather: int = DEFAULT_SEAM_FEATHER,
) -> Dict[str, Any]:
    """Merge tiles taken from several runs (per-tile source override).

    ``entries`` is ``[{"dir": <run dir>, "index": <tile index>,
    "priority": <int>}]``.  When the same part exists in more than one run the
    highest priority wins, so "body from the first bake, head from the second"
    never produces two overlapping faces.  Tiles that stay on the same run are
    still blended across their overlap, so the seam stays invisible.

    The first entry supplies the canvas size and the original image.
    """
    if not entries:
        raise ValueError("no tile sources given")
    sources: List[Dict[str, Any]] = []
    base_dir = ""
    base_manifest: Dict[str, Any] = {}
    for entry in entries:
        raw = str(entry.get("dir") or "").strip()
        if not raw:
            continue
        path = raw if os.path.isabs(raw) else os.path.join(output_root, raw)
        path = os.path.abspath(path)
        manifest_path = os.path.join(path, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError("manifest.json not found in %s" % path)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        index = int(entry.get("index", -1))
        tile = next((t for t in manifest.get("tiles") or []
                     if int(t.get("index", -1)) == index), None)
        if tile is None:
            raise ValueError("tile %d is not in %s" % (index, path))
        if not base_dir:
            base_dir = path
            base_manifest = manifest
        sources.append({
            "dir": path,
            "tile": tile,
            "priority": int(entry.get("priority", 0)),
            "wave": str(entry.get("wave") or os.path.basename(path)),
        })
    if not sources:
        raise ValueError("no readable tile sources")
    info = merge_tiles(
        base_dir,
        base_manifest,
        out_name=out_name,
        merge_parts=bool(merge_parts),
        alpha_threshold=int(alpha_threshold),
        group_layers=bool(group_layers),
        include_depth=bool(include_depth),
        include_original=bool(include_original),
        tile_sources=sources,
        seam_feather=int(seam_feather),
    )
    info["mixed"] = True
    info["sources"] = [{
        "dir": os.path.relpath(s["dir"], output_root).replace(os.sep, "/"),
        "index": int(s["tile"]["index"]),
        "priority": int(s["priority"]),
    } for s in sources]
    return info


# ---------------------------------------------------------------------------
# Grid documents -- the split every later stage refers to
# ---------------------------------------------------------------------------

GRID_KIND = "stlu_tile_grid"
IMAGES_KIND = "stlu_tile_images"


def grid_document(
    source_size: Sequence[int],
    tile_size: Sequence[int],
    overlap: int,
    source_image: str = "",
    name: str = "grid",
    created: str = "",
    offset_x: int = 0,
    offset_y: int = 0,
) -> Dict[str, Any]:
    """The global split: source size, tile geometry and every tile's box.

    Later stages (selection, save, tile PSD, merge) take this document by path
    so the grid is computed once and never re-derived differently.  ``offset``
    is the phase of the split lines: it is stored next to the coordinates so
    an old run can be recognised as a different grid even when the size,
    overlap and seed all match.
    """
    width, height = max(1, int(source_size[0])), max(1, int(source_size[1]))
    tile_w, tile_h = max(1, int(tile_size[0])), max(1, int(tile_size[1]))
    offset_x, offset_y = int(offset_x), int(offset_y)
    coords = grid_coords(width, height, tile_w, tile_h, int(overlap),
                         offset_x, offset_y)
    tiles: List[Dict[str, Any]] = []
    for index, (x0, y0, x1, y1) in enumerate(coords):
        row, col = tile_rc(coords, index)
        tiles.append({
            "index": int(index),
            "row": int(row),
            "col": int(col),
            "box": [int(x0), int(y0), int(x1), int(y1)],
        })
    return {
        "version": 1,
        "kind": GRID_KIND,
        "name": str(name or "grid"),
        "created": str(created or ""),
        "source": {
            "image": str(source_image or ""),
            "width": int(width),
            "height": int(height),
        },
        "tile_size": [int(tile_w), int(tile_h)],
        "overlap": int(overlap),
        "offset": [offset_x, offset_y],
        "rows": len({t["row"] for t in tiles}),
        "columns": len({t["col"] for t in tiles}),
        "count": len(tiles),
        "coords": [[int(v) for v in c] for c in coords],
        "tiles": tiles,
        "selection": "all",
    }


def load_grid_doc(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict) or not isinstance(doc.get("coords"), list):
        raise ValueError("not a STLU grid document: %s" % path)
    return doc


def load_grid_ref(ref: str) -> Dict[str, Any]:
    """A grid document from inline JSON or from an older ``grids/*.json``.

    Stage one no longer writes the split to disk: it hands the document on in
    memory, and the wave is the only place a run leaves files.  Paths still
    load, so documents saved by older versions keep working.
    """
    text = str(ref or "").strip()
    if not text:
        raise ValueError("empty grid reference")
    if text.startswith("{"):
        doc = json.loads(text)
        if not isinstance(doc, dict) or not isinstance(doc.get("coords"), list):
            raise ValueError("not a STLU grid document")
        return doc
    return load_grid_doc(text)


def update_grid_selection(path: str, selection: str) -> bool:
    """Persist the checked tiles back into the grid document."""
    try:
        doc = load_grid_doc(path)
    except Exception:  # noqa: BLE001
        return False
    doc["selection"] = str(selection or "all")
    try:
        json_io.write_json_atomic(path, doc)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# Tile images -- stage two's product, so stage three can run without diffusing
# ---------------------------------------------------------------------------

def _save_png(path: str, arr: np.ndarray, mode: Optional[str] = None) -> bool:
    try:
        from PIL import Image
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        if mode == "RGBA":
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
            if arr.ndim != 3 or arr.shape[-1] != 4:
                return False
        image = Image.fromarray(arr, mode) if mode else Image.fromarray(arr)
        image.save(path, compress_level=3)
        return True
    except Exception:  # noqa: BLE001
        return False


def write_tile_images(
    parts_list: Sequence[Dict[str, Any]],
    output_root: str,
    out_dir: str = "SeeThrough/Tiles",
    filename_prefix: str = "tile",
    stamp: Optional[str] = None,
    grid_doc: Optional[Dict[str, Any]] = None,
    tile_index_map: Optional[Sequence[int]] = None,
    source_size: Optional[Sequence[int]] = None,
    original_rgba: Optional[np.ndarray] = None,
    run_dir_abs: Optional[str] = None,
    original_rel: Optional[str] = None,
    write_index: bool = True,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> Dict[str, Any]:
    """Write every part of every tile as a plain layered image folder.

    One sub folder per tile (``layers.json`` + ``<tag>.png`` + depth PNGs) plus
    an ``images.json`` index.  Stage three rebuilds the tile PSDs from this
    without re-running SeeThrough, exactly like pointing at an existing result.

    ``run_dir_abs`` lets the caller put this folder inside the wave folder
    (``<wave>/images``) so one run is one self-contained directory.

    ``original_rel`` points at a reference the caller already wrote (the wave
    root's ``original.png``); it is recorded in ``images.json`` instead of
    copying the same picture into ``images/`` a second time.

    ``write_index=False`` skips ``images.json``: the per-tile node writes one
    tile at a time, and the collector node owns the final index.
    """
    if not parts_list:
        raise ValueError("no tiles to save")

    grid = grid_doc or {}
    if run_dir_abs:
        run_dir = str(run_dir_abs)
    else:
        base = os.path.join(output_root, out_dir.strip("/\\"))
        run_dir = os.path.join(base, "%s_%s_images" % (filename_prefix, stamp or "run"))
    os.makedirs(run_dir, exist_ok=True)

    coords = grid.get("coords") or []
    grid_source = grid.get("source") or {}
    src = list(source_size) if source_size else [
        int(grid_source.get("width") or 0), int(grid_source.get("height") or 0)]
    tile_size = [int(v) for v in (grid.get("tile_size") or [0, 0])]

    original_name = ""
    if original_rel:
        original_name = original_rel
    elif original_rgba is not None and _save_png(os.path.join(run_dir, "original.png"), original_rgba, "RGBA"):
        original_name = "original.png"

    entries: List[Dict[str, Any]] = []
    for slot, parts in enumerate(parts_list):
        index = int(tile_index_map[slot]) if tile_index_map and slot < len(tile_index_map) else slot
        tag2pinfo = (parts or {}).get("tag2pinfo") or {}
        frame = (parts or {}).get("frame_size") or (0, 0)
        frame_h, frame_w = int(frame[0]), int(frame[1])
        if frame_h <= 0 or frame_w <= 0:
            continue
        row, col = tile_rc(coords, index) if coords else (0, 0)
        folder = ("tile_%02d_r%d_c%d" % (index, row, col)) if coords else ("tile_%02d" % index)
        tile_dir = os.path.join(run_dir, folder)
        os.makedirs(tile_dir, exist_ok=True)

        layers: List[Dict[str, Any]] = []
        ordered = sorted(
            ((tag, pinfo) for tag, pinfo in tag2pinfo.items()
             if isinstance(pinfo, dict) and pinfo.get("img") is not None),
            key=lambda item: _depth(item[1]),
            reverse=True,  # far -> near, the PSD order
        )
        for tag, pinfo in ordered:
            img = np.asarray(pinfo["img"], dtype=np.uint8)
            if img.ndim != 3 or img.shape[-1] != 4:
                continue
            # Same dust/wash scrub the PSD path gets, so the on-disk layer
            # PNGs never carry hidden colour under alpha 0 or a wash layer.
            scrubbed = _clean_part({"img": img}, int(alpha_threshold))
            if scrubbed is None:
                continue
            img = scrubbed[0]
            stem = _safe_stem(str(tag))
            filename = stem + ".png"
            if not _save_png(os.path.join(tile_dir, filename), img, "RGBA"):
                continue
            left, top = (int(v) for v in (pinfo.get("xyxy") or [0, 0])[:2])
            entry = {
                "name": str(tag),
                "filename": filename,
                "left": left,
                "top": top,
                "right": left + int(img.shape[1]),
                "bottom": top + int(img.shape[0]),
                "depth_median": float(_depth(pinfo)),
            }
            depth = pinfo.get("depth")
            if depth is not None:
                depth = np.asarray(depth)
                if depth.ndim == 2 and depth.shape[:2] == img.shape[:2]:
                    depth_name = stem + "_depth.png"
                    if _save_png(os.path.join(tile_dir, depth_name), depth, "L"):
                        entry["depth_filename"] = depth_name
            layers.append(entry)

        json_io.write_json_atomic(os.path.join(tile_dir, "layers.json"), {
            "kind": "stlu_tile_layers",
            "tile_index": int(index),
            "width": int(frame_w),
            "height": int(frame_h),
            "layers": layers,
        })

        entries.append({
            "index": int(index),
            "slot": int(slot),
            "dir": folder,
            "canvas": [int(frame_w), int(frame_h)],
            "tile": [int(c) for c in coords[index]] if index < len(coords) else [0, 0, frame_w, frame_h],
            "layers": len(layers),
        })

    if not entries:
        raise ValueError("every tile ended up empty")

    # Without a grid the tile canvas *is* the whole picture (single frame).
    if int(src[0]) <= 0 or int(src[1]) <= 0:
        src = [entries[0]["canvas"][0], entries[0]["canvas"][1]]
    if int(tile_size[0]) <= 0 or int(tile_size[1]) <= 0:
        tile_size = [entries[0]["canvas"][0], entries[0]["canvas"][1]]

    manifest = {
        "version": 1,
        "kind": IMAGES_KIND,
        "dir": os.path.relpath(run_dir, output_root).replace(os.sep, "/"),
        "prefix": filename_prefix,
        "source": [int(src[0]), int(src[1])],
        "tile_size": tile_size,
        "overlap": int(grid.get("overlap") or 0),
        "offset": _int_pair(grid.get("offset")),
        "original": original_name,
        "tiles": entries,
        "count": len(entries),
    }
    if write_index:
        json_io.write_json_atomic(os.path.join(run_dir, "images.json"), manifest)
    return manifest


def _safe_stem(name: str) -> str:
    try:
        from .core_plan import safe_name as _safe
    except ImportError:  # pragma: no cover - standalone import
        from core_plan import safe_name as _safe
    return _safe(name)


# ---------------------------------------------------------------------------
# Reading tile images back into parts documents
# ---------------------------------------------------------------------------

def _read_layers_doc(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict) or not isinstance(doc.get("layers"), list):
        raise ValueError("not a layers document: %s" % path)
    return doc


def _fit_layer_image(arr: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    """Make a saved layer PNG match the box recorded in layers.json."""
    target_w = max(1, int(right) - int(left))
    target_h = max(1, int(bottom) - int(top))
    height, width = arr.shape[:2]
    if width == target_w and height == target_h:
        return arr
    if width >= int(right) and height >= int(bottom):
        return arr[int(top):int(bottom), int(left):int(right)]
    if width >= target_w and height >= target_h:
        return arr[:target_h, :target_w]
    try:
        from PIL import Image
        mode = "L" if arr.ndim == 2 else None
        image = Image.fromarray(np.ascontiguousarray(arr), mode) if mode else Image.fromarray(
            np.ascontiguousarray(arr))
        return np.array(image.resize((target_w, target_h), Image.Resampling.LANCZOS))
    except Exception:  # noqa: BLE001
        return arr


def parts_from_layers_doc(doc: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    """Rebuild a SeeThrough-shaped parts document from saved layer PNGs."""
    tag2pinfo: Dict[str, Dict[str, Any]] = {}
    width = int(doc.get("width") or 0)
    height = int(doc.get("height") or 0)
    for entry in doc.get("layers") or []:
        filename = str(entry.get("filename") or "").strip()
        if not filename:
            continue
        path = filename if os.path.isabs(filename) else os.path.join(base_dir, filename)
        arr = _read_rgba_png(path)
        if arr is None:
            continue
        left = int(entry.get("left") or 0)
        top = int(entry.get("top") or 0)
        right = int(entry.get("right") or (left + arr.shape[1]))
        bottom = int(entry.get("bottom") or (top + arr.shape[0]))
        img = _fit_layer_image(arr, left, top, right, bottom)
        name = str(entry.get("name") or "layer")
        if name in tag2pinfo:
            suffix = 2
            while "%s_%d" % (name, suffix) in tag2pinfo:
                suffix += 1
            name = "%s_%d" % (name, suffix)
        pinfo: Dict[str, Any] = {
            "img": img,
            "xyxy": [left, top],
            "tag": name,
            "depth_median": float(entry.get("depth_median", 1.0)),
        }
        depth_file = str(entry.get("depth_filename") or "").strip()
        if depth_file:
            depth_path = depth_file if os.path.isabs(depth_file) else os.path.join(base_dir, depth_file)
            depth = _read_depth_png(depth_path)
            if depth is not None:
                depth = _fit_layer_image(depth, left, top, right, bottom)
                if depth.ndim == 2:
                    pinfo["depth"] = depth
        tag2pinfo[name] = pinfo
        width = max(width, left + int(img.shape[1]))
        height = max(height, top + int(img.shape[0]))
    if width <= 0 or height <= 0:
        raise ValueError("layers document has no usable layers")
    return {"tag2pinfo": tag2pinfo, "frame_size": (int(height), int(width))}


def _layers_json_files(directory: str) -> List[str]:
    found = []
    try:
        names = os.listdir(directory)
    except OSError:
        return found
    for name in names:
        if not name.lower().endswith("_layers.json"):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        found.append((mtime, name, path))
    found.sort()
    return [path for _m, _n, path in found]


def _image_files(directory: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    found = []
    try:
        names = os.listdir(directory)
    except OSError:
        return found
    for name in names:
        if name.lower().endswith(exts):
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                found.append((name.lower(), path))
    found.sort()
    return [path for _n, path in found]


def _single_image_parts(path: str) -> Dict[str, Any]:
    """A plain picture becomes a one-layer tile (depth/original unavailable)."""
    arr = _read_rgba_png(path)
    if arr is None:
        raise ValueError("cannot read image: %s" % path)
    name = os.path.splitext(os.path.basename(path))[0]
    return {
        "tag2pinfo": {
            name: {"img": arr, "xyxy": [0, 0], "tag": name, "depth_median": 1.0},
        },
        "frame_size": (int(arr.shape[0]), int(arr.shape[1])),
    }


def load_image_source(path: str) -> Dict[str, Any]:
    """Resolve a path/folder of generated tile images into parts documents.

    Accepted sources, in order of preference:

    * a folder written by :func:`write_tile_images` (has ``images.json``)
    * a single SeeThrough / STLU ``*_layers.json`` file
    * a folder of ``*_layers.json`` files (one tile each, ordered by mtime)
    * a folder of plain PNGs (one single-layer tile each)
    """
    path = os.path.abspath(path)
    if os.path.isfile(path):
        if path.lower().endswith(".json"):
            doc = _read_layers_doc(path)
            tile_index = doc.get("tile_index")
            parts = parts_from_layers_doc(doc, os.path.dirname(path))
            return {
                "kind": "layers",
                "dir": os.path.dirname(path),
                "parts": [parts],
                "indices": [int(tile_index) if tile_index is not None else 0],
                "source_size": [int(parts["frame_size"][1]), int(parts["frame_size"][0])],
                "tile_size": [int(parts["frame_size"][1]), int(parts["frame_size"][0])],
                "overlap": 0,
                "original": None,
            }
        parts = _single_image_parts(path)
        return {
            "kind": "image",
            "dir": os.path.dirname(path),
            "parts": [parts],
            "indices": [0],
            "source_size": [int(parts["frame_size"][1]), int(parts["frame_size"][0])],
            "tile_size": [int(parts["frame_size"][1]), int(parts["frame_size"][0])],
            "overlap": 0,
            "original": None,
        }

    if not os.path.isdir(path):
        raise FileNotFoundError("image source not found: %s" % path)

    index_path = os.path.join(path, "images.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
        parts_list = []
        indices = []
        for entry in doc.get("tiles") or []:
            tile_dir = os.path.join(path, str(entry.get("dir") or ""))
            layers_path = os.path.join(tile_dir, "layers.json")
            if not os.path.isfile(layers_path):
                continue
            parts_list.append(parts_from_layers_doc(_read_layers_doc(layers_path), tile_dir))
            indices.append(int(entry.get("index", len(indices))))
        if not parts_list:
            raise ValueError("image run has no readable tiles: %s" % path)
        source = doc.get("source") or [0, 0]
        original = doc.get("original") or ""
        return {
            "kind": "images",
            "dir": path,
            "parts": parts_list,
            "indices": indices,
            "source_size": [int(source[0]), int(source[1])],
            "tile_size": [int(v) for v in (doc.get("tile_size") or [0, 0])],
            "overlap": int(doc.get("overlap") or 0),
            "offset": _int_pair(doc.get("offset")),
            # ``../original.png`` (the wave root) is the normal layout; keep
            # the path readable so a missing reference is obvious.
            "original": os.path.normpath(os.path.join(path, original)) if original else None,
        }

    layers_files = _layers_json_files(path)
    if layers_files:
        parts_list = []
        for layers_path in layers_files:
            doc = _read_layers_doc(layers_path)
            parts_list.append(parts_from_layers_doc(doc, path))
        return _auto_grid_source("layers", path, parts_list)

    images = _image_files(path)
    if not images:
        raise ValueError("no layers.json or images found in %s" % path)
    parts_list = [_single_image_parts(p) for p in images]
    return _auto_grid_source("image", path, parts_list)


def _auto_grid_source(kind: str, directory: str, parts_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Lay per-tile documents out in a near-square grid.

    A folder of per-tile images carries no sheet geometry of its own, so the
    tiles are arranged ``ceil(sqrt(n))`` wide.  Passing a grid document into
    the node overrides this with the real layout.
    """
    import math
    count = max(1, len(parts_list))
    frame_w = max(int(p["frame_size"][1]) for p in parts_list)
    frame_h = max(int(p["frame_size"][0]) for p in parts_list)
    cols = max(1, int(math.ceil(math.sqrt(count))))
    rows = max(1, int(math.ceil(count / float(cols))))
    return {
        "kind": kind,
        "dir": directory,
        "parts": list(parts_list),
        "indices": list(range(len(parts_list))),
        "source_size": [frame_w * cols, frame_h * rows],
        "tile_size": [frame_w, frame_h],
        "overlap": 0,
        "original": None,
    }


def write_tiles_from_images(
    source: str,
    output_root: str,
    out_dir: str = "SeeThrough/Tiles",
    filename_prefix: str = "tile",
    merge_parts: bool = True,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    stamp: Optional[str] = None,
    merge_after: bool = True,
    merged_name: str = "merged",
    grid_doc: Optional[Dict[str, Any]] = None,
    tile_index_map: Optional[Sequence[int]] = None,
    write_depth: bool = True,
    merge_group_layers: bool = False,
    include_original: bool = True,
    run_dir_abs: Optional[str] = None,
    tiles: Optional[Sequence[int]] = None,
    save_images: bool = True,
) -> Dict[str, Any]:
    """Stage three offline: existing layer images -> per-tile layered PSDs.

    ``tiles`` is an optional subset of tile indices (the grid indices, not the
    list slots): the wave picker lets a re-run cover only the checked tiles.

    ``save_images`` writes the per-part PNGs of this run into ``<wave>/images``
    as well, so every wave is self-contained: a later merge or rebuild never
    has to re-run SeeThrough just to get the layer pictures back.
    """
    data = load_image_source(source)
    source_size = [int(v) for v in data["source_size"]]
    tile_size = [int(v) for v in data["tile_size"]]
    overlap = int(data["overlap"])
    indices = list(tile_index_map) if tile_index_map else list(data["indices"])
    parts_src = list(data["parts"])
    if tiles:
        wanted = {int(t) for t in tiles}
        keep = [i for i, index in enumerate(indices) if int(index) in wanted]
        if not keep:
            raise ValueError("none of the selected tiles exist in %s" % source)
        indices = [indices[i] for i in keep]
        parts_src = [parts_src[i] for i in keep]
        data = dict(data)
        data["parts"] = parts_src
    if grid_doc:
        src = (grid_doc.get("source") or {})
        if int(src.get("width") or 0) > 0 and int(src.get("height") or 0) > 0:
            source_size = [int(src["width"]), int(src["height"])]
        tile_size = [int(v) for v in (grid_doc.get("tile_size") or tile_size)]
        overlap = int(grid_doc.get("overlap") or overlap)
    if tile_size[0] <= 0 or tile_size[1] <= 0:
        tile_size = list(source_size)
    if source_size[0] <= 0 or source_size[1] <= 0:
        source_size = list(tile_size)
    if source_size[0] <= 0 or source_size[1] <= 0:
        raise ValueError("cannot determine the canvas size of %s" % source)
    # A rebuild of a phased run has to keep the same boxes, or every PSD
    # would land at offset 0.
    tile_coords = (grid_doc or {}).get("coords") or data.get("coords") or None
    grid_offset = (grid_doc or {}).get("offset") or data.get("offset") or (0, 0)

    original_rgba = None
    if include_original and data.get("original"):
        original_rgba = _read_rgba_png(data["original"])

    manifest = write_tiles(
        data["parts"],
        output_root,
        out_dir=out_dir,
        filename_prefix=filename_prefix,
        merge_parts=bool(merge_parts),
        alpha_threshold=int(alpha_threshold),
        source_size=[int(source_size[0]), int(source_size[1])],
        tile_size=[int(tile_size[0]), int(tile_size[1])],
        overlap=int(overlap),
        stamp=stamp,
        merge_after=bool(merge_after),
        merged_name=merged_name,
        tile_index_map=indices,
        write_depth=bool(write_depth),
        original_rgba=original_rgba,
        merge_group_layers=bool(merge_group_layers),
        include_original=bool(include_original),
        run_dir_abs=run_dir_abs,
        tile_coords=tile_coords,
        grid_offset=grid_offset,
    )
    manifest["source_kind"] = data["kind"]
    manifest["source_dir"] = data["dir"]
    manifest["from_images"] = True

    # Keep the layer pictures of *this* wave next to its PSDs.  Without this a
    # rebuild would produce PSDs only, and the next merge/rebuild would have to
    # run SeeThrough again just to recover the PNGs.
    if save_images:
        try:
            images_dir = os.path.join(output_root, manifest["dir"], "images")
            img_doc = write_tile_images(
                data["parts"],
                output_root,
                out_dir=out_dir,
                filename_prefix=filename_prefix,
                grid_doc=grid_doc,
                tile_index_map=indices,
                source_size=[int(source_size[0]), int(source_size[1])],
                # The wave root already keeps the reference (write_tiles ran
                # above); images.json points at it instead of copying it.
                original_rel="../original.png" if original_rgba is not None else None,
                run_dir_abs=images_dir,
            )
            manifest["images_dir"] = img_doc["dir"]
            manifest["images"] = {
                "dir": img_doc["dir"],
                "count": img_doc["count"],
                "tiles": img_doc["tiles"],
            }
        except Exception as exc:  # noqa: BLE001 - PSDs are already on disk
            manifest["images_error"] = str(exc)

    try:
        json_io.write_json_atomic(
            os.path.join(os.path.join(output_root, manifest["dir"]), "manifest.json"),
            manifest)
    except OSError:
        pass
    return manifest
