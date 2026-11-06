"""4K part masks: re-cut the split's alpha on the reference's real edges.

The 1024 split gives every part an alpha, but at a 4K canvas that alpha is a
4x4 block: four pixels of staircase per source pixel, and for parts the split
got wrong (a 17x15 iris, an eyebrow with no solid pixel at all) the *shape*
itself is wrong too.  The reference is the same artwork at 4K and carries the
real outline, so the alpha is upsampled **with the reference as a guide**
(guided filter) instead of being interpolated blindly: the matte follows the
reference's edges, not the source grid.

Parts whose shape cannot be recovered that way -- an iris whose top arc is
covered by the lash -- are completed geometrically (an ellipse fitted to the
visible arc and the sclera boundary).  That is deterministic and can be
reviewed, unlike a hallucinated guess.

Everything here is CPU (OpenCV/scipy); no model, no GPU.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    from . import core_geometry
except ImportError:  # standalone import from the tests
    import core_geometry

Rect = Tuple[int, int, int, int]

# Guided-filter window at the destination resolution.  Radius 8 covers the 4x
# staircase of a 1024 alpha; eps decides how strongly an edge in the guide
# stops the filter from smoothing across it.
DEFAULT_GUIDE_RADIUS = 8
DEFAULT_GUIDE_EPS = 2.0e-4
DEFAULT_MASK_THRESHOLD = 0.5
# Above this luma a pixel is "sclera/highlight"; below the dark one it is
# "lash/pupil".  Both are measured on the 0..255 reference.
SCLERA_LUMA = 190.0
DARK_LUMA = 90.0
# The iris is "red" when R clearly dominates both other channels.  The split's
# own iris colour is (117, 33, 45), so 18 is comfortable and still rejects the
# black lash and the white sclera.
RED_DOMINANCE = 18


def _as_rgb(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba)
    if arr.ndim == 2:
        return np.dstack([arr, arr, arr])
    return arr[..., :3]


def _luma(rgb: np.ndarray) -> np.ndarray:
    r = np.asarray(rgb, dtype=np.float32)
    return r[..., 0] * 0.299 + r[..., 1] * 0.587 + r[..., 2] * 0.114


def _alpha_of(rgba: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(rgba)
    if arr.ndim >= 3 and arr.shape[-1] >= 4:
        return arr[..., 3].astype(np.uint8)
    return None


def _reference_rect(box: Rect, reference_size: Tuple[int, int],
                    canvas: Tuple[int, int]) -> Tuple[Rect, float]:
    k = core_geometry.context_scale(reference_size, canvas)
    rect = core_geometry.scale_rect(box, k if k > 0 else 1.0, reference_size)
    return rect, (k if k > 0 else 1.0)


def guided_matte(source_alpha: np.ndarray, guide_rgb: np.ndarray,
                 radius: int = DEFAULT_GUIDE_RADIUS,
                 eps: float = DEFAULT_GUIDE_EPS) -> np.ndarray:
    """Upsample ``source_alpha`` to the guide's size, snapped to its edges.

    ``source_alpha`` must already be the size of ``guide_rgb``.  The returned
    matte is float32 in 0..1.  Falls back to the bilinear resize when OpenCV's
    guided filter is unavailable.
    """
    guide = np.ascontiguousarray(_as_rgb(guide_rgb), dtype=np.float32) / 255.0
    src = np.ascontiguousarray(np.asarray(source_alpha, dtype=np.float32))
    if src.shape[:2] != guide.shape[:2]:
        import cv2
        src = cv2.resize(src, (guide.shape[1], guide.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    src = np.clip(src, 0.0, 1.0)
    try:
        import cv2
        matte = cv2.ximgproc.guidedFilter(guide, src, int(max(1, radius)),
                                          float(max(1e-7, eps)))
    except Exception:  # noqa: BLE001 - no contrib module: keep the bilinear resize
        return src
    return np.clip(matte, 0.0, 1.0)


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sharpen_matte(matte: np.ndarray, lo: float = 0.35, hi: float = 0.65) -> np.ndarray:
    """Steepen the anti-aliased band without moving the 0.5 contour."""
    return _smoothstep(np.asarray(matte, dtype=np.float32), lo, hi)


def snap_mask_to_edges(mask: np.ndarray, guide_rgb: np.ndarray,
                       max_dist: int = 4, smooth: int = 5,
                       canny: Tuple[int, int] = (50, 150)) -> np.ndarray:
    """Move every contour point onto the nearest strong edge (within a limit).

    The guided filter sharpens a matte but does not *translate* it: a split
    alpha that sits two pixels off the drawn line stays two pixels off.  This
    snaps the contour itself.  Points with no edge within ``max_dist`` are left
    alone, so a hidden boundary (no line under a body) keeps its prior shape.
    """
    import cv2
    from scipy import ndimage
    mask = np.asarray(mask, dtype=np.uint8)
    if not mask.any():
        return mask > 0
    gray = cv2.cvtColor(np.ascontiguousarray(_as_rgb(guide_rgb), dtype=np.uint8),
                        cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(canny[0]), int(canny[1])) > 0
    if not edges.any():
        return mask > 0
    dist, (iy, ix) = ndimage.distance_transform_edt(~edges, return_indices=True)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask > 0
    h, w = mask.shape[:2]
    kernel = np.ones(max(1, int(smooth))) / float(max(1, int(smooth)))
    outer, holes = [], []
    for i, contour in enumerate(contours):
        pts = contour.reshape(-1, 2).astype(np.float64)
        xs = np.clip(pts[:, 0].astype(np.int64), 0, w - 1)
        ys = np.clip(pts[:, 1].astype(np.int64), 0, h - 1)
        near = dist[ys, xs] <= float(max_dist)
        if near.any():
            sx = ix[ys[near], xs[near]].astype(np.float64)
            sy = iy[ys[near], xs[near]].astype(np.float64)
            pts[near, 0] = sx
            pts[near, 1] = sy
            if int(smooth) >= 3 and pts.shape[0] >= int(smooth):
                for ch in range(2):
                    value = pts[:, ch]
                    ext = np.r_[value[-int(smooth):], value, value[:int(smooth)]]
                    pts[:, ch] = np.convolve(ext, kernel, mode="same")[int(smooth):-int(smooth)]
        is_hole = hierarchy is not None and int(hierarchy[0, i, 3]) != -1
        (holes if is_hole else outer).append(pts.astype(np.int32))
    out = np.zeros(mask.shape, np.uint8)
    if outer:
        cv2.fillPoly(out, outer, 1)
    if holes:
        cv2.fillPoly(out, holes, 0)
    return out > 0


def part_mask_at_reference(alpha_src: np.ndarray, box: Rect, reference: np.ndarray,
                           canvas: Tuple[int, int],
                           radius: int = DEFAULT_GUIDE_RADIUS,
                           eps: float = DEFAULT_GUIDE_EPS,
                           threshold: float = DEFAULT_MASK_THRESHOLD,
                           close_px: int = 0,
                           snap_px: int = 0
                           ) -> Dict[str, np.ndarray]:
    """Mask one layer at the reference's resolution.

    ``alpha_src`` is the split layer's alpha (box-sized).  Returns a dict with
    ``mask`` (bool), ``matte`` (float32 0..1), ``rect`` (in reference pixels)
    and ``guide`` (the RGB crop used as edge guide).
    """
    ref = np.asarray(reference)
    rect, _ = _reference_rect(box, (ref.shape[1], ref.shape[0]), canvas)
    guide = ref[rect[1]:rect[3], rect[0]:rect[2]]
    h, w = guide.shape[:2]
    import cv2
    src = cv2.resize(np.asarray(alpha_src, dtype=np.float32) / 255.0, (w, h),
                     interpolation=cv2.INTER_LINEAR)
    alpha_up = src.copy()
    matte_raw = guided_matte(src, guide, radius, eps)
    matte = sharpen_matte(matte_raw)
    mask = matte >= float(threshold)
    if not mask.any():
        # A faint part (the nose is max alpha 50 at 1024) never reaches the
        # 0.5 contour; keep the split's own soft shape instead of losing it.
        mask = src >= max(0.08, float(threshold) * 0.2)
    if int(close_px) > 0:
        k = np.ones((int(close_px), int(close_px)), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    if int(snap_px) > 0:
        mask = snap_mask_to_edges(mask, guide, max_dist=int(snap_px))
    # The split's alpha may reach past the character; the reference knows where
    # the character actually ends.
    alpha_ref = _alpha_of(ref[rect[1]:rect[3], rect[0]:rect[2]])
    if alpha_ref is not None:
        mask &= alpha_ref > 8
    return {"mask": mask, "matte": matte, "matte_raw": matte_raw,
            "alpha_up": alpha_up, "rect": rect, "guide": guide}


def edge_alignment(mask: np.ndarray, guide_rgb: np.ndarray,
                   canny: Tuple[int, int] = (40, 120)) -> Optional[Dict[str, float]]:
    """Distance from the mask contour to the nearest strong image edge.

    A mask cut on the real outline has mean distance well below one destination
    pixel; a 4x staircase sits at ~1.5-2 px.  This is how "the edge is accurate"
    is measured instead of eyeballed.
    """
    try:
        import cv2
        from scipy import ndimage
    except Exception:  # noqa: BLE001
        return None
    gray = cv2.cvtColor(np.ascontiguousarray(_as_rgb(guide_rgb), dtype=np.uint8),
                        cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(canny[0]), int(canny[1])) > 0
    if not edges.any():
        return None
    dist = ndimage.distance_transform_edt(~edges)
    mask = np.asarray(mask, dtype=bool)
    contour = mask ^ ndimage.binary_erosion(mask)
    values = dist[contour]
    if values.size == 0:
        return None
    return {
        "contour_px": int(values.size),
        "mean_px": round(float(values.mean()), 3),
        "p95_px": round(float(np.percentile(values, 95)), 3),
    }


def character_inside(guide_rgba: np.ndarray) -> np.ndarray:
    """True where the reference actually contains the character.

    The reference is RGBA, but a ComfyUI ``LoadImage`` drops the alpha and
    leaves the transparent background as pure white.  The flattened background
    is then found by flood-filling pure white *from the crop border*: an
    enclosed white highlight is character detail, not background, and a crop
    that does not touch the background is entirely inside.
    """
    alpha = _alpha_of(guide_rgba)
    if alpha is not None and int(alpha.min()) < 200:
        return alpha > 8
    rgb = np.asarray(guide_rgba, dtype=np.float32)[..., :3]
    luma = _luma(rgb)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    white = (luma > 250.0) & (sat < 5.0)
    border_white = float(white[0].mean() + white[-1].mean() +
                         white[:, 0].mean() + white[:, -1].mean()) / 4.0
    if border_white < 0.15:
        return np.ones(luma.shape, dtype=bool)
    try:
        from scipy import ndimage
        labels, _ = ndimage.label(white)
        border = np.unique(np.concatenate([labels[0], labels[-1],
                                           labels[:, 0], labels[:, -1]]))
        border = border[border > 0]
        outside = np.isin(labels, border) if border.size else np.zeros_like(white)
    except Exception:  # noqa: BLE001 - scipy missing: a plain cut is enough
        outside = np.zeros_like(white)
    return ~outside


def _colour_masks(guide_rgba: np.ndarray,
                  inside: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    rgb = np.asarray(guide_rgba, dtype=np.float32)[..., :3]
    if inside is None:
        inside = character_inside(guide_rgba)
    inside = np.asarray(inside, dtype=bool)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = _luma(rgb)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    # Skin is also red-dominant (247, 202, 180); the caller always intersects
    # this with the part's own (sharpened) prior, which is what keeps it out.
    red = (r > g + RED_DOMINANCE) & (r > b + RED_DOMINANCE) & inside
    return {
        "red": red,
        "sclera": (mn > SCLERA_LUMA) & inside,
        "dark": (mx < DARK_LUMA) & inside,
        "inside": inside,
        "luma": luma,
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    import cv2
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def _fit_circle(points: np.ndarray, iters: int = 6,
                keep_frac: float = 0.85) -> Optional[Tuple[Tuple[float, float, float], np.ndarray]]:
    """Robust least-squares circle (Kasa) with trimming.

    Anime irises are drawn as circles, and the visible lower arc determines one
    well.  A free ellipse is not stable here: the red waterline under the lid
    is also "red" and bends a plain fit into a heart, while the hidden top arc
    leaves the vertical axis under-constrained.  A circle is the honest prior.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 5:
        return None
    cx = cy = r = 0.0
    for _ in range(max(1, int(iters))):
        lhs = np.column_stack([2.0 * pts[:, 0], 2.0 * pts[:, 1],
                               np.ones(len(pts))])
        rhs = pts[:, 0] ** 2 + pts[:, 1] ** 2
        sol, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        r = float(np.sqrt(max(0.0, float(sol[2]) + cx * cx + cy * cy)))
        resid = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        keep = resid <= float(np.quantile(resid, float(keep_frac)))
        if int(keep.sum()) < 5:
            break
        pts = pts[keep]
    if r < 2.0:
        return None
    return (cx, cy, r), pts


def complete_iris_mask(guide_rgba: np.ndarray, prior_matte: np.ndarray,
                       min_points: int = 8) -> Optional[Dict]:
    """Full iris circle from the visible arc.

    The lash covers the top of the iris, so the reference only shows a partial
    ring.  The boundary between iris and sclera is still visible on the sides
    and bottom; a circle fitted to those points is the *complete* iris, and the
    pixels under the lash are exactly what a Live2D part has to contain.
    """
    import cv2
    cols = _colour_masks(guide_rgba)
    prior = np.asarray(prior_matte, dtype=np.float32) > 0.05
    red = cols["red"] & prior
    if not red.any():
        return None
    red = _largest_component(red)
    contours, _ = cv2.findContours(red.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=lambda c: cv2.contourArea(c))
    points = contour.reshape(-1, 2)

    # Keep the arc that borders the sclera: those are the true iris boundary.
    sclera_near = cv2.dilate(cols["sclera"].astype(np.uint8),
                             np.ones((9, 9), np.uint8)) > 0
    keep = np.array([p for p in points if sclera_near[int(p[1]), int(p[0])]],
                    dtype=np.int32)
    if keep.shape[0] < int(min_points):
        keep = points

    fit = _fit_circle(keep)
    if fit is None:
        return None
    (cx, cy, r), inliers = fit
    out = np.zeros(cols["inside"].shape, np.uint8)
    cv2.circle(out, (int(round(cx)), int(round(cy))),
               int(np.ceil(r + 0.75)), 255, -1)
    return {
        "mask": out > 0,
        "circle": (cx, cy, r),
        "visible": red,
        "points": keep,
        "source": "circle",
    }


def eye_opening_mask(guide_rgba: np.ndarray, iris: Optional[np.ndarray] = None,
                     base: Optional[np.ndarray] = None,
                     close_px: int = 3) -> Optional[np.ndarray]:
    """The eyeball shape: everything inside the lids (sclera + iris + pupil).

    Used for the eyewhite layer, which has to stay complete under the iris: a
    rig that moves the iris must not reveal a hole in the sclera.  ``base`` is
    the guided split alpha for the eyewhite, used as an upper bound (dilated):
    the bright skin around the eye is sclera-like, and only the prior knows
    where the eyelid is.
    """
    import cv2
    from scipy import ndimage
    content = None
    if base is not None:
        content = np.asarray(base, dtype=bool).copy()
    if iris is not None:
        iris = np.asarray(iris, dtype=bool)
        content = iris.copy() if content is None else (content | iris)
    if content is None or not content.any():
        cols = _colour_masks(guide_rgba)
        content = cols["sclera"] | cols["red"]
    # The eyeball is the convex envelope of the visible opening: the lid notch
    # (where the lash crosses the sclera) is not a hole in the eye, and the
    # pupil/iris are inside it by construction.
    content = ndimage.binary_fill_holes(content)
    contours, _ = cv2.findContours(content.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    hull = cv2.convexHull(np.vstack(contours))
    out = np.zeros(content.shape, np.uint8)
    cv2.fillPoly(out, [hull], 1)
    filled = out > 0
    if int(close_px) > 0:
        k = np.ones((int(close_px), int(close_px)), np.uint8)
        filled = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    return filled if filled.any() else None


def stroke_mask(guide_rgba: np.ndarray, prior_matte: np.ndarray,
                luma_max: float = 110.0, near_px: int = 9,
                close_px: int = 2, prior_threshold: float = 0.02) -> Optional[np.ndarray]:
    """Dark strokes (lash, brow) inside the part's own neighbourhood.

    The split's lash layer is only 98 solid pixels at 1024 and the eyebrow has
    no solid pixel at all, so their alpha cannot describe the stroke.  The
    reference's dark pixels do -- but only close to where the split says the
    stroke is, or the neighbouring hair would be swallowed.
    """
    import cv2
    cols = _colour_masks(guide_rgba)
    near = cv2.dilate((np.asarray(prior_matte, dtype=np.float32) > float(prior_threshold)).astype(np.uint8),
                      np.ones((max(1, int(near_px)), max(1, int(near_px))), np.uint8)) > 0
    dark = (cols["luma"] < float(luma_max)) & cols["inside"] & near
    if not dark.any():
        return None
    dark = _largest_component(dark)
    if int(close_px) > 0:
        k = np.ones((int(close_px), int(close_px)), np.uint8)
        dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    return dark if dark.any() else None


def lash_mask(guide_rgba: np.ndarray, prior_matte: np.ndarray) -> Optional[np.ndarray]:
    return stroke_mask(guide_rgba, prior_matte, luma_max=110.0, near_px=9, close_px=2)


MASK_MODES = ("auto", "part", "iris", "eye", "lash", "brow")


def _auto_kind(name: str) -> str:
    base = str(name or "").lower()
    if base.startswith("irides"):
        return "iris"
    if base.startswith("eyewhite"):
        return "eye"
    if base.startswith("eyebrow"):
        return "brow"
    if base.startswith("eyelash"):
        return "lash"
    return "part"


def mask_for_layer(name: str, alpha_src: np.ndarray, box: Rect,
                   reference: np.ndarray, canvas: Tuple[int, int],
                   mode: str = "auto",
                   radius: int = DEFAULT_GUIDE_RADIUS,
                   eps: float = DEFAULT_GUIDE_EPS,
                   threshold: float = DEFAULT_MASK_THRESHOLD,
                   close_px: int = 0,
                   snap_px: int = 0) -> Dict:
    """The part's complete mask at the reference's resolution.

    ``mode`` selects how the shape is decided: ``auto`` follows the part name
    (iris / eye / lash / brow / plain part).  An iris or eyebrow the split got
    wrong falls back to the guided part mask if the geometric completion finds
    nothing usable.
    """
    mode = str(mode or "auto").lower()
    if mode not in MASK_MODES:
        mode = "auto"
    pm = part_mask_at_reference(alpha_src, box, reference, canvas,
                                radius=radius, eps=eps, threshold=threshold,
                                close_px=close_px, snap_px=snap_px)
    guide, rect = pm["guide"], pm["rect"]
    matte_raw = pm["matte_raw"]
    alpha_up = pm["alpha_up"]
    inside = character_inside(guide)
    kind = _auto_kind(name) if mode == "auto" else mode
    mask = pm["mask"]
    stats = {"kind": kind, "requested_mode": mode}

    if kind == "iris":
        # The sharpened matte is the gate: the raw one is soft enough that
        # red neighbours (skin) would join the ellipse fit.
        done = complete_iris_mask(guide, pm["matte"])
        if done is not None:
            mask = done["mask"]
            stats["circle"] = [round(float(v), 2) for v in done["circle"]]
            stats["hidden_px"] = int((done["mask"] & ~done["visible"]).sum())
        else:
            kind = "part"
    elif kind == "eye":
        done = complete_iris_mask(guide, pm["matte"])
        iris = done["mask"] if done is not None else None
        socket = eye_opening_mask(guide, iris, base=pm["mask"])
        if socket is not None:
            mask = socket
        else:
            kind = "part"
    elif kind == "lash":
        stroke = stroke_mask(guide, alpha_up, luma_max=100.0, near_px=5,
                             close_px=2, prior_threshold=0.3)
        if stroke is not None:
            mask = stroke
        else:
            kind = "part"
    elif kind == "brow":
        stroke = stroke_mask(guide, alpha_up, luma_max=150.0, near_px=3,
                             close_px=3, prior_threshold=0.4)
        if stroke is not None:
            mask = stroke
        else:
            kind = "part"

    mask = np.asarray(mask, dtype=bool) & inside
    stats.update({
        "kind": kind,
        "mask_px": int(mask.sum()),
        "rect": [int(v) for v in rect],
        "size": [int(mask.shape[1]), int(mask.shape[0])],
        "edge": edge_alignment(mask, guide),
    })
    return {"mask": mask, "rect": rect, "guide": guide, "stats": stats}


def mask_overlay(guide_rgba: np.ndarray, mask: np.ndarray,
                 colour: Tuple[int, int, int] = (255, 0, 0),
                 alpha: float = 0.45) -> np.ndarray:
    """RGB preview of ``guide`` with ``mask`` tinted (for the node's preview)."""
    rgb = np.array(_as_rgb(guide_rgba), dtype=np.float32)
    tint = np.array(colour, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    rgb[m] = rgb[m] * (1.0 - float(alpha)) + tint * float(alpha)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
