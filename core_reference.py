"""Reference-guided layer reconstruction (pure numpy/PIL, no diffusion).

When the full-body reference is a *higher-resolution version of the same
artwork* as the split (``context_scale > 1``), the reference already contains
the true linework and colour of every visible part.  Re-cutting a layer from it
-- alpha from the split, colour from the reference inside that alpha -- gives a
crisp, clean layer with genuine detail.

This is deliberately *not* a diffusion pass: a 17x15 pupil sent through an edit
model only gets smoothed/hallucinated.  Sampling the reference keeps the line
art exactly where the artist drew it.

Three things previously kept the result soft or dirty, and are handled here:

1. **Dirty edge colours.**  SeeThrough leaves a pale fill in the transparent
   area of every layer.  Resampling straight RGBA lets Lanczos drag that fill
   into the visible edge (a light halo around hair, a white blob where the mouth
   should be).  Layers are therefore resampled *premultiplied* and converted
   back, so transparent pixels contribute nothing.
2. **Rejected detail.**  The old colour gate compared the reference with the
   upscaled (blurry) layer, so every dark line -- i.e. exactly the detail we
   want -- was "too different" and discarded.  The gate now only rejects pixels
   that are *strongly saturated in a different hue* (a neighbouring part really
   showing through), which accepts line art and shading while still refusing a
   green/blue occluder.
3. **Soft silhouette.**  The split alpha is 1024; at a 4096 canvas its edge is
   four pixels wide.  ``alpha_sharpen`` narrows the transition around the 0.5
   contour without moving it, so the outline stays exactly where it was.
4. **Dotted halo outside the outline.**  The split's soft layer edges bleed a
   pixel or two past the character; at 4x that becomes a dotted white/grey
   line along hair and the headphone.  The reference is the same artwork at
   full resolution, so its (hole-filled) silhouette clips every layer: nothing
   is painted outside the character, and no inner pocket is punched through.
5. **Neighbour colour in the soft rim.**  Inside a part's own alpha the
   reference still shows the *neighbour* wherever the split alpha is soft -- a
   pale skin rim under a hair lock, white under a pupil.  That rim is what a
   rig reveals as soon as the part moves, so it is repainted with the part's
   own nearest solid colour (``_recolour_fringe``).

Everything here is numpy/PIL so it can be tested without ComfyUI or torch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from . import core_geometry, core_io
except ImportError:  # standalone import from the tests
    import core_geometry
    import core_io

Rect = Tuple[int, int, int, int]

# Chroma difference above which a *saturated* reference colour is assumed to be
# another part showing through (an occluder) rather than this layer's art.
DEFAULT_COLOR_TOLERANCE = 26.0
# Strength of the silhouette steepening around the 0.5 contour.  1.0 keeps the
# upscaled alpha as is; 1.6 makes a 4x-upscaled edge about 1.6x crisper.
DEFAULT_ALPHA_SHARPEN = 1.6
# SeeThrough leaves alpha 1..15 "dust" across the empty background.
DEFAULT_ALPHA_FLOOR = 8
# Reference alpha below which a pixel counts as "outside the character".
DEFAULT_SILHOUETTE_ALPHA = 32
# Alpha from which a pixel counts as "solidly inside the part"; softer pixels
# take their colour from the nearest one of these instead of from the
# reference (which shows the neighbour there).
DEFAULT_CRISP_ALPHA = 224

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
# HSV chroma (max - min) above which a pixel counts as "saturated".  Achromatic
# line art sits far below this even when it is almost black.
SATURATED_LEVEL = 46.0


def _flatten_white(window: np.ndarray) -> np.ndarray:
    """Turn an RGBA window into opaque RGB over white, like Qwen's vision tower."""
    rgb = window[..., :3].astype(np.float32)
    if window.shape[-1] < 4:
        return rgb
    alpha = window[..., 3:4].astype(np.float32) / 255.0
    return rgb * alpha + 255.0 * (1.0 - alpha)


def reference_scale(reference_size: Tuple[int, int],
                    canvas: Tuple[int, int]) -> float:
    """Reference pixels per source-canvas pixel (``1.0`` when not usable)."""
    return float(core_geometry.context_scale(reference_size, canvas))


def _clean_dust(layer: np.ndarray, floor: int) -> np.ndarray:
    if int(floor) <= 0:
        return layer
    out = np.array(layer, dtype=np.uint8, copy=True)
    out[..., 3][out[..., 3] <= int(floor)] = 0
    return out


def _sharpen_alpha(alpha: np.ndarray, gain: float) -> np.ndarray:
    """Steepen the alpha ramp around 0.5 without moving the contour."""
    if gain is None or gain <= 1.0001:
        return alpha
    a = alpha.astype(np.float32) / 255.0
    steep = np.clip((a - 0.5) * float(gain) + 0.5, 0.0, 1.0)
    return np.clip(np.rint(steep * 255.0), 0, 255).astype(np.uint8)


def _colour_mismatch(base_rgb: np.ndarray, ref_rgb: np.ndarray,
                     tolerance: float) -> np.ndarray:
    """True where the reference is a *different, saturated* colour.

    Dark line art and shading are luminance changes of the layer's own colour:
    their chroma is close to the base (or neutral), so they pass.  A strongly
    saturated colour of another hue (blue hair over skin, a green object) has a
    very different chroma and is rejected.
    """
    luma_b = base_rgb @ _LUMA
    luma_r = ref_rgb @ _LUMA
    chroma_b = base_rgb - luma_b[..., None]
    chroma_r = ref_rgb - luma_r[..., None]
    chroma_diff = np.abs(chroma_r - chroma_b).mean(axis=2)

    sat_b = base_rgb.max(axis=2) - base_rgb.min(axis=2)
    sat_r = ref_rgb.max(axis=2) - ref_rgb.min(axis=2)
    raw_diff = np.abs(ref_rgb - base_rgb).mean(axis=2)

    saturated = (sat_b > SATURATED_LEVEL) & (sat_r > SATURATED_LEVEL)
    return saturated & (chroma_diff > float(tolerance)) & (raw_diff > float(tolerance))


def reference_silhouette(reference_rgba: Optional[np.ndarray],
                         threshold: int = DEFAULT_SILHOUETTE_ALPHA) -> Optional[np.ndarray]:
    """Boolean mask of the character in ``reference_rgba`` (holes filled).

    Interior transparent pockets of the reference are filled so clipping a
    layer against this mask can never punch a hole *inside* the character.
    Returns ``None`` when the reference carries no alpha channel.
    """
    if reference_rgba is None:
        return None
    ref = np.asarray(reference_rgba)
    if ref.ndim < 3 or ref.shape[-1] < 4:
        return None
    mask = ref[..., 3] > int(threshold)
    if not mask.any():
        return None
    try:
        from scipy import ndimage  # optional: only used to fill interior holes
        mask = ndimage.binary_fill_holes(mask)
    except Exception:  # noqa: BLE001 - scipy missing is fine
        pass
    return mask


def _silhouette_window(silhouette: np.ndarray, box: Rect, k: float,
                       dest_w: int, dest_h: int) -> Optional[np.ndarray]:
    win = core_geometry.scale_rect(box, k, (silhouette.shape[1], silhouette.shape[0]))
    sub = silhouette[win[1]:win[3], win[0]:win[2]]
    if sub.size == 0:
        return None
    if (sub.shape[1], sub.shape[0]) != (dest_w, dest_h):
        sub = core_io.resize_mask(sub, dest_w, dest_h)
    return sub


def _reference_window(reference_rgba: np.ndarray, box: Rect, k: float,
                      dest_w: int, dest_h: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Cut ``box`` out of the reference at its native resolution.

    Returns ``(rgb_float32, valid_mask)`` already resampled to the destination
    size, or ``(None, None)`` when the reference has no usable alpha.
    """
    win = core_geometry.scale_rect(box, k, (reference_rgba.shape[1], reference_rgba.shape[0]))
    window = reference_rgba[win[1]:win[3], win[0]:win[2]]
    if window.size == 0:
        return None, None
    ctx = _flatten_white(window)
    valid = np.ones(ctx.shape[:2], dtype=bool)
    if reference_rgba.shape[-1] >= 4:
        # Transparent reference pixels carry no colour; flattening them to
        # white must not be mistaken for a white detail.
        valid = window[..., 3] > 8
    if (ctx.shape[1], ctx.shape[0]) != (dest_w, dest_h):
        rgba = np.dstack([np.clip(np.rint(ctx), 0, 255).astype(np.uint8),
                          np.full(ctx.shape[:2], 255, np.uint8)])
        ctx = core_io.resize_rgba(rgba, dest_w, dest_h)[..., :3].astype(np.float32)
        valid = core_io.resize_alpha(valid.astype(np.uint8) * 255, dest_w, dest_h) > 8
    return ctx, valid


def reconstruct_layer(
    layer_rgba: np.ndarray,
    box: Rect,
    reference_rgba: Optional[np.ndarray],
    canvas: Tuple[int, int],
    dest_size: Tuple[int, int],
    color_tolerance: float = DEFAULT_COLOR_TOLERANCE,
    alpha_floor: int = DEFAULT_ALPHA_FLOOR,
    alpha_sharpen: float = DEFAULT_ALPHA_SHARPEN,
    silhouette: Optional[np.ndarray] = None,
    crisp_alpha: int = DEFAULT_CRISP_ALPHA,
) -> Tuple[np.ndarray, dict]:
    """Return this layer rebuilt at ``dest_size`` (W, H).

    ``layer_rgba`` is the layer cropped to ``box`` (source-canvas pixels).
    ``reference_rgba`` is the full reference frame (may be larger than the
    canvas, or ``None``).  The returned stats record how much of the layer was
    taken from the reference, so the node can report it.
    """
    dest_w = max(1, int(dest_size[0]))
    dest_h = max(1, int(dest_size[1]))
    box = tuple(int(v) for v in box)
    src_w = max(1, int(box[2]) - int(box[0]))
    src_h = max(1, int(box[3]) - int(box[1]))

    layer = np.asarray(layer_rgba, dtype=np.uint8)
    if layer.ndim == 2:
        layer = np.dstack([layer, layer, layer, np.full(layer.shape, 255, np.uint8)])
    elif layer.shape[-1] == 3:
        layer = np.dstack([layer, np.full(layer.shape[:2], 255, np.uint8)])
    if (layer.shape[1], layer.shape[0]) != (src_w, src_h):
        layer = core_io.resize_rgba(layer, src_w, src_h)

    # Kill the dust *before* upscaling, so a 2..15 alpha speck cannot become a
    # visible smudge at 4x.  Then resample premultiplied: the leftover fill
    # colour of the transparent area must not bleed into the silhouette.
    layer = _clean_dust(layer, alpha_floor)
    base = core_io.resize_rgba_premultiplied(layer, dest_w, dest_h)
    out_rgb = base[..., :3].astype(np.float32)
    alpha = base[..., 3].copy()

    stats = {"scale": 1.0, "trusted": 0.0, "sharpen": round(float(alpha_sharpen or 1.0), 3)}
    if reference_rgba is not None:
        ref = np.asarray(reference_rgba, dtype=np.uint8)
        if ref.ndim >= 3 and ref.shape[-1] >= 3:
            k = reference_scale((ref.shape[1], ref.shape[0]), canvas)
            stats["scale"] = round(k, 4)
            if k > 1.02:
                ref_dest, valid = _reference_window(ref, box, k, dest_w, dest_h)
                if ref_dest is not None:
                    # Never sample outside our own silhouette: the split alpha
                    # is the contour the client rig expects.
                    inside = alpha >= 6
                    trust = inside & valid
                    trust &= ~_colour_mismatch(out_rgb, ref_dest,
                                               float(color_tolerance))
                    out_rgb = np.where(trust[..., None], ref_dest, out_rgb)
                    # Fraction of the layer's own visible pixels taken from
                    # the reference (sparse layers would read ~0 otherwise).
                    inside_n = int(inside.sum())
                    stats["trusted"] = round(
                        float(trust.sum()) / inside_n if inside_n else 0.0, 4)

    alpha = _sharpen_alpha(alpha, float(alpha_sharpen or 1.0))
    if silhouette is not None:
        sil = np.asarray(silhouette, dtype=bool)
        k_sil = float(sil.shape[1]) / max(1.0, float(canvas[0]))
        if k_sil > 1.02:
            sub = _silhouette_window(sil, box, k_sil, dest_w, dest_h)
            if sub is not None:
                # The split's soft edge may reach past the character; the
                # reference knows where the character actually ends.
                stats["clipped"] = int((alpha[~sub] > 0).sum())
                alpha[~sub] = 0
    if int(alpha_floor) > 0:
        # A second pass on the destination: Lanczos ringing next to a dust
        # speck can lift it above the source floor.
        alpha[alpha <= int(alpha_floor)] = 0

    out_rgb, stats["fringe"] = _recolour_fringe(out_rgb, alpha, crisp_alpha)

    out = np.concatenate([np.clip(np.rint(out_rgb), 0, 255).astype(np.uint8),
                          alpha[..., None]], axis=2)
    return out, stats


def fill_hidden(rgba: np.ndarray, covered: np.ndarray, margin: int) -> Tuple[np.ndarray, int]:
    """Rebuild the part behind everything that covers it.

    A Live2D part has to stay complete when the rig moves it: the pixels the
    neighbour covered must still hold *this* part.  The split's own alpha makes
    that area look like the part, and the reference hands over the *neighbour's*
    colour there (skin under hair, the eyelid inside a pupil), so both are
    replaced by a copy of the part's nearest genuinely visible pixel.

    ``covered`` marks what an opaque higher layer paints, so this cannot change
    the assembled image; ``margin`` caps how far the rebuild may reach.
    """
    margin = int(margin)
    out = np.array(rgba, dtype=np.uint8, copy=True)
    if covered is None or covered.size == 0:
        return out, 0
    covered = np.asarray(covered, dtype=bool)
    own = out[..., 3] > 0
    visible = own & ~covered
    if not visible.any() or not covered.any():
        return out, 0
    try:
        from scipy import ndimage  # optional dependency
    except Exception:  # noqa: BLE001 - no scipy: no hidden fill
        return out, 0
    dist, idx = ndimage.distance_transform_edt(~visible, return_indices=True)
    # Every pixel the split assigned to this part but that is hidden gets the
    # part's own colour, however far from the visible part it sits: the face
    # inside "back hair" is the classic case, and its distance is irrelevant.
    target = covered & own
    if margin > 0:
        # Optionally also grow into what the split left empty.  Capped at a
        # fifth of the part's own size, or a lash would balloon into a disc.
        short = int(min(out.shape[0], out.shape[1]))
        cap = max(2, short // 5)
        target |= covered & ~own & (dist <= min(margin, cap))
    count = int(target.sum())
    if count == 0:
        return out, 0
    out[target] = out[idx[0][target], idx[1][target]]
    out[target, 3] = 255
    return out, count


def _recolour_fringe(rgb: np.ndarray, alpha: np.ndarray, crisp: int) -> Tuple[np.ndarray, int]:
    """Paint the soft rim with the part's own colour.

    At the split's fringe the reference shows whatever lies *behind* the part
    (skin under a hair lock, white under a pupil), so copying it there leaves a
    pale rim inside the layer -- the part looks dirty as soon as the rig moves
    it away from its original neighbour.  Every pixel with ``0 < alpha <
    crisp`` instead takes the colour of the nearest pixel that is `crisp`, i.e.
    the part's own interior.
    """
    crisp = int(max(1, min(255, crisp)))
    solid = alpha >= crisp
    if not solid.any():
        return rgb, 0
    rim = (alpha > 0) & ~solid
    count = int(rim.sum())
    if count == 0:
        return rgb, 0
    try:
        from scipy import ndimage  # optional dependency
    except Exception:  # noqa: BLE001 - no scipy: leave the rim as it is
        return rgb, 0
    idx = ndimage.distance_transform_edt(~solid, return_distances=False,
                                        return_indices=True)
    out = rgb.copy()
    out[rim] = rgb[idx[0][rim], idx[1][rim]]
    return out, count
