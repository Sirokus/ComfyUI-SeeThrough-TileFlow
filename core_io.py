"""Small file helpers shared by the SeeThrough layer-upscale nodes."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

try:
    from .core_plan import AUTO_OUT_SUBDIR, normalize_out_dir, safe_name
except ImportError:  # standalone import from the tests
    from core_plan import AUTO_OUT_SUBDIR, normalize_out_dir, safe_name


def out_dir_abs(output_root: str, out_dir: str) -> str:
    # No source folder is known here, so an empty ("自动") out_dir can only
    # fall back to a plain UpScale under the output root.  The batch nodes
    # always pass the path the plan already resolved.
    rel = normalize_out_dir(out_dir) or AUTO_OUT_SUBDIR
    path = os.path.join(output_root, rel)
    os.makedirs(path, exist_ok=True)
    return path


def read_rgba(path: str) -> np.ndarray:
    """Read an image as an (H, W, 4) uint8 array."""
    with Image.open(path) as im:
        return np.array(im.convert("RGBA"), dtype=np.uint8)


def write_rgba(path: str, arr: np.ndarray, compress_level: int = 6) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im = Image.fromarray(np.asarray(arr, dtype=np.uint8), "RGBA")
    im.save(path, format="PNG", compress_level=int(compress_level))
    return path


def read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def layer_output_path(output_root: str, out_dir: str, name: str) -> str:
    return os.path.join(out_dir_abs(output_root, out_dir), safe_name(name) + ".png")


def find_layer_output(output_root: str, out_dir: str, name: str) -> Optional[str]:
    path = layer_output_path(output_root, out_dir, name)
    return path if os.path.isfile(path) else None


def tensor_to_rgba(tensor) -> np.ndarray:
    """Convert a ComfyUI IMAGE tensor ([B,H,W,3/4], float 0..1) to uint8 RGBA."""
    arr = tensor[0].detach().cpu().numpy()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    rgb = np.clip(np.rint(arr[..., :3] * 255.0), 0, 255).astype(np.uint8)
    if arr.shape[-1] >= 4:
        alpha = np.clip(np.rint(arr[..., 3] * 255.0), 0, 255).astype(np.uint8)
    else:
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    return np.dstack([rgb, alpha])


def mask_to_alpha(mask) -> np.ndarray:
    """ComfyUI MASK is 1 = transparent, so alpha = 255 * (1 - mask)."""
    arr = mask[0].detach().cpu().numpy() if mask is not None else None
    if arr is None:
        return None
    arr = np.clip(arr, 0.0, 1.0)
    return np.clip(np.rint((1.0 - arr) * 255.0), 0, 255).astype(np.uint8)


def resize_rgba(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resample an HxWx4 uint8 image with Lanczos filtering."""
    width = max(1, int(width))
    height = max(1, int(height))
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.shape[:2] == (height, width):
        return arr
    im = Image.fromarray(arr, "RGBA")
    im = im.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(im, dtype=np.uint8)


def resize_rgba_premultiplied(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resample RGBA *premultiplied* by alpha, then un-premultiply.

    PIL filters the four channels independently.  On a straight-alpha layer
    whose transparent pixels still hold some leftover colour (SeeThrough writes
    a pale fill there), that colour gets smeared into the visible edge by the
    Lanczos kernel -- the "dirty halo" around every part.  Premultiplying makes
    those pixels carry zero weight, so the edge keeps the part's own colour.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.shape[:2] == (height, width):
        return arr
    a = arr[..., 3].astype(np.uint16)
    pre = np.empty_like(arr)
    pre[..., :3] = ((arr[..., :3].astype(np.uint16) * a[..., None] + 127) // 255).astype(np.uint8)
    pre[..., 3] = arr[..., 3]
    up = resize_rgba(pre, width, height)
    a_up = up[..., 3].astype(np.float32)
    rgb = np.zeros(up[..., :3].shape, dtype=np.float32)
    visible = a_up > 0.0
    safe = np.maximum(a_up, 1.0)
    rgb[visible] = (up[..., :3][visible].astype(np.float32) * 255.0
                    / safe[visible][..., None])
    out = np.empty_like(up)
    out[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    out[..., 3] = up[..., 3]
    return out


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resample a boolean mask with nearest neighbour (no soft, no holes)."""
    width = max(1, int(width))
    height = max(1, int(height))
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[:2] == (height, width):
        return mask
    im = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    im = im.resize((width, height), Image.Resampling.NEAREST)
    return np.array(im, dtype=np.uint8) > 127


def resize_alpha(alpha: np.ndarray, width: int, height: int,
                 sharp: bool = False) -> np.ndarray:
    """Resample an HxW uint8 alpha map with bilinear filtering.

    ``sharp`` swaps the kernel for Lanczos: a source silhouette upscaled from
    crop size keeps a much narrower transition, which is what a locked
    outline needs (bilinear spreads a hard edge over the whole scale factor).
    """
    width = max(1, int(width))
    height = max(1, int(height))
    alpha = np.asarray(alpha, dtype=np.uint8)
    if alpha.shape[:2] == (height, width):
        return alpha
    im = Image.fromarray(alpha, "L")
    kernel = Image.Resampling.LANCZOS if sharp else Image.Resampling.BILINEAR
    im = im.resize((width, height), kernel)
    return np.array(im, dtype=np.uint8)


def extend_colour(rgb: np.ndarray, alpha: np.ndarray,
                  solid: int = 200) -> np.ndarray:
    """Fill non-solid pixels with the nearest solid pixel's colour.

    Transparent background carries whatever colour the model painted there
    (Qwen 2.1 decodes alpha 0 as noisy purple); premultiplied resampling would
    bleed that colour into the silhouette edge.  Every pixel below ``solid``
    takes the colour of its closest opaque neighbour instead, so a resize only
    ever mixes the part's own colours.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    a = np.asarray(alpha, dtype=np.uint8)
    if a.ndim == 3:
        a = a[..., 0]
    solid_mask = a >= int(solid)
    if not solid_mask.any() or solid_mask.all():
        return rgb
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - scipy ships with ComfyUI
        out = rgb.copy()
        known = solid_mask.copy()
        for _ in range(max(rgb.shape[:2])):
            if known.all():
                break
            grow = known.copy()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                shifted = np.roll(known, (dy, dx), (0, 1))
                source = np.roll(out, (-dy, -dx), (0, 1))
                sel = ~grow & shifted
                out[sel] = source[sel]
                grow |= shifted
            known = grow
        return out
    idx = ndimage.distance_transform_edt(~solid_mask, return_distances=False,
                                         return_indices=True)
    return rgb[idx[0], idx[1]]


def merge_alpha(rgba: np.ndarray, mask_alpha, keep_input_alpha: bool = False) -> np.ndarray:
    """Return RGBA where the model alpha wins unless it is missing."""
    rgba = np.array(rgba, dtype=np.uint8, copy=True)
    if mask_alpha is not None:
        mask_alpha = np.asarray(mask_alpha, dtype=np.uint8)
        if mask_alpha.ndim == 3:
            mask_alpha = mask_alpha[..., 0]
        if mask_alpha.shape[:2] != rgba.shape[:2]:
            # The model decodes at its own size and the target size generally
            # differs from the source layer.  Resample the mask instead of
            # dropping it: dropping it let a whole invented frame be saved as
            # an opaque layer.
            mask_alpha = resize_alpha(mask_alpha, rgba.shape[1], rgba.shape[0])
    if mask_alpha is not None and mask_alpha.shape[:2] == rgba.shape[:2]:
        if keep_input_alpha:
            rgba[..., 3] = mask_alpha
        else:
            # Combine: a pixel is visible only if both the model and the source
            # mask agree it belongs to the part.  The source mask is used to
            # strip background the model may have painted.
            combined = (rgba[..., 3].astype(np.float32) / 255.0) * (mask_alpha.astype(np.float32) / 255.0)
            rgba[..., 3] = np.clip(np.rint(combined * 255.0), 0, 255).astype(np.uint8)
    return rgba
