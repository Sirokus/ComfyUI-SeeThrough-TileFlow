"""Layer assembly / PSD prep helpers (Pillow + numpy only)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from .core_io import find_layer_output, read_rgba
    from .core_plan import safe_name
except ImportError:  # standalone import from the tests
    from core_io import find_layer_output, read_rgba
    from core_plan import safe_name

DEFAULT_ALPHA_THRESHOLD = 8


def alpha_bbox(alpha: np.ndarray, threshold: int = DEFAULT_ALPHA_THRESHOLD) -> Optional[Tuple[int, int, int, int]]:
    """Return (left, top, right, bottom) of pixels with alpha > threshold."""
    mask = alpha > int(threshold)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _resize_rgba(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    if arr.shape[1] == width and arr.shape[0] == height:
        return arr
    im = Image.fromarray(np.asarray(arr, dtype=np.uint8), "RGBA")
    im = im.resize((max(1, int(width)), max(1, int(height))), Image.Resampling.LANCZOS)
    return np.array(im, dtype=np.uint8)


def _resize_alpha(alpha: np.ndarray, width: int, height: int) -> np.ndarray:
    if alpha.shape[1] == width and alpha.shape[0] == height:
        return alpha
    im = Image.fromarray(np.asarray(alpha, dtype=np.uint8), "L")
    im = im.resize((max(1, int(width)), max(1, int(height))), Image.Resampling.BILINEAR)
    return np.array(im, dtype=np.uint8)


def _clean_alpha(arr: np.ndarray, floor: int) -> np.ndarray:
    """Drop near-transparent "dust" so the PSD has no faint speckles.

    SeeThrough's split leaves alpha values of 1..15 scattered across the
    otherwise empty background of every layer.  They are invisible on white
    but show up as dirt in a layered PSD, so anything at or below ``floor``
    is zeroed.
    """
    if int(floor) <= 0:
        return arr
    out = np.array(arr, dtype=np.uint8, copy=True)
    out[..., 3][out[..., 3] <= int(floor)] = 0
    return out


def _opaque_background(alpha: np.ndarray, bbox: Tuple[int, int, int, int], threshold: int) -> bool:
    """True when the model returned a nearly solid backdrop instead of alpha."""
    if bbox is None:
        return True
    h, w = alpha.shape[:2]
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if bw >= w * 0.95 and bh >= h * 0.95:
        coverage = float((alpha > threshold).mean())
        return coverage > 0.80
    return False


def prepare_layer_pixels(
    source_path: str,
    upscaled_path: Optional[str],
    display_w: int,
    display_h: int,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    full_frame: bool = False,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """Return the RGBA pixels to place in the PSD for one layer.

    ``full_frame`` means the source layer already spans the whole character
    canvas (its bbox is the full canvas).  In that case the model saw the full
    framing, so the whole upscaled image is rescaled into the display box.
    For cropped layers the model saw only the part and the non-transparent
    bbox is rescaled into the part's display box instead.
    """
    notes: List[str] = []
    display_w = max(1, int(display_w))
    display_h = max(1, int(display_h))

    if upscaled_path and os.path.isfile(upscaled_path):
        try:
            arr = read_rgba(upscaled_path)
        except Exception as exc:  # noqa: BLE001
            notes.append("failed to read %s: %s" % (upscaled_path, exc))
            arr = None
        if arr is not None:
            arr = _clean_alpha(arr, alpha_threshold)
            alpha = arr[..., 3]
            bbox = alpha_bbox(alpha, alpha_threshold)
            if bbox is None:
                notes.append("upscaled layer %s is fully transparent; using the source" % os.path.basename(upscaled_path))
                arr = None
            elif _opaque_background(alpha, bbox, alpha_threshold):
                notes.append(
                    "upscaled layer %s has an opaque backdrop; cutting it with the source mask"
                    % os.path.basename(upscaled_path)
                )
                try:
                    src = read_rgba(source_path)
                    src_alpha = _resize_alpha(src[..., 3], arr.shape[1], arr.shape[0])
                    arr = np.array(arr, dtype=np.uint8, copy=True)
                    arr[..., 3] = np.minimum(arr[..., 3], src_alpha)
                    bbox = alpha_bbox(arr[..., 3], alpha_threshold)
                except Exception as exc:  # noqa: BLE001
                    notes.append("failed to read source mask %s: %s" % (source_path, exc))
                if bbox is None:
                    arr = None
            if arr is not None:
                if full_frame:
                    # Keep the reference framing: resize the whole frame.
                    return _resize_rgba(arr, display_w, display_h), notes
                content = arr[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                return _resize_rgba(content, display_w, display_h), notes

    # Fallback: original (cropped) layer image stretched into the display box.
    try:
        src = read_rgba(source_path)
    except Exception as exc:  # noqa: BLE001
        notes.append("failed to read source %s: %s" % (source_path, exc))
        return None, notes
    src = _clean_alpha(src, alpha_threshold)
    return _resize_rgba(src, display_w, display_h), notes


def _alpha_over(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Composite ``src`` (straight alpha) over ``dst`` in place at (x, y)."""
    h, w = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1 = min(dst.shape[1], x + w)
    y1 = min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y
    src_c = src[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)].astype(np.float32) / 255.0
    dst_c = dst[y0:y1, x0:x1].astype(np.float32) / 255.0

    sa = src_c[..., 3:4]
    da = dst_c[..., 3:4]
    out_a = sa + da * (1.0 - sa)
    safe_a = np.where(out_a > 1e-6, out_a, 1.0)
    out_rgb = (src_c[..., :3] * sa + dst_c[..., :3] * da * (1.0 - sa)) / safe_a
    out = np.concatenate([out_rgb, out_a], axis=-1)
    dst[y0:y1, x0:x1] = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def build_canvas(
    plan: Dict[str, Any],
    output_root: str,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
    """Compose every layer of the plan and return (canvas RGBA, psd layers, notes).

    The returned ``psd_layers`` are ordered bottom -> top and contain cropped
    RGBA pixel blocks with their canvas offsets.
    """
    canvas_size = int(plan.get("canvas_size") or 2048)
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    out_dir = plan.get("out_dir", "SeeThrough/UpScale")
    src_w = int((plan.get("source") or {}).get("width") or 1024)
    src_h = int((plan.get("source") or {}).get("height") or src_w)

    psd_layers: List[Dict[str, Any]] = []
    notes: List[str] = []

    for layer in plan.get("layers", []):
        name = layer["name"]
        upscaled = find_layer_output(output_root, out_dir, name)
        display_box = layer.get("display_box") or [0, 0, 1, 1]
        left, top, right, bottom = [int(v) for v in display_box]
        dw, dh = max(1, right - left), max(1, bottom - top)

        if layer.get("empty") and not layer.get("selected"):
            # Nearly fully transparent part (no wings on this character, ...):
            # skip it entirely instead of spending time/space on a blank layer.
            notes.append("%s: empty layer skipped" % name)
            continue

        box = layer.get("box") or [0, 0, src_w, src_h]
        full_frame = (box[0] <= 0 and box[1] <= 0 and box[2] >= src_w and box[3] >= src_h)
        pixels, layer_notes = prepare_layer_pixels(
            layer["file"], upscaled, dw, dh, alpha_threshold, full_frame=full_frame
        )
        notes.extend("%s: %s" % (name, n) for n in layer_notes)
        if pixels is None:
            continue

        # Clamp into the canvas.
        x0 = max(0, left)
        y0 = max(0, top)
        x1 = min(canvas_size, left + pixels.shape[1])
        y1 = min(canvas_size, top + pixels.shape[0])
        if x1 <= x0 or y1 <= y0:
            notes.append("%s: display box outside canvas" % name)
            continue
        crop = pixels[y0 - top:y1 - top, x0 - left:x1 - left]
        _alpha_over(canvas, crop, x0, y0)

        psd_layers.append({
            "name": name,
            "left": int(x0),
            "top": int(y0),
            "pixels": crop,
            "source": upscaled or layer["file"],
            "upscaled": bool(upscaled),
        })

    return canvas, psd_layers, notes


def canvas_to_preview(canvas: np.ndarray, background: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Flatten an RGBA canvas onto a solid colour for ImageCompare."""
    arr = canvas.astype(np.float32) / 255.0
    a = arr[..., 3:4]
    bg = np.array(background, dtype=np.float32) / 255.0
    rgb = arr[..., :3] * a + bg * (1.0 - a)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def write_layers_json(
    plan: Dict[str, Any],
    psd_layers: List[Dict[str, Any]],
    path: str,
) -> str:
    """Write a SeeThrough frontend compatible layers.json (near -> far order).

    The ag-psd button draws the first entry on top, so this list is reversed
    relative to the PSD writer's bottom -> top order.
    """
    import json

    entries = []
    for layer in reversed(psd_layers):
        h, w = layer["pixels"].shape[:2]
        entries.append({
            "name": layer["name"],
            "filename": "%s/%s.png" % (plan.get("out_dir", "SeeThrough/UpScale"), safe_name(layer["name"])),
            "left": int(layer["left"]),
            "top": int(layer["top"]),
            "right": int(layer["left"] + w),
            "bottom": int(layer["top"] + h),
        })
    doc = {
        "prefix": plan.get("source", {}).get("prefix", "seethrough"),
        "timestamp": plan.get("source", {}).get("timestamp", ""),
        "width": int(plan.get("canvas_size") or 2048),
        "height": int(plan.get("canvas_size") or 2048),
        "upscaled": True,
        "layers": entries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path
