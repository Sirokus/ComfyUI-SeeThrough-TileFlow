"""CPU-side helpers for the PSD -> Qwen edit -> PSD upscale flow.

The finished piece arrives as one layered PSD (folders, blend modes, layer
opacity).  This module does everything that does not need a model:

* ``scan_psd``      read the PSD and keep its record order (bottom -> top),
                    folders included, as plain dicts;
* ``prepare_plan``  write every layer out as its own PNG, compute each
                    layer's upscale size (global factor, per-layer pixel cap)
                    and store the whole thing as ``psd_plan.json``;
* ``write_crops``   white-composited crops (plus the original alpha) that the
                    edit model will see;
* ``assemble_psd``  put the saved layer PNGs back on an upscaled canvas,
                    keeping folder markers / blend / opacity / hidden flags;
* ``preview_rgba``  a small merged preview for the node's output image.

Nothing here touches torch, so it can be tested offline with plain numpy.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

if __package__:  # ComfyUI loads the folder as a package
    from . import core_io, core_plan, psd_reader, psd_writer
else:  # standalone import used by the tests and tools
    import core_io
    import core_plan
    import psd_reader
    import psd_writer

MODE = "psd_upscale"
PLAN_NAME = "psd_plan.json"
LAYERS_SUBDIR = "layers"
CROPS_SUBDIR = "crops"
UPSCALED_SUBDIR = "upscaled"
REFS_SUBDIR = "refs"
PREVIEW_NAME = "preview.png"

DEFAULT_FACTOR = 2.0
DEFAULT_MAX_SIDE = 4096
# Context around a part that the edit model gets to see.  It keeps a limb from
# being cut at its silhouette, which is what makes the model invent a new edge.
DEFAULT_MARGIN_RATIO = 0.08
MIN_MARGIN = 32
PREVIEW_MAX_SIDE = 1280

GROUP_OPEN = "group_open"
GROUP_END = "group_end"
LAYER = "layer"


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _blend_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("latin-1", "replace")
    return str(value or "norm")


def scan_psd(path: str) -> Dict[str, Any]:
    """Read ``path`` and return ``{width, height, records}`` (bottom -> top).

    Record dicts mirror ``psd_reader`` but stay JSON friendly: blend is a
    four character string, pixels are only attached for real layers.
    """
    doc = psd_reader.read_psd(path)
    records: List[Dict[str, Any]] = []
    for layer in doc["layers"]:
        kind = layer.get("kind") or LAYER
        record = {
            "kind": kind,
            "name": str(layer.get("name") or ""),
            "blend": _blend_text(layer.get("blend")),
            "opacity": int(layer.get("opacity", 255)),
            "hidden": bool(layer.get("hidden")),
        }
        if kind == LAYER:
            record["box"] = [int(layer["left"]), int(layer["top"]),
                             int(layer["right"]), int(layer["bottom"])]
            record["pixels"] = np.asarray(layer["pixels"], dtype=np.uint8)
        records.append(record)
    return {"width": int(doc["width"]), "height": int(doc["height"]),
            "records": records}


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def parse_selection(select: str, names: Sequence[str]) -> List[str]:
    """``all`` / comma names / ``-name`` exclusions -> ordered layer names.

    Matching is case-insensitive and falls back to a unique substring match so
    a hand typed name still hits.  When nothing at all is recognised the whole
    catalogue is returned: a stale value must never silently run a partial job.
    """
    catalogue = [str(n) for n in names]
    lowered = [n.lower() for n in catalogue]
    text = str(select or "").strip()
    if not text or text.lower() == "all":
        return list(catalogue)
    if text.lower() == "none":
        return []

    tokens = [t.strip() for t in text.split(",")] if "," in text else text.split()
    chosen: List[str] = []
    removed = set()
    explicit = False
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        drop = token.startswith("-")
        if drop:
            token = token[1:].strip()
        key = token.lower()
        if key == "all":
            explicit = True
            for name in catalogue:
                if name not in chosen:
                    chosen.append(name)
            continue
        hits = [n for n, low in zip(catalogue, lowered) if low == key]
        if not hits:
            hits = [n for n, low in zip(catalogue, lowered) if key and key in low]
        if not hits:
            continue
        explicit = True
        if drop:
            removed.update(hits)
        else:
            for name in hits:
                if name not in chosen:
                    chosen.append(name)
    if not explicit:
        return list(catalogue)
    return [name for name in catalogue if name in chosen and name not in removed]


def unique_stems(names: Iterable[str]) -> List[str]:
    """One safe, unique file stem per layer name (duplicates get ``_2``)."""
    used: Dict[str, int] = {}
    stems: List[str] = []
    for name in names:
        base = core_plan.safe_name(name)
        count = used.get(base.lower(), 0) + 1
        used[base.lower()] = count
        stems.append(base if count == 1 else "%s_%d" % (base, count))
    return stems


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def layer_stem(plan: Dict[str, Any], index: int) -> str:
    return plan["layers"][index]["stem"]


def plan_layer_paths(out_dir_abs: str, stem: str) -> Dict[str, str]:
    return {
        "source": os.path.join(out_dir_abs, LAYERS_SUBDIR, stem + ".png"),
        "crop": os.path.join(out_dir_abs, CROPS_SUBDIR, stem + "_crop.png"),
        "upscaled": os.path.join(out_dir_abs, UPSCALED_SUBDIR, stem + ".png"),
        "ref": os.path.join(out_dir_abs, REFS_SUBDIR, stem + "_ref.png"),
    }


def _targets(width: int, height: int, factor: float, max_side: int) -> Tuple[Tuple[int, int], float, bool]:
    """Per-layer upscale size: global factor, clamped by ``max_side``."""
    factor = max(0.1, float(factor))
    longest = max(1, max(width, height))
    scale = factor
    capped = False
    if max_side and longest * factor > max_side:
        scale = float(max_side) / float(longest)
        capped = scale < factor - 1e-6
    return ((max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            scale, capped)


def prepare_plan(
    psd_path: str,
    out_dir_abs: str,
    factor: float = DEFAULT_FACTOR,
    max_side: int = DEFAULT_MAX_SIDE,
    select: str = "all",
) -> Dict[str, Any]:
    """Write every layer PNG and return the JSON-safe plan document."""
    psd_path = os.path.abspath(psd_path)
    if not os.path.isfile(psd_path):
        raise ValueError("PSD 文件不存在: %s" % psd_path)
    scanned = scan_psd(psd_path)
    width, height = scanned["width"], scanned["height"]
    if width <= 0 or height <= 0:
        raise ValueError("PSD 画布尺寸无效")

    layer_names = [r["name"] for r in scanned["records"] if r["kind"] == LAYER]
    selected = set(parse_selection(select, layer_names))
    stems = unique_stems(layer_names)

    canvas = (max(1, int(round(width * max(0.1, float(factor))))),
              max(1, int(round(height * max(0.1, float(factor))))))
    layers: List[Dict[str, Any]] = []
    stem_index = 0
    for record in scanned["records"]:
        name = record["name"]
        item = {
            "kind": record["kind"],
            "name": name,
            "blend": record["blend"],
            "opacity": record["opacity"],
            "hidden": record["hidden"],
        }
        if record["kind"] == LAYER:
            stem = stems[stem_index]
            stem_index += 1
            pixels = record["pixels"]
            h, w = pixels.shape[:2]
            if w <= 0 or h <= 0:
                continue
            box = record["box"]
            target, scale, capped = _targets(w, h, factor, max_side)
            paths = plan_layer_paths(out_dir_abs, stem)
            core_io.write_rgba(paths["source"], pixels)
            item.update({
                "stem": stem,
                "box": box,
                "size": [w, h],
                "target": [int(target[0]), int(target[1])],
                "scale": round(float(scale), 6),
                "capped": bool(capped),
                "file": paths["source"],
                "crop_file": paths["crop"],
                "out_file": paths["upscaled"],
                "selected": name in selected,
            })
        layers.append(item)

    plan = {
        "version": 1,
        "mode": MODE,
        "psd": psd_path,
        "out_dir": os.path.abspath(out_dir_abs),
        "source": {"width": width, "height": height},
        "canvas": [canvas[0], canvas[1]],
        "factor": float(factor),
        "max_side": int(max_side),
        "select": str(select or "all"),
        "layers": layers,
        "count": len([L for L in layers if L.get("selected")]),
    }
    write_plan(out_dir_abs, plan)
    return plan


def plan_path(out_dir_abs: str) -> str:
    return os.path.join(out_dir_abs, PLAN_NAME)


def write_plan(out_dir_abs: str, plan: Dict[str, Any]) -> str:
    os.makedirs(out_dir_abs, exist_ok=True)
    path = plan_path(out_dir_abs)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=1)
    return path


def load_plan(ref: str) -> Dict[str, Any]:
    """Accept the plan document itself, a plan JSON path or a layer folder."""
    if isinstance(ref, dict):
        return ref
    text = str(ref or "").strip()
    if text.startswith("{"):
        return json.loads(text)
    if os.path.isdir(text):
        text = plan_path(text)
    with open(text, "r", encoding="utf-8") as handle:
        return json.load(handle)


def selected_layers(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [layer for layer in plan["layers"]
            if layer.get("kind") == LAYER and layer.get("selected")]


def all_layer_records(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [layer for layer in plan["layers"] if layer.get("kind") == LAYER]


def capped_report(plan: Dict[str, Any]) -> str:
    capped = [layer["name"] for layer in all_layer_records(plan) if layer.get("capped")]
    if not capped:
        return ""
    return ("以下图层超过单层像素上限 %d，使用 %.2f 倍: %s"
            % (int(plan.get("max_side") or 0),
               min(layer["scale"] for layer in all_layer_records(plan)
                   if layer.get("capped")),
               "、".join(capped)))


# ---------------------------------------------------------------------------
# crops
# ---------------------------------------------------------------------------

def crop_box(box: Sequence[int], size: Tuple[int, int], margin: int) -> Tuple[int, int, int, int]:
    """Content box grown by ``margin`` px, clamped inside the canvas."""
    width, height = size
    left, top, right, bottom = (int(v) for v in box)
    x0 = max(0, left - int(margin))
    y0 = max(0, top - int(margin))
    x1 = min(int(width), right + int(margin))
    y1 = min(int(height), bottom + int(margin))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def write_crops(
    plan: Dict[str, Any],
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
) -> List[Dict[str, Any]]:
    """White-composited crop PNG per selected layer; returns crop jobs.

    The PNG keeps the source alpha inside the crop, so the tail of the edit
    chain can restore the exact silhouette after the model returns.
    """
    width = int(plan["source"]["width"])
    height = int(plan["source"]["height"])
    jobs: List[Dict[str, Any]] = []
    for layer in selected_layers(plan):
        pixels = core_io.read_rgba(layer["file"])
        h, w = pixels.shape[:2]
        margin = max(MIN_MARGIN, int(round(max(w, h) * max(0.0, float(margin_ratio)))))
        x0, y0, x1, y1 = crop_box(layer["box"], (width, height), margin)
        crop_w, crop_h = x1 - x0, y1 - y0

        crop = np.zeros((crop_h, crop_w, 4), np.uint8)
        crop[..., :3] = 255
        left, top = int(layer["box"][0]), int(layer["box"][1])
        px, py = left - x0, top - y0
        alpha = pixels[..., 3:4].astype(np.float32) / 255.0
        colour = (pixels[..., :3].astype(np.float32) * alpha +
                  255.0 * (1.0 - alpha))
        crop[py:py + h, px:px + w, :3] = np.clip(np.rint(colour), 0, 255).astype(np.uint8)
        crop[py:py + h, px:px + w, 3] = pixels[..., 3]
        core_io.write_rgba(layer["crop_file"], crop)
        jobs.append({
            "name": layer["name"],
            "stem": layer["stem"],
            "crop_file": layer["crop_file"],
            "crop_box": [x0, y0, x1, y1],
            "crop_size": [crop_w, crop_h],
            "target": [int(layer["target"][0]), int(layer["target"][1])],
            "out_file": layer["out_file"],
        })
    return jobs


def write_references(plan: Dict[str, Any], reference_path: str) -> List[Dict[str, Any]]:
    """Cut every crop box out of the high-resolution reference image.

    The reference is assumed to show the same frame as the PSD, just at a
    different resolution, so each axis maps linearly onto the canvas.  A cut
    is written white-flattened and fully opaque: it is a *colour guide* for
    the edit model, the silhouette still comes from the layer itself.
    """
    path = os.path.abspath(str(reference_path or "").strip().strip('"'))
    if not path or not os.path.isfile(path):
        raise ValueError("参考原图不存在: %s" % (path or "<empty>"))
    ref = core_io.read_rgba(path)
    ref_h, ref_w = ref.shape[:2]
    src_w = max(1, int(plan["source"]["width"]))
    src_h = max(1, int(plan["source"]["height"]))
    kx, ky = ref_w / float(src_w), ref_h / float(src_h)

    jobs: List[Dict[str, Any]] = []
    for job in plan.get("crops") or []:
        x0, y0, x1, y1 = (int(v) for v in job["crop_box"])
        rx0 = max(0, min(ref_w - 1, int(round(x0 * kx))))
        ry0 = max(0, min(ref_h - 1, int(round(y0 * ky))))
        rx1 = max(rx0 + 1, min(ref_w, int(round(x1 * kx))))
        ry1 = max(ry0 + 1, min(ref_h, int(round(y1 * ky))))
        window = ref[ry0:ry1, rx0:rx1]
        alpha = window[..., 3:4].astype(np.float32) / 255.0
        colour = (window[..., :3].astype(np.float32) * alpha +
                  255.0 * (1.0 - alpha))
        flat = np.clip(np.rint(colour), 0, 255).astype(np.uint8)
        out = np.dstack([flat, np.full(flat.shape[:2], 255, np.uint8)])
        ref_file = (job.get("ref_file") or
                    plan_layer_paths(str(plan["out_dir"]), str(job["stem"]))["ref"])
        core_io.write_rgba(ref_file, out)
        job["ref_file"] = ref_file
        job["ref_box"] = [rx0, ry0, rx1, ry1]
        jobs.append(job)

    plan["reference"] = {
        "file": path,
        "size": [ref_w, ref_h],
        "scale": [round(kx, 6), round(ky, 6)],
        "count": len(jobs),
    }
    return jobs


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------

def work_size(crop_size: Sequence[int], size: Sequence[int],
              target: Sequence[int]) -> Tuple[int, int]:
    """Crop size *after* its layer is scaled to the target size.

    The edit model gets the crop upscaled to this size, so it never edits a
    16 px eye at 16 px.
    """
    cw = max(1, int(crop_size[0]))
    ch = max(1, int(crop_size[1]))
    w = max(1, int(size[0]))
    h = max(1, int(size[1]))
    tw = max(1, int(target[0]))
    th = max(1, int(target[1]))
    return (max(1, int(round(cw * tw / float(w)))),
            max(1, int(round(ch * th / float(h)))))


def restore_layer(
    edited: np.ndarray,
    alpha: np.ndarray,
    crop_size: Sequence[int],
    crop_box: Sequence[int],
    box: Sequence[int],
    size: Sequence[int],
    target: Sequence[int],
) -> np.ndarray:
    """Cut the edited crop back to one target-sized RGBA layer.

    ``edited`` is the model's crop result (RGB or RGBA), ``alpha`` the source
    crop's alpha at ``crop_size``.  The crop is resampled to the layer's
    upscaled scale, clipped by the real silhouette and the white context is
    un-composited, so the layer keeps straight alpha without a white halo.

    When ``alpha`` already has the target size it is used as-is: it is then
    the layer's own silhouette resampled once, instead of the crop alpha
    bounced through the work size (that bounce softens a hard edge by a
    pixel).
    """
    work_w, work_h = work_size(crop_size, size, target)
    target_w = max(1, int(target[0]))
    target_h = max(1, int(target[1]))
    sx = target_w / float(max(1, int(size[0])))
    sy = target_h / float(max(1, int(size[1])))

    edited = np.asarray(edited, dtype=np.uint8)
    if edited.ndim == 3 and edited.shape[-1] >= 3:
        rgb = np.ascontiguousarray(edited[..., :3])
    else:
        rgb = np.repeat(edited[..., None], 3, axis=2)
    stacked = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    up = core_io.resize_rgba(stacked, work_w, work_h)
    aligned = np.asarray(alpha).shape[:2] == (target_h, target_w)
    a_up = (np.asarray(alpha, dtype=np.float32) if aligned
            else core_io.resize_alpha(np.asarray(alpha, dtype=np.uint8),
                                      work_w, work_h).astype(np.float32))

    left, top = int(box[0]), int(box[1])
    ox = int(round((left - int(crop_box[0])) * sx))
    oy = int(round((top - int(crop_box[1])) * sy))

    out = np.zeros((target_h, target_w, 4), np.uint8)
    cx0, cy0 = max(0, ox), max(0, oy)
    cx1, cy1 = min(work_w, ox + target_w), min(work_h, oy + target_h)
    if cx1 <= cx0 or cy1 <= cy0:
        return out
    dx, dy = cx0 - ox, cy0 - oy
    src = up[cy0:cy1, cx0:cx1, :3].astype(np.float32)
    if aligned:
        a = a_up[dy:dy + (cy1 - cy0), dx:dx + (cx1 - cx0)] / 255.0
    else:
        a = a_up[cy0:cy1, cx0:cx1] / 255.0
    # Undo the white compositing the crop went through: rgb = white*(1-a) +
    # layer*a, so layer = (rgb - white*(1-a)) / a.
    vis = a > 1e-3
    unmixed = np.zeros_like(src)
    unmixed[vis] = ((src[vis] - 255.0 * (1.0 - a[vis][:, None]))
                    / a[vis][:, None])
    out[dy:dy + (cy1 - cy0), dx:dx + (cx1 - cx0), :3] = np.clip(
        np.rint(unmixed), 0, 255).astype(np.uint8)
    out[dy:dy + (cy1 - cy0), dx:dx + (cx1 - cx0), 3] = np.clip(
        np.rint(a * 255.0), 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def _resize(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    width, height = max(1, int(width)), max(1, int(height))
    if pixels.shape[:2] == (height, width):
        return pixels
    return core_io.resize_rgba_premultiplied(pixels, width, height)


def assemble_psd(plan: Dict[str, Any], out_psd: str, compression: str = "rle") -> Dict[str, Any]:
    """Write the upscaled layered PSD; returns a small report.

    Every record keeps its name, blend, opacity and hidden flag.  A layer whose
    saved PNG is missing (not edited / not selected) is resampled from the
    original with Lanczos, so an unedited run still produces a complete file.
    """
    canvas_w, canvas_h = (int(v) for v in plan["canvas"])
    src_w = max(1, int(plan["source"]["width"]))
    src_h = max(1, int(plan["source"]["height"]))
    sx = canvas_w / float(src_w)
    sy = canvas_h / float(src_h)

    records: List[Dict[str, Any]] = []
    used: List[str] = []
    fallback: List[str] = []
    for layer in plan["layers"]:
        kind = layer.get("kind")
        if kind in (GROUP_OPEN, GROUP_END):
            records.append({
                "type": psd_writer.GROUP_START if kind == GROUP_OPEN else psd_writer.GROUP_END,
                "name": layer["name"],
                "blend": layer.get("blend", "norm"),
                "opacity": int(layer.get("opacity", 255)),
                "hidden": bool(layer.get("hidden")),
            })
            continue
        if kind != LAYER:
            continue
        left, top, right, bottom = (int(v) for v in layer["box"])
        x0 = int(round(left * sx))
        y0 = int(round(top * sy))
        x1 = int(round(right * sx))
        y1 = int(round(bottom * sy))
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)

        source = layer.get("out_file") if os.path.isfile(layer.get("out_file") or "") else layer["file"]
        if source == layer.get("out_file"):
            used.append(layer["name"])
        else:
            fallback.append(layer["name"])
        pixels = core_io.read_rgba(source)
        pixels = _resize(pixels, width, height)
        records.append({
            "name": layer["name"],
            "left": x0,
            "top": y0,
            "pixels": pixels,
            "blend": layer.get("blend", "norm"),
            "opacity": int(layer.get("opacity", 255)),
            "hidden": bool(layer.get("hidden")),
        })

    os.makedirs(os.path.dirname(os.path.abspath(out_psd)), exist_ok=True)
    psd_writer.write_psd(out_psd, records, canvas_w, canvas_h, compression=compression)
    return {
        "path": out_psd,
        "canvas": [canvas_w, canvas_h],
        "upscaled": used,
        "fallback": fallback,
        "fallback_count": len(fallback),
    }


def preview_rgba(plan: Dict[str, Any], max_side: int = PREVIEW_MAX_SIDE) -> np.ndarray:
    """Merged preview of the plan's layers (honours hidden + opacity)."""
    canvas_w, canvas_h = (int(v) for v in plan["canvas"])
    scale = min(1.0, float(max_side) / float(max(canvas_w, canvas_h)))
    width = max(1, int(round(canvas_w * scale)))
    height = max(1, int(round(canvas_h * scale)))
    src_w = max(1, int(plan["source"]["width"]))
    src_h = max(1, int(plan["source"]["height"]))
    sx = width / float(src_w)
    sy = height / float(src_h)

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    suppressed = psd_writer.hidden_record_mask(plan["layers"])
    for index, layer in enumerate(plan["layers"]):
        if layer.get("kind") != LAYER or layer.get("hidden") or suppressed[index]:
            continue
        left, top, right, bottom = (int(v) for v in layer["box"])
        w = max(1, int(round(right * sx)) - int(round(left * sx)))
        h = max(1, int(round(bottom * sy)) - int(round(top * sy)))
        source = layer.get("out_file") if os.path.isfile(layer.get("out_file") or "") else layer["file"]
        pixels = _resize(core_io.read_rgba(source), w, h)
        opacity = int(layer.get("opacity", 255))
        if opacity < 255:
            pixels = pixels.copy()
            pixels[..., 3] = (pixels[..., 3].astype(np.uint16) * max(0, opacity) // 255).astype(np.uint8)
        canvas.alpha_composite(Image.fromarray(pixels, "RGBA"),
                               (int(round(left * sx)), int(round(top * sy))))
    return np.array(canvas, dtype=np.uint8)


# ---------------------------------------------------------------------------
# sampler options
# ---------------------------------------------------------------------------

def scale_sigmas(text: str, denoise: float) -> str:
    """Scale a sigma schedule by ``denoise`` (0.45 keeps 45 % of the steps)."""
    strength = max(0.0, min(1.0, float(denoise)))
    values: List[float] = []
    for token in str(text or "").replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token) * strength)
        except ValueError:
            continue
    if not values:
        return str(text or "")
    return ", ".join("%.6g" % v for v in values)
