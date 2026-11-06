"""Pure-python planning logic for the SeeThrough layer upscale batch.

No ComfyUI, torch or PIL imports here so the module can be unit tested
standalone.  All paths are resolved against an explicit ``output_root`` that
the caller supplies (normally ``folder_paths.get_output_directory()``).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# With no rule at all a layer is rendered at the PSD display scale
# (canvas_size / source canvas), so no work is wasted on over-large rasters.
# Tiny face parts get a fixed long side so they still carry enough detail.
DEFAULT_SIZE_RULES: Dict[str, Any] = {
    "*": {},
    "eyebrow-*": {"long_side": 512},
    "eyelash-*": {"long_side": 512},
    "irides-*": {"long_side": 512},
    "eyewhite-*": {"long_side": 512},
    "ears-*": {"long_side": 512},
}
DEFAULT_CANVAS_SIZE = 2048
MIN_SIDE = 128
MAX_SIDE = 8192

# Local crop-and-paste-back refinement.  Feeding a 17x15 pupil to an edit model
# as a whole frame makes it complete the frame (it invents a head); feeding it
# a crop of the surrounding character keeps the edit inside the part.
FOCUS_MARGIN = 32
# Reference images are resized to about resolution x resolution by the encoder;
# 1024 is the size the Qwen edit model was trained at.
FOCUS_RESOLUTION = 1024
DEFAULT_FOCUS_PROMPT = (
    "Upscale <image1>, a cropped region of a character illustration. Keep the exact "
    "composition, framing, colors and shapes. Do NOT add, remove or extend anything. "
    "Enhance only sharpness and fine detail."
)
# Qwen reference images are resized to multiples of 32 and the VAE prefers
# 32-aligned lattices, so plan targets are rounded to 32 rather than 8.
STEP = 32

# Windows device names that cannot be used as a file stem.
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *{"COM%d" % i for i in range(1, 10)},
    *{"LPT%d" % i for i in range(1, 10)},
}
_ILLEGAL_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def round_step(value: float, step: int = STEP) -> int:
    """Round to the nearest multiple of ``step`` (never returns 0)."""
    return max(step, int(round(float(value) / step)) * step)


def clamp_side(value: float) -> int:
    return max(MIN_SIDE, min(MAX_SIDE, round_step(value)))


def safe_name(name: str) -> str:
    """Turn a layer name into a Windows-safe file stem (spaces / CJK kept)."""
    stem = _ILLEGAL_FILENAME_CHARS.sub("_", str(name)).strip().rstrip(". ")
    if not stem or stem in (".", ".."):
        stem = "_"
    if stem.split(".")[0].upper() in _RESERVED_NAMES:
        stem = "_" + stem
    return stem


AUTO_OUT_SUBDIR = "UpScale"
# Early versions hard-coded this as the widget default.  It now means "自动"
# as well, so a workflow saved back then starts writing to
# <拆层文件夹>/UpScale without being edited first.
LEGACY_OUT_DIR = "SeeThrough/UpScale"


def normalize_out_dir(out_dir: str) -> str:
    """Validate a workspace-relative output sub directory.

    Rejects absolute paths, drive letters, UNC paths and any ``..`` traversal.
    Returns a forward-slash relative path; an empty value stays empty, which
    means "自动" (see :func:`resolve_out_dir`).
    """
    raw = (out_dir or "").strip().replace("\\", "/")
    if not raw or raw.lower() == LEGACY_OUT_DIR.lower():
        return ""
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw) or raw.startswith("//"):
        raise ValueError("out_dir must be relative to the ComfyUI output directory: %r" % out_dir)
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError("out_dir may not contain '..': %r" % out_dir)
    if not parts:
        raise ValueError("out_dir is empty: %r" % out_dir)
    return "/".join(parts)


def auto_out_dir(source_dir: Optional[str], output_root: str) -> str:
    """``<拆层文件夹>/UpScale`` as a path relative to the output root.

    Each split folder keeps its own upscale output, so two characters that
    live in different sub folders never overwrite each other.  When the
    source folder is unknown or lies outside the output directory there is no
    sensible relative path; fall back to a plain ``UpScale`` at the root.
    """
    if not source_dir:
        return AUTO_OUT_SUBDIR
    try:
        rel = os.path.relpath(os.path.abspath(source_dir), os.path.abspath(output_root))
    except ValueError:  # different drives on Windows
        return AUTO_OUT_SUBDIR
    rel = rel.replace("\\", "/").strip("/")
    if not rel or rel == "." or rel.startswith(".."):
        return AUTO_OUT_SUBDIR
    return rel + "/" + AUTO_OUT_SUBDIR


def auto_basis(doc: Dict[str, Any]) -> Optional[str]:
    """Folder the auto ``UpScale`` folder belongs to.

    That is the folder holding the layers.json -- not necessarily the folder
    the user browsed to, which may be a parent holding one sub folder per
    character.  Bare-PNG folders have no layers.json, so the browsed folder
    itself is used there.
    """
    path = str(doc.get("path") or "")
    if path and os.path.isfile(path):
        return os.path.dirname(os.path.abspath(path))
    return doc.get("source_dir")


def resolve_out_dir(
    out_dir: str,
    source_dir: Optional[str],
    output_root: str,
) -> str:
    """Return the effective relative output folder.

    An explicit ``out_dir`` wins; empty ("自动") means the ``UpScale`` sub
    folder of the split folder being processed.
    """
    return normalize_out_dir(out_dir) or auto_out_dir(source_dir, output_root)


def resolve_path(path: str, output_root: str) -> str:
    """Resolve ``path`` relative to output_root unless it is absolute."""
    path = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(output_root, path))


def _is_upscale_doc(path: str) -> bool:
    """True for layers.json files produced by the PSD assembler."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        return bool(doc.get("upscaled"))
    except Exception:  # noqa: BLE001
        return False


def find_latest_layers_json(output_root: str) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    for root, _dirs, files in os.walk(output_root):
        for f in files:
            if f.lower().endswith("_layers.json") and not f.lower().endswith("_psdinfo.json"):
                p = os.path.join(root, f)
                if _is_upscale_doc(p):
                    continue
                try:
                    candidates.append((os.path.getmtime(p), p))
                except OSError:
                    continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def file_fingerprint(path: str) -> str:
    try:
        st = os.stat(path)
        return "%s:%d:%d" % (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return "missing:%s" % path


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


# ---------------------------------------------------------------------------
# layers.json
# ---------------------------------------------------------------------------

_ALPHA_CACHE: Dict[str, Tuple[Tuple[int, int], Tuple[Optional[float], Optional[int], Optional[int], Optional[float]]]] = {}

# SeeThrough leaves alpha 1..15 noise on the otherwise empty background of every
# layer.  Anything above this is real content.
VISIBLE_ALPHA = 32
# Connected blobs smaller than this are stray specks, not a part.  A real but
# faint part (the nose shadow is only alpha 28..50) is a compact blob of a
# dozen pixels and must survive.
MIN_COMPONENT = 4
# Fewer kept pixels than this means the layer is genuinely empty.
EMPTY_PIXELS = 4


def _alpha_stats(img_path: str) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[float]]:
    """Return (coverage, max alpha, visible pixel count, fraction of the file).

    ``coverage`` is the fraction of the part's own tight bbox that is opaque,
    not the fraction of the canvas.  Small parts (a mouth, an iris) live in
    full-canvas layers, and measuring them against 1024x1024 marked them
    "empty" -- dropping the mouth from the PSD and exposing the pale fill
    underneath it.  Dust is below ``VISIBLE_ALPHA`` and does not count.

    ``fraction`` is the visible share of the whole file (the canvas for
    full-canvas layers).  It is small for the mouth/nose layers, whose box is
    the full canvas even though the part itself is tiny; they must still be
    composited, but regenerating a whole frame around them would only make the
    model invent the character.

    Missing/broken files return (None, None, None, None).  The result is cached
    per (path, mtime, size): the Plan panel calls this for every layer on each
    refresh, and decoding ~30 full-size PNGs per keystroke would crawl.
    """
    try:
        st = os.stat(img_path)
    except OSError:
        return None, None, None, None
    key = os.path.abspath(img_path)
    stamp = (int(st.st_mtime_ns), int(st.st_size))
    cached = _ALPHA_CACHE.get(key)
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        from PIL import Image
        import numpy as np

        with Image.open(img_path) as im:
            alpha = np.asarray(im.convert("RGBA"))[..., 3]
        visible = _drop_specks(alpha > VISIBLE_ALPHA)
        pixels = int(visible.sum())
        if pixels:
            ys, xs = np.nonzero(visible)
            area = ((int(xs.max()) - int(xs.min()) + 1)
                    * (int(ys.max()) - int(ys.min()) + 1))
            coverage = pixels / float(max(1, area))
        else:
            coverage = 0.0
        fraction = pixels / float(max(1, alpha.shape[0] * alpha.shape[1]))
        result = (float(coverage), int(alpha.max()), pixels, float(fraction))
    except Exception:  # noqa: BLE001
        result = (None, None, None, None)
    if len(_ALPHA_CACHE) > 2000:
        _ALPHA_CACHE.clear()
    _ALPHA_CACHE[key] = (stamp, result)
    return result


def _drop_specks(mask):
    """Remove isolated specks, keeping every compact blob (dust cleanup).

    SeeThrough marks stray pixels of the empty background with alpha 17..32;
    without this a 3-pixel speck in a corner stretches the bbox over the whole
    canvas, the fill ratio collapses, and a real part is called "empty" and
    dropped from the PSD.
    """
    import numpy as np

    if not mask.any():
        return mask
    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask)
        if count <= 0:
            return mask
        sizes = np.bincount(labels.ravel())
        keep_ids = np.nonzero(sizes >= MIN_COMPONENT)[0]
        keep_ids = keep_ids[keep_ids != 0]
        if keep_ids.size == 0:
            return np.zeros_like(mask)
        return np.isin(labels, keep_ids) & mask
    except Exception:  # noqa: BLE001 - scipy is optional
        return mask


def _is_empty(coverage: Optional[float], pixels: Optional[int],
              threshold: float = 0.0005) -> bool:
    if pixels is None or coverage is None:
        return False
    return int(pixels) < EMPTY_PIXELS or float(coverage) < float(threshold)


def find_layers_json_in_dir(directory: str, depth: int = 2) -> Optional[str]:
    """Newest ``*_layers.json`` in ``directory`` (optionally one level down)."""
    if not os.path.isdir(directory):
        return None
    candidates: List[Tuple[float, str]] = []
    base_depth = directory.rstrip("\\/").count(os.sep)
    for root, dirs, files in os.walk(directory):
        if root.rstrip("\\/").count(os.sep) - base_depth >= depth:
            dirs[:] = []
            continue
        for f in files:
            if f.lower().endswith("_layers.json") and not f.lower().endswith("_psdinfo.json"):
                p = os.path.join(root, f)
                if _is_upscale_doc(p):
                    continue
                try:
                    candidates.append((os.path.getmtime(p), p))
                except OSError:
                    continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _png_sizes(path: str) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:  # noqa: BLE001
        return None


def synthesize_doc_from_pngs(directory: str) -> Dict[str, Any]:
    """Build a layers document from bare PNGs when no layers.json exists.

    Positioning information is unavailable, so every layer is placed at the
    origin with its own size and z order follows the file name.  This keeps
    the batch usable for folders produced by other tools.
    """
    files = sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(".png") and not f.lower().endswith("_depth.png")
    )
    if not files:
        raise ValueError("no PNG layers found in %s" % directory)

    entries: List[Dict[str, Any]] = []
    canvas_w = canvas_h = 0
    for idx, filename in enumerate(files):
        path = os.path.join(directory, filename)
        size = _png_sizes(path)
        if not size:
            continue
        w, h = size
        canvas_w = max(canvas_w, w)
        canvas_h = max(canvas_h, h)
        coverage, alpha_max, alpha_pixels, alpha_fraction = _alpha_stats(path)
        entries.append({
            "name": os.path.splitext(filename)[0],
            "index": idx,
            "file": os.path.abspath(path),
            "filename": filename,
            "box": [0, 0, w, h],
            "depth_median": 1.0 - (idx / max(1, len(files))),
            "depth_filename": None,
            "alpha_coverage": coverage,
            "alpha_max": alpha_max,
            "alpha_pixels": alpha_pixels,
            "alpha_fraction": alpha_fraction,
            "empty": _is_empty(coverage, alpha_pixels),
            "synthetic": True,
        })
    if not entries:
        raise ValueError("no readable PNG layers found in %s" % directory)
    return {
        "path": os.path.abspath(directory),
        "prefix": os.path.basename(directory.rstrip("\\/")) or "layers",
        "timestamp": "",
        "width": canvas_w or 1024,
        "height": canvas_h or 1024,
        "layers": entries,
        "warnings": ["no *_layers.json found; positions default to 0,0"],
        "synthetic": True,
    }


def resolve_layers_source(source: str, output_root: str) -> Dict[str, Any]:
    """Resolve a folder / file / empty value into a layers document.

    * folder  -> newest ``*_layers.json`` inside (or a synthesized document)
    * file    -> that layers.json
    * empty   -> newest ``*_layers.json`` anywhere under output/
    """
    text = (source or "").strip()
    path = resolve_path(text, output_root) if text else None

    if path and os.path.isdir(path):
        found = find_layers_json_in_dir(path)
        if found:
            doc = load_layers_json(found)
            doc["source_dir"] = os.path.abspath(path)
            return doc
        doc = synthesize_doc_from_pngs(path)
        doc["source_dir"] = os.path.abspath(path)
        return doc

    if path and os.path.isfile(path):
        doc = load_layers_json(path)
        doc["source_dir"] = os.path.dirname(os.path.abspath(path))
        return doc

    if text:
        # A non-empty path that does not exist is a typo/selection mistake;
        # do not silently fall back to some other run's layers.json.
        raise ValueError("path not found: %s" % path)

    latest = find_latest_layers_json(output_root)
    if latest:
        doc = load_layers_json(latest)
        doc["source_dir"] = os.path.dirname(os.path.abspath(latest))
        return doc

    raise ValueError("no layers.json / PNG layers found for %r" % source)


def load_layers_json(path: str) -> Dict[str, Any]:
    """Parse a SeeThrough ``*_layers.json`` file.

    Returns a normalised document with absolute layer image paths and an
    explicit bbox for every layer.
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    layers_in = doc.get("layers")
    if not isinstance(layers_in, list) or not layers_in:
        raise ValueError("layers.json has no layers: %s" % path)

    canvas_w = int(doc.get("width") or 1024)
    canvas_h = int(doc.get("height") or canvas_w)
    base_dir = os.path.dirname(os.path.abspath(path))
    warnings: List[str] = []
    layers: List[Dict[str, Any]] = []

    for idx, item in enumerate(layers_in):
        name = str(item.get("name") or "").strip()
        if not name:
            warnings.append("layer #%d has no name, skipped" % idx)
            continue
        filename = item.get("filename")
        if not filename:
            warnings.append("layer %r has no filename" % name)
            continue
        img_path = filename if os.path.isabs(filename) else os.path.join(base_dir, filename)
        img_path = os.path.normpath(img_path)
        if not os.path.isfile(img_path):
            warnings.append("layer %r image not found: %s" % (name, img_path))

        left = int(item.get("left") or 0)
        top = int(item.get("top") or 0)
        right = int(item.get("right") or 0)
        bottom = int(item.get("bottom") or 0)
        if right <= left or bottom <= top:
            # Fall back to the full canvas when the bbox is missing/empty.
            left, top, right, bottom = 0, 0, canvas_w, canvas_h
            warnings.append("layer %r has an empty bbox, using full canvas" % name)
        left = max(0, min(canvas_w, left))
        top = max(0, min(canvas_h, top))
        right = max(left + 1, min(canvas_w, right))
        bottom = max(top + 1, min(canvas_h, bottom))

        try:
            depth = float(item.get("depth_median", 1.0))
        except (TypeError, ValueError):
            depth = 1.0

        coverage, alpha_max, alpha_pixels, alpha_fraction = _alpha_stats(img_path)
        layers.append({
            "name": name,
            "index": idx,
            "file": img_path,
            "filename": filename,
            "box": [left, top, right, bottom],
            "depth_median": depth,
            "depth_filename": item.get("depth_filename"),
            "alpha_coverage": coverage,
            "alpha_max": alpha_max,
            "alpha_pixels": alpha_pixels,
            "alpha_fraction": alpha_fraction,
            "empty": _is_empty(coverage, alpha_pixels),
        })

    if not layers:
        raise ValueError("layers.json contains no usable layers: %s" % path)

    return {
        "path": os.path.abspath(path),
        "prefix": doc.get("prefix", "seethrough"),
        "timestamp": doc.get("timestamp", ""),
        "width": canvas_w,
        "height": canvas_h,
        "layers": layers,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def _matches(name: str, pattern: str) -> bool:
    pattern = pattern.strip()
    if not pattern:
        return False
    if len(pattern) > 2 and pattern.startswith("/") and pattern.endswith("/"):
        try:
            return re.search(pattern[1:-1], name, re.IGNORECASE) is not None
        except re.error:
            return False
    if any(ch in pattern for ch in "*?["):
        return fnmatch.fnmatchcase(name.lower(), pattern.lower())
    return name.lower() == pattern.lower()


def _parse_patterns(text: str) -> Tuple[List[str], List[str]]:
    """Split a comma separated list into (positive, negative) patterns.

    ``!pattern`` (or a leading ``-``) marks an exclusion.
    """
    pos, neg = [], []
    for raw in str(text or "").split(","):
        item = raw.strip()
        if not item:
            continue
        if item.startswith("!") or item.startswith("-"):
            item = item[1:].strip()
            if item:
                neg.append(item)
        else:
            pos.append(item)
    return pos, neg


def select_layer_names(names: Iterable[str], select: str, exclude: str = "") -> List[str]:
    """Return the selected layer names, preserving the input order."""
    names = list(names)
    pos, neg = _parse_patterns(select)
    extra_pos, extra_neg = _parse_patterns(exclude)
    # Everything in ``exclude`` is an exclusion, whether or not it is prefixed.
    neg.extend(extra_pos)
    neg.extend(extra_neg)

    select_all = not pos or any(p.lower() in ("all", "*") for p in pos)

    selected: List[str] = []
    for name in names:
        if select_all:
            chosen = True
        else:
            chosen = any(_matches(name, p) for p in pos)
        if chosen and any(_matches(name, p) for p in neg):
            chosen = False
        if chosen:
            selected.append(name)
    return selected


# ---------------------------------------------------------------------------
# size rules
# ---------------------------------------------------------------------------

def parse_size_rules(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse the JSON ``size_rules`` widget into ordered (glob, settings)."""
    raw = (text or "").strip()
    if not raw:
        return [(k, dict(v)) for k, v in DEFAULT_SIZE_RULES.items()]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("size_rules is not valid JSON: %s" % exc)
    if not isinstance(data, dict):
        raise ValueError("size_rules must be a JSON object {glob: settings}")

    rules: List[Tuple[str, Dict[str, Any]]] = []
    for key, value in data.items():
        if isinstance(value, (int, float)):
            settings: Dict[str, Any] = {"long_side": float(value)}
        elif isinstance(value, dict):
            settings = dict(value)
        else:
            raise ValueError("size_rules[%r] must be a number or object" % key)
        rules.append((str(key), settings))
    return rules


def merge_size_settings(name: str, rules: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Merge every matching rule for ``name``; later rules win."""
    merged: Dict[str, Any] = {}
    for pattern, settings in rules:
        if _matches(name, pattern):
            merged.update(settings)
    return merged


def compute_target_size(
    lw: int,
    lh: int,
    settings: Dict[str, Any],
    base_factor: float = 1.0,
) -> Tuple[int, int]:
    """Work out the upscaled raster size for a layer of width ``lw`` x ``lh``.

    ``base_factor`` is the PSD display scale (canvas / source canvas).  A layer
    with no explicit rule is rendered at exactly that scale, which keeps the
    raster in sync with the PSD and avoids over-rendering small parts.
    """
    lw = max(1, int(lw))
    lh = max(1, int(lh))

    explicit_w = settings.get("width")
    explicit_h = settings.get("height")
    if explicit_w and explicit_h:
        return clamp_side(explicit_w), clamp_side(explicit_h)

    factor: Optional[float] = None
    if "scale" in settings:
        factor = float(settings["scale"])
    elif "long_side" in settings:
        factor = float(settings["long_side"]) / max(lw, lh)
    if factor is None:
        factor = float(base_factor)
    if "short_side" in settings:
        short = float(settings["short_side"])
        factor = max(factor, short / max(1, min(lw, lh)))

    tw = float(explicit_w) if explicit_w else lw * factor
    th = float(explicit_h) if explicit_h else lh * factor

    # Never let a non-square result be smaller than the minimum side.
    tw = max(tw, MIN_SIDE)
    th = max(th, MIN_SIDE)

    max_pixels = settings.get("max_pixels")
    if max_pixels:
        limit = float(max_pixels) * 1_000_000.0
        if tw * th > limit:
            shrink = (limit / (tw * th)) ** 0.5
            tw *= shrink
            th *= shrink

    return clamp_side(tw), clamp_side(th)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def build_plan(
    layers_json: str,
    output_root: str,
    select: str = "all",
    exclude: str = "",
    size_rules_text: str = "",
    out_dir: str = "",
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    skip_empty: bool = True,
    empty_coverage: float = 0.0005,
    focus_crop: bool = True,
    focus_margin: int = FOCUS_MARGIN,
    focus_context: str = "backdrop",
    focus_prompt: str = DEFAULT_FOCUS_PROMPT,
) -> Dict[str, Any]:
    """Build the plan dictionary consumed by the other nodes."""
    doc = resolve_layers_source(layers_json, output_root)
    path = doc["path"]
    warnings: List[str] = list(doc["warnings"])

    # z order: far (depth ~1) first, near (depth ~0) last, stable on ties.
    ordered = sorted(
        doc["layers"],
        key=lambda layer: (-float(layer["depth_median"]), layer["index"]),
    )
    names = [layer["name"] for layer in ordered]
    selected_names = set(select_layer_names(names, select, exclude))
    # An explicit name/glob list means the user chose those layers on purpose;
    # only the blanket "all" path auto-drops nearly-empty parts.
    pos_patterns, _ = _parse_patterns(select)
    select_all = not pos_patterns or any(p.lower() in ("all", "*") for p in pos_patterns)
    explicit_selection = not select_all

    try:
        rules = parse_size_rules(size_rules_text)
    except ValueError as exc:
        warnings.append(str(exc))
        rules = parse_size_rules("")

    explicit_out = normalize_out_dir(out_dir)
    out_dir_rel = explicit_out or auto_out_dir(auto_basis(doc), output_root)
    out_dir_abs = os.path.join(output_root, out_dir_rel)
    # Created up front (and again on every save): the folder follows the split
    # folder by default, so it may not exist yet on the first run.
    os.makedirs(out_dir_abs, exist_ok=True)

    try:
        canvas = clamp_side(canvas_size)
    except (TypeError, ValueError):
        canvas = DEFAULT_CANVAS_SIZE
    canvas = clamp_side(canvas)

    src_w = max(1, int(doc["width"]))
    src_h = max(1, int(doc["height"]))
    scale = float(canvas) / float(src_w)

    used_stems: Dict[str, int] = {}
    layers_out: List[Dict[str, Any]] = []
    selected_out: List[str] = []
    empty_skipped: List[str] = []
    # Full-canvas layers holding only a tiny part (mouth, nose).  They stay in
    # the PSD, but auto-selecting them would make the edit model regenerate a
    # whole 1024x1024 frame around a 30-pixel mouth.
    tiny_skipped: List[str] = []

    for z, layer in enumerate(ordered):
        name = layer["name"]
        left, top, right, bottom = layer["box"]
        lw, lh = right - left, bottom - top
        is_selected = name in selected_names

        coverage = layer.get("alpha_coverage")
        if is_selected and skip_empty and not explicit_selection:
            if _is_empty(coverage, layer.get("alpha_pixels"), empty_coverage):
                is_selected = False
                empty_skipped.append(name)
            else:
                fraction = layer.get("alpha_fraction")
                if fraction is not None and float(fraction) < float(empty_coverage):
                    is_selected = False
                    tiny_skipped.append(name)

        settings = merge_size_settings(name, rules)
        if is_selected:
            try:
                tw, th = compute_target_size(lw, lh, settings, base_factor=scale)
            except (TypeError, ValueError) as exc:
                warnings.append("bad size rule for %r: %s" % (name, exc))
                tw, th = compute_target_size(lw, lh, {}, base_factor=scale)
        else:
            tw, th = lw, lh

        stem = safe_name(name)
        if stem in used_stems:
            used_stems[stem] += 1
            stem = "%s_%d" % (stem, used_stems[stem])
        else:
            used_stems[stem] = 1

        disp_w = max(1, int(round(lw * scale)))
        disp_h = max(1, int(round(lh * scale)))
        disp_left = int(round(left * scale))
        disp_top = int(round(top * scale))

        entry = {
            "name": name,
            "z": z,
            "selected": bool(is_selected),
            "depth_median": float(layer["depth_median"]),
            "alpha_coverage": coverage,
            "alpha_pixels": layer.get("alpha_pixels"),
            "alpha_fraction": layer.get("alpha_fraction"),
            "empty": bool(layer.get("empty")),
            "file": layer["file"],
            "box": [left, top, right, bottom],
            "display_box": [disp_left, disp_top, disp_left + disp_w, disp_top + disp_h],
            "target": {"w": int(tw), "h": int(th)},
            "target_file": "%s/%s.png" % (out_dir_rel, stem),
            "settings": settings,
        }
        layers_out.append(entry)
        if is_selected:
            selected_out.append(name)

    plan = {
        "version": 1,
        "source": {
            "layers_json": path,
            "source_dir": doc.get("source_dir"),
            "synthetic": bool(doc.get("synthetic")),
            "prefix": doc["prefix"],
            "timestamp": doc["timestamp"],
            "width": src_w,
            "height": src_h,
        },
        "out_dir": out_dir_rel,
        "out_dir_abs": os.path.abspath(out_dir_abs),
        # True when out_dir came from the split folder rather than the widget.
        "out_dir_auto": not explicit_out,
        "canvas_size": int(canvas),
        "scale": scale,
        "total": len(selected_out),
        "selected": selected_out,
        "empty_skipped": empty_skipped,
        "tiny_skipped": tiny_skipped,
        "layers": layers_out,
        "warnings": warnings,
        "focus": {
            "crop": bool(focus_crop),
            "margin": _clamp_margin(focus_margin),
            "context": str(focus_context) if str(focus_context) in ("backdrop", "reference") else "backdrop",
            "prompt": str(focus_prompt or ""),
            "resolution": int(FOCUS_RESOLUTION),
        },
    }
    return plan


def _clamp_margin(value: Any) -> int:
    try:
        margin = int(round(float(value)))
    except (TypeError, ValueError):
        return FOCUS_MARGIN
    return max(0, min(256, margin))


def plan_to_text(plan: Dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))


def plan_from_text(text: str) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    text = (text or "").strip()
    if not text:
        raise ValueError("empty plan")
    if not text.startswith("{"):
        with open(text, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(text)


def plan_report(plan: Dict[str, Any]) -> str:
    lines = [
        "layers.json: %s" % plan["source"]["layers_json"],
        "canvas: %d  scale: %.3f  selected: %d/%d" % (
            plan["canvas_size"], plan["scale"], plan["total"], len(plan["layers"]),
        ),
        "out_dir: %s%s" % (plan["out_dir"], "  (auto)" if plan.get("out_dir_auto") else ""),
    ]
    if plan.get("empty_skipped"):
        lines.append("empty layers skipped: %s" % ", ".join(plan["empty_skipped"]))
    if plan.get("tiny_skipped"):
        lines.append("tiny parts kept (not AI-upscaled): %s" % ", ".join(plan["tiny_skipped"]))
    for layer in plan["layers"]:
        mark = "*" if layer["selected"] else " "
        target = layer["target"]
        lines.append("%s %-16s %4dx%-4d -> %4dx%-4d  %s" % (
            mark, layer["name"],
            layer["box"][2] - layer["box"][0], layer["box"][3] - layer["box"][1],
            target["w"], target["h"], layer["target_file"],
        ))
    if plan.get("warnings"):
        lines.append("warnings:")
        lines.extend("  - %s" % w for w in plan["warnings"])
    return "\n".join(lines)
