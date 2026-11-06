"""Live2D part extraction with an image-edit model (no SeeThrough weights).

The batch node asks Qwen-Image-Edit to isolate one part per sample on a flat
chroma background, keys that background out on the CPU, and writes layers that
are laid out exactly like a SeeThrough run: same part names, same
``*_layers.json`` document, same PSD layer order (far -> near, bottom -> top).

Nothing here loads a model, so every function is testable without CUDA.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import core_io
    from .core_compose import _alpha_over
    from .core_plan import safe_name
except ImportError:  # pragma: no cover - standalone import from the tests
    import core_io
    from core_compose import _alpha_over
    from core_plan import safe_name


# ---------------------------------------------------------------------------
# Part catalogue
# ---------------------------------------------------------------------------
#
# The names are the SeeThrough tags (spaces and ``-l`` / ``-r`` suffixes are
# part of the format, not a display choice).  ``depth`` copies the ordering a
# SeeThrough run produces for a standard standing character, so the assembled
# PSD stacks far-to-near exactly like the original pipeline:
# the first entry is the bottom-most layer.
#
# ``-r`` / ``-l`` are the *character's* right / left, which puts them on the
# left / right side of the picture.  The prompt says "left side of the image"
# so the edit model cannot mirror them.

PARTS: List[Tuple[str, float, str]] = [
    ("neckwear", 1.0, "the neckwear (necklace, collar or scarf) around the neck"),
    ("tail", 1.0, "the character's tail"),
    ("wings", 1.0, "the character's wings"),
    ("objects", 1.0, "the prop or object the character is holding"),
    ("eyewear", 1.0, "the eyewear (glasses, sunglasses or goggles) on the face"),
    ("earwear", 1.0, "the earrings and ear accessories"),
    ("back hair", 0.96875, "the back hair behind the head and body (excluding the front bangs)"),
    ("headwear", 0.765625, "the headwear (hat, hair accessory or head ornament) worn on the head"),
    ("legwear", 0.7578125, "the legwear (stockings, thigh-highs or leg accessories)"),
    ("handwear-r", 0.75390625, "the character's right hand and forearm, on the left side of the image"),
    ("footwear", 0.7421875, "the footwear (shoes or boots)"),
    ("handwear-l", 0.7421875, "the character's left hand and forearm, on the right side of the image"),
    ("neck", 0.7109375, "the character's neck and upper chest skin (excluding the face and clothing)"),
    ("bottomwear", 0.703125, "the bottom clothing (skirt, shorts or trousers)"),
    ("topwear", 0.64453125, "the top clothing (shirt, jacket or upper garment)"),
    ("eyebrow-r", 0.568359375, "the character's right eyebrow, on the left side of the image"),
    ("ears-r", 0.5234375, "the character's right ear, on the left side of the image"),
    ("ears-l", 0.51171875, "the character's left ear, on the right side of the image"),
    ("eyebrow-l", 0.490234375, "the character's left eyebrow, on the right side of the image"),
    ("face", 0.478515625,
     "the bare face skin (forehead, cheeks, chin and jaw), excluding hair, eyes, "
     "eyebrows, nose, mouth and ears"),
    ("nose", 0.477515625, "the character's nose"),
    ("mouth", 0.477515625, "the character's mouth and lips"),
    ("eyewhite-r", 0.314453125,
     "the white of the character's right eye (sclera only), on the left side of the image"),
    ("eyewhite-l", 0.314453125,
     "the white of the character's left eye (sclera only), on the right side of the image"),
    ("eyelash-l", 0.27734375,
     "the eyelashes and upper eyelid line of the character's left eye, on the right "
     "side of the image"),
    ("irides-r", 0.2734375,
     "the iris (colored part only) of the character's right eye, on the left side of "
     "the image"),
    ("irides-l", 0.2734375,
     "the iris (colored part only) of the character's left eye, on the right side of "
     "the image"),
    ("eyelash-r", 0.26171875,
     "the eyelashes and upper eyelid line of the character's right eye, on the left "
     "side of the image"),
    ("front hair", 0.109375,
     "the front hair / bangs and side locks covering the forehead and framing the "
     "face (excluding the back hair; the face behind the bangs must stay hidden and "
     "must not be drawn)"),
]

PART_NAMES: List[str] = [name for name, _depth, _desc in PARTS]
PART_DEPTH: Dict[str, float] = {name: depth for name, depth, _desc in PARTS}
PART_DESC: Dict[str, str] = {name: desc for name, _depth, desc in PARTS}

# UI labels / grouping (the panel lists these; the PSD still uses PART_NAMES).
PART_LABEL: Dict[str, str] = {
    "neckwear": "颈饰", "tail": "尾巴", "wings": "翅膀", "objects": "手持物",
    "eyewear": "眼镜", "earwear": "耳饰", "back hair": "后发",
    "headwear": "头饰", "legwear": "腿饰", "handwear-r": "右手",
    "footwear": "鞋", "handwear-l": "左手", "neck": "脖子",
    "bottomwear": "下装", "topwear": "上装", "eyebrow-r": "右眉",
    "ears-r": "右耳", "ears-l": "左耳", "eyebrow-l": "左眉",
    "face": "脸", "nose": "鼻子", "mouth": "嘴", "eyewhite-r": "右眼白",
    "eyewhite-l": "左眼白", "eyelash-l": "左睫毛", "irides-r": "右虹膜",
    "irides-l": "左虹膜", "eyelash-r": "右睫毛", "front hair": "前发",
}

PART_GROUP: Dict[str, str] = {
    "back hair": "hair", "front hair": "hair",
    "eyebrow-r": "face", "eyebrow-l": "face", "ears-r": "face",
    "ears-l": "face", "eyewhite-r": "face", "eyewhite-l": "face",
    "eyelash-l": "face", "eyelash-r": "face", "irides-r": "face",
    "irides-l": "face", "face": "face", "nose": "face", "mouth": "face",
    "eyewear": "face", "earwear": "face",
    "topwear": "wear", "bottomwear": "wear", "legwear": "wear",
    "footwear": "wear", "handwear-r": "wear", "handwear-l": "wear",
    "neckwear": "wear",
}

GROUPS: List[Tuple[str, str]] = [
    ("hair", "头发"),
    ("face", "面部"),
    ("wear", "服装 / 手"),
    ("body", "身体 / 其它"),
]


def catalogue() -> List[Dict[str, Any]]:
    """Part table for the graphical selector."""
    return [{
        "name": name,
        "label": PART_LABEL.get(name, name),
        "desc": PART_DESC.get(name, ""),
        "group": PART_GROUP.get(name, "body"),
        "optional": name in OPTIONAL_PARTS,
    } for name in PART_NAMES]

# Optional parts are very often absent; the prompt tells the model to return a
# completely keyed frame, and the PSD pass drops the empty layer just like
# SeeThrough's ``skip_empty``.
OPTIONAL_PARTS = {"tail", "wings", "objects", "eyewear", "earwear", "neckwear",
                  "headwear", "legwear", "footwear", "topwear", "bottomwear"}


KEY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "green": (0, 255, 0),
    "magenta": (255, 0, 255),
    "blue": (0, 96, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

KEY_NAME: Dict[str, str] = {
    "green": "pure bright green",
    "magenta": "pure magenta",
    "blue": "pure blue",
    "white": "pure white",
    "black": "pure black",
}

PROMPT_TEMPLATE = (
    "Extract only {desc} from <image1>. Keep its exact shape, position, scale, "
    "colors, shading and line art; do not redraw, move or reshape it, and do not "
    "add or remove details. The output must show only this part on a flat solid "
    "{key_name} background ({key_hex}). {exclude} must be completely replaced by "
    "that flat {key_name} color. Never paint the extracted part itself with the "
    "key color. If the character does not have this part, output a completely "
    "{key_name} image."
)


def _exclude_for(part: str) -> str:
    """What must be keyed out for this part.

    Without a per-part list the model tends to keep the whole face under the
    bangs (or the whole body for a tiny eye part), which lands in the layer as
    unwanted content.
    """
    if part in ("front hair", "back hair"):
        return ("the face, skin, eyes, eyebrows, nose, mouth, ears, neck, body "
                "and clothing")
    if part == "face":
        return ("hair, eyebrows, eyes, nose, mouth, ears, neck, body, clothing "
                "and background")
    if part in ("nose", "mouth"):
        return ("the rest of the face, the skin around it, eyes, eyebrows, "
                "hair, neck, body, clothing and background")
    if part.startswith(("eyewhite", "irides", "eyelash", "eyebrow", "ear")):
        return ("face skin, hair, the other eye, nose, mouth, neck, body, "
                "clothing and background")
    if part in ("handwear-l", "handwear-r"):
        return ("face, hair, eyes, clothing fabric, torso, the other arm and "
                "the background")
    if part == "neck":
        return ("the face, hair, eyes, clothing, arms, torso and background")
    if part in ("topwear", "bottomwear", "legwear", "footwear", "neckwear",
                "headwear", "eyewear", "earwear"):
        return "the character's skin, face, hair, eyes and the background"
    return ("every other body part, skin, hair, eyes, clothing and the "
            "background")


def resolve_key(key: str) -> Tuple[str, Tuple[int, int, int]]:
    key = str(key or "green").strip().lower()
    if key not in KEY_COLORS:
        key = "green"
    return key, KEY_COLORS[key]


def hex_rgb(rgb: Sequence[int]) -> str:
    return "#%02X%02X%02X" % tuple(int(v) for v in rgb[:3])


def build_prompt(part: str, key: str = "green") -> str:
    key_id, rgb = resolve_key(key)
    return PROMPT_TEMPLATE.format(
        desc=PART_DESC.get(part, part),
        exclude=_exclude_for(part),
        key_name=KEY_NAME[key_id],
        key_hex=hex_rgb(rgb),
    )


def parse_select(select: str) -> List[str]:
    """Turn the UI ``select`` string into part names.

    ``all`` / empty keeps the whole catalogue, ``none`` selects nothing.
    ``-name`` removes a part, so ``all -wings -tail`` works without listing
    27 names.  A comma means the pieces are whole part names, which is how
    the graphical panel writes names that contain spaces (``back hair``);
    without a comma the old whitespace splitting is kept.  Unknown names are
    ignored -- but if *nothing* was recognised the old "all" fallback stays,
    so a stale/foreign value never silently runs a partial job.
    """
    text = str(select or "").strip()
    if not text or text.lower() == "all":
        return list(PART_NAMES)
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
        if token.startswith("-"):
            removed.add(token[1:].strip().lower())
            explicit = True
            continue
        if token.lower() == "all":
            for name in PART_NAMES:
                if name not in chosen:
                    chosen.append(name)
            explicit = True
            continue
        match = _match_part(token.lower())
        if match:
            explicit = True
            if match not in chosen:
                chosen.append(match)
    if not explicit:
        return list(PART_NAMES)
    # Always return the catalogue order: layers.json (and therefore the PSD
    # layer stack) must not depend on the order the user ticked the boxes.
    wanted = {name for name in chosen if name.lower() not in removed}
    return [name for name in PART_NAMES if name in wanted]


def _match_part(token: str) -> Optional[str]:
    for name in PART_NAMES:
        if name.lower() == token:
            return name
    # Accept underscores / hyphens swapped and the ``_l`` spelling.
    norm = token.replace("_", " ").replace("-", " ")
    for name in PART_NAMES:
        if name.lower().replace("-", " ").replace("_", " ") == norm:
            return name
    return None


# ---------------------------------------------------------------------------
# Chroma key -> alpha
# ---------------------------------------------------------------------------

def _chroma(rgb: np.ndarray) -> np.ndarray:
    total = rgb.sum(axis=-1, keepdims=True)
    return rgb / np.maximum(total, 1e-6)


def chroma_alpha(
    rgb: np.ndarray,
    key_rgb: Sequence[int],
    tolerance: float = 12.0,
    softness: float = 6.0,
    reference_rgb: Optional[np.ndarray] = None,
    protect: bool = True,
) -> np.ndarray:
    """Return a float32 alpha in 0..1 for an RGB uint8 image.

    Distance is a blend of chroma distance (catches shaded / gradient key
    background) and plain RGB distance (catches desaturated key pixels).  A
    pixel that the model kept identical to the reference *and* that was already
    key-coloured in the reference is protected, so green eyes or a green
    garment are not eaten by a green screen.
    """
    img = np.asarray(rgb, dtype=np.float32) / 255.0
    key = np.asarray(key_rgb, dtype=np.float32) / 255.0

    # Chroma decides for saturated keys: a shaded / gradient key background
    # keeps the key's hue but not its brightness, and a plain RGB distance
    # would leave all of it opaque.  Desaturated keys (white / black) have no
    # useful hue, so they fall back to plain RGB distance.
    key_saturation = float(key.max() - key.min())
    rgb_weight = 0.3 if key_saturation > 0.35 else 1.0
    de = np.linalg.norm(img - key, axis=-1)
    dc = np.linalg.norm(_chroma(img) - _chroma(key), axis=-1) * 1.6
    dist = np.maximum(dc, de * rgb_weight)

    t0 = max(0.01, float(tolerance) / 100.0)
    t1 = max(t0 + 0.01, t0 + float(softness) / 100.0)
    alpha = np.clip((dist - t0) / (t1 - t0), 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep

    if protect and reference_rgb is not None:
        ref = np.asarray(reference_rgb, dtype=np.float32) / 255.0
        if ref.shape[:2] == img.shape[:2]:
            ref_dist = np.maximum(
                np.linalg.norm(_chroma(ref) - _chroma(key), axis=-1) * 1.6,
                np.linalg.norm(ref - key, axis=-1) * rgb_weight,
            )
            close_to_ref = np.linalg.norm(img - ref, axis=-1) < 0.10
            keep = close_to_ref & (ref_dist < t0)
            alpha = np.where(keep, 1.0, alpha)
    return alpha.astype(np.float32)


def despill(rgb: np.ndarray, alpha: np.ndarray, key_rgb: Sequence[int],
            strength: float = 0.65) -> np.ndarray:
    """Pull the key hue out of partially transparent edge pixels."""
    img = np.asarray(rgb, dtype=np.float32) / 255.0
    key = np.asarray(key_rgb, dtype=np.float32) / 255.0
    key_norm = float(np.dot(key, key))
    if key_norm <= 1e-6:
        return np.asarray(rgb, dtype=np.uint8)
    edge = (alpha > 0.0) & (alpha < 0.98)
    if not np.any(edge):
        return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    proj = np.clip(img @ key / key_norm, 0.0, 1.0)[..., None]
    spill = key[None, None, :] * proj
    fixed = img - spill * (float(strength) * (1.0 - alpha[..., None]))
    return np.clip(np.rint(fixed * 255.0), 0, 255).astype(np.uint8)


def key_out(rgb: np.ndarray, key: str = "green", tolerance: float = 12.0,
            softness: float = 6.0, reference_rgb: Optional[np.ndarray] = None,
            protect: bool = True, despill_strength: float = 0.65,
            alpha_floor: int = 8, alpha_ceiling: int = 250) -> np.ndarray:
    """Full key: RGB uint8 -> RGBA uint8 with a cleaned alpha channel."""
    _key_id, key_rgb = resolve_key(key)
    alpha = chroma_alpha(rgb, key_rgb, tolerance, softness, reference_rgb, protect)
    fixed = despill(rgb, alpha, key_rgb, despill_strength)
    a8 = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    a8[a8 <= int(alpha_floor)] = 0
    a8[a8 >= int(alpha_ceiling)] = 255

    # Second pass: the model sometimes paints key-colour streaks *inside*
    # the part (green lines along the hair).  ``despill`` only touches the
    # soft edge, so clamp the green channel where the reference image has no
    # green at the same pixel -- a genuinely green eye or dyed hair stays.
    if reference_rgb is not None:
        ref = np.asarray(reference_rgb, dtype=np.float32) / 255.0
        f = fixed.astype(np.float32) / 255.0
        painted = (f[..., 1] - np.maximum(f[..., 0], f[..., 2])) > 0.06
        ref_green = (ref[..., 1] - np.maximum(ref[..., 0], ref[..., 2])) > 0.06
        fix = painted & ~ref_green & (a8 > 0)
        if np.any(fix):
            f[..., 1] = np.where(
                fix,
                np.minimum(f[..., 1], np.maximum(f[..., 0], f[..., 2])),
                f[..., 1],
            )
            fixed = np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)
    return np.dstack([fixed, a8])


def coverage(rgba: np.ndarray, threshold: int = 8) -> float:
    if rgba is None or rgba.size == 0:
        return 0.0
    return float(np.mean(rgba[..., 3] > int(threshold)))


# ---------------------------------------------------------------------------
# SeeThrough-compatible layers.json
# ---------------------------------------------------------------------------

def layers_json_doc(prefix: str, timestamp: str,
                    entries: Sequence[Dict[str, Any]],
                    width: int, height: int) -> Dict[str, Any]:
    return {
        "prefix": str(prefix),
        "timestamp": str(timestamp),
        "layers": [dict(e) for e in entries],
        "width": int(width),
        "height": int(height),
    }


def write_layers_json(path: str, doc: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path


def read_layers_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc.get("layers"), list) or not doc["layers"]:
        raise ValueError("layers.json has no layers: %s" % path)
    return doc


def layer_entry(name: str, out_dir_rel: str, filename: str,
                left: int, top: int, right: int, bottom: int,
                depth: Optional[float] = None) -> Dict[str, Any]:
    return {
        "name": name,
        "filename": "%s/%s" % (out_dir_rel.strip("/").replace("\\", "/"), filename),
        "left": int(left),
        "top": int(top),
        "right": int(right),
        "bottom": int(bottom),
        "depth_median": float(PART_DEPTH.get(name, depth if depth is not None else 0.5)),
    }


# ---------------------------------------------------------------------------
# Assemble PSD from a layers.json
# ---------------------------------------------------------------------------

def assemble_psd(
    layers_json_path: str,
    output_root: str,
    psd_path: str,
    alpha_threshold: int = 8,
    skip_empty: bool = True,
    max_coverage: float = 0.9,
) -> Dict[str, Any]:
    """Read layers.json, composite its PNGs and write a PSD.

    ``layers`` in the document are far -> near (first entry bottom-most), which
    is exactly the PSD writer's bottom -> top order.  Missing / empty / broken
    parts are skipped, the same way SeeThrough's assembler drops empty layers.
    """
    doc = read_layers_json(layers_json_path)
    width = int(doc.get("width") or 0)
    height = int(doc.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("layers.json has no canvas size")

    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    psd_layers: List[Dict[str, Any]] = []
    notes: List[str] = []
    kept: List[str] = []

    for entry in doc["layers"]:
        name = str(entry.get("name") or "").strip()
        rel = str(entry.get("filename") or "")
        path = rel
        if not os.path.isabs(path):
            path = os.path.join(output_root, rel)
        if not name or not os.path.isfile(path):
            notes.append("%s: missing file" % (name or rel or "?"))
            continue
        rgba = core_io.read_rgba(path)
        if rgba.shape[1] != width or rgba.shape[0] != height:
            rgba = core_io.resize_rgba(rgba, width, height)
        cov = coverage(rgba, alpha_threshold)
        if skip_empty and cov <= 0.0005:
            notes.append("%s: empty layer skipped" % name)
            continue
        if cov >= float(max_coverage):
            notes.append("%s: %.0f%% coverage, probably a failed extraction"
                         % (name, cov * 100.0))
            continue
        _alpha_over(canvas, rgba, 0, 0)
        psd_layers.append({
            "name": name,
            "pixels": rgba,
            "left": 0,
            "top": 0,
            "depth_median": float(entry.get("depth_median", 0.5)),
        })
        kept.append(name)

    if not psd_layers:
        raise ValueError("no non-empty layers to assemble")

    try:
        from . import psd_writer
    except ImportError:  # pragma: no cover - standalone import
        import psd_writer
    os.makedirs(os.path.dirname(psd_path) or ".", exist_ok=True)
    psd_writer.write_psd(psd_path, psd_layers, width, height, "rle")

    return {
        "psd": psd_path,
        "canvas": [width, height],
        "layers": kept,
        "notes": notes,
    }


def parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on", "y")


def safe_timestamp(text: str) -> str:
    """Keep a timestamp filename-safe (the UI passes one through literal)."""
    keep = []
    for ch in str(text or ""):
        keep.append(ch if ch.isalnum() or ch in "_-" else "_")
    return "".join(keep) or "run"


TEMPLATE_NAME = "Live2D图编辑抠层.json"


def default_template_path() -> str:
    """Locate the edit subgraph template next to the user's other workflows."""
    here = os.path.dirname(os.path.abspath(__file__))
    comfy_root = os.path.dirname(os.path.dirname(here))
    root = os.path.join(comfy_root, "user", "default", "workflows")
    preferred = os.path.join(root, "Comfy-My-Workflow", "图片", "Live2D拆层",
                             TEMPLATE_NAME)
    if os.path.isfile(preferred):
        return preferred
    for dirpath, _dirs, files in os.walk(root):
        if TEMPLATE_NAME in files:
            return os.path.join(dirpath, TEMPLATE_NAME)
    return preferred
