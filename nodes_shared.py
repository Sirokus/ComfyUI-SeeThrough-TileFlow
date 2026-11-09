"""Shared constants and helpers for the SeeThrough LayerUpscale nodes.

Every node module imports from here instead of from each other, so the
dependency graph stays ``nodes_* -> nodes_shared -> tile_psd/core_*``.
The path and prompt-id hooks are looked up through this module at call time,
which lets the offline tests rebind them in one place.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import traceback
import uuid

import numpy as np
import torch
import torch.nn.functional as F

import folder_paths

if __package__:  # ComfyUI loads the folder as a package
    from . import (core_compose, core_geometry, core_io, core_mask4k, core_plan,
                   core_reference, edit_expand, edit_parts, psd_reader,
                   psd_writer, tile_psd, tile_strips)
    from .stlu import json_io, limits as stlu_limits
else:  # standalone import used by the tests and tools
    import core_compose
    import core_geometry
    import core_io
    import core_mask4k
    import core_plan
    import core_reference
    import edit_expand
    import edit_parts
    import psd_reader
    import psd_writer
    import tile_psd
    import tile_strips
    from stlu import json_io, limits as stlu_limits

CATEGORY = "SeeThrough/LayerUpscale"

# Conservative defaults: 2048 long side is already a 2x upscale of the 1024
# source canvas.  4096 is available but heavy on this machine (see the PCIe /
# power notes), so it is opt-in through the size_rules widget.
DEFAULT_SIZE_RULES = json.dumps(
    core_plan.DEFAULT_SIZE_RULES, ensure_ascii=False, indent=2
)

# A folder (relative to output/) is enough: the plan finds the newest
# *_layers.json inside it, or falls back to the PNGs in that folder.
DEFAULT_LAYERS_JSON = "SeeThrough"

def _output_root() -> str:
    return folder_paths.get_output_directory()


def _notify(event: str, payload: dict) -> None:
    """Push a UI event; never let a missing socket fail the run."""
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(event, payload)
    except Exception:  # noqa: BLE001
        pass


def _prompt_id() -> str:
    """The queue prompt id, used to name one output wave.

    ComfyUI exposes it through the current node context; older builds do not,
    in which case ``tile_psd.wave_id`` falls back to a timestamp.
    """
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
        value = getattr(context, "prompt_id", "") if context is not None else ""
        return str(value or "")
    except Exception:  # noqa: BLE001 - the node must run without it
        return ""


def _image_fingerprint(frame) -> str:
    """Stable id of the reference image, so cached tiles are not reused after
    the user swaps the picture while keeping the same output folder."""
    try:
        arr = np.asarray(frame.detach().cpu().numpy(), dtype=np.float32)
        arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
        return hashlib.sha1(arr.tobytes()).hexdigest()
    except Exception:  # noqa: BLE001 - the fingerprint is an optimisation
        return ""


def _model_identity(model) -> str:
    """Stable-enough identity of a loaded SeeThrough pipeline.

    The loader nodes return diffusers pipelines, so the model repo/folder and
    the UNet/VAE submodels expose ``_name_or_path`` through their configs.
    Quantization is appended because bf16 and nf4 are different numerics.
    """
    if model is None:
        return ""
    names = []
    for obj in (model, getattr(model, "unet", None),
                getattr(model, "vae", None), getattr(model, "trans_vae", None)):
        if obj is None:
            continue
        name = ""
        config = getattr(obj, "config", None)
        if config is not None:
            try:
                name = str(config.get("_name_or_path") or "")
            except Exception:  # noqa: BLE001
                name = ""
        if not name:
            name = str(getattr(obj, "_name_or_path", "") or "")
        if name and name not in names:
            names.append(name)
    quant = "bf16"
    unet = getattr(model, "unet", None)
    try:
        for _, module in unet.named_modules():
            if "4bit" in type(module).__name__.lower():
                quant = "nf4"
                break
    except Exception:  # noqa: BLE001
        pass
    names.append("quant:" + quant)
    return "|".join(names) or type(model).__name__


def _output_rel(path: str) -> tuple:
    """Return ``(subfolder, filename)`` relative to the ComfyUI output dir."""
    root = os.path.realpath(_output_root())
    full = os.path.realpath(path)
    try:
        if os.path.commonpath([root, full]) == root:
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if "/" in rel:
                folder, name = rel.rsplit("/", 1)
                return folder, name
            return "", rel
    except ValueError:
        pass
    return "", os.path.basename(full)


def _plan_fingerprint(plan_ref: str) -> str:
    text = (plan_ref or "").strip()
    if not text:
        return "empty"
    if text.startswith("{"):
        return core_plan.sha1_text(text)
    path = core_plan.resolve_path(text, _output_root())
    return core_plan.file_fingerprint(path)


def _load_plan(plan_ref: str) -> dict:
    """Parse a plan from inline JSON or from a path (relative to output/)."""
    text = (plan_ref or "").strip()
    if text and not text.startswith("{") and not os.path.isabs(text):
        text = core_plan.resolve_path(text, _output_root())
    return core_plan.plan_from_text(text)


def _empty_image():
    image = torch.zeros((1, 8, 8, 4), dtype=torch.float32)
    mask = torch.ones((1, 8, 8), dtype=torch.float32)
    return image, mask

def _alpha_over(canvas: np.ndarray, rgba: np.ndarray, box) -> None:
    """Composite straight-alpha ``rgba`` over an RGB float canvas in place."""
    height, width = canvas.shape[:2]
    left, top, right, bottom = (int(v) for v in box)
    lw, lh = right - left, bottom - top
    if lw <= 0 or lh <= 0:
        return
    if rgba.shape[0] != lh or rgba.shape[1] != lw:
        rgba = core_io.resize_rgba(rgba, lw, lh)
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, right), min(height, bottom)
    if x1 <= x0 or y1 <= y0:
        return
    src = rgba[y0 - top:y1 - top, x0 - left:x1 - left].astype(np.float32)
    alpha = src[..., 3:4] / 255.0
    dst = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = src[..., :3] * alpha + dst * (1.0 - alpha)


def _numpy_rgba(image) -> np.ndarray:
    return core_io.tensor_to_rgba(image)


def _sample_region(src: torch.Tensor, fx0: float, fy0: float, fx1: float, fy1: float,
                   out_w: int, out_h: int) -> torch.Tensor:
    """Sample the fractional rect ``(fx0, fy0, fx1, fy1)`` of ``src`` ([C,H,W])."""
    _, height, width = src.shape
    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))
    if width < 1 or height < 1:
        return torch.zeros((src.shape[0], out_h, out_w), dtype=src.dtype, device=src.device)
    # Centre-based sampling, matching PIL: output centre i+0.5 maps to
    # fx0 + (i+0.5)/out_w * (fx1-fx0) of the source.
    xs = fx0 + (torch.arange(out_w, dtype=torch.float32, device=src.device) + 0.5) / out_w * (fx1 - fx0)
    ys = fy0 + (torch.arange(out_h, dtype=torch.float32, device=src.device) + 0.5) / out_h * (fy1 - fy0)
    grid = torch.stack(torch.meshgrid(xs * 2.0 - 1.0, ys * 2.0 - 1.0, indexing="xy"), dim=-1)[None]
    sampled = F.grid_sample(src[None], grid, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled[0]

def _resize_mask(mask, width, height):
    width, height = int(width), int(height)
    if mask is None:
        return torch.zeros((1, height, width), dtype=torch.float32)
    mask = mask[:1].float()
    if (int(mask.shape[1]), int(mask.shape[2])) == (height, width):
        return mask.clamp(0.0, 1.0)
    resized = F.interpolate(mask[:, None], size=(height, width), mode="bilinear",
                            align_corners=False)
    return resized[:, 0].clamp(0.0, 1.0)

def _first(value, default=None):
    """INPUT_IS_LIST nodes receive list-or-scalar for every socket."""
    if isinstance(value, list):
        return value[0] if value else default
    return value


def _list_values(value):
    """Flatten a socket value into a plain list (INPUT_IS_LIST sockets)."""
    if isinstance(value, (list, tuple)):
        return [item for item in value if item is not None]
    return [] if value is None else [value]


def _resolve_workspace_path(raw: str) -> str:
    """Absolute path, or one relative to output/ or input/."""
    text = os.path.expandvars(os.path.expanduser(str(raw or "").strip()))
    if not text:
        raise FileNotFoundError("empty path")
    if os.path.isabs(text):
        return os.path.normpath(text)
    output_root = _output_root()
    for root in (output_root, folder_paths.get_input_directory()):
        candidate = os.path.normpath(os.path.join(root, text))
        if os.path.exists(candidate):
            return candidate
    return os.path.normpath(os.path.join(output_root, text))


def _load_grid(grid_ref: str):
    """Load a grid document from a widget; ``None`` when not set.

    The value is either the inline JSON stage one returns now, or the path of
    a ``grids/*.json`` document written by an older version.
    """
    text = str(grid_ref or "").strip()
    if not text or text.lower() in ("none", "auto"):
        return None, ""
    if text.startswith("{"):
        try:
            return tile_psd.load_grid_ref(text), ""
        except Exception as exc:  # noqa: BLE001
            print("[STLU] cannot read inline grid: %s" % exc, flush=True)
            return None, ""
    path = _resolve_workspace_path(text)
    if not os.path.isfile(path):
        print("[STLU] grid document not found: %s" % path, flush=True)
        return None, text
    try:
        return tile_psd.load_grid_doc(path), path
    except Exception as exc:  # noqa: BLE001
        print("[STLU] cannot read grid %s: %s" % (path, exc), flush=True)
        return None, text


def _tile_index_map(raw_indices, parts_list, grid_doc=None):
    """Map the tile list onto original grid indices (or ``None``)."""
    raw = str(raw_indices or "").strip()
    if raw and raw.lower() not in ("all", "none"):
        parsed = _parse_index_list(raw)
        if len(parsed) == len(parts_list):
            return parsed
        print("[STLU] index list %r has %d entries but %d tiles arrived; "
              "falling back to sequential numbering"
              % (raw, len(parsed), len(parts_list)), flush=True)
    if grid_doc:
        selection = str(grid_doc.get("selection") or "").strip()
        if selection and selection.lower() not in ("all", "none"):
            parsed = _parse_index_list(selection)
            if len(parsed) == len(parts_list):
                return parsed
    return None

def _wave_doc(value) -> dict:
    """``STLU_Wave``'s document from a socket value (list or scalar)."""
    raw = _first(value, "")
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw or "").strip()
    if not text:
        raise ValueError("missing wave_doc: connect STLU_Wave")
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise ValueError("wave_doc is not JSON: %s" % exc)
    if not isinstance(doc, dict) or not doc.get("dir"):
        raise ValueError("wave_doc has no dir")
    return doc


def _grid_out(doc: dict) -> dict:
    """The output block ``STLU_PipelineInput`` stores inside ``grid_doc``."""
    if not isinstance(doc, dict):
        return {}
    out = doc.get("out")
    return out if isinstance(out, dict) else {}


def _previous_plan_boxes(previous):
    """The full old selection geometry, not just the tiles that were written.

    A previous partial run may have written only two of ten grid tiles; the
    visible ``select`` string still refers to the ten-index grid.  Reconstruct
    the grid from the manifest's geometry (or use the old strip regions), so
    mapping the user's old choice cannot turn "top two tiles" into "all
    strips".
    """
    if not isinstance(previous, dict):
        return []
    plan = previous.get("partition_plan") or {}
    regions = plan.get("regions") or []
    boxes = []
    for region in regions:
        box = region.get("box") if isinstance(region, dict) else region
        if box is not None and len(box) >= 4:
            boxes.append(tuple(int(v) for v in box[:4]))
    if boxes:
        return boxes
    source = previous.get("source") or [0, 0]
    tile_size = previous.get("tile_size") or [0, 0]
    try:
        width, height = int(source[0]), int(source[1])
        tile_w, tile_h = int(tile_size[0]), int(tile_size[1])
        overlap = int(previous.get("overlap") or 0)
        offset = previous.get("offset") or [0, 0]
        ox, oy = int(offset[0]), int(offset[1])
        if width > 0 and height > 0 and tile_w > 0 and tile_h > 0:
            return [tuple(int(v) for v in box)
                    for box in tile_psd.grid_coords(width, height, tile_w, tile_h,
                                                    overlap, ox, oy)]
    except (TypeError, ValueError, IndexError):
        pass
    boxes = []
    for entry in previous.get("tiles") or []:
        box = entry.get("tile") if isinstance(entry, dict) else None
        if box is not None and len(box) >= 4:
            boxes.append(tuple(int(v) for v in box[:4]))
    return boxes


def _map_selection_by_area(old_boxes, old_picked, new_coords):
    """Map checked areas from an old grid/plan onto new region boxes.

    A mode/plan change must not reinterpret old tile indices as new strip
    indices: that both runs the wrong strips and silently widens the user's
    selection.  A new box is kept only when at least half of it was already
    checked -- strip boxes overlap by design (519 px in a 1662 px strip), so
    plain rectangle intersection would widen one checked strip into every
    neighbour sharing the overlap band.  An all-selected old set stays
    all-selected; an empty set stays empty.
    """
    new_coords = [tuple(int(v) for v in box[:4]) for box in (new_coords or [])]
    old_boxes = [tuple(int(v) for v in box[:4]) for box in (old_boxes or [])]
    if not new_coords:
        return []
    if not old_boxes:
        return list(range(len(new_coords)))
    picked = [int(i) for i in (old_picked or []) if 0 <= int(i) < len(old_boxes)]
    if len(picked) >= len(old_boxes):
        return list(range(len(new_coords)))
    if not picked:
        return []
    checked = [old_boxes[i] for i in picked]
    mapped = []
    for index, box in enumerate(new_coords):
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        covered = 0
        for other in checked:
            w = min(box[2], other[2]) - max(box[0], other[0])
            h = min(box[3], other[3]) - max(box[1], other[1])
            if w > 0 and h > 0:
                covered += w * h
        if min(1.0, covered / float(area)) >= 0.5:
            mapped.append(index)
    return mapped


def _flatten_background(rgb: np.ndarray, subject: dict, arr: np.ndarray):
    """Paint a detected flat background white so only the subject reaches the model.

    ``arr`` is the full frame exactly as handed to us (alpha kept when the
    loader preserved it); only the RGB channels of background pixels change.
    Returns ``None`` when the subject mask is not trustworthy.
    """
    if str(subject.get("source") or "") not in ("background", "alpha"):
        return None
    fg = subject.get("foreground")
    if fg is None:
        return None
    mask = np.asarray(fg, dtype=bool)
    if mask.ndim != 2 or mask.shape[:2] != arr.shape[:2]:
        return None
    if not mask.any() or mask.all():
        return None
    flat = arr.copy()
    flat[..., :3][~mask] = 255
    return flat


def _plan_strips_frame(frame, width: int, height: int, *,
                       overlap: int, max_side: int,
                       margin: int, tile_width: int, tile_height: int,
                       resolution: int, grid_offset_x: int, grid_offset_y: int,
                       created: str) -> dict:
    """Plan strips from the executed frame, or build the visible fallback."""
    flat = None
    try:
        arr = np.clip(np.rint(frame.detach().cpu().numpy() * 255.0),
                      0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError("input is not an RGB image")
        rgb = np.ascontiguousarray(arr[..., :3])
        subject = tile_strips.detect_subject(rgb)
        plan = tile_strips.plan_from_subject(
            width, height, subject,
            overlap=int(overlap), max_side=int(max_side),
            margin=int(margin), resolution=int(resolution),
        )
        # ComfyUI's LoadImage drops alpha and keeps the RGB of transparent
        # pixels, so a cut-out character often reaches us over a solid
        # leftovers colour (here: magenta).  SeeThrough then paints that
        # colour as if it were part of the character -- on a strip that is
        # mostly background it swallows the subject entirely.  Flatten the
        # detected background to white before any tile is cut.
        flat = _flatten_background(rgb, subject, arr)
    except Exception as exc:  # noqa: BLE001 - visible fallback, not a crash
        plan = {
            "ok": False, "fallback": "uniform",
            "error": "分区规划失败 / partition planning failed: %s" % exc, "warnings": [],
        }

    if plan.get("ok"):
        doc = tile_strips.strips_document(
            plan, width, height, source_image="", name="grid", created=created)
        meta = dict(doc.get("partition_plan") or {})
        print("[STLU] subject strips: %d regions, side=%d, resolution=%d, "
              "density=%.2f src px/mask px" % (
                  len(doc["coords"]), int(plan["side"]), int(plan["resolution"]),
                  float(plan.get("density") or 0.0)), flush=True)
        if flat is not None:
            print("[STLU] background flattened to white before cutting strips "
                  "(alpha-less input kept a solid leftovers colour)", flush=True)
        return {
            "doc": doc,
            "coords": doc["coords"],
            "resolution": int(plan["resolution"]),
            "plan_meta": meta,
            "warnings": list(plan.get("warnings") or []),
            "fallback": None,
            "flat": flat,
        }

    # Explicit, visible fallback: never silently cut a too-wide subject with a
    # vertical seam.  Keep the user's grid phase so the panel's fallback grid
    # and the crops are the same geometry.
    doc = tile_psd.grid_document(
        (width, height), (tile_width, tile_height), int(overlap),
        source_image="", name="grid", created=created,
        offset_x=int(grid_offset_x), offset_y=int(grid_offset_y),
    )
    fallback = {
        "code": "planner",
        "message": str(plan.get("error") or "无法用方形分区覆盖主体"
                                          " / cannot cover the subject with square partitions"),
    }
    warnings = list(plan.get("warnings") or []) + [fallback["message"]]
    meta = {
        "mode": "uniform_fallback",
        "fallback": fallback,
        "warnings": warnings,
        "subject": plan.get("subject") or {},
        "subject_detection": plan.get("subject_detection") or {},
    }
    doc["mode"] = "uniform_fallback"
    doc["partition_plan"] = meta
    print("[STLU] subject strips fallback -> uniform grid: %s"
          % fallback["message"], flush=True)
    return {
        "doc": doc,
        "coords": doc["coords"],
        "resolution": int(resolution),
        "plan_meta": meta,
        "warnings": warnings,
        "fallback": fallback,
    }


def _resolve_picked_selection(raw_select: str, coords, *, is_strips: bool,
                              planned_id: str, select_plan: str,
                              previous_dir: str, width: int, height: int,
                              tile_width: int, tile_height: int, overlap: int,
                              grid_offset_x: int, grid_offset_y: int,
                              is_custom: bool = False):
    """Resolve the checked regions without misreading an old index space.

    ``select_plan`` says which geometry the selection string was written for.
    When it does not match this run's plan (subject strips or user-placed
    windows), map the old checked AREA onto the new boxes.  The previous
    manifest is the best source; if the folder is new (or was cleared), the
    selection was made against the uniform grid this node is currently
    configured with, so reconstruct that grid instead of treating e.g.
    ``"0-1"`` as the first two windows.  An empty selection stays empty:
    this helper never widens the user's choice.
    """
    planned_id = str(planned_id or "")
    select_plan = str(select_plan or "").strip()
    if not ((is_strips or is_custom) and planned_id and select_plan != planned_id):
        return parse_tile_selection(raw_select, len(coords))

    old_boxes = []
    source = ""
    if previous_dir:
        try:
            old_boxes = _previous_plan_boxes(tile_psd.read_manifest(previous_dir))
            source = "manifest"
        except Exception as exc:  # noqa: BLE001 - use the configured grid
            print("[STLU] cannot read previous plan geometry: %s" % exc, flush=True)

    if not old_boxes and width > 0 and height > 0:
        try:
            old_boxes = [
                tuple(int(v) for v in box)
                for box in tile_psd.grid_coords(
                    int(width), int(height), int(tile_width), int(tile_height),
                    int(overlap), int(grid_offset_x), int(grid_offset_y))
            ]
            source = "configured uniform grid"
        except (TypeError, ValueError, IndexError):
            old_boxes = []

    if not old_boxes:
        return parse_tile_selection(raw_select, len(coords))

    old_picked = parse_tile_selection(raw_select, len(old_boxes))
    picked = _map_selection_by_area(old_boxes, old_picked, coords)
    if not picked and old_picked:
        # Old boxes barely cover the new plan (e.g. a single uniform tile in a
        # much taller strip): keep the literal indices rather than dropping the
        # choice.  Still never wider than what the user wrote.
        picked = parse_tile_selection(raw_select, len(coords))
    print("[STLU] mapped old selection (%s) from %d %s boxes -> %s"
          % (raw_select, len(old_boxes), source, picked or "none"), flush=True)
    return picked

def _seethrough_node_class(name: str):
    """Fetch a SeeThrough node class at execution time (no import cycle)."""
    try:
        import nodes as comfy_nodes  # ComfyUI's global node registry module
        return comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    except Exception:  # noqa: BLE001
        return None

def _parse_index_list(text: str) -> List[int]:
    """Parse a comma list such as "0-2,5,7" into ascending indices."""
    picked = set()
    for chunk in str(text or "").replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            picked.update(range(lo, hi + 1))
        else:
            try:
                picked.add(int(chunk))
            except ValueError:
                continue
    return sorted(picked)


def encode_tile_selection(indices: List[int]) -> str:
    """Compress indices into "all"/"none"/"0-2,5"."""
    arr = sorted({int(i) for i in indices})
    if not arr:
        return "none"
    parts = []
    start = prev = arr[0]
    for value in arr[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append(str(start) if start == prev else "%d-%d" % (start, prev))
        start = prev = value
    parts.append(str(start) if start == prev else "%d-%d" % (start, prev))
    return ",".join(parts)


def parse_tile_selection(text: str, count: int) -> List[int]:
    """Parse "all", "none" or a comma list such as "0-2,5,7"."""
    raw = str(text or "all").strip().lower()
    if not raw or raw == "all":
        return list(range(count))
    if raw == "none":
        return []
    return [i for i in _parse_index_list(raw) if 0 <= i < count]

__all__ = [
    "CATEGORY",
    "DEFAULT_SIZE_RULES",
    "DEFAULT_LAYERS_JSON",
    "_output_root",
    "_notify",
    "_prompt_id",
    "_image_fingerprint",
    "_model_identity",
    "_output_rel",
    "_plan_fingerprint",
    "_load_plan",
    "_empty_image",
    "_alpha_over",
    "_numpy_rgba",
    "_sample_region",
    "_resize_mask",
    "_first",
    "_list_values",
    "_resolve_workspace_path",
    "_load_grid",
    "_tile_index_map",
    "_wave_doc",
    "_grid_out",
    "_previous_plan_boxes",
    "_map_selection_by_area",
    "_flatten_background",
    "_plan_strips_frame",
    "_resolve_picked_selection",
    "_seethrough_node_class",
    "_parse_index_list",
    "encode_tile_selection",
    "parse_tile_selection",
]
