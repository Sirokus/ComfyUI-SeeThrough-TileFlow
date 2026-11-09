"""Stage logic for the transparent SeeThrough tile pipeline.

The four stages used to live inside ``STLU_PipelineInput`` / ``STLU_TilePSD`` /
``STLU_WaveIndex``.  They are plain functions here so that

* the compact nodes in ``nodes_pipeline`` keep their old, tested behaviour by
  calling them, and
* ``nodes_transparent`` can expose every step as its own small node without a
  second implementation drifting away from the first.

Nothing in this module talks to ComfyUI: no widgets, no UI payloads.  The only
side effects are the ones the pipeline always had (writing the generation
fingerprint, clearing a stale run folder, writing tile files).
"""

from __future__ import annotations

import datetime
import json
import os
import time
import traceback

import numpy as np
import torch

import folder_paths

if __package__:  # ComfyUI loads the folder as a package
    from . import core_compose, core_io, region_plan, tile_psd
    from .stlu import limits as stlu_limits
    from . import nodes_shared as _shared
else:  # standalone import used by the tests and tools
    import core_compose
    import core_io
    import region_plan
    import tile_psd
    from stlu import limits as stlu_limits
    import nodes_shared as _shared


def _first(value, default=None):
    return _shared._first(value, default)


def _list_values(value):
    return _shared._list_values(value)


def _output_root() -> str:
    return _shared._output_root()


def _notify(event, payload):
    return _shared._notify(event, payload)


# Settings carried by ``STLU_PipelineSettings``.  The order is the widget order
# of the old ``STLU_PipelineInput`` -- workflow JSON stores widget values by
# position, so it must not change.
SETTINGS_KEYS = (
    "tile_width", "tile_height", "overlap", "select", "resolution",
    "num_inference_steps", "depth_resolution", "random_seed", "tblr_split",
    "use_lama", "alpha_threshold", "out_dir", "group", "filename_prefix",
    "merge_name", "group_layers", "include_depth", "include_original",
    "guide_context", "guide_feather", "guide_strength", "guide_tolerance",
    "seam_feather", "grid_offset_x", "grid_offset_y", "partition_mode",
    "strip_max_side", "strip_max_resolution", "strip_margin", "select_plan",
    "context_pad", "custom_regions", "head_scope", "tile_tags",
    "depth_random_seed",
)

# SeeThrough v3 runs a fixed second pass over the head tags for every tile,
# whether or not a head is actually in it -- a leg/shirt tile gets an
# invented face, eyes and mouth.  These are the tags that pass produces; they
# are dropped from tiles the user did not mark as head tiles.  v2 has no such
# pass and must keep every tag.
HEAD_SCOPE_TOP_ROW = "自动（最上一行）"
HEAD_SCOPE_ALL = "全部块保留"
HEAD_SCOPE_NONE = "全部块去掉"
HEAD_SCOPE_CHOICES = (HEAD_SCOPE_TOP_ROW, HEAD_SCOPE_ALL, HEAD_SCOPE_NONE)
V3_HEAD_TAGS = ("headwear", "face", "irides", "eyebrow", "eyewhite",
                "eyelash", "eyewear", "ears", "earwear", "nose", "mouth")
V3_BODY_TAGS = ("front hair", "back hair", "head", "neck", "neckwear",
                "topwear", "handwear", "bottomwear", "legwear", "footwear",
                "tail", "wings", "objects")
# v2 runs a single pass over one fixed list (third-party nodes.py
# VALID_BODY_PARTS_V2); the user can whitelist a subset of it per tile.
V2_BODY_TAGS = ("hair", "headwear", "face", "eyes", "eyewear", "ears",
                "earwear", "nose", "mouth", "neck", "neckwear", "topwear",
                "handwear", "bottomwear", "legwear", "footwear", "tail",
                "wings", "objects")

DEFAULT_SETTINGS = {
    "tile_width": 1024, "tile_height": 1024, "overlap": 128, "select": "all",
    "resolution": 1024, "num_inference_steps": 30, "depth_resolution": -1,
    "random_seed": 0, "tblr_split": True, "use_lama": True,
    "alpha_threshold": 8, "out_dir": "SeeThrough/我的自设/", "group": "",
    "filename_prefix": "tile", "merge_name": "merged", "group_layers": True,
    "include_depth": False, "include_original": False, "guide_context": False,
    "guide_feather": 32, "guide_strength": 40, "guide_tolerance": 16,
    "seam_feather": 32, "grid_offset_x": 0, "grid_offset_y": 0,
    "partition_mode": "均匀网格", "strip_max_side": 2048,
    "strip_max_resolution": 768, "strip_margin": 32, "select_plan": "",
    "context_pad": 0, "custom_regions": "[]",
    "head_scope": HEAD_SCOPE_TOP_ROW, "tile_tags": "{}",
    # -1 keeps older folders compatible: carried-forward values mean "follow
    # the layer seed" instead of a separate depth seed.
    "depth_random_seed": -1,
}


def normalize_settings(**kwargs) -> dict:
    """A settings dict with every key present and the old types/floors applied.

    Missing keys fall back to the shipped defaults; values are only coerced
    where the old execute() coerced them (int/bool/str casts), so a workflow
    saved by an older version keeps behaving exactly as before.
    """
    out = dict(DEFAULT_SETTINGS)
    for key in SETTINGS_KEYS:
        if key in kwargs and kwargs[key] is not None:
            out[key] = kwargs[key]
    ints = ("tile_width", "tile_height", "overlap", "resolution",
            "num_inference_steps", "depth_resolution", "random_seed",
            "alpha_threshold", "guide_feather", "guide_strength",
            "guide_tolerance", "seam_feather", "grid_offset_x",
            "grid_offset_y", "strip_max_side", "strip_max_resolution",
            "strip_margin", "context_pad", "depth_random_seed")
    for key in ints:
        out[key] = int(out[key])
    for key in ("tblr_split", "use_lama", "group_layers", "include_depth",
                "include_original", "guide_context"):
        out[key] = bool(out[key])
    for key in ("select", "out_dir", "group", "filename_prefix", "merge_name",
                "partition_mode", "select_plan", "custom_regions",
                "head_scope", "tile_tags"):
        out[key] = str(out[key] if out[key] is not None else "")
    if out["head_scope"] not in HEAD_SCOPE_CHOICES:
        out["head_scope"] = HEAD_SCOPE_TOP_ROW
    return out


def _head_tile_set(coords, scope):
    """Grid indices that may keep their v3 head layers.

    ``None`` means "every tile"; an empty set means "none".  The default
    scope keeps only the top row of the grid -- the library is upright
    full-body art, so the head is always there.
    """
    total = len(coords or [])
    if scope == HEAD_SCOPE_ALL:
        return None
    if scope == HEAD_SCOPE_NONE:
        return set()
    ys = []
    for box in coords or []:
        try:
            ys.append(int(box[1]))
        except (TypeError, IndexError, ValueError):
            ys.append(0)
    if not ys:
        return set()
    top = min(ys)
    return {i for i, y in enumerate(ys) if y == top} if total else set()


def _tile_tag_config(raw):
    """Per-tile tag whitelist, normalised to ``{index: {"body": ..., "head": ...}}``.

    ``raw`` is the ``tile_tags`` widget string: ``{"tiles": {"<index>": {...}}}``.
    A bare ``{index: {...}}`` map is accepted too.  A group key that is
    *absent* means "automatic" (keep everything the model produced); an
    explicit list -- even an empty one -- is the whitelist of tags to keep.
    Unknown tags are dropped so a stale workflow cannot invent new ones.
    """
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw) if raw.strip() else {}
        except (TypeError, ValueError):
            data = {}
    if not isinstance(data, dict):
        return {}
    tiles = data.get("tiles", data)
    if not isinstance(tiles, dict):
        return {}
    cfg = {}
    for key, value in tiles.items():
        if not isinstance(value, dict):
            continue
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        entry = {}
        for group, universe in (("body", V3_BODY_TAGS + V2_BODY_TAGS),
                                ("head", V3_HEAD_TAGS)):
            if group not in value:
                continue
            keep = value[group]
            if not isinstance(keep, (list, tuple)):
                continue
            entry[group] = frozenset(str(t) for t in keep if str(t) in universe)
        if entry:
            cfg[index] = entry
    return cfg


def _tile_drop_tags(index, cfg, head_tiles, is_v3):
    """Tags to pop from this tile's layer_dict, or ``None`` for "keep all".

    An explicit per-tile whitelist always wins.  Otherwise the v3 head pass is
    dropped from tiles the head scope did not mark (unless it marked every
    tile / ``head_tiles is None``).
    """
    tile_cfg = (cfg or {}).get(index) or {}
    body_universe = V3_BODY_TAGS if is_v3 else V2_BODY_TAGS
    head_universe = V3_HEAD_TAGS if is_v3 else ()
    drop = set()
    body_keep = tile_cfg.get("body")
    if body_keep is not None:
        drop.update(t for t in body_universe if t not in body_keep)
    head_keep = tile_cfg.get("head")
    if head_keep is not None:
        drop.update(t for t in head_universe if t not in head_keep)
    elif is_v3 and head_tiles is not None and index not in head_tiles:
        drop.update(head_universe)
    return tuple(sorted(drop)) or None


# ---------------------------------------------------------------------------
# head-pass skip hook
# ---------------------------------------------------------------------------
# The v3 branch of the third-party generator always runs a second, full
# diffusion pass over the head crop -- on every tile, head or not.  On tiles
# where every head tag ends up dropped that pass is pure wasted compute (a
# whole 30-step run per tile).  Instead of forking the third-party source, we
# wrap the pipeline's ``__call__`` once per process: while our flag is set, a
# call with ``group_index=1`` hands back transparent images, and the
# generator's head block then writes layers we discard anyway.
_HEAD_PASS_SKIP = {"active": False}
_HOOKED_PIPELINES = set()


def _empty_head_output(fullpage):
    """Stand-in for the head-group pipeline output (transparent canvases)."""
    arr = np.asarray(fullpage)
    blank = np.zeros(arr.shape[:2] + (4,), dtype=np.uint8)
    images = [blank.copy() for _ in range(len(V3_HEAD_TAGS))]
    try:
        from modules.layerdiffuse.diffusers_kdiffusion_sdxl import (
            LayerdiffPipelineOutput)
    except Exception:  # noqa: BLE001 - upstream import layout may change
        LayerdiffPipelineOutput = None
    if LayerdiffPipelineOutput is not None:
        return LayerdiffPipelineOutput(images=images, vis_list=[])
    import types
    return types.SimpleNamespace(images=images, vis_list=[])


def _install_head_skip_hook(pipeline):
    """Route the head-group call through the skip check (idempotent)."""
    cls = type(pipeline)
    if cls in _HOOKED_PIPELINES:
        return
    original = cls.__call__
    if not callable(original):  # pragma: no cover - defensive
        return

    def __call__(self, *args, **kwargs):
        if _HEAD_PASS_SKIP["active"] and kwargs.get("group_index") == 1:
            return _empty_head_output(kwargs.get("fullpage"))
        return original(self, *args, **kwargs)

    cls.__call__ = __call__
    _HOOKED_PIPELINES.add(cls)


def _skips_head_pass(drop_tags, is_v3):
    """True when this tile keeps no head tag, so the second pass is wasted."""
    if not is_v3 or not drop_tags:
        return False
    return all(tag in drop_tags for tag in V3_HEAD_TAGS)


# ---------------------------------------------------------------------------
# ① planning
# ---------------------------------------------------------------------------

def make_plan(image, settings, layerdiff_model=None, depth_model=None) -> dict:
    """Cut the source geometry into a plan: grid / strips / custom windows.

    Pure except for the (cheap) subject detection that drives the strip
    planner.  No file is written and no folder is cleared -- that is
    :func:`prepare_run`'s job, so the two are visibly separate steps.
    """
    settings = normalize_settings(**dict(settings or {}))
    if image is None or not hasattr(image, "shape") or image.ndim < 3:
        raise ValueError("STLU ①: 需要连接 image（标准 LoadImage 即可）/ image input required")
    try:
        (resolution, num_inference_steps, depth_resolution,
         random_seed) = stlu_limits.check_all(
            settings["resolution"], settings["num_inference_steps"],
            settings["depth_resolution"], settings["random_seed"])
        # -1 means "follow the layer seed" (the pre-split behaviour); the plan
        # carries the number actually used so previews can show it back.
        depth_seed = stlu_limits.resolve_depth_seed(
            stlu_limits.check_depth_seed(settings["depth_random_seed"]),
            random_seed)
    except ValueError as exc:
        # 旧节点名保留在信息里，测试与用户报错都按这个名字找。
        raise ValueError("STLU_PipelineInput: %s" % exc) from exc

    batch = image if image.ndim == 4 else image[None]
    frame = batch[0].detach().float().cpu()
    height, width = int(frame.shape[0]), int(frame.shape[1])
    created = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    partition_mode = str(settings["partition_mode"] or "").strip()
    is_strips = partition_mode == "方形全宽分区"
    is_custom = partition_mode == region_plan.CUSTOM_LABEL
    effective_resolution = int(resolution)
    effective_depth_resolution = int(depth_resolution)
    plan_meta: dict = {}
    plan_warnings: list = []
    fallback = None
    flat = None

    # The run folder is needed before selection mapping (the previous manifest
    # is the authoritative old geometry).
    wave_dir = tile_psd.flat_directory(
        _output_root(), settings["out_dir"], settings["group"])

    if is_strips:
        planned = _shared._plan_strips_frame(
            frame, width, height,
            overlap=settings["overlap"], max_side=settings["strip_max_side"],
            margin=settings["strip_margin"],
            tile_width=settings["tile_width"], tile_height=settings["tile_height"],
            resolution=settings["resolution"],
            grid_offset_x=settings["grid_offset_x"],
            grid_offset_y=settings["grid_offset_y"],
            created=created)
        doc = planned["doc"]
        coords = planned["coords"]
        effective_resolution = int(planned["resolution"])
        plan_meta = planned["plan_meta"]
        plan_warnings = list(planned["warnings"])
        fallback = planned["fallback"]
        flat = planned.get("flat")
        if effective_depth_resolution > effective_resolution:
            effective_depth_resolution = effective_resolution
            plan_warnings.append(
                "深度分辨率高于分区层分辨率，已跟随层分辨率 %d。"
                " / depth resolution was above the partition resolution;"
                " it now follows %d." % (effective_resolution,
                                         effective_resolution))
    elif is_custom:
        custom_warnings = []
        regions = region_plan.parse_custom_regions(
            settings["custom_regions"], width, height, custom_warnings)
        doc = region_plan.custom_document(
            regions, width, height,
            global_resolution=int(effective_resolution),
            created=created, overlap=settings["overlap"],
            warnings=custom_warnings)
        coords = doc["coords"]
        plan_meta = dict(doc["partition_plan"])
        plan_warnings = list(custom_warnings)
        fallback = None
    else:
        doc = tile_psd.grid_document(
            (width, height), (settings["tile_width"], settings["tile_height"]),
            settings["overlap"], source_image="", name="grid", created=created,
            offset_x=settings["grid_offset_x"],
            offset_y=settings["grid_offset_y"])
        coords = doc["coords"]

    # 块内推理分辨率低于块本身时，模型先下采样再上采样，细节与语义都会退化。
    # 上游 SeeThrough 默认 1280，这里只在明显不匹配时提示，不改用户选择。
    try:
        biggest = max(max(int(c[2]) - int(c[0]), int(c[3]) - int(c[1]))
                      for c in coords) if coords else 0
    except Exception:  # noqa: BLE001
        biggest = 0
    if biggest and effective_resolution < biggest:
        plan_warnings.append(
            "推理分辨率 %d 小于最大块边长 %d：每块会先缩小再放大，"
            "拆层容易整块丢失。建议 resolution 调到 %d。"
            " / resolution %d < max tile side %d; raise it to %d."
            % (effective_resolution, biggest, biggest,
               effective_resolution, biggest, biggest))

    return {
        "settings": settings,
        "doc": doc,
        "coords": coords,
        "width": int(width),
        "height": int(height),
        "wave_dir": wave_dir,
        "is_strips": bool(is_strips),
        "is_custom": bool(is_custom),
        "resolution": int(effective_resolution),
        "depth_resolution": int(effective_depth_resolution),
        "num_inference_steps": int(num_inference_steps),
        "seed": int(random_seed),
        "depth_seed": int(depth_seed),
        "partition": plan_meta,
        "warnings": plan_warnings,
        "fallback": fallback,
        "flat": flat,
        "created": created,
        "layerdiff_model_id": _shared._model_identity(layerdiff_model),
        "depth_model_id": _shared._model_identity(depth_model),
        "picked": None,
        "indices": None,
        "selection": None,
        "run": None,
        "reset": None,
    }


def pick_tiles(plan) -> dict:
    """Resolve the checked tiles, remapping an old selection space if needed."""
    plan = dict(plan)
    settings = plan["settings"]
    coords = plan["coords"]
    raw_select = str(settings["select"] if settings["select"] is not None
                     else "all").strip()
    picked = _shared._resolve_picked_selection(
        raw_select, coords, is_strips=plan["is_strips"],
        is_custom=plan["is_custom"],
        planned_id=str((plan["partition"] or {}).get("plan_id") or ""),
        select_plan=str(settings["select_plan"] or ""),
        previous_dir=plan["wave_dir"], width=plan["width"], height=plan["height"],
        tile_width=settings["tile_width"], tile_height=settings["tile_height"],
        overlap=settings["overlap"], grid_offset_x=settings["grid_offset_x"],
        grid_offset_y=settings["grid_offset_y"])
    if plan["is_custom"]:
        # Custom windows run in the order the user arranged them, not in
        # index order.  A checked subset keeps that order too, because the
        # guide only flows forward through the run.
        order = list((plan["partition"] or {}).get("order") or range(len(coords)))
        chosen = set(int(i) for i in picked)
        picked = [int(i) for i in order if int(i) in chosen]
    if not picked:
        mode_hint = "方形分区" if plan["is_strips"] else "均匀网格"
        mode_en = "square partitions" if plan["is_strips"] else "uniform grid"
        raise ValueError(
            "STLU ①: 当前%s没有勾选任何块/条（select=%r）"
            " / no tiles selected in %s (select=%r)"
            " / click Select all in the ① panel or pick the regions to generate, then queue."
            % (mode_hint, raw_select, mode_en, raw_select))
    encoded = ("all" if len(picked) == len(coords)
               else _shared.encode_tile_selection(picked))
    doc = dict(plan["doc"])
    doc["selection"] = encoded
    plan["doc"] = doc
    plan["picked"] = [int(i) for i in picked]
    plan["indices"] = [str(int(i)) for i in picked]
    plan["selection"] = encoded
    return plan


def plan_context(plan) -> dict:
    """Grow each selected tile's input window by ``context_pad`` pixels."""
    plan = dict(plan)
    settings = plan["settings"]
    coords = plan["coords"]
    doc = dict(plan["doc"])
    pad = max(0, min(512, int(settings["context_pad"] or 0)))
    context_boxes = {}
    if pad > 0:
        for index in plan["picked"] or []:
            x0, y0, x1, y1 = (int(v) for v in coords[index])
            box = [max(0, x0 - pad), max(0, y0 - pad),
                   min(plan["width"], x1 + pad), min(plan["height"], y1 + pad)]
            if box != [x0, y0, x1, y1]:
                context_boxes[str(index)] = box
    doc["context_boxes"] = context_boxes
    doc["context_pad"] = int(pad)
    plan["doc"] = doc
    return plan


def _generation_block(plan) -> dict:
    settings = plan["settings"]
    doc = plan["doc"]
    guide_block = {
        "enabled": bool(settings["guide_context"]),
        "feather": max(0, int(settings["guide_feather"])),
        "strength": min(100, max(0, int(settings["guide_strength"]))),
        "tolerance": max(0, min(255, int(settings["guide_tolerance"]))),
    }
    # Everything that can change the pixels of a tile.  Merge-only options
    # (group_layers / depth / original / seam_feather) are deliberately
    # absent: changing them only needs ④ to run again, not a re-diffusion.
    # This is a record for tracing a folder back to its settings; it is never
    # compared automatically -- only structure decides whether a folder is
    # cleared (see tile_psd.cache_mismatch_reason).
    # Only the tiles being generated matter: editing the whitelist of an
    # un-ticked tile must not invalidate the thumbnails of the ticked ones.
    cfg = _tile_tag_config(settings["tile_tags"])
    picked = plan.get("picked")
    tile_tags = {}
    for index in (picked if picked is not None else sorted(cfg)):
        entry = cfg.get(int(index))
        if not entry:
            continue
        tile_tags[str(index)] = {
            group: sorted(entry[group])
            for group in ("body", "head") if group in entry
        }
    generation = {
        "version": 1,
        # Bumped whenever tile pixels can change; kept in the record so a
        # folder's tiles can be traced back to the code that made them.
        "algo_version": int(tile_psd.ALGO_VERSION),
        "source": plan.get("fingerprint") or "",
        "source_size": [int(plan["width"]), int(plan["height"])],
        "tile_size": [int(v) for v in doc["tile_size"]],
        "overlap": int(doc["overlap"]),
        "resolution": int(plan["resolution"]),
        "num_inference_steps": int(plan["num_inference_steps"]),
        "depth_resolution": int(plan["depth_resolution"]),
        "random_seed": int(plan["seed"]),
        "depth_random_seed": int(plan.get("depth_seed", plan["seed"])),
        "tblr_split": bool(settings["tblr_split"]),
        "use_lama": bool(settings["use_lama"]),
        "alpha_threshold": int(settings["alpha_threshold"]),
        "context_pad": int(settings["context_pad"] or 0),
        "head_scope": str(settings["head_scope"]),
        "tile_tags": tile_tags,
        "guide": guide_block,
        "layerdiff_model": plan.get("layerdiff_model_id") or "",
        "depth_model": plan.get("depth_model_id") or "",
    }
    meta = plan["partition"] or {}
    if plan["is_strips"]:
        # Only geometry goes into the record: warnings/notes can change
        # without changing a single pixel.
        generation["partition"] = {
            key: meta.get(key)
            for key in ("mode", "side", "resolution", "density", "overlap",
                        "requested_overlap", "margin", "plan_id", "regions")
            if meta.get(key) is not None
        }
        if meta.get("fallback"):
            generation["partition"]["fallback"] = meta["fallback"]
        # A fallback run really is the uniform grid, including its phase.
        if (str(doc.get("mode") or "") == "uniform_fallback"
                and (int(settings["grid_offset_x"]) or int(settings["grid_offset_y"]))):
            generation["offset"] = [int(settings["grid_offset_x"]),
                                    int(settings["grid_offset_y"])]
    elif plan["is_custom"]:
        # The region list is in run order, so moving a window up/down or
        # changing a per-window resolution is recorded too: both change what
        # the guide sees.
        generation["partition"] = {
            "mode": region_plan.CUSTOM_MODE,
            "plan_id": str(meta.get("plan_id") or ""),
            "regions": [{"box": list(r["box"]),
                         "resolution": int(r["resolution"])}
                        for r in (meta.get("regions") or [])],
        }
    # Only when the grid is actually shifted: keeping the key out at 0 means
    # folders written by the previous version stay compatible.
    elif int(settings["grid_offset_x"]) or int(settings["grid_offset_y"]):
        generation["offset"] = [int(settings["grid_offset_x"]),
                                int(settings["grid_offset_y"])]
    return generation, guide_block


def prepare_run(plan, image=None) -> dict:
    """Record the generation settings and settle reuse of the folder.

    The folder is the cache: every finished tile in it joins this run whatever
    the diffusion settings were.  Only a structural mismatch -- another
    partition type, another source image, a grid whose boxes moved -- clears
    it, because those tiles could never sit in this run's boxes.
    """
    plan = dict(plan)
    settings = plan["settings"]
    doc = dict(plan["doc"])
    if image is None:
        image = plan.get("source_frame")
    if image is None or not hasattr(image, "shape") or image.ndim < 3:
        raise ValueError("STLU ①: 准备运行需要连接 image / Prepare-run needs the image input")
    batch = image if image.ndim == 4 else image[None]
    source_frame = batch[0].detach().float().cpu()
    fingerprint = _shared._image_fingerprint(source_frame)
    plan["fingerprint"] = fingerprint

    wave_dir = plan["wave_dir"]
    rel = os.path.relpath(wave_dir, _output_root()).replace(os.sep, "/")
    # Drop a locked-parts store left behind by an older version of this pack.
    tile_psd.discard_legacy_locks(wave_dir)
    wave = os.path.basename(wave_dir.rstrip("/\\")) or "run"
    doc.setdefault("source", {})["fingerprint"] = fingerprint
    generation, guide_block = _generation_block(plan)
    doc["guide"] = guide_block
    doc["generation"] = generation

    # Nothing in the folder is touched for a settings change: only the ticked
    # tiles run, and every other tile already there is reused as it is.  A
    # structural mismatch (another partition type / another source image /
    # moved boxes) is different -- those tiles would land in the wrong place,
    # so the whole folder is cleared and the user is told why.
    reset_info = None
    picked = list(plan["picked"])
    previous = tile_psd.read_manifest(wave_dir)
    mismatch = tile_psd.cache_mismatch_reason(previous, doc)
    if mismatch:
        tile_psd.clear_run_outputs(wave_dir, previous)
        tile_psd.verify_run_cleared(wave_dir)
        # The cleared tiles are gone, so a subset would leave the merge
        # incomplete: this one automatic case runs the whole grid.  (The
        # selection itself is never widened without clearing.)
        everything = list(range(len(doc.get("coords") or [])))
        if picked != everything:
            picked = everything
            plan["picked"] = list(everything)
            plan["indices"] = [str(i) for i in everything]
            plan["selection"] = "all"
            doc["selection"] = "all"
        reset_info = {
            "required": False, "changed": True, "reason": mismatch,
            "keys": [], "tiles": len(picked),
        }
        _notify("stlu.cache_reset", {
            "dir": wave_dir, "rel": rel, "reason": mismatch,
            "tiles": len(picked),
        })
        print("[STLU] %s in %s -> cleared the folder before running; "
              "all %d tile(s) will be generated again"
              % (mismatch, rel, len(picked)), flush=True)

    out_info = {
        "dir": wave_dir,
        "rel": rel,
        "wave": wave,
        "flat": True,
        "fingerprint": fingerprint,
        "out_dir": str(settings["out_dir"] or "SeeThrough"),
        "group": str(settings["group"] or "").strip(),
        "prefix": str(settings["filename_prefix"] or "tile"),
        "alpha_threshold": int(settings["alpha_threshold"]),
        "merge_name": str(settings["merge_name"] or "merged"),
        "group_layers": bool(settings["group_layers"]),
        "include_depth": bool(settings["include_depth"]),
        "include_original": bool(settings["include_original"]),
        "seam_feather": max(0, int(settings["seam_feather"])),
    }
    if reset_info:
        out_info["reset"] = reset_info
    doc["out"] = out_info
    try:
        tile_psd.write_generation(wave_dir, generation)
    except OSError as exc:
        raise RuntimeError(
            "STLU ①: 无法写入生成参数 %s：%s。"
            "请检查输出目录权限后重试。"
            " / cannot write the generation settings %s: %s;"
            " check the output folder's permissions and retry."
            % (wave_dir, exc, wave_dir, exc)) from exc

    plan["doc"] = doc
    plan["run"] = {
        "dir": wave_dir, "rel": rel, "wave": wave,
        "fingerprint": fingerprint, "out": out_info,
        "generation": generation,
    }
    plan["reset"] = reset_info
    _notify("stlu.grid", grid_info(plan))
    return plan


def crop_tiles(image, plan) -> tuple:
    """Cut the selected tiles out of the source image (pure)."""
    batch = image if image.ndim == 4 else image[None]
    frame = batch[0].detach().float().cpu()
    flat = plan.get("flat")
    if flat is not None:
        frame = torch.from_numpy(np.asarray(flat, dtype=np.float32) / 255.0).to(
            frame.device)
    coords = plan["coords"]
    context_boxes = (plan["doc"] or {}).get("context_boxes") or {}
    tiles = []
    indices = []
    for index in plan["picked"]:
        box = context_boxes.get(str(index))
        if box:
            x0, y0, x1, y1 = box
        else:
            x0, y0, x1, y1 = (int(v) for v in coords[index])
        crop = frame[int(y0):int(y1), int(x0):int(x1), :]
        tiles.append(crop[None].contiguous())
        indices.append(str(index))
    return tiles, indices


def grid_info(plan) -> dict:
    """The payload the ① panel consumes after a run."""
    settings = plan["settings"]
    doc = plan["doc"]
    run = plan.get("run") or {}
    out_info = run.get("out") or (doc.get("out") or {})
    rel = run.get("rel") or out_info.get("rel") or ""
    return {
        "grid": "",
        "grid_doc": "",
        "dir": rel,
        "mode": str(doc.get("mode") or "uniform"),
        "partition": plan["partition"],
        "warnings": list(plan["warnings"]),
        "fallback": plan["fallback"],
        "count": int(doc.get("count") or 0),
        "selected": len(plan.get("picked") or []),
        "indices": list(plan.get("indices") or []),
        "rows": int(doc.get("rows") or 0),
        "columns": int(doc.get("columns") or 0),
        "source": [int(plan["width"]), int(plan["height"])],
        "tile_size": [int(v) for v in (doc.get("tile_size") or [0, 0])],
        "overlap": int(doc.get("overlap") or 0),
        "resolution": int(plan["resolution"]),
        "selection": plan.get("selection") or "all",
        "out": out_info,
    }


def describe_run(plan) -> None:
    """One line in the log naming what this run will do."""
    settings = plan["settings"]
    doc = plan["doc"]
    picked = plan.get("picked") or []
    rel = (plan.get("run") or {}).get("rel") or ""
    if plan["is_custom"]:
        print("[STLU] input: %d/%d custom windows (%s) from %dx%d -> %s "
              "(%s, order %s)" % (
                  len(picked), doc["count"],
                  region_plan.summaries(plan["partition"].get("regions") or [],
                                        int(plan["resolution"])),
                  plan["width"], plan["height"], rel, plan.get("selection"),
                  ",".join(str(i) for i in picked)), flush=True)
    elif plan["is_strips"] and str(doc.get("mode") or "") == "subject_strips":
        print("[STLU] input: %d/%d strips (side %d, resolution %d) from %dx%d -> %s (%s)" % (
            len(picked), doc["count"], doc["tile_size"][0],
            int(plan["resolution"]), plan["width"], plan["height"], rel,
            plan.get("selection")), flush=True)
    else:
        print("[STLU] input: %d/%d tiles (%dx%d) from %dx%d -> %s (%s)" % (
            len(picked), doc["count"], doc["tile_size"][0], doc["tile_size"][1],
            plan["width"], plan["height"], rel, plan.get("selection")),
            flush=True)


# ---------------------------------------------------------------------------
# ② per-tile loop support
# ---------------------------------------------------------------------------

def guide_state(plan) -> dict:
    """Everything the guide needs from the plan, resolved once per run."""
    doc = plan.get("doc") or {}
    guide = dict(doc.get("guide") or {})
    regions_by_index = {}
    for region in ((doc.get("partition_plan") or {}).get("regions") or []):
        if not isinstance(region, dict):
            continue
        try:
            regions_by_index[int(region["index"])] = int(region["resolution"])
        except (KeyError, TypeError, ValueError):
            continue
    context_boxes = {}
    for key, box in (doc.get("context_boxes") or {}).items():
        try:
            context_boxes[int(key)] = [int(v) for v in box[:4]]
        except (TypeError, ValueError, IndexError):
            continue
    coords = list(doc.get("coords") or [])
    generation = doc.get("generation") or {}
    scope = str(generation.get("head_scope") or HEAD_SCOPE_TOP_ROW)
    return {
        "guide": guide,
        "coords": coords,
        "context_boxes": context_boxes,
        "regions_by_index": regions_by_index,
        "alpha_threshold": int(((doc.get("out") or {}).get("alpha_threshold")) or 8),
        "head_tiles": _head_tile_set(coords, scope),
        "tile_tags": _tile_tag_config(generation.get("tile_tags")),
    }


def guide_tile(tile, index, state, generated, fixed=None):
    """Paste finished neighbours into one tile's overlap before generation."""
    return _apply_tile_guide(
        tile, int(index), state["coords"], generated, state["guide"],
        state["alpha_threshold"], fixed=fixed,
        input_box=state["context_boxes"].get(int(index)),
        done=set(generated))


def crop_back(parts, composite, index, state):
    """Cut a context-padded result back to the strip box."""
    input_box = state["context_boxes"].get(int(index))
    if input_box:
        return _crop_tile_parts(
            parts, composite, state["coords"][int(index)], input_box)
    return parts, composite


def run_guided_tiles(tiles, indices, state, run_dir, layerdiff_model,
                     depth_model, seed, resolution, steps, depth_resolution,
                     tblr_split, use_lama, progress=None, depth_seed=None) -> tuple:
    """The sequential per-tile loop: guide -> chain -> crop back.

    This is the one place a loop is genuinely required: tile N has to see the
    pixels tile N-1 produced, which no acyclic node graph can express.  Every
    step it calls is a separate function (``guide_tile`` / ``_run_seethrough_tile``
    / ``crop_back``), so the graph still shows what happens.

    ``progress`` is an optional callable receiving one dict per milestone
    (``phase`` / ``pos`` / ``total`` / ``index`` / ``done`` / ``seconds`` /
    ``tile_seconds`` / ``cached`` / ``empty``).  The node passes
    the websocket notifier here; plain callers omit it.  Every aspect of a
    tile's work inside this loop (model reload included) takes minutes, so
    without these events the UI sits on one unchanged line.
    """
    import comfy.model_management
    import comfy.utils

    layer_cls = _shared._seethrough_node_class("SeeThrough_GenerateLayers")
    depth_cls = _shared._seethrough_node_class("SeeThrough_GenerateDepth")
    post_cls = _shared._seethrough_node_class("SeeThrough_PostProcess")
    if not (layer_cls and depth_cls and post_cls):
        raise RuntimeError(
            "STLU ②: 找不到 SeeThrough 节点（ComfyUI-See-through 未加载？）"
            " / STLU ②: SeeThrough nodes not found (is ComfyUI-See-through not loaded?)")

    layer_node, depth_node, post_node = layer_cls(), depth_cls(), post_cls()

    model = _first(layerdiff_model)
    depth_model = _first(depth_model)
    seed = int(_first(seed, 42))
    resolution = int(_first(resolution, 1024))
    steps = int(_first(steps, 30))
    depth_res = int(_first(depth_resolution, -1))
    try:
        (resolution, steps, depth_res,
         seed) = stlu_limits.check_all(resolution, steps, depth_res, seed)
        raw_depth_seed = (stlu_limits.DEPTH_SEED_FOLLOW if depth_seed is None
                          else int(_first(depth_seed,
                                          stlu_limits.DEPTH_SEED_FOLLOW)))
        depth_seed = stlu_limits.resolve_depth_seed(
            stlu_limits.check_depth_seed(raw_depth_seed), seed)
    except ValueError as exc:
        raise ValueError("STLU ②: %s" % exc) from exc
    tblr = bool(_first(tblr_split, True))
    lama = bool(_first(use_lama, True))

    guide = state["guide"]
    coords = state["coords"]
    regions_by_index = state["regions_by_index"]
    context_boxes = state["context_boxes"]
    alpha_threshold = state["alpha_threshold"]
    head_tiles = state.get("head_tiles")
    tile_tags = state.get("tile_tags") or {}
    # v3 alone has the forced second pass; v2's single pass legitimately sees
    # head tags, so its tiles must be left untouched unless the user set an
    # explicit per-tile whitelist.
    try:
        tag_version = str(model.unet.get_tag_version()) if model is not None else ""
    except Exception:  # noqa: BLE001 - unknown model object: keep old behaviour
        tag_version = ""
    is_v3 = tag_version == "v3"
    if is_v3 and head_tiles is not None:
        print("[STLU] guided: v3 head layers kept only on tiles %s" % (
            sorted(head_tiles) if head_tiles else "[]"), flush=True)
    if tile_tags:
        print("[STLU] guided: per-tile tag whitelist on tiles %s" % (
            sorted(tile_tags), ), flush=True)

    def _drop_for(tile_index):
        return _tile_drop_tags(int(tile_index), tile_tags, head_tiles, is_v3)
    if not coords:
        raise ValueError(
            "STLU ②: 计划里没有块坐标，无法做引导。"
            "请确认 ① 已正常完成并把 PLAN 接到 ②。"
            " / STLU ②: the plan has no tile coordinates, so guidance cannot"
            " run; make sure ① finished normally and its PLAN is wired to ②.")

    tiles = _list_values(tiles)
    raw_indices = _list_values(indices)
    total = len(tiles)
    if not total:
        return ([], [], [], [])

    started = time.monotonic()

    def _progress(phase, **info):
        if progress is None:
            return
        info["phase"] = phase
        info["total"] = total
        info["seconds"] = round(time.monotonic() - started, 1)
        try:
            progress(info)
        except Exception:  # noqa: BLE001 - a bad UI hook must not fail the run
            pass

    index_order = []
    for pos in range(total):
        raw_index = raw_indices[pos] if pos < len(raw_indices) else pos
        try:
            index_order.append(int(str(_first(raw_index, pos)).strip()))
        except (TypeError, ValueError):
            index_order.append(pos)

    # Tiles an earlier run already finished in this same folder are real
    # generated content: merge them as guide now, and step ③ puts them back
    # into the manifest so a one-tile re-run still merges complete.
    cached: dict = {}
    if run_dir and guide.get("enabled", True):
        try:
            entries = tile_psd.cached_tile_entries(
                run_dir, state.get("doc"), skip=index_order)
        except Exception as exc:  # noqa: BLE001
            entries = []
            print("[STLU] guided: cannot read cached tiles: %s" % exc, flush=True)
        for entry in entries:
            rgba = tile_psd.composite_psd(
                os.path.join(run_dir, str(entry.get("file") or "")))
            if rgba is not None:
                cached[int(entry["index"])] = rgba
        if cached:
            print("[STLU] guided: reusing %d finished tiles from %s" % (
                len(cached), run_dir), flush=True)

    print("[STLU] guided: %d tiles, context=%s feather=%s strength=%s" % (
        total, guide.get("enabled", True), guide.get("feather"),
        guide.get("strength")), flush=True)

    _progress("start", done=len(cached), cached=len(cached),
              order=list(index_order))

    pbar = comfy.utils.ProgressBar(total)
    generated: dict = dict(cached)
    fixed = set(cached)
    parts_out: list = []
    indices_out: list = []
    guided_out: list = []
    composite_out: list = []

    def _composite_tensor(rgba):
        if rgba is None:
            return None
        return torch.from_numpy(
            np.ascontiguousarray(rgba.astype(np.float32) / 255.0))[None]

    for pos, tile in enumerate(tiles):
        index = index_order[pos]
        comfy.model_management.throw_exception_if_processing_interrupted()
        tile_started = time.monotonic()
        _progress("guide", pos=pos + 1, index=index, done=pos,
                  tile_seconds=0.0)

        # Per-window resolution when the document names one; otherwise the
        # single value (uniform grid / subject strips).
        tile_res = int(regions_by_index.get(index, resolution))
        tile_depth = depth_res if depth_res == -1 else min(depth_res, tile_res)
        input_box = context_boxes.get(index)
        input_tile = guide_tile(tile, index, state, generated, fixed=fixed)
        guided_out.append(input_tile.detach().cpu().clamp(0, 1))

        def _phase(phase, _pos=pos + 1, _index=index, _done=pos):
            _progress(phase, pos=_pos, index=_index, done=_done,
                      tile_seconds=round(time.monotonic() - tile_started, 1))

        try:
            parts, composite = _run_seethrough_tile(
                layer_node, depth_node, post_node, input_tile, model,
                depth_model, seed, tile_res, steps, tile_depth, tblr,
                lama, alpha_threshold, on_phase=_phase,
                drop_tags=_drop_for(index), depth_seed=depth_seed)
        except comfy.model_management.InterruptProcessingException:
            raise
        except Exception as exc:  # noqa: BLE001 - add the tile number
            raise RuntimeError(
                "STLU ②: 第 %d 块生成失败（%s: %s）。\n%s"
                " / STLU ②: tile %d failed to generate (%s: %s).\n%s"
                % (index, type(exc).__name__, exc,
                   traceback.format_exc(), index, type(exc).__name__, exc,
                   traceback.format_exc())) from exc
        if composite["status"] == GUIDE_ERROR:
            raise RuntimeError(
                "STLU ②: 第 %d 块合成失败（%s）。这是代码/部件数据错误，"
                "不是模型生成的空块；已停止本次运行。\n%s"
                " / STLU ②: tile %d failed to composite (%s). This is a"
                " code/part-data error, not an empty tile from the model;"
                " the run has been stopped.\n%s"
                % (index, composite.get("error") or "unknown",
                   composite.get("traceback") or "",
                   index, composite.get("error") or "unknown",
                   composite.get("traceback") or ""))

        if composite["status"] == GUIDE_EMPTY:
            print("[STLU] guided: tile %d came back empty (seed %d / depth %d)"
                  % (index, seed, depth_seed), flush=True)
        else:
            if input_box:
                # The model saw the padded window; everything downstream
                # (merge geometry, PSD size, guides) wants the strip box.
                _phase("crop")
                parts, composite = _crop_tile_parts(
                    parts, composite, coords[index], input_box)
            generated[index] = composite["rgba"]
        parts_out.append(parts)
        indices_out.append(str(index))
        composite_out.append(_composite_tensor(composite.get("rgba")))
        pbar.update(1)
        _progress("tile_done", pos=pos + 1, index=index, done=pos + 1,
                  tile_seconds=round(time.monotonic() - tile_started, 1),
                  empty=composite["status"] == GUIDE_EMPTY,
                  cached=len(cached))
        print("[STLU] guided: tile %d done (%d/%d)" % (index, pos + 1, total),
              flush=True)

    _progress("end", done=total, cached=len(cached))
    return (parts_out, indices_out, guided_out, composite_out)


# ---------------------------------------------------------------------------
# Guide / crop helpers (moved from nodes_pipeline so both facades share them)
# ---------------------------------------------------------------------------

def _guide_feather_mask(height: int, width: int, feather: int) -> np.ndarray:
    """1.0 in the middle of the overlap, fading to 0 at its edge."""
    height = max(1, int(height))
    width = max(1, int(width))
    feather = max(0, int(feather))
    if feather <= 0:
        return np.ones((height, width), dtype=np.float32)
    fy = min(float(feather), max(0.0, (height - 1) / 2.0))
    fx = min(float(feather), max(0.0, (width - 1) / 2.0))
    yy = np.arange(height, dtype=np.float32)
    xx = np.arange(width, dtype=np.float32)
    # A 1-2 px span cannot hold a ramp: treat that axis as fully guided.
    wy = (np.ones(height, dtype=np.float32) if fy < 1.0
          else np.clip(np.minimum(yy, height - 1 - yy) / fy, 0.0, 1.0))
    wx = (np.ones(width, dtype=np.float32) if fx < 1.0
          else np.clip(np.minimum(xx, width - 1 - xx) / fx, 0.0, 1.0))
    return (wy[:, None] * wx[None, :]).astype(np.float32)


def _box_blur_3x3(arr: np.ndarray) -> np.ndarray:
    """Edge-padded 3x3 mean, so a per-pixel gate stays spatially coherent."""
    padded = np.pad(arr, 1, mode="edge")
    return (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]) / 9.0


GUIDE_OK = "ok"
GUIDE_EMPTY = "empty"
GUIDE_ERROR = "error"

def _guide_composite_result(parts: dict, alpha_threshold: int = 8) -> dict:
    """Flatten one tile's parts, distinguishing a real empty result from a bug.

    A fully transparent tile is a known model failure and is reported as
    ``empty`` so the loop can log it and move on.  An exception or a malformed
    ``frame_size`` is not an empty tile: it is reported as ``error`` with a
    traceback.
    """
    try:
        layers = tile_psd.build_tile_layers(
            parts, merge_parts=True, alpha_threshold=int(alpha_threshold))
    except Exception as exc:  # noqa: BLE001 - reported to the node layer
        return {
            "status": GUIDE_ERROR,
            "rgba": None,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }
    frame_h, frame_w = (int(v) for v in (parts.get("frame_size") or (0, 0))[:2])
    if frame_h <= 0 or frame_w <= 0:
        return {
            "status": GUIDE_ERROR,
            "rgba": None,
            "error": "parts have an invalid frame_size: %r"
                     % ((parts.get("frame_size"),)),
            "traceback": "",
        }
    if not layers:
        return {"status": GUIDE_EMPTY, "rgba": None, "error": "", "traceback": ""}
    canvas = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    for layer in layers:
        core_compose._alpha_over(canvas, layer["pixels"],
                                 int(layer["left"]), int(layer["top"]))
    if not canvas[..., 3].any():
        return {"status": GUIDE_EMPTY, "rgba": None, "error": "", "traceback": ""}
    return {"status": GUIDE_OK, "rgba": canvas, "error": "", "traceback": ""}


def _guide_composite(parts: dict, alpha_threshold: int = 8):
    """Backwards-compatible: the RGBA canvas, or ``None`` when unusable."""
    return _guide_composite_result(parts, alpha_threshold).get("rgba")


def _run_seethrough_tile(layer_node, depth_node, post_node, input_tile, model,
                         depth_model, seed, resolution, steps, depth_res,
                         tblr, lama, alpha_threshold, on_phase=None,
                         drop_tags=None, skip_head=False, depth_seed=None):
    """Generate -> depth -> post for one tile; return (parts, composite result).

    ``on_phase(name)`` is called before each of the three model steps.  They
    are where the minutes go (GenerateLayers alone can reload the text
    encoder), so the UI reports them by name instead of a frozen bar.

    ``seed`` drives the layer diffusion, ``depth_seed`` the depth estimate;
    ``None`` makes depth follow the layer seed (the old single-seed setup).

    ``drop_tags`` removes tags from the generated layer set *before* depth is
    estimated, so a tile the model was forced to hallucinate a face on simply
    ends up without those layers (and without depth/post/merge passes for
    them).  See :data:`V3_HEAD_TAGS`.

    ``skip_head`` also skips the head-group diffusion itself (a second full
    pass) -- only valid when every head tag is in ``drop_tags``.
    """
    if on_phase:
        on_phase("layers")
    _HEAD_PASS_SKIP["active"] = bool(skip_head)
    try:
        layers_data, _preview = layer_node.generate(
            input_tile, model, seed, resolution, steps)
    finally:
        _HEAD_PASS_SKIP["active"] = False
    if drop_tags:
        layer_dict = getattr(layers_data, "layer_dict", None)
        if isinstance(layer_dict, dict):
            removed = [tag for tag in drop_tags if tag in layer_dict]
            for tag in removed:
                layer_dict.pop(tag, None)
            if removed:
                print("[STLU] tile: dropped %d tag(s) (%s)" % (
                    len(removed), ", ".join(removed)), flush=True)
    # GenerateDepth returns (layers_depth, preview), not a bare object.
    if on_phase:
        on_phase("depth")
    depth_data, _preview_depth = depth_node.generate(
        layers_data, depth_model,
        seed if depth_seed is None else depth_seed, depth_res)
    if on_phase:
        on_phase("post")
    parts, _preview2 = post_node.process(depth_data, tblr, lama)
    return parts, _guide_composite_result(parts, alpha_threshold)

def _apply_tile_guide(tile, index: int, coords, generated: dict, guide: dict,
                      alpha_threshold: int = 8, fixed=None, input_box=None,
                      done=None):
    """Paste already-generated neighbouring tiles into this tile's overlap.

    ``fixed`` holds tiles cached from an earlier run: they are complete, so
    they may guide a tile that comes before them in grid order too.  Tiles
    produced earlier in *this* run are only used forward.

    ``done`` names the tiles finished in this run so far.  The guided node
    passes it because its tiles run in a user-defined order, where "index
    smaller than mine" no longer says which tile came first.  Standalone
    callers that omit it keep the original index-order contract.

    ``coords`` describes the content (strip) boxes the generated tiles cover;
    ``input_box`` is the larger window ① fed to the model when context
    padding is on.  They are the same box when padding is off.
    """
    if not guide.get("enabled", True) or not generated:
        return tile
    strength = min(100.0, max(0.0, float(guide.get("strength", 40)))) / 100.0
    if strength <= 0.0:
        return tile
    if index < 0 or index >= len(coords):
        return tile
    feather = int(guide.get("feather", 32))

    device, dtype = tile.device, tile.dtype
    arr = (tile[0].detach().float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    height, width = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in (input_box or coords[index]))
    tile_w, tile_h = max(1, x1 - x0), max(1, y1 - y0)
    scale_x = width / float(tile_w)
    scale_y = height / float(tile_h)

    changed = False
    fixed = fixed or set()
    done_set = None if done is None else set(done)
    # Earlier tiles only, unless the neighbour is a finished tile cached from
    # a previous run -- then its position in the queue does not matter.
    for other in sorted(generated):
        if other == index:
            # A cached copy of this very tile must not seed itself.
            continue
        if done_set is not None:
            if other not in done_set and other not in fixed:
                continue
        elif other >= index and other not in fixed:
            continue
        source = generated[other]
        ox0, oy0 = max(x0, int(coords[other][0])), max(y0, int(coords[other][1]))
        ox1, oy1 = min(x1, int(coords[other][2])), min(y1, int(coords[other][3]))
        if ox0 >= ox1 or oy0 >= oy1:
            continue

        sx0 = int(coords[other][0])
        sy0 = int(coords[other][1])
        other_w = max(1, int(coords[other][2]) - sx0)
        other_h = max(1, int(coords[other][3]) - sy0)
        rgba = source
        if rgba.shape[0] != other_h or rgba.shape[1] != other_w:
            rgba = core_io.resize_rgba_premultiplied(rgba, other_w, other_h)

        piece = rgba[oy0 - sy0:oy1 - sy0, ox0 - sx0:ox1 - sx0]
        if piece.size == 0:
            continue
        dx0 = int(round((ox0 - x0) * scale_x))
        dy0 = int(round((oy0 - y0) * scale_y))
        dx1 = int(round((ox1 - x0) * scale_x))
        dy1 = int(round((oy1 - y0) * scale_y))
        dst_w, dst_h = dx1 - dx0, dy1 - dy0
        if dst_w <= 0 or dst_h <= 0:
            continue
        if piece.shape[0] != dst_h or piece.shape[1] != dst_w:
            piece = core_io.resize_rgba(piece, dst_w, dst_h)

        weight = _guide_feather_mask(dst_h, dst_w, feather)
        alpha = (piece[..., 3].astype(np.float32) / 255.0) * weight * strength
        # Only carry forward pixels where the earlier tile's *reconstruction*
        # agrees with what this tile's source image really shows.  SeeThrough
        # repaints/completes its input per tile; pasting those invented pixels
        # into the neighbour makes the model split the wrong thing (a half
        # face, stray clothing).  Large disagreements stay on the source.
        tolerance = float(guide.get("tolerance", 16) or 0)
        if tolerance > 0.0:
            source = arr[dy0:dy1, dx0:dx1].astype(np.float32)
            diff = np.abs(piece[..., :3].astype(np.float32) - source).mean(axis=-1)
            gate = np.clip(1.0 - diff / tolerance, 0.0, 1.0).astype(np.float32)
            # Smooth the gate: a raw per-pixel threshold leaves salt-and-pepper
            # holes right at the limit, which the model reads as a broken image.
            alpha = alpha * _box_blur_3x3(gate)
        if not np.any(alpha > 0.0):
            continue
        dst = arr[dy0:dy1, dx0:dx1].astype(np.float32)
        src = piece[..., :3].astype(np.float32)
        a = alpha[..., None]
        arr[dy0:dy1, dx0:dx1] = np.clip(dst * (1.0 - a) + src * a, 0, 255).astype(np.uint8)
        changed = True

    if not changed:
        return tile
    out = torch.from_numpy(arr.astype(np.float32) / 255.0)[None]
    return out.to(device=device, dtype=dtype)


def _crop_tile_parts(parts, composite, content_box, input_box):
    """Crop a context-padded tile's result back to its strip box.

    ① may feed the model a window larger than the strip; PSD size, merge
    geometry and guides all expect the strip box, so every part image/depth
    and the composite RGBA are cut back here.  Parts that do not carry the
    padded frame are left untouched (they were not produced from it).
    """
    if not input_box or not content_box:
        return parts, composite
    cx0, cy0, cx1, cy1 = (int(v) for v in content_box[:4])
    ix0, iy0, ix1, iy1 = (int(v) for v in input_box[:4])
    pad_w, pad_h = ix1 - ix0, iy1 - iy0
    if pad_w <= 0 or pad_h <= 0 or cx1 <= cx0 or cy1 <= cy0:
        return parts, composite

    # The parts are in the MODEL's pixel space, which is not the padded
    # window's pixel size: a 1950 px window comes back at ``resolution``
    # (512).  Map the strip box into that space first.  Cutting with source
    # pixels would slice a 512 px result at 128..1822, and then declaring the
    # frame to be 1694 px puts that sliver in the PSD's top-left corner.
    frame = parts.get("frame_size") or (0, 0)
    frame_h, frame_w = int(frame[0]), int(frame[1])
    if frame_w <= 0 or frame_h <= 0:
        frame_w, frame_h = pad_w, pad_h
    scale_x = frame_w / float(pad_w)
    scale_y = frame_h / float(pad_h)
    dx = max(0, int(round((cx0 - ix0) * scale_x)))
    dy = max(0, int(round((cy0 - iy0) * scale_y)))
    width = min(frame_w, int(round((cx1 - ix0) * scale_x))) - dx
    height = min(frame_h, int(round((cy1 - iy0) * scale_y))) - dy
    if width <= 0 or height <= 0:
        return parts, composite
    if dx == 0 and dy == 0 and (frame_w, frame_h) == (width, height):
        return parts, composite

    out = dict(parts)
    tag2pinfo = {}
    for tag, pinfo in (parts.get("tag2pinfo") or {}).items():
        if not isinstance(pinfo, dict):
            tag2pinfo[tag] = pinfo
            continue
        item = dict(pinfo)
        for key in ("img", "depth"):
            value = item.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if (arr.ndim >= 2
                    and arr.shape[0] == frame_h and arr.shape[1] == frame_w):
                item[key] = np.ascontiguousarray(arr[dy:dy + height, dx:dx + width])
        xyxy = item.get("xyxy")
        if xyxy is not None and len(xyxy) >= 2:
            moved = [min(width, max(0, int(xyxy[0]) - dx)),
                     min(height, max(0, int(xyxy[1]) - dy))]
            if len(xyxy) >= 4:
                # Keep the box ordered: a part that sat fully outside the
                # strip must not come back with its right/top below its left.
                moved += [min(width, max(moved[0], int(xyxy[2]) - dx)),
                          min(height, max(moved[1], int(xyxy[3]) - dy))]
            else:
                moved += list(xyxy[2:])
            item["xyxy"] = moved
        tag2pinfo[tag] = item
    out["tag2pinfo"] = tag2pinfo
    out["frame_size"] = (height, width)

    if isinstance(composite, dict):
        composite = dict(composite)
        rgba = composite.get("rgba")
        if rgba is not None:
            arr = np.asarray(rgba)
            if (arr.ndim == 3
                    and arr.shape[0] == frame_h and arr.shape[1] == frame_w):
                composite["rgba"] = np.ascontiguousarray(
                    arr[dy:dy + height, dx:dx + width])
    return out, composite


# ---------------------------------------------------------------------------
# ③ saving one tile / the wave manifest
# ---------------------------------------------------------------------------

def tile_index_of(index) -> int:
    try:
        return int(str(_first(index, "0")).strip())
    except (TypeError, ValueError):
        return 0


def tile_context(plan, index) -> dict:
    """Folder, prefix and alpha cutoff a tile is written with."""
    doc = plan.get("doc") or {}
    out = doc.get("out") or {}
    wave_dir = str(out.get("dir") or "")
    if not wave_dir:
        raise ValueError(
            "STLU ③: PLAN 里没有运行目录。请确认 ① 的 PrepareRun 已接上并完成。"
            " / STLU ③: PLAN has no run directory; make sure ①'s PrepareRun"
            " is connected and finished.")
    return {
        "doc": doc,
        "out": out,
        "wave_dir": wave_dir,
        "prefix": str(out.get("prefix") or "tile"),
        "alpha_threshold": int(out.get("alpha_threshold", 8)),
        "index": tile_index_of(index),
    }




def tile_is_empty(parts) -> bool:
    return not isinstance(parts, dict) or not (parts.get("tag2pinfo") or {})


def write_tile_psd(parts, plan, index) -> dict:
    """Write this tile's layered PSD and return its manifest record."""
    ctx = tile_context(plan, index)
    tile_index = ctx["index"]
    doc = ctx["doc"]
    wave_dir = ctx["wave_dir"]
    prefix = ctx["prefix"]
    alpha_threshold = ctx["alpha_threshold"]
    if not isinstance(parts, dict):
        raise ValueError(
            "STLU ③: tile %d 没有收到部件（上游返回 %r）。请检查 ② 是否正常完成。"
            " / STLU ③: tile %d received no parts (upstream returned %r);"
            " check whether ② finished normally."
            % (tile_index, parts, tile_index, parts))
    if str(parts.get("status") or "") == "error":
        raise RuntimeError("STLU ③ tile %d upstream error: %s"
                           % (tile_index, parts.get("error") or "unknown"))
    if not (parts.get("tag2pinfo") or {}):
        # Still nothing to write: skip the tile (an edge tile with no content).
        return {"index": tile_index, "status": "empty"}

    frame = (parts.get("frame_size") or (0, 0))
    if (int(frame[0]) <= 0 or int(frame[1]) <= 0) and doc.get("tile_size"):
        tw, th = (int(v) for v in (doc["tile_size"] or [0, 0])[:2])
        frame = (th, tw)
        parts["frame_size"] = frame
    tile_size = tuple(int(v) for v in (doc.get("tile_size") or (frame[1], frame[0])))
    overlap = int(doc.get("overlap") or 0)
    src = doc.get("source") or {}
    source_size = (int(src.get("width") or 0), int(src.get("height") or 0))
    if source_size[0] <= 0 or source_size[1] <= 0:
        source_size = None

    try:
        psd_manifest = tile_psd.write_tiles(
            [parts], _output_root(),
            filename_prefix=prefix,
            merge_parts=True,
            alpha_threshold=int(alpha_threshold),
            source_size=source_size,
            tile_size=tile_size,
            overlap=overlap,
            tile_index_map=[tile_index],
            write_depth=True,
            original_rgba=None,
            merge_group_layers=True,
            include_original=True,
            run_dir_abs=wave_dir,
            write_manifest=False,
            # The grid's own boxes, so a phased grid lands where it was cut.
            tile_coords=doc.get("coords") or None,
            grid_offset=doc.get("offset") or (0, 0),
            # Per-tile seed record for the ⑤ preview ("种子 42 / 深度 7").
            generation=doc.get("generation") or None,
        )
    except ValueError as exc:
        # Say which tile came up empty: "every tile ended up empty" alone
        # gives no way to tell this apart from a layout bug.
        raise ValueError("STLU ③: tile %d: %s" % (tile_index, exc)) from exc
    entry = psd_manifest["tiles"][0]

    print("[STLU] tile %d -> %s" % (tile_index, entry.get("file")), flush=True)
    return {"index": int(tile_index), "status": "ok", "entry": entry,
            "file": entry.get("file")}


def write_tile_images(parts, plan, index) -> dict:
    """Write this tile's per-layer PNGs (the images/ folder)."""
    ctx = tile_context(plan, index)
    tile_index = ctx["index"]
    doc = ctx["doc"]
    wave_dir = ctx["wave_dir"]
    if not isinstance(parts, dict) or not (parts.get("tag2pinfo") or {}):
        return {"index": tile_index, "status": "empty"}
    src = doc.get("source") or {}
    source_size = (int(src.get("width") or 0), int(src.get("height") or 0))
    if source_size[0] <= 0 or source_size[1] <= 0:
        source_size = None
    images_manifest = tile_psd.write_tile_images(
        [parts], _output_root(),
        filename_prefix=ctx["prefix"],
        grid_doc=doc,
        tile_index_map=[tile_index],
        source_size=source_size,
        original_rel="../original.png",
        run_dir_abs=os.path.join(wave_dir, "images"),
        write_index=False,
        alpha_threshold=int(ctx["alpha_threshold"]),
    )
    return {"index": int(tile_index), "status": "ok",
            "entry": images_manifest["tiles"][0]}


def save_original(image, plan) -> str:
    """Write the untouched source once per run as ``original.png``."""
    doc = plan.get("doc") or {}
    out = doc.get("out") or {}
    wave_dir = str(out.get("dir") or "")
    if not wave_dir:
        raise ValueError("STLU ③: PLAN 里没有运行目录 / PLAN has no run directory; cannot write original.png.")
    frame = _first(image)
    if frame is None or not hasattr(frame, "shape") or frame.ndim < 3:
        return ""
    try:
        px = frame if frame.ndim == 3 else frame[0]
        px = px.detach().cpu().numpy() if hasattr(px, "detach") else np.asarray(px)
        rgba = np.clip(np.rint(np.asarray(px, dtype=np.float32) * 255.0),
                       0, 255).astype(np.uint8)
        if rgba.ndim == 3 and rgba.shape[-1] == 3:
            rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, np.uint8)])
        from PIL import Image
        Image.fromarray(np.ascontiguousarray(rgba), "RGBA").save(
            os.path.join(wave_dir, "original.png"), compress_level=3)
        return "original.png"
    except Exception as exc:  # noqa: BLE001 - tiles still matter more
        print("[STLU] original.png failed: %s" % exc, flush=True)
        return ""


def write_manifest(records, image_records, original_name, plan) -> dict:
    """Fold this run's tiles together with what the folder already holds."""
    raw_records = [r for r in _list_values(records) if isinstance(r, dict)]
    empty_tiles = [int(r.get("index", -1)) for r in raw_records
                   if r.get("status") == "empty" or r.get("empty")]
    records = [r for r in raw_records
               if r.get("status") not in ("empty", "error")
               and not r.get("empty")]
    image_records = [r for r in _list_values(image_records) if isinstance(r, dict)]
    image_records = [r for r in image_records
                     if r.get("status") not in ("empty", "error")]

    doc = dict(plan.get("doc") or {})
    out = doc.get("out") or {}
    wave_dir = str(out.get("dir") or "")
    if not wave_dir:
        raise ValueError("STLU ③: PLAN 里没有运行目录 / PLAN has no run directory; cannot write the manifest.")
    rel = str(out.get("rel") or "")
    prefix = str(out.get("prefix") or "tile")
    group_name = str(out.get("group") or "")
    wave_id_value = str(out.get("wave") or "")
    raw_seam = out.get("seam_feather")
    if raw_seam is None:
        raw_seam = tile_psd.DEFAULT_SEAM_FEATHER
    raw_alpha = out.get("alpha_threshold")
    alpha_threshold = (tile_psd.DEFAULT_ALPHA_THRESHOLD if raw_alpha is None
                       else int(raw_alpha))
    os.makedirs(wave_dir, exist_ok=True)

    # The folder is fixed now, so it can already hold tiles from earlier runs.
    # Read that manifest BEFORE overwriting it: those tiles join this run's
    # manifest (unless re-generated now) and therefore the automatic merge,
    # which is what makes a partial re-run complete.
    previous = tile_psd.read_manifest(wave_dir)

    generation = dict(doc.get("generation") or {})
    if not generation:
        stored = tile_psd.read_generation(wave_dir) or {}
        generation = dict(stored.get("generation") or {})
    cache_doc = dict(doc)
    cache_doc["generation"] = generation
    fingerprint = str(out.get("fingerprint") or "")

    entries = []
    run_indices = set()
    for record in records:
        entry = dict(record.get("entry") or {})
        entries.append(entry)
        run_indices.add(int(entry.get("index", -1)))

    image_entries = []
    for record in image_records:
        image_entry = dict(record.get("entry") or record)
        if image_entry and "index" in image_entry:
            image_entries.append(image_entry)

    # Finished tiles from earlier runs in this same folder.
    try:
        cached = tile_psd.cached_tile_entries(
            wave_dir, cache_doc, skip=run_indices, manifest=previous)
    except Exception as exc:  # noqa: BLE001
        cached = []
        print("[STLU] cannot reuse cached tiles: %s" % exc, flush=True)
    reused_indices = []
    if cached:
        entries.extend(cached)
        reused_indices = [int(e.get("index", -1)) for e in cached]
        prev_images = {}
        images_path = os.path.join(wave_dir, "images", "images.json")
        if os.path.isfile(images_path):
            try:
                with open(images_path, "r", encoding="utf-8") as handle:
                    prev_doc = json.load(handle)
                prev_images = {int(t.get("index", -1)): t
                               for t in prev_doc.get("tiles") or []}
            except Exception:  # noqa: BLE001
                prev_images = {}
        have_images = {int(e.get("index", -1)) for e in image_entries}
        for entry in cached:
            index = int(entry.get("index", -1))
            image_entry = prev_images.get(index)
            if image_entry and index not in have_images:
                image_entries.append(dict(image_entry))

    # Cached tiles may come from a folder written before the
    # seeds were recorded per entry; the manifest's generation is the same
    # for every tile, so it can fill the gap.
    if generation:
        for entry in entries:
            if not entry.get("seeds"):
                entry["seeds"] = {
                    "layer": generation.get(tile_psd.SEED_KEY),
                    "depth": generation.get(tile_psd.DEPTH_SEED_KEY),
                }
    entries.sort(key=lambda e: int(e.get("index", -1)))
    for slot, entry in enumerate(entries):
        entry["slot"] = int(slot)
    image_entries.sort(key=lambda e: int(e.get("index", -1)))
    for slot, entry in enumerate(image_entries):
        entry["slot"] = int(slot)

    if not entries:
        raise ValueError("STLU ③: every tile ended up empty")
    if reused_indices:
        print("[STLU] reusing %d tiles from an earlier run: %s" % (
            len(reused_indices), reused_indices), flush=True)

    src = doc.get("source") or {}
    if int(src.get("width") or 0) > 0 and int(src.get("height") or 0) > 0:
        source_size = [int(src["width"]), int(src["height"])]
    else:
        source_size = [
            max(int(e["tile"][2]) for e in entries),
            max(int(e["tile"][3]) for e in entries),
        ]
    tile_size = [int(v) for v in (doc.get("tile_size") or entries[0]["canvas"])]
    overlap = int(doc.get("overlap") or 0)

    richest = max(entries, key=lambda e: len(e.get("layers") or []))
    preview_name = str(richest.get("preview") or "")
    images_dir = os.path.relpath(os.path.join(wave_dir, "images"),
                                 _output_root()).replace(os.sep, "/")

    manifest = {
        "dir": rel,
        "prefix": prefix,
        "merge_parts": True,
        "source": source_size,
        "tile_size": tile_size,
        "overlap": overlap,
        "offset": tile_psd._int_pair(doc.get("offset")),
        "preview": preview_name,
        "original": original_name or "",
        "has_depth": any(e.get("depth") for e in entries),
        "group_layers": True,
        "seam_feather": max(0, int(raw_seam)),
        "alpha_threshold": int(alpha_threshold),
        "mode": str(doc.get("mode") or "uniform"),
        "partition_plan": doc.get("partition_plan") or None,
        "source_fingerprint": fingerprint,
        "tiles": entries,
        "count": len(entries),
        "run_tiles": sorted(run_indices),
        "reused_tiles": sorted(reused_indices),
        "generation": generation,
        "images_dir": images_dir,
        "group": group_name,
        "wave": wave_id_value,
        "prompt_id": _shared._prompt_id(),
    }
    manifest_path = os.path.join(wave_dir, "manifest.json")
    _shared.json_io.write_json_atomic(manifest_path, manifest)

    images_doc = {
        "version": 1,
        "kind": tile_psd.IMAGES_KIND,
        "dir": images_dir,
        "prefix": prefix,
        "source": source_size,
        "tile_size": tile_size,
        "overlap": overlap,
        "offset": tile_psd._int_pair(doc.get("offset")),
        "original": "../original.png" if original_name else "",
        "tiles": image_entries,
        "count": len(image_entries),
    }
    images_dir_abs = os.path.join(wave_dir, "images")
    os.makedirs(images_dir_abs, exist_ok=True)
    _shared.json_io.write_json_atomic(
        os.path.join(images_dir_abs, "images.json"), images_doc)

    total = 0
    for entry in entries:
        path = os.path.join(wave_dir, str(entry.get("file") or ""))
        if os.path.isfile(path):
            total += os.path.getsize(path)

    preview = None
    preview_path = os.path.join(wave_dir, preview_name) if preview_name else ""
    if preview_path and os.path.isfile(preview_path):
        preview = {
            "filename": preview_name,
            "subfolder": rel,
            "type": "output",
            "v": int(os.path.getmtime(preview_path) * 1000),
        }
    info = {
        "dir": rel,
        "subfolder": rel,
        "filename": entries[0].get("file"),
        "count": len(entries),
        "layers": len(entries[0].get("layers") or []),
        "canvas": entries[0].get("canvas"),
        "size": total,
        "merged": True,
        "merged_psd": None,
        "merge_error": None,
        "preview": preview,
        "images_dir": images_dir,
        "images_error": None,
        "group": group_name,
        "wave": wave_id_value,
        "run_tiles": sorted(run_indices),
        "reused_tiles": sorted(reused_indices),
        "empty_tiles": empty_tiles,
        "algo_version": (generation or {}).get("algo_version"),
        "error": None,
    }
    _notify("stlu.tiles", info)
    print("[STLU] %d tile PSDs written: %s" % (len(entries), wave_dir), flush=True)
    return {
        "info": info,
        "manifest": manifest,
        "merge_doc": {
            "dir": rel,
            "name": str(out.get("merge_name") or "merged"),
            "group_layers": bool(out.get("group_layers", True)),
            "include_depth": bool(out.get("include_depth", False)),
            "include_original": bool(out.get("include_original", False)),
            "seam_feather": max(0, int(raw_seam)),
            "alpha_threshold": int(alpha_threshold),
            "wave": wave_id_value,
            "prompt_id": _shared._prompt_id(),
        },
    }


# ---------------------------------------------------------------------------
# ④ merge
# ---------------------------------------------------------------------------

def merge_run_doc(merge_doc) -> dict:
    """Merge one run folder from its ``merge_doc`` (no JSON round-trip)."""
    doc = dict(merge_doc or {})
    rel = str(doc.get("dir") or "").strip()
    if not rel:
        raise ValueError("STLU ④: merge_doc 里没有目录 / merge_doc has no directory; cannot merge.")
    path = _shared._resolve_workspace_path(rel)
    if not os.path.isdir(path):
        raise FileNotFoundError("tile run folder not found: %s" % path)
    if not os.path.isfile(os.path.join(path, "manifest.json")):
        raise FileNotFoundError("manifest.json not found in %s" % path)
    name = str(doc.get("name") or "merged")
    group_layers = bool(doc.get("group_layers", True))
    include_depth = bool(doc.get("include_depth", False))
    include_original = bool(doc.get("include_original", False))
    seam_feather = max(0, int(doc.get("seam_feather") or 0))
    raw_alpha = doc.get("alpha_threshold")
    info = tile_psd.merge_run(
        path,
        name=name,
        alpha_threshold=None if raw_alpha is None else int(raw_alpha),
        group_layers=group_layers,
        include_depth=include_depth,
        include_original=include_original,
        seam_feather=seam_feather,
    )
    info["dir"] = os.path.relpath(path, _output_root()).replace(os.sep, "/")
    psd_path = os.path.join(path, info.get("file") or "")
    payload = {
        "dir": info["dir"], "name": name,
        "group_layers": group_layers, "include_depth": include_depth,
        "include_original": include_original,
        "file": info.get("file"), "preview": info.get("preview"),
        "depth": info.get("depth"), "depth_psd": info.get("depth_psd"),
        "original": info.get("original"), "canvas": info.get("canvas"),
        "layers": info.get("layers"), "groups": info.get("groups"),
        "tiles": info.get("tiles"), "notes": info.get("notes"),
    }
    _notify("stlu.tile_merge", payload)
    print("[STLU] merged %d tiles -> %s"
          % (len(info.get("tiles") or []), psd_path), flush=True)
    return {"psd_path": psd_path, "payload": payload}
