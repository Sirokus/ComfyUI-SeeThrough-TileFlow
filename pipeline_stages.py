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
    "context_pad", "custom_regions",
)

DEFAULT_SETTINGS = {
    "tile_width": 1024, "tile_height": 1024, "overlap": 128, "select": "all",
    "resolution": 512, "num_inference_steps": 30, "depth_resolution": -1,
    "random_seed": 0, "tblr_split": True, "use_lama": True,
    "alpha_threshold": 8, "out_dir": "SeeThrough/我的自设/", "group": "",
    "filename_prefix": "tile", "merge_name": "merged", "group_layers": True,
    "include_depth": False, "include_original": False, "guide_context": True,
    "guide_feather": 32, "guide_strength": 40, "guide_tolerance": 16,
    "seam_feather": 32, "grid_offset_x": 0, "grid_offset_y": 0,
    "partition_mode": "均匀网格", "strip_max_side": 2048,
    "strip_max_resolution": 768, "strip_margin": 32, "select_plan": "",
    "context_pad": 0, "custom_regions": "[]",
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
            "strip_margin", "context_pad")
    for key in ints:
        out[key] = int(out[key])
    for key in ("tblr_split", "use_lama", "group_layers", "include_depth",
                "include_original", "guide_context"):
        out[key] = bool(out[key])
    for key in ("select", "out_dir", "group", "filename_prefix", "merge_name",
                "partition_mode", "select_plan", "custom_regions"):
        out[key] = str(out[key] if out[key] is not None else "")
    return out


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
    generation = {
        "version": 1,
        # Bumped whenever tile pixels can change; joins generation_id so a
        # code update cannot silently reuse tiles made by older logic.
        "algo_version": int(tile_psd.ALGO_VERSION),
        "source": plan.get("fingerprint") or "",
        "source_size": [int(plan["width"]), int(plan["height"])],
        "tile_size": [int(v) for v in doc["tile_size"]],
        "overlap": int(doc["overlap"]),
        "resolution": int(plan["resolution"]),
        "num_inference_steps": int(plan["num_inference_steps"]),
        "depth_resolution": int(plan["depth_resolution"]),
        "random_seed": int(plan["seed"]),
        "tblr_split": bool(settings["tblr_split"]),
        "use_lama": bool(settings["use_lama"]),
        "alpha_threshold": int(settings["alpha_threshold"]),
        "context_pad": int(settings["context_pad"] or 0),
        "guide": guide_block,
        "layerdiff_model": plan.get("layerdiff_model_id") or "",
        "depth_model": plan.get("depth_model_id") or "",
    }
    meta = plan["partition"] or {}
    if plan["is_strips"]:
        # Only geometry goes into the fingerprint: warnings/notes can change
        # without changing a single pixel, and a warning-only difference must
        # not clear a finished folder.
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
        # changing a per-window resolution changes the fingerprint too:
        # both change what the guide sees, and old tiles cannot be reused.
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
    """Fingerprint the run, clear a stale folder, write ``generation.json``.

    This is the one stage ① step with side effects, which is why it is its own
    node: the graph now shows when a folder is about to be reset.
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
    wave = os.path.basename(wave_dir.rstrip("/\\")) or "run"
    doc.setdefault("source", {})["fingerprint"] = fingerprint
    generation, guide_block = _generation_block(plan)
    doc["guide"] = guide_block
    gid = tile_psd.generation_id(generation)
    doc["generation"] = generation
    doc["generation_id"] = gid

    # A changed pixel-affecting parameter means the old tiles cannot be
    # trusted: the folder is cleared before anything runs, so stale tiles from
    # another generation can never be mixed into the merge.  The selection
    # still decides how much work happens -- an explicit subset is the user's
    # instruction and is never widened to the whole grid, it only regenerates
    # the checked tiles into the now-empty folder.
    reset_info = None
    picked = list(plan["picked"])
    total = len(plan["coords"])
    explicit_subset = 0 < len(picked) < total
    previous = tile_psd.read_manifest(wave_dir)
    reset_keys = []
    reset_reason = ""
    old_generation = None
    if previous is not None:
        old_generation = previous.get("generation")
    else:
        # An interrupted run leaves tiles and generation.json behind but no
        # manifest; that fingerprint still describes those tiles.
        old_generation = (tile_psd.read_generation(wave_dir) or {}).get("generation")
    if previous is not None and (not isinstance(old_generation, dict)
                                 or not old_generation):
        reset_reason = "旧结果没有参数指纹"
    elif isinstance(old_generation, dict) and old_generation:
        # Seed / context padding are deliberately excluded: changing them
        # regenerates only the checked tiles and leaves the rest of the
        # folder usable.
        reset_keys = [key for key in generation
                      if key not in tile_psd.LOCAL_RERUN_KEYS
                      and old_generation.get(key) != generation[key]]
        # A key that only the old run has still counts: that is what moving
        # the grid back to offset 0 looks like.
        reset_keys += [key for key in old_generation
                       if key not in tile_psd.LOCAL_RERUN_KEYS
                       and key not in generation]
        if reset_keys:
            reset_reason = "生成参数已变"
    if reset_reason and explicit_subset:
        cleared = tile_psd.clear_run_outputs(wave_dir, previous)
        tile_psd.verify_run_cleared(wave_dir)
        reset_info = {
            "required": True, "partial": True, "reason": reset_reason,
            "keys": reset_keys, "tiles": len(picked),
            "removed": int(cleared.get("removed") or 0),
        }
        _notify("stlu.cache_reset", {
            "dir": wave_dir, "rel": rel, "reason": reset_reason,
            "keys": reset_keys, "tiles": len(picked),
            "partial": True, "removed": reset_info["removed"],
        })
        print("[STLU] generation changed (%s) -> cleared %s, re-running only the "
              "%d selected tiles" % (
                  ", ".join(reset_keys) or reset_reason, rel, len(picked)),
              flush=True)
    elif reset_reason:
        picked = list(range(len(plan["coords"])))
        encoded = "all"
        doc["selection"] = encoded
        plan["picked"] = picked
        plan["indices"] = [str(int(i)) for i in picked]
        plan["selection"] = encoded
        cleared = tile_psd.clear_run_outputs(wave_dir, previous)
        tile_psd.verify_run_cleared(wave_dir)
        reset_info = {
            "required": True, "reason": reset_reason, "keys": reset_keys,
            "tiles": len(picked), "removed": int(cleared.get("removed") or 0),
        }
        _notify("stlu.cache_reset", {
            "dir": wave_dir, "rel": rel, "reason": reset_reason,
            "keys": reset_keys, "tiles": len(picked),
            "removed": reset_info["removed"],
        })
        print("[STLU] generation changed -> cleared %s, regenerating all %d tiles (%s)" % (
            rel, len(picked), ", ".join(reset_keys) or reset_reason), flush=True)

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
        tile_psd.write_generation(wave_dir, generation, gid)
    except OSError as exc:
        raise RuntimeError(
            "STLU ①: 无法写入生成指纹 %s：%s。"
            "请检查输出目录权限后重试。"
            " / cannot write the generation fingerprint %s: %s;"
            " check the output folder's permissions and retry."
            % (wave_dir, exc, wave_dir, exc)) from exc

    plan["doc"] = doc
    plan["run"] = {
        "dir": wave_dir, "rel": rel, "wave": wave,
        "fingerprint": fingerprint, "out": out_info,
        "generation": generation, "generation_id": gid,
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
    return {
        "guide": guide,
        "coords": list(doc.get("coords") or []),
        "context_boxes": context_boxes,
        "regions_by_index": regions_by_index,
        "alpha_threshold": int(((doc.get("out") or {}).get("alpha_threshold")) or 8),
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
                     tblr_split, use_lama) -> tuple:
    """The sequential per-tile loop: guide -> chain -> retry -> crop back.

    This is the one place a loop is genuinely required: tile N has to see the
    pixels tile N-1 produced, which no acyclic node graph can express.  Every
    step it calls is a separate function (``guide_tile`` / ``_run_seethrough_tile``
    / ``crop_back``), so the graph still shows what happens.
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
    resolution = int(_first(resolution, 512))
    steps = int(_first(steps, 30))
    depth_res = int(_first(depth_resolution, -1))
    try:
        (resolution, steps, depth_res,
         seed) = stlu_limits.check_all(resolution, steps, depth_res, seed)
    except ValueError as exc:
        raise ValueError("STLU ②: %s" % exc) from exc
    tblr = bool(_first(tblr_split, True))
    lama = bool(_first(use_lama, True))

    guide = state["guide"]
    coords = state["coords"]
    regions_by_index = state["regions_by_index"]
    context_boxes = state["context_boxes"]
    alpha_threshold = state["alpha_threshold"]
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

        # Per-window resolution when the document names one; otherwise the
        # single value (uniform grid / subject strips).
        tile_res = int(regions_by_index.get(index, resolution))
        tile_depth = depth_res if depth_res == -1 else min(depth_res, tile_res)
        input_box = context_boxes.get(index)
        input_tile = guide_tile(tile, index, state, generated, fixed=fixed)
        guided_out.append(input_tile.detach().cpu().clamp(0, 1))

        try:
            parts, composite = _run_seethrough_tile(
                layer_node, depth_node, post_node, input_tile, model,
                depth_model, seed, tile_res, steps, tile_depth, tblr,
                lama, alpha_threshold)
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

        # An all-transparent result is a model collapse, not a real empty
        # tile: retry this one tile on a new seed before giving up.  The
        # input tile stays the same -- nothing about the guide changes.
        attempt = 0
        while composite["status"] == GUIDE_EMPTY and attempt < GUIDED_EMPTY_RETRIES:
            attempt += 1
            retry_seed = (seed + attempt * 0x9E3779B1) % (2 ** 32)
            print("[STLU] guided: tile %d came back empty with seed %d, "
                  "retrying on seed %d (%d/%d)" % (
                      index, seed, retry_seed, attempt,
                      GUIDED_EMPTY_RETRIES), flush=True)
            comfy.model_management.throw_exception_if_processing_interrupted()
            try:
                parts, composite = _run_seethrough_tile(
                    layer_node, depth_node, post_node, input_tile, model,
                    depth_model, retry_seed, tile_res, steps, tile_depth,
                    tblr, lama, alpha_threshold)
            except comfy.model_management.InterruptProcessingException:
                raise
            except Exception as exc:  # noqa: BLE001 - add the tile number
                raise RuntimeError(
                    "STLU ②: 第 %d 块在重试时生成失败（%s: %s）。\n%s"
                    " / STLU ②: tile %d failed to generate on retry (%s: %s).\n%s"
                    % (index, type(exc).__name__, exc,
                       traceback.format_exc(), index, type(exc).__name__, exc,
                       traceback.format_exc())) from exc
            if composite["status"] == GUIDE_ERROR:
                raise RuntimeError(
                    "STLU ②: 第 %d 块在重试时合成失败（%s）。\n%s"
                    " / STLU ②: tile %d failed to composite on retry (%s).\n%s"
                    % (index, composite.get("error") or "unknown",
                       composite.get("traceback") or "",
                       index, composite.get("error") or "unknown",
                       composite.get("traceback") or ""))
        if composite["status"] == GUIDE_EMPTY:
            print("[STLU] guided: tile %d is still empty after %d "
                  "retries" % (index, GUIDED_EMPTY_RETRIES), flush=True)
        else:
            if input_box:
                # The model saw the padded window; everything downstream
                # (merge geometry, PSD size, guides) wants the strip box.
                parts, composite = _crop_tile_parts(
                    parts, composite, coords[index], input_box)
            generated[index] = composite["rgba"]
        parts_out.append(parts)
        indices_out.append(str(index))
        composite_out.append(_composite_tensor(composite.get("rgba")))
        pbar.update(1)
        print("[STLU] guided: tile %d done (%d/%d)" % (index, pos + 1, total),
              flush=True)

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

    A fully transparent tile is a known model failure and is retried on a new
    seed.  An exception or a malformed ``frame_size`` is not: retrying it only
    burns two more diffusion runs and hides the real error, so it is reported
    as ``error`` with a traceback.
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

GUIDED_EMPTY_RETRIES = 2


def _run_seethrough_tile(layer_node, depth_node, post_node, input_tile, model,
                         depth_model, seed, resolution, steps, depth_res,
                         tblr, lama, alpha_threshold):
    """Generate -> depth -> post for one tile; return (parts, composite result)."""
    layers_data, _preview = layer_node.generate(
        input_tile, model, seed, resolution, steps)
    # GenerateDepth returns (layers_depth, preview), not a bare object.
    depth_data, _preview_depth = depth_node.generate(
        layers_data, depth_model, seed, depth_res)
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


def _restore_locked_orphans(wave_dir, doc, prefix, entries, image_entries,
                            alpha_threshold):
    """Rebuild tiles this run did not select from their locked-part snapshots.

    A reset clears the folder down to the tiles this queue click generated, so
    a locked part living in any other tile would disappear with it.  The
    ``_locks/`` store survives the reset, so those tiles are written back with
    just their locked layers and join the manifest (and therefore the merge).
    """
    locks = tile_psd.tile_locks(wave_dir)
    if not locks:
        return [], []
    have = set()
    for entry in entries:
        try:
            have.add(int(entry.get("index", -1)))
        except (TypeError, ValueError):
            continue
    pending = [index for index in sorted(locks) if index not in have]
    if not pending:
        return [], []
    coords = (doc or {}).get("coords") or []
    src = (doc or {}).get("source") or {}
    source = None
    try:
        source = [int(src["width"]), int(src["height"])]
    except (KeyError, TypeError, ValueError):
        source = None
    tile_size = []
    try:
        tile_size = [int(v) for v in ((doc or {}).get("tile_size") or [])
                     if v is not None][:2]
    except (TypeError, ValueError):
        tile_size = []
    made_entries = []
    made_images = []
    for index in pending:
        parts = tile_psd.snapshot_parts(
            wave_dir, index, coords=coords, source=source)
        if not parts:
            print("[STLU] locked tile %d not restored: its crop layout changed"
                  % index, flush=True)
            continue
        frame_h, frame_w = (int(v) for v in parts["frame_size"][:2])
        size_pair = tile_size if len(tile_size) == 2 else [frame_w, frame_h]
        try:
            manifest = tile_psd.write_tiles(
                [parts], _output_root(),
                filename_prefix=prefix,
                merge_parts=True,
                alpha_threshold=int(alpha_threshold),
                source_size=tuple(source) if source else None,
                tile_size=tuple(size_pair),
                overlap=int((doc or {}).get("overlap") or 0),
                tile_index_map=[index],
                write_depth=True,
                original_rgba=None,
                merge_group_layers=True,
                include_original=False,
                run_dir_abs=wave_dir,
                write_manifest=False,
                tile_coords=coords or None,
                grid_offset=(doc or {}).get("offset") or (0, 0),
            )
            image_manifest = tile_psd.write_tile_images(
                [parts], _output_root(),
                filename_prefix=prefix,
                grid_doc=doc,
                tile_index_map=[index],
                source_size=source,
                original_rel="../original.png",
                run_dir_abs=os.path.join(wave_dir, "images"),
                write_index=False,
                alpha_threshold=int(alpha_threshold),
            )
        except (ValueError, OSError) as exc:
            print("[STLU] could not restore locked tile %d: %s" % (index, exc),
                  flush=True)
            continue
        made_entries.append(manifest["tiles"][0])
        made_images.append(image_manifest["tiles"][0])
        print("[STLU] restored tile %d from locked parts: %s" % (
            index, ", ".join(sorted((parts.get("tag2pinfo") or {}).keys()))),
            flush=True)
    return made_entries, made_images


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


def restore_locks(parts, plan, index) -> dict:
    """Bring back the parts the ⑤ panel locked for this tile.

    The ``_locks/`` snapshot survives a folder reset, so it is tried first;
    the previous tile PSD is only a fallback for locks saved before the store
    existed.  Either source can revive a tile the new run left empty.
    """
    ctx = tile_context(plan, index)
    tile_index = ctx["index"]
    doc = ctx["doc"]
    wave_dir = ctx["wave_dir"]
    prefix = ctx["prefix"]
    if not isinstance(parts, dict):
        raise ValueError(
            "STLU ③: tile %d 没有收到部件（上游返回 %r）。请检查 ② 是否正常完成。"
            " / STLU ③: tile %d received no parts (upstream returned %r);"
            " check whether ② finished normally."
            % (tile_index, parts, tile_index, parts))
    if str(parts.get("status") or "") == "error":
        raise RuntimeError("STLU ③ tile %d upstream error: %s"
                           % (tile_index, parts.get("error") or "unknown"))
    coords = doc.get("coords") or []
    src = doc.get("source") or {}
    source = None
    try:
        if int(src.get("width") or 0) > 0 and int(src.get("height") or 0) > 0:
            source = [int(src["width"]), int(src["height"])]
    except (TypeError, ValueError):
        source = None
    locks = []
    seen_locks = set()
    for name in tile_psd.tile_locks(wave_dir).get(tile_index, []):
        name = str(name).strip()
        if name and name.lower() not in seen_locks:
            seen_locks.add(name.lower())
            locks.append(name)
    if not locks:
        return parts
    parts = dict(parts)
    restored, loaded = tile_psd.restore_locked_parts(
        parts, wave_dir, tile_index, locks, coords=coords, source=source)
    missing = [name for name in locks if name.strip().lower() not in loaded]
    if missing:
        old_path = tile_psd.find_tile_psd(wave_dir, prefix, tile_index)
        if old_path:
            previous = tile_psd.read_manifest(wave_dir) or {}
            old_entry = None
            for entry in previous.get("tiles") or []:
                try:
                    if int(entry.get("index", -1)) == tile_index:
                        old_entry = entry
                        break
                except (TypeError, ValueError):
                    continue
            try:
                row, col = (tile_psd.tile_rc(coords, tile_index)
                            if coords else (0, 0))
                tile_folder = ("tile_%02d_r%d_c%d" % (tile_index, row, col)
                               if coords else "tile_%02d" % tile_index)
            except Exception:  # noqa: BLE001 - fall back to no depth reuse
                tile_folder = "tile_%02d" % tile_index
            images_dir = os.path.join(wave_dir, "images", tile_folder)
            restored.extend(tile_psd.replace_locked_parts(
                parts, old_path, missing, old_entry, images_dir=images_dir))
    if restored:
        print("[STLU] tile %d: kept locked parts: %s" % (
            tile_index, ", ".join(sorted(set(restored)))), flush=True)
    return parts


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
        # Still nothing to write: skip the tile (an edge tile with no content
        # and no locks to restore).
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
        )
    except ValueError as exc:
        # Say which tile came up empty: "every tile ended up empty" alone
        # gives no way to tell this apart from a layout bug.
        raise ValueError("STLU ③: tile %d: %s" % (tile_index, exc)) from exc
    entry = psd_manifest["tiles"][0]

    # Refresh the lock snapshots from what was actually written.  This
    # migrates locks that came from the old PSD and keeps every locked layer
    # safe for the next time a parameter change clears the folder.
    source = None
    try:
        if int(src.get("width") or 0) > 0 and int(src.get("height") or 0) > 0:
            source = [int(src["width"]), int(src["height"])]
    except (TypeError, ValueError):
        source = None
    locks = [str(name).strip()
             for name in tile_psd.tile_locks(wave_dir).get(tile_index, [])]
    present = {str(tag).lower() for tag in (parts.get("tag2pinfo") or {})}
    lock_names = [name for name in locks if name.strip().lower() in present]
    if lock_names:
        tile_psd.store_locked_parts(
            wave_dir, tile_index, parts, lock_names,
            tile_box=entry.get("tile"), canvas=entry.get("canvas"),
            source=source)

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
    generation_id = str(doc.get("generation_id") or "")
    if not generation:
        stored = tile_psd.read_generation(wave_dir) or {}
        generation = dict(stored.get("generation") or {})
        generation_id = str(stored.get("generation_id") or "")
    cache_doc = dict(doc)
    cache_doc["generation"] = generation
    cache_doc["generation_id"] = generation_id
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

    # Tiles the current selection did not touch (after a reset the folder
    # holds only the selected ones).  Locked parts keep their own snapshot, so
    # those tiles are rebuilt instead of silently vanishing.
    restored_entries, restored_images = _restore_locked_orphans(
        wave_dir, doc, prefix, entries, image_entries, alpha_threshold)
    restored_indices = []
    if restored_entries:
        entries.extend(restored_entries)
        image_entries.extend(restored_images)
        restored_indices = [int(e.get("index", -1)) for e in restored_entries]

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
        "restored_tiles": sorted(restored_indices),
        "generation": generation,
        "generation_id": generation_id,
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
        "restored_tiles": sorted(restored_indices),
        "empty_tiles": empty_tiles,
        "generation_id": generation_id,
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
