"""CPU-only checks for user-placed square windows in the four-step pipeline.

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_pipeline_custom_regions.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (COMFY_ROOT, PKG_DIR, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import nodes as stlu  # noqa: E402
import nodes_shared  # noqa: E402
import region_plan  # noqa: E402
import tile_psd  # noqa: E402

_failures = []
CUSTOM = region_plan.CUSTOM_LABEL

# Three windows of different sides; the middle one overrides the model size.
WINDOWS = [
    {"x": 100, "y": 40, "size": 1024, "res": 0, "name": "头"},
    {"x": 150, "y": 1500, "size": 1536, "res": 1024, "name": "身"},
    {"x": 50, "y": 2800, "size": 768, "res": 0, "name": "腿"},
]


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def synthetic_frame(width=1751, height=4096):
    image = np.full((height, width, 3), 255, np.uint8)
    image[43:3990, 34:1696] = (200, 40, 40)
    return torch.from_numpy(image.astype(np.float32) / 255.0)[None]


def write_old_manifest(run_dir):
    """A uniform-grid manifest from a previous version (no partition plan)."""
    os.makedirs(run_dir, exist_ok=True)
    generation = {
        "version": 1,
        "source": "old-source",
        "source_size": [1751, 4096],
        "tile_size": [1024, 1024],
        "overlap": 256,
        "resolution": 512,
        "num_inference_steps": 30,
        "depth_resolution": -1,
        "random_seed": 0,
        "tblr_split": True,
        "use_lama": True,
        "alpha_threshold": 8,
        "guide": {"enabled": True, "feather": 32, "strength": 100, "tolerance": 48},
        "layerdiff_model": "",
        "depth_model": "",
    }
    tiles = [
        {"index": 0, "file": "tile_00_r0_c0.psd", "canvas": [512, 512],
         "tile": [0, 0, 1024, 1024], "layers": []},
        {"index": 1, "file": "tile_01_r0_c1.psd", "canvas": [512, 512],
         "tile": [727, 0, 1751, 1024], "layers": []},
    ]
    manifest = {
        "source": [1751, 4096],
        "tile_size": [1024, 1024],
        "overlap": 256,
        "offset": [0, 0],
        "generation": generation,
        "tiles": tiles,
        "count": 2,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


def run_node(frame, select, out_dir, regions=None, **kwargs):
    node = stlu.STLU_PipelineInput()
    args = dict(
        image=frame, tile_width=1024, tile_height=1024, overlap=256,
        select=select, resolution=512, num_inference_steps=30,
        depth_resolution=-1, random_seed=0, tblr_split=True, use_lama=True,
        alpha_threshold=8, out_dir=out_dir, group="", filename_prefix="tile",
        merge_name="merged", group_layers=True, include_depth=False,
        include_original=False, partition_mode=CUSTOM,
        custom_regions=json.dumps(
            regions if regions is not None else WINDOWS, ensure_ascii=False),
    )
    args.update(kwargs)
    out = node.execute(**args)
    (tiles, indices, grid_doc, _original, res, _steps, depth,
     _seed, _tblr, _lama) = out["result"]
    doc = tile_psd.load_grid_ref(grid_doc)
    return out, tiles, indices, doc, res, depth


def expected_boxes(regions):
    boxes = []
    for region in regions:
        x, y, size = int(region["x"]), int(region["y"]), int(region["size"])
        boxes.append([x, y, x + size, y + size])
    return boxes


def write_custom_manifest(run_dir, regions):
    """A finished custom run from before, as ③ would have left it."""
    os.makedirs(run_dir, exist_ok=True)
    side = max(int(region["size"]) for region in regions)
    tiles = [
        {"index": index, "file": "tile_%02d.psd" % index,
         "canvas": [512, 512], "tile": list(box)}
        for index, box in enumerate(expected_boxes(regions))
    ]
    manifest = {
        "source": [1751, 4096],
        "tile_size": [side, side],
        "overlap": 256,
        "offset": [0, 0],
        "mode": region_plan.CUSTOM_MODE,
        "partition_plan": region_plan.custom_document(
            regions, 1751, 4096)["partition_plan"],
        "tiles": tiles,
        "count": len(tiles),
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


def test_custom_run():
    print("[① custom windows: geometry, per-window resolution, fingerprint]")
    frame = synthetic_frame()
    out, tiles, indices, doc, res, depth = run_node(
        frame, "all", "SeeThrough/Custom")
    check(doc.get("mode") == region_plan.CUSTOM_MODE,
          "doc mode is custom_regions")
    check(doc["count"] == 3 and len(doc["coords"]) == 3,
          "three user windows planned")
    check(doc["coords"] == expected_boxes(WINDOWS),
          "coords are the windows in array order (%s)" % doc["coords"])
    check(doc["tile_size"] == [1536, 1536],
          "compat tile size is the largest side")
    check(indices == ["0", "1", "2"], "every window is handed on in order")
    check([tuple(tile.shape) for tile in tiles] == [
        (1, 1024, 1024, 3), (1, 1536, 1536, 3), (1, 768, 768, 3)],
        "each crop is its own square (%s)" % [tuple(t.shape) for t in tiles])
    check(res == 512, "the global resolution stays the pipeline resolution")
    check(depth == -1, "depth still follows the layer resolution")
    plan = doc["partition_plan"]
    check(plan["plan_id"] == region_plan.plan_key(WINDOWS, 1751, 4096),
          "plan_id is the shared geometry key")
    check(plan["order"] == [0, 1, 2], "run order is the array order")
    check([r["resolution"] for r in plan["regions"]] == [512, 1024, 512],
          "per-window resolutions resolve against the global one")
    check(plan["regions"][1]["res_follow"] is False,
          "the explicit window is not marked follow-global")
    generation = doc["generation"]
    check(generation["partition"]["mode"] == region_plan.CUSTOM_MODE,
          "generation carries the custom mode")
    check(generation["partition"]["plan_id"] == plan["plan_id"],
          "generation carries the plan id")
    check(generation["partition"]["regions"] == [
        {"box": [100, 40, 1124, 1064], "resolution": 512},
        {"box": [150, 1500, 1686, 3036], "resolution": 1024},
        {"box": [50, 2800, 818, 3568], "resolution": 512},
    ], "generation fingerprints every box and its model size")
    info = out["ui"]["stlu_grid"][0]
    check(info.get("mode") == region_plan.CUSTOM_MODE and
          len((info.get("partition") or {}).get("regions") or []) == 3,
          "UI payload carries the executed windows")
    check(doc["selection"] == "all", "selection is all")
    return doc


def test_subset_does_not_clear(doc, plan_id):
    print("[① checked subset: one window, order kept, cache untouched]")
    frame = synthetic_frame()
    out, tiles, indices, doc2, _res, _depth = run_node(
        frame, "1", "SeeThrough/Custom", select_plan=plan_id)
    check(indices == ["1"] and len(tiles) == 1,
          "only the checked window is handed on (%s)" % indices)
    check(tuple(tiles[0].shape) == (1, 1536, 1536, 3),
          "the checked crop is the 1536 px window")
    check(doc2["selection"] == "1", "the subset is encoded")
    check((doc2.get("out") or {}).get("reset") is None,
          "an unchanged geometry does not clear the folder")
    return doc2


def test_size_change_resets(before, plan_id, tmp):
    print("[① resizing a window moves its box, so the folder is cleared]")
    frame = synthetic_frame()
    run_dir = os.path.join(tmp, "SeeThrough", "Custom")
    write_custom_manifest(run_dir, WINDOWS)
    resized = [dict(region) for region in WINDOWS]
    resized[0]["size"] = 1152
    out, tiles, indices, doc, _res, _depth = run_node(
        frame, "all", "SeeThrough/Custom", regions=resized,
        select_plan=plan_id)
    plan = doc["partition_plan"]
    check(plan["plan_id"] != before["partition_plan"]["plan_id"],
          "the plan id changes with the geometry")
    reset = (doc.get("out") or {}).get("reset") or {}
    check(reset.get("required") is False and reset.get("changed") is True,
          "the change only reports a reset (%s)" % (reset,))
    check("位置" in str(reset.get("reason") or ""),
          "the moved box is the stated reason (%s)" % (reset.get("reason"),))
    check(len(indices) == 3,
          "a cleared folder runs every window, whatever was ticked")
    check(not os.path.isfile(os.path.join(run_dir, "manifest.json")),
          "the old manifest is gone until ③ writes the new one")
    check(doc["coords"][0] == [100, 40, 1252, 1192],
          "the resized box is what gets cropped")


def test_order_change():
    print("[① reordering the array changes which crop runs first]")
    swapped = [WINDOWS[1], WINDOWS[0], WINDOWS[2]]
    frame = synthetic_frame()
    out, tiles, indices, doc, _res, _depth = run_node(
        frame, "all", "SeeThrough/Reordered", regions=swapped)
    check(indices == ["0", "1", "2"], "indices follow the new array")
    check([tuple(tile.shape) for tile in tiles] == [
        (1, 1536, 1536, 3), (1, 1024, 1024, 3), (1, 768, 768, 3)],
        "the first crop is now the 1536 px window (%s)"
        % [tuple(t.shape) for t in tiles])
    check(doc["coords"][0] == [150, 1500, 1686, 3036],
          "the first box is the reordered window")
    check(doc["partition_plan"]["plan_id"] !=
          region_plan.plan_key(WINDOWS, 1751, 4096),
          "the plan id changes with the order")
    check([r["resolution"] for r in doc["partition_plan"]["regions"]] ==
          [1024, 512, 512],
          "per-window sizes travel with their window")


def test_append_window_is_incremental(tmp):
    print("[① appending one more window keeps every existing tile]")
    frame = synthetic_frame()
    run_dir = os.path.join(tmp, "SeeThrough", "Append")
    write_custom_manifest(run_dir, WINDOWS)
    grown = WINDOWS + [
        {"x": 0, "y": 1000, "size": 1728, "res": 0, "name": "新"},
    ]
    plan_id = region_plan.custom_document(
        grown, 1751, 4096, global_resolution=512,
        created=0)["partition_plan"]["plan_id"]
    out, tiles, indices, doc, _res, _depth = run_node(
        frame, "3", "SeeThrough/Append", regions=grown, select_plan=plan_id)
    check((doc.get("out") or {}).get("reset") is None,
          "a bigger appended window does not clear the folder")
    check(os.path.isfile(os.path.join(run_dir, "manifest.json")),
          "the existing manifest survives")
    check(indices == ["3"] and len(tiles) == 1,
          "only the new window is handed on (%s)" % indices)
    check(tuple(tiles[0].shape) == (1, 1728, 1728, 3),
          "the new crop is the 1728 px window (%s)" % (tuple(tiles[0].shape),))


def test_reorder_keeps_tiles(tmp):
    print("[① reordering the array keeps the cached tiles]")
    frame = synthetic_frame()
    run_dir = os.path.join(tmp, "SeeThrough", "Reorder")
    write_custom_manifest(run_dir, WINDOWS)
    swapped = [WINDOWS[1], WINDOWS[0], WINDOWS[2]]
    plan_id = region_plan.plan_key(swapped, 1751, 4096)
    out, tiles, indices, doc, _res, _depth = run_node(
        frame, "0", "SeeThrough/Reorder", regions=swapped, select_plan=plan_id)
    check((doc.get("out") or {}).get("reset") is None,
          "a reorder does not clear the folder")
    check(os.path.isfile(os.path.join(run_dir, "manifest.json")),
          "the existing manifest survives")
    check(indices == ["0"], "only the checked window runs (%s)" % indices)


def test_box_remap():
    print("[cache: adopted tiles are re-keyed by box, not by old index]")
    entries = [{"index": index, "tile": list(box)}
               for index, box in enumerate(expected_boxes(WINDOWS))]
    swapped = [WINDOWS[1], WINDOWS[0], WINDOWS[2]]
    doc = {"mode": region_plan.CUSTOM_MODE,
           "coords": expected_boxes(swapped)}
    remapped = tile_psd._remap_custom_entries(entries, doc)
    check([entry["index"] for entry in remapped] == [1, 0, 2],
          "each tile follows its box (%s)"
          % [entry["index"] for entry in remapped])
    check([entry["tile"] for entry in remapped] == expected_boxes(WINDOWS),
          "boxes are untouched by the remap")


def test_mapping_from_uniform(tmp):
    print("[① an old uniform folder is cleared when the partition type changes]")
    frame = synthetic_frame()
    old_dir = os.path.join(tmp, "SeeThrough", "Mapped")
    write_old_manifest(old_dir)
    out, tiles, indices, doc, _res, _depth = run_node(
        frame, "0,1", "SeeThrough/Mapped", select_plan="")
    reset = (doc.get("out") or {}).get("reset") or {}
    check(reset.get("required") is False and reset.get("changed") is True,
          "switching from the uniform grid reports a reset (%s)" % (reset,))
    check("分区方式" in str(reset.get("reason") or ""),
          "the mode change is the stated reason (%s)" % (reset.get("reason"),))
    check(indices == ["0", "1", "2"],
          "the cleared folder runs every window (%s)" % indices)
    check(not os.path.isfile(os.path.join(old_dir, "manifest.json")),
          "the old uniform manifest is cleared")

    # An old bottom tile barely touches the windows: mapping is empty, so the
    # literal index is kept -- still a subset, never the whole layout.
    out, tiles, indices, doc2, _res, _depth = run_node(
        frame, "2", "SeeThrough/Mapped", select_plan="")
    check(indices == ["2"] and len(tiles) == 1,
          "an unmappable old index falls back to itself, not to all windows "
          "(%s)" % indices)


def test_empty_layout_raises():
    print("[① an empty layout fails loudly instead of falling back]")
    frame = synthetic_frame()
    try:
        run_node(frame, "all", "SeeThrough/Empty", regions=[])
    except ValueError as exc:
        check("自定义分区" in str(exc) and "正方形" in str(exc),
              "the error names the missing windows (%s)" % exc)
        return
    check(False, "an empty custom_regions layout raises")


def main():
    with tempfile.TemporaryDirectory(prefix="stlu_custom_") as tmp:
        nodes_shared._output_root = lambda: tmp
        doc = test_custom_run()
        plan_id = doc["partition_plan"]["plan_id"]
        test_subset_does_not_clear(doc, plan_id)
        test_size_change_resets(doc, plan_id, tmp)
        test_order_change()
        test_append_window_is_incremental(tmp)
        test_reorder_keeps_tiles(tmp)
        test_box_remap()
        test_mapping_from_uniform(tmp)
        test_empty_layout_raises()
    print()
    if _failures:
        print("FAILED %d check(s):" % len(_failures))
        for item in _failures:
            print("  - " + item)
        return 1
    print("ALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
