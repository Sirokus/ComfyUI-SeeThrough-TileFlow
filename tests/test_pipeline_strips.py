"""CPU-only checks for the subject-strip mode in the four-step pipeline.

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_pipeline_strips.py
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
import psd_reader  # noqa: E402
import tile_psd  # noqa: E402
import tile_strips  # noqa: E402

_failures = []
STRIPS = "方形全宽分区"


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def synthetic_frame(width=1751, height=4096):
    image = np.full((height, width, 3), 255, np.uint8)
    image[43:3990, 34:1696] = (200, 40, 40)
    image[4000:4020, 1731:1751] = (200, 40, 40)
    return torch.from_numpy(image.astype(np.float32) / 255.0)[None]


def make_part(width, height, left, top, value=(200, 40, 40)):
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[..., :3] = value
    pixels[..., 3] = 255
    depth = np.full((height, width), 128, np.uint8)
    return {
        "img": pixels,
        "depth": depth,
        "xyxy": [left, top, left + width, top + height],
        "depth_median": 0.5,
    }


def make_parts(resolution=768):
    return {
        "tag2pinfo": {"face": make_part(32, 32, 100, 100)},
        "frame_size": (resolution, resolution),
    }


def write_old_manifest(run_dir):
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


def run_node(frame, select, out_dir, **kwargs):
    node = stlu.STLU_PipelineInput()
    args = dict(
        image=frame, tile_width=1024, tile_height=1024, overlap=256,
        select=select, resolution=512, num_inference_steps=30,
        depth_resolution=-1, random_seed=0, tblr_split=True, use_lama=True,
        alpha_threshold=8, out_dir=out_dir, group="", filename_prefix="tile",
        merge_name="merged", group_layers=True, include_depth=False,
        include_original=False, partition_mode=STRIPS,
        strip_max_side=2048, strip_max_resolution=768, strip_margin=32,
    )
    args.update(kwargs)
    out = node.execute(**args)
    tiles, indices, grid_doc, original, res, steps, depth, seed, tblr, lama = out["result"]
    doc = tile_psd.load_grid_ref(grid_doc)
    return out, tiles, indices, grid_doc, original, doc, res, depth


def test_subject_run(tmp):
    print("[① subject strips: plan, crop and generation]")
    frame = synthetic_frame()
    out, tiles, indices, grid_doc, original, doc, res, depth = run_node(
        frame, "0", "SeeThrough/Strips", grid_offset_x=123, grid_offset_y=45)
    check(doc.get("mode") == "subject_strips", "doc mode is subject_strips")
    check(doc["count"] == 3 and len(doc["coords"]) == 3, "three strips planned")
    check(doc["tile_size"] == [1726, 1726], "doc tile size is the square side")
    check(indices == ["0"] and len(tiles) == 1, "only the checked strip is handed on")
    check(tuple(tiles[0].shape) == (1, 1726, 1726, 3),
          "the crop is a square region (%s)" % (tuple(tiles[0].shape),))
    check(res == 512, "the strip uses the quality card resolution (%s)" % res)
    check(depth == -1, "depth follows the strip resolution")
    check(doc["selection"] == "0", "selection is encoded for the strip plan")
    generation = doc["generation"]
    check("partition" in generation and generation["partition"]["plan_id"],
          "generation fingerprint carries the partition plan")
    check(generation["tile_size"] == [1726, 1726] and
          generation["resolution"] == 512,
          "generation carries the strip geometry/resolution")
    check("offset" not in generation, "grid phase is not part of a strip run")
    info = out["ui"]["stlu_grid"][0]
    check(info.get("mode") == "subject_strips" and
          (info.get("partition") or {}).get("plan_id"),
          "UI payload carries the executed plan")
    plan = doc["partition_plan"]
    check(plan["side"] == 1726 and plan["resolution"] == 512,
          "partition plan reports side/resolution")
    check(doc["coords"][0][0] <= 34 and doc["coords"][0][2] >= 1696,
          "the first strip spans the subject width")
    return doc


def test_current_selection(doc):
    print("[① select_plan marks an already-migrated selection]")
    frame = synthetic_frame()
    plan_id = (doc.get("partition_plan") or {}).get("plan_id") or ""
    out, tiles, indices, _grid_doc, _original, doc2, _res, _depth = run_node(
        frame, "2", "SeeThrough/Strips", select_plan=plan_id)
    check(indices == ["2"] and len(tiles) == 1,
          "a selection already tagged with the current plan is used as-is")
    check(doc2["selection"] == "2", "the current selection survives")
    check(doc2["coords"][2] == doc["coords"][2], "the third strip is the same box")


def test_old_selection_mapping(tmp):
    print("[① an old uniform folder is cleared when the partition type changes]")
    frame = synthetic_frame()
    old_dir = os.path.join(tmp, "SeeThrough", "Mapped")
    write_old_manifest(old_dir)
    out, tiles, indices, _grid_doc, _original, doc, _res, _depth = run_node(
        frame, "0,1", "SeeThrough/Mapped", select_plan="")
    reset = (doc.get("out") or {}).get("reset") or {}
    check(reset.get("required") is False and reset.get("changed") is True,
          "switching from uniform reports a reset (%s)" % (reset,))
    check("分区方式" in str(reset.get("reason") or ""),
          "the mode change is the stated reason (%s)" % (reset.get("reason"),))
    check(indices == ["0", "1", "2"],
          "the cleared folder runs every strip (%s)" % indices)
    check(not os.path.isfile(os.path.join(old_dir, "manifest.json")),
          "the old uniform manifest is cleared")
    rgb = np.clip(np.rint(synthetic_frame()[0].numpy() * 255.0),
                  0, 255).astype(np.uint8)
    expected = tile_strips.plan_from_rgb(
        rgb, overlap=256, max_side=2048, max_resolution=768, margin=32)
    check(doc["coords"][0] == expected["coords"][0],
          "the mapped strip is the planned first region")


def test_old_all_mapping(tmp):
    print("[① an old all-selection still means all new strips]")
    frame = synthetic_frame()
    old_dir = os.path.join(tmp, "SeeThrough", "MappedAll")
    write_old_manifest(old_dir)
    out, tiles, indices, _grid_doc, _original, doc, _res, _depth = run_node(
        frame, "all", "SeeThrough/MappedAll", select_plan="")
    check(indices == ["0", "1", "2"] and len(tiles) == 3,
          "all old tiles map to all strips (%s)" % indices)
    check(doc["selection"] == "all", "the all-selection is preserved")


def test_chain_and_merge(tmp):
    print("[③/④: square strips flow through PSD, manifest and merge]")
    frame = synthetic_frame()
    out, tiles, indices, grid_doc, original, doc, res, _depth = run_node(
        frame, "all", "SeeThrough/Chain")
    records = []
    tile_node = stlu.STLU_TilePSD()
    for index in indices:
        record = tile_node.execute(parts=make_parts(res),
                                   index=index, grid_doc=grid_doc)["result"][0]
        check(not record.get("empty"), "strip %s writes a PSD" % index)
        records.append(record)
    wave_dir, report_json, merge_doc_json = stlu.STLU_WaveIndex().execute(
        tiles=records, original=[original], grid_doc=[grid_doc])["result"]
    report = json.loads(report_json)
    check(report.get("mode") == "subject_strips", "manifest carries the strip mode")
    check(bool(report.get("partition_plan")), "manifest carries the partition plan")
    boxes = [entry["tile"] for entry in report.get("tiles") or []]
    check(boxes == doc["coords"], "manifest tile boxes are the strip boxes")
    merge_doc = json.loads(merge_doc_json)
    merge_doc["dir"] = wave_dir
    merged_path = stlu.STLU_TileMerge().execute(
        merge_doc=json.dumps(merge_doc, ensure_ascii=False))["result"][0]
    check(bool(merged_path) and os.path.isfile(merged_path), "④ merges the strips")
    merged = psd_reader.read_psd(merged_path)
    check([merged["width"], merged["height"]] == [1751, 4096],
          "merged canvas is the source size")
    check(len(merged["layers"]) >= 1, "merged PSD has layers")


def test_fallback(tmp):
    print("[① explicit fallback when a square cannot cover the subject]")
    # A 1662 px wide subject on a 1600 px tall canvas cannot fit in any square.
    frame = synthetic_frame()[:, :1600]
    out, tiles, _indices, _grid_doc, _original, doc, res, _depth = run_node(
        frame, "0", "SeeThrough/Fallback",
        grid_offset_x=123, grid_offset_y=45)
    check(doc.get("mode") == "uniform_fallback",
          "an impossible square falls back to the uniform grid")
    check(bool((doc.get("partition_plan") or {}).get("fallback")),
          "the fallback reason is recorded")
    check("partition" in doc["generation"], "fallback mode is fingerprinted")
    check(doc["generation"].get("offset") == [123, 45],
          "the fallback fingerprint keeps the grid phase")
    check(res == 512, "fallback uses the uniform resolution")
    expected = [list(box) for box in
                tile_psd.grid_coords(1751, 1600, 1024, 1024, 256, 123, 45)]
    check(doc["coords"] == expected,
          "the fallback document matches the panel's uniform geometry")
    check(len(tiles) == 1 and tuple(tiles[0].shape)[1:3] == (1024, 1024),
          "fallback crops are the uniform tiles")


def test_mapping_without_manifest(tmp):
    print("[① a mode switch with no previous manifest maps the uniform area]")
    frame = synthetic_frame()
    out, tiles, indices, _grid_doc, _original, doc, _res, _depth = run_node(
        frame, "0,1", "SeeThrough/Fresh",
        select_plan="", grid_offset_x=0, grid_offset_y=0)
    check(indices == ["0"] and len(tiles) == 1,
          "top-row uniform indices map to the first strip, not strips 0-1 (%s)"
          % indices)
    check(doc["selection"] == "0", "the mapped selection is written into the doc")


def test_mapping_helper():
    print("[selection mapper: conservative by area]")
    old = [[0, 0, 1024, 1024], [727, 0, 1751, 1024],
           [0, 768, 1024, 1792], [727, 768, 1751, 1792]]
    new = [[2, 11, 1728, 1737], [2, 1154, 1728, 2880], [2, 2296, 1728, 4022]]
    check(stlu._map_selection_by_area(old, [0, 1], new) == [0],
          "the old top row maps to the first strip")
    check(stlu._map_selection_by_area(old, [0, 1, 2, 3], new) == [0, 1, 2],
          "an all-old selection maps to all strips")
    check(stlu._map_selection_by_area(old, [], new) == [],
          "an empty selection stays empty")
    check(stlu._map_selection_by_area(old, [0], []) == [],
          "no new geometry maps to nothing")


def main():
    with tempfile.TemporaryDirectory(prefix="stlu_strips_") as tmp:
        nodes_shared._output_root = lambda: tmp
        doc = test_subject_run(tmp)
        test_current_selection(doc)
        test_old_selection_mapping(tmp)
        test_old_all_mapping(tmp)
        test_chain_and_merge(tmp)
        test_fallback(tmp)
        test_mapping_without_manifest(tmp)
        test_mapping_helper()
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
