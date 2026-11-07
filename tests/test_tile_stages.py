"""CPU-only checks for the four-step tile pipeline.

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_tile_stages.py

No GPU, no server and no network are touched: the nodes are called directly
and every file lands in a temporary folder.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile

import numpy as np
import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import nodes as stlu  # noqa: E402
import nodes_shared  # noqa: E402
import psd_reader  # noqa: E402
import psd_writer  # noqa: E402
import tile_psd  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def make_part(width, height, left, top, value=200, depth_value=128):
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[..., :3] = value
    pixels[..., 3] = 255
    depth = np.full((height, width), depth_value, np.uint8)
    return {
        "img": pixels,
        "depth": depth,
        "xyxy": [left, top, left + width, top + height],
        "depth_median": 0.5,
    }


def make_parts(width=1024, height=1024, face_value=200, face_depth=128, eyewear=False):
    parts = {
        "face": make_part(80, 60, 100, 120, face_value, face_depth),
        "hairf": make_part(40, 30, 60, 40, 90, face_depth),
        "hairb": make_part(40, 30, 60, 80, 120, face_depth),
    }
    if eyewear:
        parts["eyewear"] = make_part(120, 20, 80, 170, 20, face_depth)
    return {"tag2pinfo": parts, "frame_size": (height, width)}


def make_wide_parts():
    """Parts that span their whole crop, so each tile owns a piece of them."""
    parts = {
        "face": make_part(900, 60, 60, 120, 200, 128),
        "hairf": make_part(900, 30, 60, 40, 90, 128),
        "hairb": make_part(900, 30, 60, 80, 120, 128),
    }
    return {"tag2pinfo": parts, "frame_size": (1024, 1024)}


def test_signatures():
    print("[signatures]")
    bad = 0
    for name, cls in stlu.NODE_CLASS_MAPPINGS.items():
        types = cls.INPUT_TYPES()
        wanted = list(types.get("required", {})) + list(types.get("optional", {}))
        signature = inspect.signature(cls.execute).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.values()):
            continue  # **options takes the whole widget set (STLU_PipelineInput)
        params = set(signature)
        params.discard("self")
        missing = [key for key in wanted if key not in params]
        if missing:
            bad += 1
            print("  BAD %-22s missing %s" % (name, missing))
    check(bad == 0, "every INPUT_TYPES key reaches execute (%d bad)" % bad)


def test_tile_grid(tmp):
    print("[① tile grid]")
    frame = torch.rand(1, 900, 1300, 3)
    grid = stlu.STLU_TileGrid()
    out = grid.execute(image=frame, tile_width=1024, tile_height=1024, overlap=128)
    tiles, indices, grid_doc, original = out["result"]
    check(len(tiles) == len(indices) and len(tiles) >= 2,
          "every tile came out (%d)" % len(tiles))
    check(indices == [str(i) for i in range(len(tiles))],
          "indices align with the tiles")
    # Stage one hands the split over in memory: no grids/ folder on disk.
    check(isinstance(grid_doc, str) and grid_doc.startswith("{"),
          "the grid document comes back inline")
    doc = tile_psd.load_grid_ref(grid_doc)
    check(doc["selection"] == "all", "stage one hands over every tile")
    check(not os.path.exists(os.path.join(tmp, "SeeThrough", "grids")),
          "no grids/ folder under the output root")
    check(not any(name.endswith(".json") and "grid" in name
                  for name in os.listdir(tmp)), "nothing grid-shaped at the output top")
    check(doc["count"] == len(tiles), "grid count matches the tiles")
    check(tuple(original.shape) == (1, 900, 1300, 3), "original image is returned")

    info = out["ui"]["stlu_grid"][0]
    check(info["count"] == doc["count"], "ui payload reports the grid size")
    return grid_doc


def test_tile_grid_select():
    print("[① cut + pick in one node]")
    frame = torch.rand(1, 900, 1300, 3)
    grid = stlu.STLU_TileGrid()

    out = grid.execute(image=frame, tile_width=1024, tile_height=1024,
                       overlap=128, select="1")
    tiles, indices, grid_doc, _ = out["result"]
    check(len(tiles) == 1 and indices == ["1"],
          "only the checked tile comes out (%r)" % (indices,))
    doc = tile_psd.load_grid_ref(grid_doc)
    check(doc["selection"] == "1", "the selection is written into grid_doc")
    info = out["ui"]["stlu_grid"][0]
    check(info["count"] == 2 and info["selected"] == 1,
          "ui reports selected/total")

    full = grid.execute(image=frame, tile_width=1024, tile_height=1024,
                        overlap=128, select="0-1")["result"]
    check(tile_psd.load_grid_ref(full[2])["selection"] == "all",
          "picking every tile is stored as all")

    try:
        grid.execute(image=frame, tile_width=1024, tile_height=1024,
                     overlap=128, select="none")
        check(False, "an empty selection should fail")
    except ValueError:
        check(True, "an empty selection fails instead of running nothing")


def test_select_tiles(grid_doc):
    print("[①b select tiles]")
    frame = torch.rand(1, 900, 1300, 3)
    grid = stlu.STLU_TileGrid()
    tiles, indices, grid_doc, _ = grid.execute(
        image=frame, tile_width=1024, tile_height=1024, overlap=128)["result"]
    sel = stlu.STLU_SelectTiles()
    out = sel.execute(tiles=tiles, indices=indices, grid_doc=[grid_doc],
                      image=[frame], select=["0-1"])
    picked, picked_indices, selected_doc = out["result"]
    check(len(picked) == 2, "two tiles kept")
    check(picked_indices == ["0", "1"], "original grid indices are preserved")
    doc = tile_psd.load_grid_ref(selected_doc)
    check(doc["selection"] == "0-1", "the selection is written into the grid document")
    info = out["ui"]["stlu_tile_select"][0]
    check(info["selected"] == 2 and info["count"] == len(tiles),
          "ui reports selected/total")
    try:
        sel.execute(tiles=tiles, indices=indices, grid_doc=[grid_doc], select=["none"])
        check(False, "an empty selection should fail")
    except ValueError:
        check(True, "an empty selection fails instead of silently doing nothing")
    return picked_indices, selected_doc


def test_wave_nodes(grid_doc, tmp):
    print("[③ wave / per-tile PSD / index]")
    frame = torch.rand(1, 900, 1300, 3)
    wave = stlu.STLU_Wave()
    wave_dir, wave_doc = wave.execute(out_dir="SeeThrough/New", group="角色B",
                                      filename_prefix="tile")["result"]
    check(not os.path.exists(wave_dir), "STLU_Wave names the folder without creating it")

    tile_node = stlu.STLU_TilePSD()
    records = []
    for index in ("0", "1"):
        rec = tile_node.execute(parts=make_parts(), wave_doc=wave_doc, index=index,
                                grid_doc=grid_doc, alpha_threshold=8)["result"][0]
        records.append(rec)
    check(all(not r.get("empty") for r in records), "two tile records came out")
    check(os.path.isdir(wave_dir), "the per-tile node creates the wave")
    psd_files = [f for f in os.listdir(wave_dir) if f.endswith(".psd")]
    check(len(psd_files) == 2, "two tile PSDs written (%s)" % psd_files)

    index_node = stlu.STLU_WaveIndex()
    out = index_node.execute(tiles=records, original=[frame], grid_doc=[grid_doc],
                             wave_doc=[wave_doc])
    run_dir, report_json, merge_doc = out["result"]
    report = json.loads(report_json)
    check(json.loads(merge_doc).get("dir") == report.get("dir"),
          "the index node also hands the folder + merge options to stage four")
    check(os.path.normpath(run_dir) == os.path.normpath(wave_dir),
          "the index node reports the same wave")
    check(report.get("count") == 2 and report.get("group") == "角色B",
          "manifest counts the tiles and keeps the group")
    check([t["index"] for t in report.get("tiles") or []] == [0, 1],
          "manifest keeps the original grid indices")
    check(os.path.isfile(os.path.join(run_dir, "manifest.json")),
          "manifest.json written")
    check(os.path.isfile(os.path.join(run_dir, "original.png")),
          "one original at the wave root")

    images_dir = os.path.join(run_dir, "images")
    check(os.path.isfile(os.path.join(images_dir, "images.json")),
          "images.json written")
    with open(os.path.join(images_dir, "images.json"), "r", encoding="utf-8") as handle:
        img_doc = json.load(handle)
    orig_rel = img_doc.get("original") or ""
    orig_path = os.path.normpath(os.path.join(images_dir, orig_rel)) if orig_rel else ""
    check(orig_rel == "../original.png" and os.path.isfile(orig_path),
          "images.json points at the wave root copy (%s)" % orig_rel)
    check(not os.path.isfile(os.path.join(images_dir, "original.png")),
          "no second identical original inside images/")
    sub = [d for d in os.listdir(images_dir) if d.startswith("tile_")]
    check(len(sub) == 2 and all(os.path.isfile(os.path.join(images_dir, d, "layers.json"))
                                for d in sub),
          "each tile has its own layers.json + PNGs")

    info = out["ui"]["stlu_tiles"][0]
    check(info.get("count") == 2 and info.get("preview", {}).get("filename"),
          "ui payload names the wave preview")

    merge = stlu.STLU_TileMerge()
    merged = merge.execute(out_dir="SeeThrough/New", name="merged",
                           group_layers=False, include_depth=True,
                           include_original=True, run_dir=run_dir)
    merged_path = merged["result"][0]
    check(bool(merged_path) and os.path.isfile(merged_path),
          "④ merges the new wave")
    if merged_path and os.path.isfile(merged_path):
        doc = psd_reader.read_psd(merged_path)
        check(doc["width"] == 1300 and doc["height"] == 900,
              "merged canvas is the source size")
        names = [layer["name"] for layer in doc["layers"]]
        check("face" in names and "depth" not in names and "original" in names,
              "merged layers keep colour/depth/original apart (%s)" % names)
    return run_dir, images_dir


def test_pipeline_input(tmp):
    print("[① pipeline input: the picked folder is the run folder]")
    frame = torch.rand(1, 900, 1300, 3)
    node = stlu.STLU_PipelineInput()
    out = node.execute(
        image=frame, tile_width=1024, tile_height=1024, overlap=128, select="1",
        resolution=512, num_inference_steps=20, depth_resolution=768, random_seed=42,
        tblr_split=False, use_lama=True, alpha_threshold=10,
        out_dir="SeeThrough/New", group="角色B", filename_prefix="tile",
        merge_name="merged_auto", group_layers=True, include_depth=False,
        include_original=True,
    )
    tiles, indices, grid_doc, original, res, steps, depth_res, seed, tblr, lama = out["result"]
    check(len(tiles) == 1 and indices == ["1"], "only the checked tile is handed on")
    check((res, steps, depth_res, seed) == (512, 20, 768, 42),
          "quality settings come out as scalars")
    check(tblr is False and lama is True, "post-process switches come out")
    doc = tile_psd.load_grid_ref(grid_doc)
    out_info = doc.get("out") or {}
    expected_dir = os.path.join(tmp, "SeeThrough", "New", "角色B")
    check(os.path.normpath(out_info.get("dir", "")) == os.path.normpath(expected_dir),
          "the picked folder is the run folder, with no wave subfolder (%s)"
          % out_info.get("dir"))
    check(out_info.get("flat") is True and bool(out_info.get("fingerprint")),
          "the doc marks the flat layout and fingerprints the source")
    check(out_info.get("alpha_threshold") == 10 and
          out_info.get("merge_name") == "merged_auto",
          "output/merge options travel with the doc")

    tile_node = stlu.STLU_TilePSD()
    record = tile_node.execute(parts=make_parts(), index="1",
                               grid_doc=grid_doc)["result"][0]
    check(not record.get("empty"), "per-tile PSD runs from grid_doc alone")
    check(os.path.isdir(out_info["dir"]), "the per-tile node creates the folder")
    psd_files = [f for f in os.listdir(out_info["dir"]) if f.endswith(".psd")]
    check(len(psd_files) == 1, "one tile PSD written")

    index_node = stlu.STLU_WaveIndex()
    run_dir, _report, merge_doc_json = index_node.execute(
        tiles=[record], original=[original], grid_doc=[grid_doc])["result"]
    check(os.path.normpath(run_dir) == os.path.normpath(out_info["dir"]),
          "the index node lands in the same folder")
    report = json.loads(_report)
    check(report.get("run_tiles") == [1] and report.get("reused_tiles") == [],
          "the first run marks tile 1 as its own work")
    merge_doc = json.loads(merge_doc_json)
    check(merge_doc.get("name") == "merged_auto" and
          merge_doc.get("include_original") is True,
          "the merge options are forwarded")
    check(merge_doc.get("alpha_threshold") == 10,
          "the merge uses the same alpha threshold as the tiles")

    # The real merge route resolves the relative dir against output/; the
    # test output root is temporary, so hand it the absolute path.
    absolute_doc = dict(merge_doc)
    absolute_doc["dir"] = out_info["dir"]
    merged = stlu.STLU_TileMerge().execute(
        merge_doc=json.dumps(absolute_doc, ensure_ascii=False))["result"][0]
    check(bool(merged) and os.path.isfile(merged),
          "④ merges automatically from merge_doc")

    # Re-run only tile 0 in the same folder: tile 1 must come back from disk
    # so the merge is complete instead of a one-tile fragment.
    print("[cache: a partial re-run completes from finished tiles]")
    out2 = node.execute(
        image=frame, tile_width=1024, tile_height=1024, overlap=128, select="0",
        resolution=512, num_inference_steps=20, depth_resolution=768, random_seed=42,
        tblr_split=False, use_lama=True, alpha_threshold=10,
        out_dir="SeeThrough/New", group="角色B", filename_prefix="tile",
        merge_name="merged_auto", group_layers=True, include_depth=False,
        include_original=True,
    )
    _tiles2, _indices2, grid_doc2, original2, *_ = out2["result"]
    record2 = tile_node.execute(parts=make_parts(), index="0",
                                grid_doc=grid_doc2)["result"][0]
    run_dir2, report_json2, merge_doc_json2 = index_node.execute(
        tiles=[record2], original=[original2], grid_doc=[grid_doc2])["result"]
    report2 = json.loads(report_json2)
    indices2 = [int(t["index"]) for t in report2.get("tiles") or []]
    check(os.path.normpath(run_dir2) == os.path.normpath(expected_dir),
          "the re-run reuses the same folder")
    check(indices2 == [0, 1], "the manifest still covers both tiles (%s)" % indices2)
    check(report2.get("run_tiles") == [0] and report2.get("reused_tiles") == [1],
          "only tile 0 was re-run; tile 1 was reused from disk")
    check(os.path.isfile(os.path.join(run_dir2, "original.png")),
          "one original stays at the folder root")
    merged2 = stlu.STLU_TileMerge().execute(
        merge_doc=json.dumps(dict(json.loads(merge_doc_json2),
                                  dir=out_info["dir"]), ensure_ascii=False))["result"][0]
    check(bool(merged2) and os.path.isfile(merged2),
          "④ merges the re-run plus the cached tile")

    # Merge-only options are not part of the generation fingerprint: they
    # must not wipe the tile cache or force every tile.
    out_merge_only = node.execute(
        image=frame, tile_width=1024, tile_height=1024, overlap=128, select="0",
        resolution=512, num_inference_steps=20, depth_resolution=768, random_seed=42,
        tblr_split=False, use_lama=True, alpha_threshold=10,
        out_dir="SeeThrough/New", group="角色B", filename_prefix="tile",
        merge_name="merged_auto", group_layers=False, include_depth=True,
        include_original=False,
    )
    check(len(out_merge_only["result"][0]) == 1,
          "merge-only option changes keep the tile cache and the selection")

    # A different source image must NOT inherit the old tiles; it also has to
    # clear the folder and hand on every tile, not just select="0".
    other = torch.rand(1, 900, 1300, 3)
    out3 = node.execute(
        image=other, tile_width=1024, tile_height=1024, overlap=128, select="0",
        resolution=512, num_inference_steps=20, depth_resolution=768, random_seed=42,
        tblr_split=False, use_lama=True, alpha_threshold=10,
        out_dir="SeeThrough/New", group="角色B", filename_prefix="tile",
        merge_name="merged_auto", group_layers=True, include_depth=False,
        include_original=True,
    )
    _t3, _i3, grid_doc3, original3, *_ = out3["result"]
    check(len(_t3) == 1,
          "a changed source clears the folder but keeps the selection")
    doc3 = tile_psd.load_grid_ref(grid_doc3)
    reset3 = (doc3.get("out") or {}).get("reset") or {}
    check(bool(doc3.get("generation_id")) and reset3.get("required") is True
          and reset3.get("partial") is True,
          "the partial reset is recorded in grid_doc")
    record3 = tile_node.execute(parts=make_parts(), index="0",
                                grid_doc=grid_doc3)["result"][0]
    _rd3, report_json3, _md3 = index_node.execute(
        tiles=[record3], original=[original3], grid_doc=[grid_doc3])["result"]
    report3 = json.loads(report_json3)
    check([int(t["index"]) for t in report3.get("tiles") or []] == [0],
          "a new source image does not inherit stale tiles")
    check(report3.get("generation_id") == doc3.get("generation_id"),
          "the manifest records the generation fingerprint")

    merged3 = stlu.STLU_TileMerge().execute(
        merge_doc=json.dumps(dict(json.loads(_md3), dir=out_info["dir"]),
                             ensure_ascii=False))["result"][0]
    check(bool(merged3) and os.path.isfile(merged3),
          "④ merges the regenerated folder")

    print("[⑤ PSD preview node]")
    preview = stlu.STLU_PSDPreview().execute(psd_path=merged3)
    tensor = preview["result"][0]
    check(tuple(tensor.shape) == (1, 900, 1300, 3),
          "the preview flattens the merged PSD at full canvas (%s)" % (tuple(tensor.shape),))
    panel_payload = (preview["ui"].get("stlu_parts") or [{}])[0]
    check(bool(panel_payload.get("path")) and os.path.isfile(panel_payload["path"]),
          "the preview hands its panel the merged PSD path")
    check("images" not in preview["ui"],
          "the preview does not draw a second copy in ComfyUI's own widget")
    by_folder = stlu.STLU_PSDPreview().execute(psd_path=out_info["dir"])
    check(tuple(by_folder["result"][0].shape) == tuple(tensor.shape),
          "pointing at the folder picks the merged PSD")

    # A resolution change is a pixel-affecting parameter too.
    out4 = node.execute(
        image=other, tile_width=1024, tile_height=1024, overlap=128, select="1",
        resolution=768, num_inference_steps=20, depth_resolution=768, random_seed=42,
        tblr_split=False, use_lama=True, alpha_threshold=10,
        out_dir="SeeThrough/New", group="角色B", filename_prefix="tile",
        merge_name="merged_auto", group_layers=True, include_depth=False,
        include_original=True,
    )
    _t4, _i4, grid_doc4, *_ = out4["result"]
    check(len(_t4) == 1,
          "a resolution change clears the folder and keeps the selection")
    doc4 = tile_psd.load_grid_ref(grid_doc4)
    check("resolution" in ((doc4.get("out") or {}).get("reset") or {}).get("keys", []),
          "the changed resolution is named in the reset report")
    gen4 = doc4.get("generation") or {}
    pixel_keys = {"source", "source_size", "tile_size", "overlap", "resolution",
                  "num_inference_steps", "depth_resolution", "random_seed",
                  "tblr_split", "use_lama", "alpha_threshold", "guide",
                  "layerdiff_model", "depth_model"}
    check(pixel_keys <= set(gen4),
          "the fingerprint covers every pixel-affecting parameter (%s)"
          % sorted(pixel_keys - set(gen4)))
    merge_keys = {"group_layers", "include_depth", "include_original",
                  "merge_name", "seam_feather"}
    check(not (merge_keys & set(gen4)),
          "merge-only options stay out of the generation fingerprint")


def test_rebuild(images_dir, tmp):
    print("[③ rebuild from images, no GPU]")
    node = stlu.STLU_RebuildPSD()
    out = node.execute(source=images_dir, out_dir="SeeThrough/Rebuilt", group="",
                       filename_prefix="tile", tiles="", alpha_threshold=8)
    run_dir, report_json = out["result"]
    report = json.loads(report_json)
    check(report.get("from_images") is True, "rebuild takes the offline path")
    check(report.get("count") == 2, "two tile PSDs rebuilt")
    check(os.path.isfile(os.path.join(run_dir, "manifest.json")),
          "rebuilt wave keeps its manifest")
    psd_files = [f for f in os.listdir(run_dir) if f.endswith(".psd")]
    check(len(psd_files) == 2, "two PSDs on disk (%s)" % psd_files)
    info = out["ui"]["stlu_tiles"][0]
    rel = info.get("images_dir") or ""
    check(bool(rel) and os.path.isdir(os.path.join(tmp, rel)),
          "rebuild keeps its own images (%s)" % rel)


def test_live_psd(grid_path, tmp):
    print("[③ live parts -> PSD + images]")
    frame = torch.rand(1, 900, 1300, 3)
    psd = stlu.STLU_TilesPSD()
    out = psd.execute(
        parts=[make_parts()], image=frame, tile_indices="0",
        out_dir="SeeThrough/Tiles", filename_prefix="tile",
        merge_parts=True, alpha_threshold=8,
        tile_width=1024, tile_height=1024, overlap=128,
        merged_psd=False, write_depth=True, group_layers=True,
        include_original=True, source="", grid=grid_path,
    )
    info = out["ui"]["stlu_tiles"][0]
    run_dir = out["result"][0]
    check(os.path.isdir(run_dir), "tile PSD run folder exists")
    check(info["count"] == 1, "one tile PSD written")
    psd_files = [f for f in os.listdir(run_dir) if f.endswith(".psd")]
    check(len(psd_files) == 1, "exactly one .psd next to the manifest")

    images_rel = info["images_dir"]
    images_dir = os.path.join(tmp, images_rel) if images_rel else ""
    check(bool(images_rel) and os.path.isdir(images_dir),
          "tile images saved automatically (%s)" % images_rel)
    # Item 5: one wave folder per run, PSDs at its top and images inside it.
    check(os.path.dirname(os.path.normpath(images_dir)) == os.path.normpath(run_dir)
          and os.path.basename(os.path.normpath(images_dir)) == "images",
          "images live in <wave>/images next to the PSDs")
    if images_dir:
        check(os.path.isfile(os.path.join(images_dir, "images.json")),
              "images.json index present")
        with open(os.path.join(images_dir, "images.json"), "r", encoding="utf-8") as handle:
            img_doc = json.load(handle)
        orig_rel = img_doc.get("original") or ""
        orig_path = os.path.normpath(os.path.join(images_dir, orig_rel)) if orig_rel else ""
        check(bool(orig_path) and os.path.isfile(orig_path),
              "images.json points at the wave's reference (%s)" % orig_rel)
        check(not os.path.isfile(os.path.join(images_dir, "original.png")),
              "no second identical original inside images/")
        check(os.path.isfile(os.path.join(run_dir, "original.png")),
              "the wave keeps exactly one original at its root")
        check(os.path.normpath(orig_path) == os.path.normpath(os.path.join(run_dir, "original.png")),
              "the reference resolves to that one file")
        sub = [d for d in os.listdir(images_dir) if d.startswith("tile_")]
        check(bool(sub) and os.path.isfile(
            os.path.join(images_dir, sub[0], "layers.json")),
            "each tile has layers.json + part PNGs")
        check(os.path.isfile(os.path.join(images_dir, sub[0], "hairf.png"))
              and os.path.isfile(os.path.join(images_dir, sub[0], "hairb.png")),
              "sub layers survive (group_layers on)")
    check(not os.path.isfile(os.path.join(run_dir, "_preview.png")),
          "no aggregate preview duplicating a tile preview")
    card = info.get("preview") or {}
    check(bool(card.get("filename")) and "tile" in card.get("filename", ""),
          "the card preview is a tile preview (%s)" % card.get("filename"))
    return run_dir, images_rel


def test_source_wins(grid_path, images_rel, tmp):
    print("[③ source overrides parts]")
    images_dir = os.path.join(tmp, images_rel)
    psd = stlu.STLU_TilesPSD()
    out = psd.execute(
        parts=[make_parts()], image=None, tile_indices="",
        out_dir="SeeThrough/Tiles", filename_prefix="fromimg",
        merge_parts=True, alpha_threshold=8,
        tile_width=1024, tile_height=1024, overlap=128,
        merged_psd=False, write_depth=True, group_layers=True,
        include_original=True, source=images_dir, grid=grid_path,
    )
    report = json.loads(out["result"][1])
    check(report.get("from_images") is True,
          "an explicit source runs the offline path even with parts connected")
    check(report.get("count") == 1, "one PSD rebuilt from the images")
    return out["result"][0]


def test_wave_group_and_subset(grid_path, images_rel, tmp):
    print("[wave id folder + group prefix + tile subset]")
    psd = stlu.STLU_TilesPSD()
    old_prompt = nodes_shared._prompt_id
    nodes_shared._prompt_id = lambda: "abcd1234-ffff-0000-1111-222233334444"
    try:
        out = psd.execute(
            parts=None, image=None, tile_indices="",
            out_dir="SeeThrough", filename_prefix="tile",
            merge_parts=True, alpha_threshold=8,
            tile_width=1024, tile_height=1024, overlap=128,
            merged_psd=False, write_depth=True, group_layers=True,
            include_original=True,
            source=os.path.join(tmp, images_rel), grid=grid_path,
            group="角色A", tiles="0",
        )
    finally:
        nodes_shared._prompt_id = old_prompt
    run_dir = out["result"][0]
    rel = os.path.relpath(run_dir, tmp).replace(os.sep, "/")
    check(rel.startswith("SeeThrough/角色A/"),
          "group prefix becomes a category folder (%s)" % rel)
    check("abcd1234" in os.path.basename(run_dir),
          "the prompt id names the wave folder (%s)" % os.path.basename(run_dir))
    check(os.path.isfile(os.path.join(run_dir, "manifest.json")),
          "offline wave keeps its manifest")
    info = out["ui"]["stlu_tiles"][0]
    images_rel = info.get("images_dir") or ""
    images_abs = os.path.join(tmp, images_rel) if images_rel else ""
    check(bool(images_abs) and os.path.isdir(images_abs),
          "offline rebuild keeps its own images (%s)" % images_rel)
    check(bool(images_abs) and os.path.normpath(images_abs).startswith(os.path.normpath(run_dir)),
          "offline images live inside the new wave, not the source folder")
    try:
        psd.execute(
            parts=None, image=None, tile_indices="",
            out_dir="SeeThrough", filename_prefix="tile",
            merge_parts=True, alpha_threshold=8,
            tile_width=1024, tile_height=1024, overlap=128,
            merged_psd=False, write_depth=True, group_layers=True,
            include_original=True, source=os.path.join(tmp, images_rel),
            grid=grid_path, group="", tiles="99",
        )
        check(False, "an unknown tile subset should fail")
    except ValueError:
        check(True, "an unknown tile subset fails instead of writing nothing")


def test_merge(run_dir):
    print("[④ merge tile PSDs]")
    merge = stlu.STLU_TileMerge()
    out = merge.execute(out_dir="SeeThrough/Tiles", name="merged",
                        group_layers=False, include_depth=True,
                        include_original=True, run_dir=run_dir)
    psd_path = out["result"][0]
    check(bool(psd_path) and os.path.isfile(psd_path), "merged PSD written")
    if not (psd_path and os.path.isfile(psd_path)):
        return
    doc = psd_reader.read_psd(psd_path)
    names = [layer["name"] for layer in doc["layers"]]
    check("face" in names, "colour layer merged")
    check("depth" not in names, "depth is NOT a layer inside the colour PSD")
    check("original" in names, "original sits at the bottom")
    check(doc["width"] == 1300 and doc["height"] == 900, "merged canvas is the source size")
    payload = out["ui"]["stlu_tile_merge"][0]
    check(payload.get("file"), "ui payload names the merged file")
    depth_psd = payload.get("depth_psd") or ""
    depth_path = os.path.join(run_dir, depth_psd) if depth_psd else ""
    check(bool(depth_psd) and os.path.isfile(depth_path),
          "depth written to its own PSD (%s)" % depth_psd)
    if depth_path and os.path.isfile(depth_path):
        ddoc = psd_reader.read_psd(depth_path)
        dnames = [layer["name"] for layer in ddoc["layers"]]
        check(dnames == ["depth"],
              "the depth PSD holds only the depth layer (%s)" % dnames)


def test_merge_group_and_flat(grid_path):
    print("[④ folder mode vs flat mode with two tiles]")
    frame = torch.rand(1, 900, 1300, 3)
    psd = stlu.STLU_TilesPSD()
    out = psd.execute(
        parts=[make_wide_parts(), make_wide_parts()], image=frame, tile_indices="0-1",
        out_dir="SeeThrough/Group", filename_prefix="tile",
        merge_parts=True, alpha_threshold=8,
        tile_width=1024, tile_height=1024, overlap=128,
        merged_psd=False, write_depth=True, group_layers=True,
        include_original=True, source="", grid=grid_path,
    )
    run_dir = out["result"][0]
    merge = stlu.STLU_TileMerge()

    grouped = merge.execute(out_dir="SeeThrough/Group", name="grouped",
                            group_layers=True, include_depth=True,
                            include_original=False, run_dir=run_dir)
    names = grouped["ui"]["stlu_tile_merge"][0].get("layers") or []
    check("face/" in names, "same part from two tiles becomes one folder (%s)" % names)
    check("face_0" in names and "face_1" in names,
          "each tile keeps its own copy inside the folder (%s)" % names)
    check("hair/" not in names,
          "front and back hair never share one folder (%s)" % names)
    check("hairf/" in names and "hairf_0" in names and "hairf_1" in names,
          "front hair keeps its own folder with per-tile copies (%s)" % names)
    check("hairb/" in names and "hairb_0" in names and "hairb_1" in names,
          "back hair keeps its own folder with per-tile copies (%s)" % names)
    check("depth" not in names, "folder merge keeps depth out of the colour PSD")
    grouped_path = grouped["result"][0]
    check(bool(grouped_path) and os.path.isfile(grouped_path), "folder-mode PSD written")
    if grouped_path and os.path.isfile(grouped_path):
        gdoc = psd_reader.read_psd(grouped_path)
        gnames = [layer["name"] for layer in gdoc["layers"]]
        check("face" in gnames and any(n.startswith("</Layer group") for n in gnames),
              "folder markers survive a re-read (%s)" % gnames)

    flat = merge.execute(out_dir="SeeThrough/Group", name="flat",
                         group_layers=False, include_depth=False,
                         include_original=False, run_dir=run_dir)
    flat_payload = flat["ui"]["stlu_tile_merge"][0]
    flat_names = flat_payload.get("layers") or []
    check(flat_names.count("face") == 1,
          "without folders the same part is composited into ONE layer (%s)" % flat_names)
    check("hairf" in flat_names and "hairb" in flat_names and "hair" not in flat_names,
          "flat mode never aliases front/back hair into one layer (%s)" % flat_names)
    check(not flat_payload.get("depth_psd"),
          "unchecked depth export writes no depth PSD")
    check(not os.path.isfile(os.path.join(run_dir, "flat_depth.psd")),
          "no stray depth PSD on disk when the export is off")


def test_merge_scaled_tile(tmp):
    print("[④ render canvas smaller than the grid tile]")
    from PIL import Image

    run_dir = os.path.join(tmp, "SeeThrough/Scale/20260101_000000_abcd1234")
    os.makedirs(run_dir, exist_ok=True)
    # The grid cut a 1024 box but SeeThrough rendered it at 512: a 20x20 part
    # at (100, 200) of the box is a 10x10 part at (50, 100) of the canvas.
    px = np.zeros((10, 10, 4), np.uint8)
    px[..., :3] = (10, 20, 30)
    px[..., 3] = 255
    psd_writer.write_psd(os.path.join(run_dir, "tile_00.psd"),
                         [{"name": "face", "pixels": px, "left": 50, "top": 100}],
                         512, 512, "rle")
    depth = np.zeros((512, 512), np.uint8)
    depth[100:110, 50:60] = 200
    Image.fromarray(depth, "L").save(os.path.join(run_dir, "tile_00_depth.png"))
    manifest = {
        "dir": "SeeThrough/Scale/20260101_000000_abcd1234",
        "source": [1024, 1024],
        "original": "",
        "tiles": [{
            "index": 0,
            "file": "tile_00.psd",
            "canvas": [512, 512],
            "tile": [0, 0, 1024, 1024],
            "depth": "tile_00_depth.png",
            "layers": [{"name": "face", "depth_median": 0.5}],
        }],
    }
    info = tile_psd.merge_tiles(run_dir, manifest, out_name="scaled",
                                group_layers=False, include_depth=True,
                                include_original=False)
    doc = psd_reader.read_psd(os.path.join(run_dir, info["file"]))
    check(doc["width"] == 1024 and doc["height"] == 1024,
          "merged canvas is the grid source size")
    face = doc["layers"][0]
    alpha = face["pixels"][..., 3]
    ys, xs = np.nonzero(alpha > 8)
    check(bool(len(xs)), "the scaled part still has pixels")
    if len(xs):
        cx = face["left"] + (int(xs.min()) + int(xs.max()) + 1) / 2.0
        cy = face["top"] + (int(ys.min()) + int(ys.max()) + 1) / 2.0
        check(abs(cx - 110) <= 1.5 and abs(cy - 210) <= 1.5,
              "the part lands at its 1024-box position, not half size top-left "
              "(centre %.1f,%.1f)" % (cx, cy))
        check(abs((int(xs.max()) - int(xs.min()) + 1) - 20) <= 2,
              "the part is scaled up to the box (%d px wide)"
              % (int(xs.max()) - int(xs.min()) + 1))
    depth_psd = info.get("depth_psd") or ""
    check(bool(depth_psd) and os.path.isfile(os.path.join(run_dir, depth_psd)),
          "depth is scaled with the layer and written to its own PSD")
    if depth_psd:
        ddoc = psd_reader.read_psd(os.path.join(run_dir, depth_psd))
        dlayer = ddoc["layers"][0]
        check(abs(dlayer["left"] - 100) <= 2 and abs(dlayer["top"] - 200) <= 2,
              "depth sits where the part sits (%d,%d)" % (dlayer["left"], dlayer["top"]))
        dpx = dlayer["pixels"]
        row = 210 - int(dlayer["top"])
        col = 110 - int(dlayer["left"])
        val = int(dpx[row, col, 0]) if (0 <= row < dpx.shape[0] and 0 <= col < dpx.shape[1]) else -1
        check(abs(val - 200) <= 6,
              "depth values survive the upscale (centre %d)" % val)


def test_mixed_merge(grid_path, tmp):
    print("[④ mixed merge: each tile from a different wave]")
    frame = torch.rand(1, 900, 1300, 3)
    psd = stlu.STLU_TilesPSD()
    base = dict(out_dir="SeeThrough/Mix", merge_parts=True, alpha_threshold=8,
                tile_width=1024, tile_height=1024, overlap=128,
                merged_psd=False, write_depth=True, group_layers=False,
                include_original=True, source="", grid=grid_path,
                tile_indices="0", image=frame)
    old_prompt = nodes_shared._prompt_id
    try:
        nodes_shared._prompt_id = lambda: "aaaa1111-0000-0000-0000-000000000000"
        out_a = psd.execute(parts=[make_parts(face_value=200, face_depth=200)],
                            filename_prefix="wavea", **base)
        nodes_shared._prompt_id = lambda: "bbbb2222-0000-0000-0000-000000000000"
        out_b = psd.execute(parts=[make_parts(face_value=100, face_depth=64, eyewear=True)],
                            filename_prefix="waveb", **base)
    finally:
        nodes_shared._prompt_id = old_prompt
    dir_a = out_a["result"][0]
    dir_b = out_b["result"][0]
    check(os.path.normpath(dir_a) != os.path.normpath(dir_b), "two separate wave folders")

    def merged(priority_a, priority_b, name):
        return tile_psd.merge_mixed(
            [{"dir": dir_a, "index": 0, "priority": priority_a},
             {"dir": dir_b, "index": 0, "priority": priority_b}],
            tmp, out_name=name, group_layers=False,
            include_depth=True, include_original=False)

    info = merged(0, 1, "b_wins.psd")
    doc = psd_reader.read_psd(os.path.join(dir_a, info["file"]))
    layers = {layer["name"]: layer for layer in doc["layers"]}
    check("eyewear" in layers, "a part only the second wave has survives the merge")
    face = layers["face"]["pixels"]
    check(abs(int(face[..., 0].mean()) - 100) <= 2,
          "the shared part comes from the higher-priority wave (face=%d)"
          % int(face[..., 0].mean()))
    depth_doc = psd_reader.read_psd(os.path.join(dir_a, info["depth_psd"]))
    dpix = depth_doc["layers"][0]["pixels"]
    dmask = dpix[..., 3] > 0
    check(bool(dmask.any()) and abs(int(dpix[..., 0][dmask].mean()) - 65) <= 4,
          "depth follows the winning wave, not the average")
    check("depth" not in [l["name"] for l in doc["layers"]],
          "mixed merge also keeps depth out of the colour PSD")

    info_b = merged(1, 0, "a_wins.psd")
    doc_b = psd_reader.read_psd(os.path.join(dir_a, info_b["file"]))
    layers_b = {layer["name"]: layer for layer in doc_b["layers"]}
    face_b = layers_b["face"]["pixels"]
    check(abs(int(face_b[..., 0].mean()) - 200) <= 2,
          "picking the first wave for the face restores its pixels (face=%d)"
          % int(face_b[..., 0].mean()))
    check("eyewear" in layers_b, "the non-conflicting part is kept either way")


def test_pipeline_offset(tmp):
    print("[① grid phase: offset moves the split line and resets the cache]")
    frame = torch.rand(1, 1024, 2048, 3)
    node = stlu.STLU_PipelineInput()
    kwargs = dict(
        image=frame, tile_width=1024, tile_height=1024, overlap=128, select="0",
        resolution=512, num_inference_steps=20, depth_resolution=768, random_seed=42,
        tblr_split=False, use_lama=True, alpha_threshold=10,
        out_dir="SeeThrough/PhaseTest", group="", filename_prefix="tile",
        merge_name="merged", group_layers=True, include_depth=False,
        include_original=False,
    )
    out = node.execute(grid_offset_x=512, grid_offset_y=0, **kwargs)
    doc = tile_psd.load_grid_ref(out["result"][2])
    check(doc.get("offset") == [512, 0], "grid_doc carries the phase")
    check((doc.get("generation") or {}).get("offset") == [512, 0],
          "the phase is part of the generation fingerprint")
    boxes = [[int(v) for v in c] for c in doc["coords"]]
    check([512, 0, 1536, 1024] in boxes, "the phase created the centred tile")
    check(len(out["result"][0]) == 1, "the explicit selection is still honoured")

    # generation.json from the run above is on disk; returning to offset 0
    # drops the key, which must still be detected as a changed fingerprint.
    out0 = node.execute(grid_offset_x=0, grid_offset_y=0, **kwargs)
    doc0 = tile_psd.load_grid_ref(out0["result"][2])
    reset = (doc0.get("out") or {}).get("reset") or {}
    check(reset.get("required") is True, "offset 0 after a phased run resets")
    check("offset" in (reset.get("keys") or []), "the dropped offset key is reported")
    check(reset.get("partial") is True, "the explicit selection is preserved")
    check((doc0.get("generation") or {}).get("offset") is None,
          "offset 0 keeps the fingerprint free of the key")

    # The whole chain on a phased grid: every tile must land at its phased
    # box, otherwise the merge would assemble the picture shifted.
    print("[① grid phase: the whole chain keeps the phased boxes]")
    frame2 = torch.rand(1, 1024, 2048, 3)
    full_kwargs = dict(kwargs)
    full_kwargs.update(image=frame2, select="all", out_dir="SeeThrough/PhaseFull")
    full = node.execute(grid_offset_x=512, grid_offset_y=0, **full_kwargs)
    tiles, indices, grid_doc3, original3, *_ = full["result"]
    doc3 = tile_psd.load_grid_ref(grid_doc3)
    boxes = [[int(v) for v in c] for c in doc3["coords"]]
    check(indices == [str(i) for i in range(len(boxes))],
          "all tiles of the phased grid are handed on")
    tile_node = stlu.STLU_TilePSD()
    records = [tile_node.execute(parts=make_parts(), index=str(i),
                                 grid_doc=grid_doc3)["result"][0]
               for i in range(len(boxes))]
    run_dir, report_json, merge_doc_json = stlu.STLU_WaveIndex().execute(
        tiles=records, original=[original3], grid_doc=[grid_doc3])["result"]
    report = json.loads(report_json)
    stored = [list(t["tile"]) for t in report["tiles"]]
    check(stored == boxes, "the manifest keeps the phased boxes (%s)" % stored[:2])
    merge_doc = json.loads(merge_doc_json)
    merged_path = stlu.STLU_TileMerge().execute(
        merge_doc=json.dumps(dict(merge_doc, dir=run_dir),
                             ensure_ascii=False))["result"][0]
    check(bool(merged_path) and os.path.isfile(merged_path),
          "the phased run merges")
    merged_doc = psd_reader.read_psd(merged_path)
    check([merged_doc["width"], merged_doc["height"]] == [2048, 1024],
          "the merged canvas is the source size")


def main():
    test_signatures()
    with tempfile.TemporaryDirectory(prefix="stlu_tiles_") as tmp:
        nodes_shared._output_root = lambda: tmp
        grid_path = test_tile_grid(tmp)
        test_tile_grid_select()
        picked_indices, selected_doc = test_select_tiles(grid_path)
        _wave_dir, wave_images = test_wave_nodes(selected_doc, tmp)
        test_pipeline_input(tmp)
        test_pipeline_offset(tmp)
        test_rebuild(wave_images, tmp)
        run_dir, images_rel = test_live_psd(selected_doc, tmp)
        test_source_wins(selected_doc, images_rel, tmp)
        test_wave_group_and_subset(selected_doc, images_rel, tmp)
        test_merge(run_dir)
        test_merge_group_and_flat(selected_doc)
        test_merge_scaled_tile(tmp)
        test_mixed_merge(selected_doc, tmp)
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
