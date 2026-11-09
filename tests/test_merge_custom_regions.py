"""CPU-only checks for merging user-placed windows with per-window canvases.

Uses the real tile writer and the real merge: two windows of the same source
pixel size but different model resolutions, so one PSD has to be scaled back
into its box.  No model, no GPU.

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_merge_custom_regions.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (COMFY_ROOT, PKG_DIR, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import region_plan  # noqa: E402
import tile_psd  # noqa: E402

_failures = []
SOURCE = (1600, 1200)
# Two 1024 px windows; the second is modelled at 512.  They overlap by 448 px.
WINDOWS = [
    {"x": 0, "y": 0, "size": 1024, "res": 1024, "name": "左"},
    {"x": 576, "y": 0, "size": 1024, "res": 512, "name": "右"},
]
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def tile_parts(resolution, color):
    """One part filling the whole tile frame in the model's pixel space."""
    pixels = np.zeros((resolution, resolution, 4), np.uint8)
    pixels[..., :3] = color
    pixels[..., 3] = 255
    return {
        "tag2pinfo": {"face": {
            "img": pixels,
            "depth": np.full((resolution, resolution), 128, np.uint8),
            "xyxy": [0, 0, resolution, resolution],
            "depth_median": 0.5,
        }},
        "frame_size": (resolution, resolution),
    }


def pixel(rgba, x, y):
    return [int(v) for v in rgba[int(y), int(x), :3]]


def test_coords_from_manifest():
    print("[manifest: the plan's boxes win over a rebuilt grid]")
    manifest = {
        "source": [1000, 1000],
        "tile_size": [1024, 1024],
        "overlap": 256,
        "partition_plan": {
            "mode": region_plan.CUSTOM_MODE,
            "tiles": [
                {"index": 0, "box": [10, 20, 180, 190]},
                {"index": 1, "box": [200, 30, 370, 200]},
            ],
        },
    }
    check(tile_psd._coords_from_manifest(manifest) ==
          [[10, 20, 180, 190], [200, 30, 370, 200]],
          "custom plan boxes are returned in index order")
    strips = {
        "source": [1000, 1000],
        "tile_size": [1024, 1024],
        "partition_plan": {
            "mode": "subject_strips",
            "regions": [
                {"index": 0, "box": [0, 0, 1000, 1000]},
                {"index": 1, "box": [0, 500, 1000, 1500]},
            ],
        },
    }
    check(tile_psd._coords_from_manifest(strips) ==
          [[0, 0, 1000, 1000], [0, 500, 1000, 1500]],
          "a strips plan falls back to its regions")
    gapped = {
        "source": [2048, 2048],
        "tile_size": [1024, 1024],
        "partition_plan": {"tiles": [{"index": 1, "box": [0, 0, 10, 10]}]},
    }
    rebuilt = tile_psd._coords_from_manifest(gapped)
    check(rebuilt and rebuilt[0] == [0, 0, 1024, 1024],
          "non-contiguous indices fall back to the uniform grid (%s)" % rebuilt[:1])


def test_merge(tmp):
    print("[merge: each window lands in its own source box, scaled to size]")
    coords = [[w["x"], w["y"], w["x"] + w["size"], w["y"] + w["size"]]
              for w in WINDOWS]
    manifest = tile_psd.write_tiles(
        [tile_parts(1024, RED), tile_parts(512, BLUE)],
        output_root=tmp, out_dir="SeeThrough/CustomMerge",
        filename_prefix="tile", source_size=SOURCE, tile_size=(1024, 1024),
        overlap=0, tile_coords=coords, tile_index_map=[0, 1],
        write_depth=False, include_original=False,
        generation={"test": 1},
    )
    run_dir = os.path.join(tmp, manifest["dir"])
    check([entry["canvas"] for entry in manifest["tiles"]] ==
          [[1024, 1024], [512, 512]],
          "the two PSDs carry their own canvases")
    check([entry["tile"] for entry in manifest["tiles"]] == coords,
          "manifest tile boxes are the window boxes")
    # ③ records the executed plan in the manifest; a reader that only has the
    # finished run must be able to recover the custom boxes from it.
    manifest["partition_plan"] = region_plan.custom_document(
        WINDOWS, SOURCE[0], SOURCE[1])["partition_plan"]
    check(tile_psd._coords_from_manifest(manifest) == coords,
          "the finished manifest alone describes the custom geometry")

    merged = tile_psd.merge_tiles(
        run_dir, manifest, out_name="merged", merge_parts=True,
        alpha_threshold=8, group_layers=False, include_depth=False,
        include_original=False, seam_feather=0,
    )
    merged_path = os.path.join(run_dir, merged["file"])
    check(merged.get("canvas") == list(SOURCE),
          "the merged canvas is the source size (%s)" % (merged.get("canvas"),))
    check(merged.get("tiles") == [0, 1],
          "the merge reports both window indices (%s)" % (merged.get("tiles"),))
    rgba = tile_psd.composite_psd(merged_path)
    check(rgba is not None and rgba.shape[:2] == (SOURCE[1], SOURCE[0]),
          "the merged PSD has the source canvas")

    left = pixel(rgba, 300, 500)
    check(left[0] > 200 and left[2] < 60,
          "the left window's colour is at its source position (%s)" % left)
    right = pixel(rgba, 1200, 500)
    check(right[2] > 200 and right[0] < 60,
          "the 512 PSD is scaled into its 1024 box at the source position (%s)"
          % right)
    top_right = pixel(rgba, 1500, 100)
    check(top_right[2] > 200 and top_right[0] < 60,
          "the scaled window reaches its far corner (%s)" % top_right)
    seam = pixel(rgba, 1024, 500)
    check(seam == list(RED) or seam == list(BLUE) or
          (seam[0] > 0 and seam[2] > 0),
          "the overlap carries one window's pixels, not a hole (%s)" % seam)
    return merged_path


def main():
    test_coords_from_manifest()
    with tempfile.TemporaryDirectory(prefix="stlu_merge_custom_") as tmp:
        test_merge(tmp)
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
