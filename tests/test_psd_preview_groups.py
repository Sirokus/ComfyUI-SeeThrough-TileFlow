"""CPU-only check for the ⑤ PSD preview's tile / part grouping.

The preview panel can list a merged PSD two ways: one thumbnail per tile
(what the tile wall generated) or one per body part (every same-named copy
stacked back together).  This builds a tiny two-tile run and checks both
answers, plus the white-backed whole-canvas preview.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import folder_paths  # noqa: E402
import routes  # noqa: E402
import tile_psd  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def make_parts(size, tags):
    info = {}
    for i, tag in enumerate(tags):
        pixels = np.zeros((size, size, 4), np.uint8)
        pixels[..., :3] = (30 + 40 * i, 120, 220)
        pixels[..., 3] = 255
        info[tag] = {
            "img": pixels,
            "depth": np.full((size, size), 128, np.uint8),
            "xyxy": [0, 0, size, size],
            "depth_median": 0.5,
            "tag": tag,
        }
    return {"tag2pinfo": info, "frame_size": (size, size)}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out_root = os.path.join(tmp, "output")
        temp_dir = os.path.join(tmp, "temp")
        os.makedirs(out_root, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        run_dir = os.path.join(out_root, "SeeThrough", "PreviewGroups")
        original_root = routes._scope_root
        original_temp = folder_paths.get_temp_directory
        routes._scope_root = lambda scope: os.path.realpath(out_root)
        folder_paths.get_temp_directory = lambda: temp_dir
        try:
            size = 64
            tile_psd.write_tiles(
                [make_parts(size, ["face", "hairf"]),
                 make_parts(size, ["face", "hairf"])],
                out_root, filename_prefix="tile", source_size=(128, 64),
                tile_size=(size, size), overlap=0, tile_index_map=[0, 1],
                write_depth=False, merge_group_layers=True,
                include_original=False, run_dir_abs=run_dir,
                write_manifest=True, merge_after=True, merged_name="merged",
                tile_coords=[(0, 0, size, size), (size, 0, 2 * size, size)])

            tiles = routes._psd_preview_build(run_dir, 768, "tiles")
            check(tiles["tiles_count"] == 2,
                  "two tiles are listed (%d)" % tiles["tiles_count"])
            check([t["index"] for t in tiles["tiles"]] == [0, 1],
                  "tiles keep their grid order")
            check(all(t.get("image", {}).get("filename") for t in tiles["tiles"]),
                  "every tile has a thumbnail")
            check(all("块" in t["label"] for t in tiles["tiles"]),
                  "tile labels use the block wording")
            check(all(t.get("split") for t in tiles["tiles"]),
                  "every tile lists its own parts")
            check([p["name"] for p in tiles["tiles"][0]["split"]] == ["face", "hairf"],
                  "tile parts keep their names (%s)"
                  % [p["name"] for p in tiles["tiles"][0]["split"]])
            check(bool(tiles.get("main")),
                  "the whole merged canvas has a preview")

            parts = routes._psd_preview_build(run_dir, 768, "parts")
            names = [p["name"] for p in parts["parts"]]
            check(sorted(names) == ["face", "hair"],
                  "copies are grouped by body part (%s)" % names)
            check(all(p["count"] == 2 for p in parts["parts"]),
                  "each part counts both tiles' copies")
            check(all(os.path.isfile(os.path.join(temp_dir, p["image"]["filename"]))
                      for p in parts["parts"]),
                  "part thumbnails exist on disk")
            check(all(len(p.get("split") or []) == 2 for p in parts["parts"]),
                  "every part lists the copy from each tile")
            check(sorted({s["tile"] for p in parts["parts"] for s in p["split"]})
                  == [0, 1],
                  "split copies record which tile they came from")
            check(all(os.path.isfile(os.path.join(temp_dir, s["image"]["filename"]))
                      for p in parts["parts"] for s in p["split"]),
                  "split thumbnails exist on disk")
            check(bool(parts.get("main")),
                  "the part view still carries the merged preview")

            psd_path = os.path.join(run_dir, "merged.psd")
            by_file = routes._psd_preview_build(psd_path, 768, "tiles")
            check(by_file["tiles_count"] == 2,
                  "pointing at the PSD file also finds the run's tiles")
        finally:
            routes._scope_root = original_root
            folder_paths.get_temp_directory = original_temp

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
