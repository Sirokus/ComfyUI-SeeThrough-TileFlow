"""CPU-only check that the PSD preview changes when the PSD changes.

The ⑤ node shows the merge through its own panel now: the panel asks
``/stlu/psd_preview`` for white-backed thumbnails.  Their filenames embed the
source PSD's mtime/size, so a re-merge must produce new URLs instead of the
browser serving the previous picture from cache.  The node itself returns no
``images`` (that drew a second, very tall copy inside the node).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import folder_paths  # noqa: E402
import nodes as stlu  # noqa: E402
import routes  # noqa: E402
import tile_psd  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def make_parts(size=128):
    pixels = np.zeros((size, size, 4), np.uint8)
    pixels[..., :3] = (30, 120, 220)
    pixels[..., 3] = 255
    return {
        "tag2pinfo": {"face": {
            "img": pixels,
            "depth": np.full((size, size), 128, np.uint8),
            "xyxy": [0, 0, size, size],
            "depth_median": 0.5,
            "tag": "face",
        }},
        "frame_size": (size, size),
    }


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out_root = os.path.join(tmp, "output")
        temp_dir = os.path.join(tmp, "temp")
        os.makedirs(out_root, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        run_dir = os.path.join(out_root, "SeeThrough", "PreviewCache")
        original_root = routes._scope_root
        original_temp = folder_paths.get_temp_directory
        routes._scope_root = lambda scope: os.path.realpath(out_root)
        folder_paths.get_temp_directory = lambda: temp_dir
        try:
            size = 128
            tile_psd.write_tiles(
                [make_parts(size)], out_root, filename_prefix="tile",
                source_size=(size, size), tile_size=(size, size), overlap=0,
                tile_index_map=[0], write_depth=False, merge_group_layers=True,
                include_original=False, run_dir_abs=run_dir,
                write_manifest=True, merge_after=True, merged_name="merged",
                tile_coords=[(0, 0, size, size)])
            psd_path = os.path.join(run_dir, "merged.psd")
            check(os.path.isfile(psd_path), "the test PSD was written")

            first = routes._psd_preview_build(run_dir, 256, "parts")
            first_part = first["parts"][0]
            first_name = first_part["image"]["filename"]
            first_v = first_part["image"].get("v")
            first_split = first_part["split"][0]["image"]["filename"]

            # Bump the mtime exactly as a re-merge would.
            stamp = time.time_ns() + 2_000_000_000
            os.utime(psd_path, ns=(stamp, stamp))
            second = routes._psd_preview_build(run_dir, 256, "parts")
            second_part = second["parts"][0]
            second_name = second_part["image"]["filename"]
            second_v = second_part["image"].get("v")
            second_split = second_part["split"][0]["image"]["filename"]

            check(first_name != second_name,
                  "the part thumbnail name changes with the PSD (%s -> %s)"
                  % (first_name, second_name))
            check(first_v is not None and second_v is not None
                  and first_v != second_v,
                  "the browser version changes too (%s -> %s)" % (first_v, second_v))
            check(first_split != second_split,
                  "split thumbnails change with the PSD too")
            check(os.path.isfile(os.path.join(temp_dir, second_name)),
                  "the new thumbnail exists on disk")

            preview = stlu.STLU_PSDPreview().execute(
                psd_path=psd_path, max_side=256, show_parts=False)
            check("images" not in (preview.get("ui") or {}),
                  "the node leaves image drawing to its own panel")
            panel = (preview.get("ui") or {}).get("stlu_parts") or [{}]
            check((panel[0].get("path") or "").endswith("merged.psd"),
                  "the panel still receives the merged PSD path")
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
