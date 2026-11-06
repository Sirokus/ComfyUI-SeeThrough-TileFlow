"""CPU-only checks that locked parts survive a reset inside the real nodes.

The preview panel's lock button writes a snapshot into ``<run>/_locks/``.
This test drives that store through ``STLU_TilePSD`` and the orphan-tile
rebuild in ``STLU_WaveIndex`` with no GPU, no server and no SeeThrough:

* a lock set before the folder is cleared still replaces the freshly
  generated layer of that same tile;
* a tile the new run did not select is rebuilt from its locked layers when
  the run folder was wiped, so the locked picture is not lost.
"""

from __future__ import annotations

import json
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

import nodes_pipeline as pipeline  # noqa: E402
import nodes_shared  # noqa: E402
import psd_reader  # noqa: E402
import tile_psd  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def solid(width, height, value):
    arr = np.zeros((height, width, 4), np.uint8)
    arr[..., :3] = value
    arr[..., 3] = 255
    return arr


def layer_named(path, name):
    data = psd_reader.read_psd(path)
    for layer in data.get("layers") or []:
        if str(layer.get("name") or "") == name:
            return layer
    return None


def main():
    with tempfile.TemporaryDirectory() as tmp:
        original_root = nodes_shared._output_root
        nodes_shared._output_root = lambda: tmp
        try:
            doc = tile_psd.grid_document((128, 128), (64, 64), 0, name="grid")
            doc["source"]["fingerprint"] = "test"
            run_dir = os.path.join(tmp, "SeeThrough", "Snapshot")
            os.makedirs(run_dir, exist_ok=True)
            doc["out"] = {
                "dir": run_dir, "rel": "SeeThrough/Snapshot", "wave": "snapshot",
                "prefix": "tile", "fingerprint": "test",
                "alpha_threshold": 8, "merge_name": "merged",
                "group_layers": True, "include_depth": False,
                "include_original": False, "seam_feather": 32,
            }
            coords = doc["coords"]
            check(len(coords) >= 3, "the test grid has enough tiles (%d)" % len(coords))

            # Locks written by the ⑤ panel, then the automatic reset.
            tile_psd.store_locked_layer(
                run_dir, 1, "face", solid(9, 7, (200, 10, 10)), 5, 6,
                tile_box=coords[1], canvas=[64, 64], source=[128, 128],
                depth=np.full((7, 9), 90, np.uint8), depth_median=0.30)
            tile_psd.store_locked_layer(
                run_dir, 2, "hair", solid(11, 8, (10, 10, 200)), 4, 9,
                tile_box=coords[2], canvas=[64, 64], source=[128, 128],
                depth_median=0.70)
            with open(os.path.join(run_dir, "tile_01_r0_c1.psd"), "wb") as handle:
                handle.write(b"stale")
            tile_psd.clear_run_outputs(run_dir)
            check(os.path.isfile(tile_psd.lock_index_path(run_dir)),
                  "the reset left the lock store alone")
            check(not os.path.isfile(os.path.join(run_dir, "tile_01_r0_c1.psd")),
                  "the reset removed the stale tile PSD")

            # ---- a selected tile is restored from the snapshot ------------
            fresh = {"frame_size": (64, 64), "tag2pinfo": {
                "face": {"img": solid(9, 7, (20, 220, 20)),
                         "xyxy": [0, 0, 9, 7], "depth_median": 0.5},
                "mouth": {"img": solid(6, 5, (0, 0, 220)),
                          "xyxy": [30, 30, 36, 35], "depth_median": 0.5},
            }}
            record = pipeline.STLU_TilePSD().execute(
                fresh, "1", json.dumps(doc, ensure_ascii=False))["result"][0]
            entry = record.get("entry") or {}
            tile_path = os.path.join(run_dir, str(entry.get("file") or ""))
            check(os.path.isfile(tile_path), "the re-generated tile PSD exists")
            face = layer_named(tile_path, "face") or {}
            pixels = face.get("pixels")
            check(pixels is not None
                  and int(pixels[..., 0].min()) == 200
                  and int(pixels[..., 1].max()) == 10,
                  "the locked face replaced the fresh green one")
            check(face.get("left") == 5 and face.get("top") == 6,
                  "the restored face kept its position (%s,%s)"
                  % (face.get("left"), face.get("top")))
            mouth = layer_named(tile_path, "mouth") or {}
            check((mouth.get("pixels") is not None
                   and int(mouth["pixels"][..., 2].min()) == 220),
                  "the unlocked mouth still uses the fresh generation")

            # ---- a tile the run did not select is rebuilt -----------------
            made_entries, made_images = pipeline._restore_locked_orphans(
                run_dir, doc, "tile", [entry], [], 8)
            check([int(e.get("index", -1)) for e in made_entries] == [2],
                  "only the unselected locked tile was rebuilt (%s)"
                  % [int(e.get("index", -1)) for e in made_entries])
            check(len(made_images) == len(made_entries),
                  "the rebuilt tile also got its images entry")
            orphan_path = os.path.join(
                run_dir, str((made_entries[0] if made_entries else {}).get("file") or ""))
            check(os.path.isfile(orphan_path),
                  "the orphan tile PSD was written back")
            hair = layer_named(orphan_path, "hair") or {}
            check(hair.get("pixels") is not None
                  and int(hair["pixels"][..., 2].min()) == 200,
                  "the rebuilt tile carries the locked hair pixels")
            check(not layer_named(orphan_path, "face"),
                  "the rebuilt tile only holds the layers locked for it")
        finally:
            nodes_shared._output_root = original_root

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
