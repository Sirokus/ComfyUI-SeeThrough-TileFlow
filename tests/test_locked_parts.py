"""CPU-only checks for locked parts on a targeted tile re-run.

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_locked_parts.py

No GPU, server or network: an older tile PSD is written to a temporary folder
and the parts document is passed through ``tile_psd.replace_locked_parts``.
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

import psd_writer  # noqa: E402
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


def main():
    tmp = tempfile.mkdtemp(prefix="stlu_locks_")
    run_dir = os.path.join(tmp, "run")
    os.makedirs(run_dir)
    old_psd = os.path.join(run_dir, "tile_00_r0_c0.psd")
    psd_writer.write_psd(old_psd, [
        {"name": "face", "pixels": solid(10, 12, (200, 0, 0)), "left": 5, "top": 6},
        {"name": "topwear", "pixels": solid(8, 9, (0, 0, 200)), "left": 20, "top": 30},
    ], 64, 64, "rle")

    images_dir = os.path.join(run_dir, "images", "tile_00_r0_c0")
    os.makedirs(images_dir)
    from PIL import Image
    Image.fromarray(np.full((12, 10), 200, np.uint8), "L").save(
        os.path.join(images_dir, tile_psd._safe_stem("face") + "_depth.png"))

    found = tile_psd.find_tile_psd(run_dir, "tile", 0)
    check(found == old_psd, "find_tile_psd locates the earlier tile PSD")
    check(tile_psd.find_tile_psd(run_dir, "tile", 3) is None,
          "a tile the earlier run did not write has no PSD")

    new_part = {"img": solid(8, 8, (0, 255, 0)), "xyxy": [0, 0, 8, 8],
                "depth_median": 0.5}
    parts = {"frame_size": (64, 64), "tag2pinfo": {
        "face": dict(new_part),
        "mouth": dict(new_part),
        "nose": dict(new_part),
    }}
    kept = tile_psd.replace_locked_parts(
        parts, old_psd, ["face", "topwear", "nose"],
        {"layers": [{"name": "face", "depth_median": 0.48},
                    {"name": "topwear", "depth_median": 0.64}]},
        images_dir=images_dir)

    check(kept == ["face", "topwear", "nose"], "every lock is reported (%s)" % kept)
    face = parts["tag2pinfo"].get("face") or {}
    pixels = face.get("img")
    check(pixels is not None and int(pixels[..., 0].max()) == 200
          and int(pixels[..., 1].max()) == 0,
          "the locked part keeps the earlier pixels")
    check(face.get("xyxy") == [5, 6, 15, 18],
          "the locked part keeps its position (%s)" % face.get("xyxy"))
    check(abs(float(face.get("depth_median", 0)) - 0.48) < 1e-6,
          "the locked part keeps its depth")
    check(face.get("depth") is not None and int(face["depth"][0, 0]) == 200,
          "the locked part keeps its depth map")
    check("topwear" in parts["tag2pinfo"],
          "a locked part absent from the new run is restored from disk")
    check("nose" not in parts["tag2pinfo"],
          "a lock the earlier tile did not have stays absent")
    check("mouth" in parts["tag2pinfo"],
          "unlocked parts keep the fresh generation")

    empty = {"frame_size": (64, 64), "tag2pinfo": {"face": dict(new_part)}}
    check(tile_psd.replace_locked_parts(empty, os.path.join(tmp, "missing.psd"),
                                        ["face"]) == [],
          "a missing earlier PSD is a no-op")
    check("face" in empty["tag2pinfo"], "the no-op leaves the fresh part in place")

    # ---- lock store: survives a cleared folder ---------------------------
    locked_pixels = solid(7, 9, (12, 220, 34))
    record = tile_psd.store_locked_layer(
        run_dir, 0, "face", locked_pixels, 3, 4,
        tile_box=[0, 0, 64, 64], canvas=[64, 64], source=[256, 256],
        depth=np.full((9, 7), 77, np.uint8), depth_median=0.42)
    check(tile_psd.lock_summary(run_dir) == {"0": ["face"]},
          "the lock summary names the locked tile and part (%s)"
          % tile_psd.lock_summary(run_dir))
    check(record.get("depth") and record.get("size") == [7, 9],
          "the snapshot keeps the depth map and the layer size")

    fresh = {"frame_size": (64, 64), "tag2pinfo": {
        "face": dict(new_part), "mouth": dict(new_part)}}
    kept, loaded = tile_psd.restore_locked_parts(
        fresh, run_dir, 0, ["face"],
        coords=[[0, 0, 64, 64]], source=[256, 256])
    check(kept == ["face"] and loaded == {"face"},
          "the snapshot restores the locked part (%s / %s)" % (kept, loaded))
    restored = fresh["tag2pinfo"]["face"]["img"]
    check(int(restored[..., 1].max()) == 220 and int(restored[..., 1].min()) == 220,
          "the restored pixels come from the snapshot")
    check(fresh["tag2pinfo"]["face"].get("xyxy") == [3, 4, 10, 13],
          "the restored layer keeps its position")
    check(fresh["tag2pinfo"]["face"].get("depth") is not None
          and int(fresh["tag2pinfo"]["face"]["depth"][0, 0]) == 77,
          "the restored layer keeps its depth map")
    check("mouth" in fresh["tag2pinfo"],
          "an unlocked part still uses the fresh generation")

    # A parameter change clears the PSDs, never the lock snapshots.
    tile_psd.clear_run_outputs(run_dir)
    check(not os.path.isfile(old_psd), "the reset removed the old tile PSD")
    check(os.path.isfile(tile_psd.lock_index_path(run_dir)),
          "the lock store survives the reset")
    after = {"frame_size": (64, 64), "tag2pinfo": {
        "face": dict(new_part), "mouth": dict(new_part)}}
    kept, loaded = tile_psd.restore_locked_parts(
        after, run_dir, 0, ["face"],
        coords=[[0, 0, 64, 64]], source=[256, 256])
    check(kept == ["face"] and loaded == {"face"},
          "the lock still restores after the folder was cleared")
    check(int(after["tag2pinfo"]["face"]["img"][..., 1].max()) == 220,
          "the pixels after the clear still come from the snapshot")

    # The same tile index on a different crop must not receive the snapshot.
    mismatch = {"frame_size": (64, 64), "tag2pinfo": {"face": dict(new_part)}}
    kept, loaded = tile_psd.restore_locked_parts(
        mismatch, run_dir, 0, ["face"],
        coords=[[100, 100, 164, 164]], source=[256, 256])
    check(kept == [] and loaded == set(),
          "a changed crop layout ignores the snapshot")
    check("face" in mismatch["tag2pinfo"],
          "the ignored tile keeps its fresh part")

    # A tile the new run did not select is rebuilt from its locks.
    orphan = tile_psd.snapshot_parts(
        run_dir, 0, coords=[[0, 0, 64, 64]], source=[256, 256])
    check(orphan is not None and "face" in (orphan.get("tag2pinfo") or {}),
          "a missing tile can be rebuilt from its locked parts")
    check(orphan.get("frame_size") == (64, 64),
          "the rebuilt tile keeps its canvas size")
    check(tile_psd.snapshot_parts(
        run_dir, 7, coords=[[0, 0, 64, 64]], source=[256, 256]) is None,
        "a tile without locks is not rebuilt")

    check(tile_psd.remove_locked_layer(run_dir, 0, "face"),
          "unlocking removes the lock")
    check(tile_psd.lock_summary(run_dir) == {}, "the summary is empty after unlock")
    check(not os.path.isfile(os.path.join(tile_psd.lock_dir(run_dir), record["file"])),
          "unlocking removes the snapshot PNG")

    print()
    if _failures:
        print("FAILED %d check(s):" % len(_failures))
        for item in _failures:
            print("  - " + item)
        raise SystemExit(1)
    print("ALL-OK")


if __name__ == "__main__":
    main()
