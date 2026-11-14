"""CPU-only checks for part locks (``<run>/locks.json`` + locked-part restore)."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import pipeline_stages  # noqa: E402
import psd_reader  # noqa: E402
import psd_writer  # noqa: E402
import tile_psd  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def _solid(h, w, rgb):
    arr = np.zeros((h, w, 4), np.uint8)
    arr[..., :3] = rgb
    arr[..., 3] = 255
    return arr


def _part(rgb, h=32, w=32, x=0, y=0, depth=0.5):
    return {
        "img": _solid(h, w, rgb),
        "xyxy": [x, y, x + w, y + h],
        "depth_median": float(depth),
        "depth": np.full((h, w), int(round(depth * 255)), np.uint8),
    }


def _parts(eyes_rgb=(255, 0, 0), hair_rgb=(0, 0, 255)):
    return {
        "frame_size": (64, 64),
        "tag2pinfo": {
            "eyes": _part(eyes_rgb, x=0, y=0, depth=0.3),
            "hair": _part(hair_rgb, x=32, y=0, depth=0.7),
        },
    }


def _plan(run_dir):
    return {"doc": {
        "out": {"dir": run_dir, "prefix": "tile", "alpha_threshold": 8},
        "coords": [[0, 0, 64, 64]],
        "tile_size": [64, 64],
        "offset": [0, 0],
        "source": {"width": 64, "height": 64},
    }}


def _layer_pixel(path, name, x, y):
    """Top-left RGBA of a visible layer named ``name`` (None if absent)."""
    for layer in psd_reader.read_psd(path)["layers"]:
        if layer.get("kind") == "layer" and layer.get("name") == name \
                and not layer.get("hidden"):
            px = layer["pixels"]
            return tuple(int(v) for v in px[0, 0])
    return None


def test_store():
    print("[locks.json store]")
    tmp = tempfile.mkdtemp(prefix="stlu_locks_")
    try:
        check(tile_psd.read_locks(tmp)["tiles"] == {}, "missing file reads empty")
        result = tile_psd.update_locks(tmp, [
            {"tile": 3, "name": "eyes", "locked": True},
            {"tile": 3, "name": "Hair", "locked": True},
            {"tile": 3, "name": "EYES", "locked": True},   # same part, new casing
            {"tile": 5, "name": " ", "locked": True},      # ignored
        ])
        check(result["done"] == 3, "blank names are skipped")
        check(tile_psd.lock_summary(tmp) == {"3": ["EYES", "Hair"]},
              "case-insensitive duplicate replaces the spelling")
        check(tile_psd.tile_locks(tmp) == {3: ["EYES", "Hair"]},
              "tile_locks keys are ints")
        check(tile_psd.read_locks(tmp)["version"] == 1, "version stays 1")

        tile_psd.update_locks(tmp, [{"tile": 3, "name": "eyes", "locked": False}])
        check(tile_psd.lock_summary(tmp) == {"3": ["Hair"]},
              "unlock is case-insensitive and keeps the other names")
        tile_psd.update_locks(tmp, [{"tile": 3, "name": "HAIR", "locked": False}])
        check(tile_psd.lock_summary(tmp) == {}, "last unlock drops the tile key")

        with open(tile_psd.locks_path(tmp), "w", encoding="utf-8") as f:
            f.write("{not json")
        check(tile_psd.read_locks(tmp)["tiles"] == {}, "corrupt file reads empty")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_clear():
    print("[clear_run_outputs]")
    tmp = tempfile.mkdtemp(prefix="stlu_locks_")
    try:
        tile_psd.update_locks(tmp, [{"tile": 0, "name": "eyes", "locked": True}])
        with open(os.path.join(tmp, "tile_00_r0_c0.psd"), "wb") as f:
            f.write(b"8BPS placeholder")
        check(os.path.isfile(tile_psd.locks_path(tmp)), "locks.json exists first")
        tile_psd.clear_run_outputs(tmp)
        check(not os.path.isfile(tile_psd.locks_path(tmp)),
              "structural reset discards locks.json")
        check(not os.path.isfile(os.path.join(tmp, "tile_00_r0_c0.psd")),
              "tile PSD discarded too")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_replace_locked_parts():
    print("[replace_locked_parts]")
    tmp = tempfile.mkdtemp(prefix="stlu_locks_")
    try:
        old_psd = os.path.join(tmp, "tile_00_r0_c0.psd")
        psd_writer.write_psd(old_psd, [
            {"name": "eyes", "pixels": _solid(32, 32, (255, 0, 0)), "left": 0, "top": 0},
            {"name": "hair", "pixels": _solid(32, 32, (0, 0, 255)), "left": 32, "top": 0},
        ], 64, 64)

        parts = _parts(eyes_rgb=(0, 255, 0), hair_rgb=(255, 255, 0))
        kept = tile_psd.replace_locked_parts(parts, old_psd, ["EYES"])
        check(kept == ["eyes"], "locked layer reported as kept")
        eyes = parts["tag2pinfo"]["eyes"]
        check(tuple(int(v) for v in eyes["img"][0, 0]) == (255, 0, 0, 255),
              "locked part keeps the old pixels")
        check(list(eyes["xyxy"]) == [0, 0, 32, 32], "locked part keeps the old box")
        check(tuple(int(v) for v in parts["tag2pinfo"]["hair"]["img"][0, 0])
              == (255, 255, 0, 255), "unlocked part keeps the fresh pixels")

        # Nothing to keep -> the fresh parts survive untouched.
        fresh = _parts(eyes_rgb=(0, 255, 0))
        check(tile_psd.replace_locked_parts(fresh, old_psd, []) == [],
              "no locks means no-op")
        check(tuple(int(v) for v in fresh["tag2pinfo"]["eyes"]["img"][0, 0])
              == (0, 255, 0, 255), "no-lock parts untouched")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_tile_psd_end_to_end():
    print("[write_tile_psd end to end]")
    tmp = tempfile.mkdtemp(prefix="stlu_locks_")
    real_root = pipeline_stages._output_root
    pipeline_stages._output_root = lambda: tmp
    try:
        run_dir = os.path.join(tmp, "run")
        os.makedirs(run_dir)
        plan = _plan(run_dir)
        first = pipeline_stages.write_tile_psd(_parts(), plan, 0)
        check(first["status"] == "ok", "first write succeeds")
        psd_path = os.path.join(run_dir, first["file"])
        check(_layer_pixel(psd_path, "eyes", 0, 0) == (255, 0, 0, 255),
              "first run writes the eyes layer")

        tile_psd.update_locks(run_dir, [{"tile": 0, "name": "eyes", "locked": True}])
        second = pipeline_stages.write_tile_psd(
            _parts(eyes_rgb=(0, 255, 0), hair_rgb=(255, 255, 0)), plan, 0)
        check(second["file"] == first["file"], "same tile writes the same PSD")
        check(_layer_pixel(psd_path, "eyes", 0, 0) == (255, 0, 0, 255),
              "rerun keeps the locked eyes layer")
        check(_layer_pixel(psd_path, "hair", 0, 0) == (255, 255, 0, 255),
              "rerun replaces the unlocked hair layer")

        # The restore marker makes the second ③ node skip a second restore
        # (the old PSD is gone by then); feeding it back must be harmless.
        parts = _parts(eyes_rgb=(0, 255, 0))
        parts["_stlu_locks_tile"] = 0
        kept_again = pipeline_stages.write_tile_psd(parts, plan, 0)
        check(kept_again["status"] == "ok", "marked parts write without restoring")
        check(_layer_pixel(psd_path, "eyes", 0, 0) == (0, 255, 0, 255),
              "marked parts are not restored twice")

        # A tile without locks must not even read the PSD.
        empty_dir = os.path.join(tmp, "run_plain")
        os.makedirs(empty_dir)
        plain = pipeline_stages.write_tile_psd(_parts(), _plan(empty_dir), 0)
        check(not os.path.isfile(tile_psd.locks_path(empty_dir)),
              "plain run writes no locks.json")
        check(_layer_pixel(os.path.join(empty_dir, plain["file"]), "eyes", 0, 0)
              == (255, 0, 0, 255), "plain run writes fresh pixels")
    finally:
        pipeline_stages._output_root = real_root
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_store()
    test_clear()
    test_replace_locked_parts()
    test_write_tile_psd_end_to_end()
    if _failures:
        print("\n%d failure(s)" % len(_failures))
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
