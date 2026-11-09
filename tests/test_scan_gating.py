"""CPU-only checks that every finished tile in a folder is reused."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest import mock

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import tile_psd  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def fake_psd(width=512, height=512):
    return {
        "width": width,
        "height": height,
        "layers": [{
            "kind": "layer", "name": "face", "hidden": False,
            "left": 0, "top": 0, "right": width, "bottom": height,
        }],
    }


def main():
    with tempfile.TemporaryDirectory() as tmp:
        generation = {"version": 1, "algo_version": 1, "random_seed": 1}
        grid = tile_psd.grid_document((1024, 1024), (512, 512), 64)
        grid["source"]["fingerprint"] = "fp"
        grid["generation"] = generation

        manifest = {
            "source": [1024, 1024],
            "tile_size": [512, 512],
            "overlap": 64,
            "offset": [0, 0],
            "source_fingerprint": "fp",
            "generation": generation,
            "tiles": [{"index": 0, "file": "tile_00.psd",
                       "preview": "", "depth": ""}],
        }
        for name in ("tile_00.psd", "tile_05.psd"):
            with open(os.path.join(tmp, name), "wb") as handle:
                handle.write(b"psd")
        manifest_path = os.path.join(tmp, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        tile_psd.write_generation(tmp, generation)

        print("[the folder is the cache]")
        with mock.patch("tile_psd.psd_reader.read_psd", side_effect=lambda _p: fake_psd()):
            entries = tile_psd.cached_tile_entries(tmp, grid)
        indices = [int(e["index"]) for e in entries]
        check(indices == [0, 5],
              "an unlisted leftover PSD is adopted too (%s)" % indices)

        print("[a different generation is still reused]")
        tile_psd.write_generation(
            tmp, {"version": 1, "algo_version": 2, "random_seed": 1})
        with mock.patch("tile_psd.psd_reader.read_psd", side_effect=lambda _p: fake_psd()):
            entries = tile_psd.cached_tile_entries(tmp, grid)
        indices = [int(e["index"]) for e in entries]
        check(indices == [0, 5],
              "settings never invalidate finished tiles (%s)" % indices)

        print("[skip wins over the folder]")
        with mock.patch("tile_psd.psd_reader.read_psd", side_effect=lambda _p: fake_psd()):
            entries = tile_psd.cached_tile_entries(tmp, grid, skip=(5,))
        indices = [int(e["index"]) for e in entries]
        check(indices == [0], "the ticked tile is regenerated (%s)" % indices)

        print("[scan with an explicit manifest gate]")
        with mock.patch("tile_psd.psd_reader.read_psd", side_effect=lambda _p: fake_psd()):
            gated = tile_psd.scan_tile_entries(
                tmp, grid, manifest, allow_unlisted=False)
            open_gate = tile_psd.scan_tile_entries(
                tmp, grid, {}, allow_unlisted=True)
        check([int(e["index"]) for e in gated] == [0],
              "allow_unlisted=False keeps only manifest tiles")
        check([int(e["index"]) for e in open_gate] == [0, 5],
              "allow_unlisted=True rebuilds unlisted tiles")

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
