"""CPU-only checks for the generation record and the structural reset rule."""

from __future__ import annotations

import os
import sys
import tempfile

import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import nodes as stlu  # noqa: E402
import nodes_shared  # noqa: E402
import tile_psd  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def _manifest(mode="uniform", size=(128, 128), tiles=(64, 64),
              overlap=0, offset=(0, 0), coords=((0, 0, 64, 64),)):
    return {
        "mode": mode,
        "source": list(size),
        "source_fingerprint": "abc",
        "tile_size": list(tiles),
        "overlap": overlap,
        "offset": list(offset),
        "tiles": [{"index": i, "tile": list(box)} for i, box in enumerate(coords)],
    }


def _doc(mode="uniform", size=(128, 128), tiles=(64, 64),
         overlap=0, offset=(0, 0), coords=((0, 0, 64, 64),)):
    return {
        "mode": mode,
        "source": {"image": "a.png", "width": size[0], "height": size[1],
                   "fingerprint": "abc"},
        "tile_size": list(tiles),
        "overlap": overlap,
        "offset": list(offset),
        "coords": [list(box) for box in coords],
    }


def main():
    print("[structural mismatch is the only automatic reset]")
    check(tile_psd.cache_mismatch_reason(_manifest(), _doc()) == "",
          "same layout -> reusable")
    check("分区方式" in tile_psd.cache_mismatch_reason(
        _manifest(mode="subject_strips"), _doc()), "another mode -> reset")
    check("原图尺寸" in tile_psd.cache_mismatch_reason(
        _manifest(size=(64, 64)), _doc(size=(128, 128))),
        "another source size -> reset")
    check("原图" in tile_psd.cache_mismatch_reason(
        dict(_manifest(), source_fingerprint="old"), _doc()),
        "another source image -> reset")
    check("分块尺寸" in tile_psd.cache_mismatch_reason(
        _manifest(tiles=(32, 32)), _doc()), "another tile size -> reset")
    check("重叠" in tile_psd.cache_mismatch_reason(
        _manifest(overlap=16), _doc()), "another overlap -> reset")
    check("偏移" in tile_psd.cache_mismatch_reason(
        _manifest(offset=(8, 0)), _doc()), "another offset -> reset")
    check("位置" in tile_psd.cache_mismatch_reason(
        _manifest(coords=((0, 0, 64, 64),)), _doc(coords=((0, 32, 64, 64),))),
        "moved box -> reset")
    # A folder whose manifest only knows some tiles (an interrupted run) keeps
    # the tiles it does know about; only the shared indices are compared.
    check(tile_psd.cache_mismatch_reason(
        _manifest(coords=((0, 0, 64, 64),)),
        _doc(coords=((0, 0, 64, 64), (0, 64, 64, 64)))) == "",
        "a partial manifest still joins the run")
    check(tile_psd.cache_mismatch_reason({}, _doc()) == "",
          "no manifest -> nothing to compare")
    check(tile_psd.cache_mismatch_reason(_manifest(), {}) == "",
          "no plan -> nothing to compare")

    with tempfile.TemporaryDirectory() as tmp:
        print("[PipelineInput writes algo_version]")
        original = nodes_shared._output_root
        nodes_shared._output_root = lambda: tmp
        try:
            out = stlu.STLU_PipelineInput().execute(
                image=torch.zeros(1, 128, 128, 3),
                tile_width=64, tile_height=64, overlap=0, select="all",
                resolution=512, num_inference_steps=1, depth_resolution=-1,
                random_seed=7, out_dir="SeeThrough/VersionTest")
        finally:
            nodes_shared._output_root = original
        doc = tile_psd.load_grid_ref(out["result"][2])
        generation = doc.get("generation") or {}
        check(generation.get("algo_version") == tile_psd.ALGO_VERSION,
              "grid_doc carries algo_version=%s" % generation.get("algo_version"))
        check("generation_id" not in doc, "grid_doc carries no fingerprint")

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
