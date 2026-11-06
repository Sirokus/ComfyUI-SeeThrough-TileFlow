"""CPU-only checks for the generation fingerprint's algorithm version."""

from __future__ import annotations

import json
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


def main():
    print("[algorithm version is part of the fingerprint]")
    old = {"version": 1, "random_seed": 1}
    new = {"version": 1, "algo_version": 1, "random_seed": 1}
    check(tile_psd.generation_id(old) != tile_psd.generation_id(new),
          "adding algo_version changes generation_id")
    check(not tile_psd.generation_matches(old, new),
          "an old folder without algo_version does not match")
    check(tile_psd.generation_matches(
        {"version": 1, "algo_version": 1, "random_seed": 1},
        {"version": 1, "algo_version": 1, "random_seed": 1}),
        "identical fingerprints match")

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
        check(doc.get("generation_id") == tile_psd.generation_id(generation),
              "generation_id matches the document")

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
