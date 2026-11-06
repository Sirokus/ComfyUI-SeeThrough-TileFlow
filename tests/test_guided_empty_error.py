"""CPU-only checks separating an empty tile from a compositing error."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest import mock

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
import tile_psd  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def solid_part(size=512):
    pixels = np.zeros((size, size, 4), np.uint8)
    pixels[..., :3] = (200, 40, 40)
    pixels[..., 3] = 255
    return {
        "img": pixels,
        "depth": np.full((size, size), 128, np.uint8),
        "xyxy": [0, 0, size, size],
        "depth_median": 0.5,
        "tag": "face",
    }


def grid_json(tmp):
    doc = tile_psd.grid_document((512, 512), (512, 512), 0, created="test")
    doc["guide"] = {"enabled": False, "feather": 0, "strength": 0, "tolerance": 0}
    doc["out"] = {"dir": tmp, "alpha_threshold": 8}
    generation = {"version": 1, "algo_version": 1, "random_seed": 42}
    doc["generation"] = generation
    doc["generation_id"] = tile_psd.generation_id(generation)
    return json.dumps(doc, ensure_ascii=False)


def main():
    print("[composite result status]")
    parts = {"tag2pinfo": {"face": solid_part()}, "frame_size": (512, 512)}
    ok = stlu._guide_composite_result(parts)
    check(ok["status"] == stlu.GUIDE_OK and ok["rgba"] is not None,
          "a normal tile is ok")
    empty = stlu._guide_composite_result(
        {"tag2pinfo": {}, "frame_size": (512, 512)})
    check(empty["status"] == stlu.GUIDE_EMPTY, "no parts is empty, not an error")
    with mock.patch("tile_psd.build_tile_layers", side_effect=ValueError("bad parts")):
        broken = stlu._guide_composite_result(parts)
    check(broken["status"] == stlu.GUIDE_ERROR, "an exception is an error")
    check("bad parts" in broken["error"] and broken["traceback"],
          "the error keeps the message and traceback")

    with tempfile.TemporaryDirectory() as tmp:
        raw_grid = grid_json(tmp)
        tile = torch.zeros(1, 512, 512, 3)

        print("[an error stops immediately]")
        calls = {"n": 0}

        class ErrorLayer:
            def generate(self, image, model, seed, resolution, steps):
                calls["n"] += 1
                return ({"image": image}, None)

        class ErrorDepth:
            def generate(self, layers, depth_model, seed, depth_resolution):
                return ({"image": layers["image"]}, None)

        class ErrorPost:
            def process(self, depth, tblr_split, use_lama):
                raise RuntimeError("post failed")

        classes = {
            "SeeThrough_GenerateLayers": ErrorLayer,
            "SeeThrough_GenerateDepth": ErrorDepth,
            "SeeThrough_PostProcess": ErrorPost,
        }
        original = nodes_shared._seethrough_node_class
        nodes_shared._seethrough_node_class = lambda name: classes[name]
        try:
            try:
                stlu.STLU_GuidedLayers().execute(
                    tiles=[tile], indices=["0"],
                    layerdiff_model=[object()], depth_model=[object()],
                    grid_doc=[raw_grid], random_seed=42)
                check(False, "a post-process error should raise")
            except RuntimeError as exc:
                check("第 0 块" in str(exc), "the error names the tile")
        finally:
            nodes_shared._seethrough_node_class = original
        check(calls["n"] == 1, "an error is not retried (%d call)" % calls["n"])

        print("[an empty tile is retried twice]")
        calls["n"] = 0

        class EmptyPost:
            def process(self, depth, tblr_split, use_lama):
                return ({"tag2pinfo": {}, "frame_size": (512, 512)}, None)

        classes["SeeThrough_PostProcess"] = EmptyPost
        nodes_shared._seethrough_node_class = lambda name: classes[name]
        try:
            parts_out, indices_out = stlu.STLU_GuidedLayers().execute(
                tiles=[tile], indices=["0"],
                layerdiff_model=[object()], depth_model=[object()],
                grid_doc=[raw_grid], random_seed=42)
        finally:
            nodes_shared._seethrough_node_class = original
        check(calls["n"] == 3, "one try plus two empty retries (%d calls)" % calls["n"])
        check(len(parts_out) == 1 and indices_out == ["0"],
              "the empty tile still reaches stage 3 as an empty record")

        print("[stage 3 rejects a missing parts document]")
        try:
            stlu.STLU_TilePSD().execute(parts=None, index="0", grid_doc=raw_grid)
            check(False, "parts=None should raise")
        except ValueError as exc:
            check("没有收到部件" in str(exc), "parts=None raises a clear error")

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
