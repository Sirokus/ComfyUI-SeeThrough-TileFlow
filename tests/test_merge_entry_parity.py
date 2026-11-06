"""CPU-only check that the ④ node and the merge route share one code path.

Both used to assemble the merge arguments and rewrite ``manifest["merged"]``
on their own, which drifted (the route stored the layer list, the node stored
a count).  ``tile_psd.merge_run`` is now the single entry point.
"""

from __future__ import annotations

import json
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

import nodes as stlu  # noqa: E402
import nodes_shared  # noqa: E402
import psd_writer  # noqa: E402
import tile_psd  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def fill(name, size, left, top, color):
    pixels = np.zeros((size[1], size[0], 4), np.uint8)
    pixels[..., :3] = color
    pixels[..., 3] = 255
    return {"name": name, "pixels": pixels, "left": int(left), "top": int(top)}


def write_run(run):
    os.makedirs(run, exist_ok=True)
    size = 512
    layers = [fill("face", (256, 256), 128, 128, (240, 210, 190))]
    psd_writer.write_psd(os.path.join(run, "tile_00.psd"),
                         layers, size, size, "rle")
    manifest = {
        "source": [size, size],
        "tile_size": [size, size],
        "overlap": 0,
        "alpha_threshold": 8,
        "merge_parts": True,
        "group_layers": True,
        "tiles": [{
            "index": 0, "file": "tile_00.psd",
            "canvas": [size, size], "tile": [0, 0, size, size],
            "layers": [{"name": "face", "depth_median": 0.5}],
        }],
    }
    with open(os.path.join(run, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


def read_merged(run):
    with open(os.path.join(run, "manifest.json"), "r", encoding="utf-8") as handle:
        return (json.load(handle).get("merged") or {})


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "SeeThrough")
        run_node = os.path.join(root, "Node")
        write_run(run_node)
        run_route = os.path.join(root, "Route")
        shutil.copytree(run_node, run_route)

        original = nodes_shared._output_root
        nodes_shared._output_root = lambda: tmp
        try:
            out = stlu.STLU_TileMerge().execute(
                name="merged", group_layers=False, include_depth=False,
                include_original=False, run_dir="SeeThrough/Node",
                seam_feather=0)
        finally:
            nodes_shared._output_root = original
        check(bool(out["result"][0]), "the ④ node wrote a merged PSD")

        info = tile_psd.merge_run(
            run_route, name="merged", group_layers=False,
            include_depth=False, include_original=False, seam_feather=0)
        check(bool(info.get("file")), "merge_run wrote a merged PSD")

        node_doc = read_merged(run_node)
        route_doc = read_merged(run_route)
        check(node_doc == route_doc,
              "both paths record the same merged block (%r vs %r)"
              % (node_doc, route_doc))
        check(node_doc.get("layers") == 1,
              "the merged block stores a layer count, not the list")
        check(sorted(os.listdir(run_node)) == sorted(os.listdir(run_route)),
              "both folders end up with the same files")

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
