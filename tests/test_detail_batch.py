"""Dry run of STLU_DetailBatch: reference rebuild with no GPU and no server."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for p in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import nodes as stlu  # noqa: E402
import nodes_shared  # noqa: E402

CANVAS = 100
BOX = [10, 10, 30, 20]
RED = (200, 40, 40)

_failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _failures.append(msg)


def main():
    tmp = tempfile.mkdtemp(prefix="stlu_detail_")
    src_dir = os.path.join(tmp, "src")
    os.makedirs(src_dir)
    part = np.zeros((10, 20, 4), np.uint8)
    part[..., :3] = RED
    part[..., 3] = 255
    part[0, :, 3] = 5                     # dust that must be cleaned
    Image.fromarray(part, "RGBA").save(os.path.join(src_dir, "part.png"))

    ref = np.zeros((400, 400, 4), np.uint8)
    ref[40:80, 40:120, :3] = RED
    ref[40:80, 40:120, 3] = 255
    ref_tensor = torch.from_numpy(ref.astype(np.float32) / 255.0)[None]

    plan = {
        "version": 1,
        "source": {"width": CANVAS, "height": CANVAS, "prefix": "t"},
        "out_dir": "UpScale",
        "canvas_size": 200,
        "scale": 2.0,
        "layers": [
            {"name": "part", "z": 0, "selected": True, "empty": False,
             "file": os.path.join(src_dir, "part.png"), "box": BOX,
             "display_box": [20, 20, 60, 40], "target": {"w": 40, "h": 20}},
            {"name": "blank", "z": 1, "selected": True, "empty": True,
             "file": os.path.join(src_dir, "part.png"),
             "box": [0, 0, CANVAS, CANVAS], "display_box": [0, 0, 200, 200],
             "target": {"w": 200, "h": 200}},
        ],
    }

    original_root = nodes_shared._output_root
    nodes_shared._output_root = lambda: tmp
    try:
        saved, report = stlu.STLU_DetailBatch().execute(
            json.dumps(plan), ref_tensor, color_tolerance=26.0, alpha_floor=12)
    finally:
        nodes_shared._output_root = original_root

    out_path = os.path.join(tmp, "UpScale", "part.png")
    check(os.path.isfile(out_path), "detail layer written")
    check(saved == out_path, "last_saved points at the last layer")
    check(not os.path.exists(os.path.join(tmp, "UpScale", "blank.png")),
          "empty layers are skipped")

    if os.path.isfile(out_path):
        arr = np.array(Image.open(out_path).convert("RGBA"))
        check(arr.shape == (20, 40, 4), "saved at display size %s" % (arr.shape,))
        check(int(arr[0, :, 3].max()) == 0, "alpha dust cleaned")
        check(tuple(arr[10, 20]) == (200, 40, 40, 255), "colour taken from reference")

    data = json.loads(report)
    check(data["total"] == 1, "only non-empty layers reported")
    check(data["layers"][0]["detail"]["trusted"] > 0.9, "reference trusted for visible pixels")

    print()
    if _failures:
        print("FAILED: %d" % len(_failures))
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
