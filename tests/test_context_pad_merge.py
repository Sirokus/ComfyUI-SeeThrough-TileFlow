"""End-to-end (CPU) check: a context-padded tile lands at the right place.

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_context_pad_merge.py

The model answers in its own pixel space, not in the padded window's pixels.
Cutting the strip with source pixels and then declaring the frame to be the
source size put a 512 px result into a 1694 px PSD: the whole figure ended up
in the top-left corner, and the merge's phantom trim removed every layer
because all of them sat on the source's flat background.  This runs the real
cut -> write_tiles -> merge_tiles chain on a tiny synthetic image so that
mismatch cannot come back silently.
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
for path in (PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import nodes as stlu  # noqa: E402
import psd_reader  # noqa: E402
import tile_psd  # noqa: E402
from PIL import Image  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def main():
    width, height = 200, 300
    content_box = (0, 100, 200, 300)   # the strip
    input_box = (0, 50, 200, 350)      # strip + 50 px context on top/bottom
    # The model renders the 200x300 window at half size.
    frame_w, frame_h = 100, 150

    rgb = np.full((height, width, 3), 255, np.uint8)
    rgb[150:250, 50:150] = (200, 20, 40)   # subject: a red block

    with tempfile.TemporaryDirectory() as tmp:
        Image.fromarray(rgb).save(os.path.join(tmp, "original.png"))

        rgba = np.zeros((frame_h, frame_w, 4), np.uint8)
        rgba[..., 3] = 255
        rgba[..., :3] = (200, 20, 40)
        # Same block in model space: x 50..150 -> 25..75, y 100..200 -> 50..100
        parts = {
            "tag2pinfo": {
                "face": {
                    "img": rgba,
                    "depth": np.full((frame_h, frame_w), 128, np.uint8),
                    "xyxy": [25, 50, 75, 100],
                    "depth_median": 0.5,
                    "tag": "face",
                },
            },
            "frame_size": (frame_h, frame_w),
        }
        composite = {"status": "ok", "rgba": rgba.copy()}

        print("[context: the strip is cut in model space]")
        cut, cut_comp = stlu._crop_tile_parts(
            parts, composite, content_box, input_box)
        check(cut["frame_size"] == (100, 100),
              "100 px of model space is the strip (%s)" % (cut["frame_size"],))
        check(np.asarray(cut["tag2pinfo"]["face"]["img"]).shape == (100, 100, 4),
              "part image follows that space")
        check(np.asarray(cut_comp["rgba"]).shape == (100, 100, 4),
              "composite follows that space")

        print("[context: the PSD keeps the model-space canvas]")
        manifest = tile_psd.write_tiles(
            [cut], tmp, filename_prefix="tile",
            source_size=(width, height), tile_size=(200, 200),
            tile_index_map=[0], write_depth=False, include_original=False,
            run_dir_abs=tmp, write_manifest=True,
            tile_coords=[content_box])
        entry = manifest["tiles"][0]
        check(entry["canvas"] == [100, 100],
              "tile canvas is the model space, not the strip size (%s)"
              % (entry["canvas"],))
        psd = psd_reader.read_psd(os.path.join(tmp, entry["file"]))
        check((psd["width"], psd["height"]) == (100, 100),
              "PSD canvas matches it (%dx%d)" % (psd["width"], psd["height"]))

        print("[context: the merge puts the strip back where it was cut]")
        info = tile_psd.merge_tiles(
            tmp, manifest, out_name="merged", group_layers=False,
            include_depth=False, include_original=False, alpha_threshold=8)
        check(info["canvas"] == [width, height],
              "merged canvas is the source size (%s)" % (info["canvas"],))
        merged = psd_reader.read_psd(os.path.join(tmp, "merged.psd"))
        layers = [l for l in merged["layers"]
                  if str(l.get("kind") or "layer") == "layer"]
        check(len(layers) == 1, "the part survives the merge (%d layers)" % len(layers))
        if layers:
            layer = layers[0]
            px = np.asarray(layer["pixels"])
            check(np.count_nonzero(px[..., 3] > 8) > 8000,
                  "the merged layer is not empty (%d px)"
                  % np.count_nonzero(px[..., 3] > 8))
            left, top = int(layer["left"]), int(layer["top"])
            cx = left + px.shape[1] // 2
            cy = top + px.shape[0] // 2
            check(abs(cx - 100) <= 2 and abs(cy - 200) <= 2,
                  "the block sits at the source position (centre %d,%d)"
                  % (cx, cy))
        with open(os.path.join(tmp, "manifest.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        check(saved.get("source") == [width, height],
              "manifest source size survives (%s)" % (saved.get("source"),))

    print()
    if _failures:
        print("FAILED:", len(_failures))
        for item in _failures:
            print("  -", item)
        return 1
    print("ALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
