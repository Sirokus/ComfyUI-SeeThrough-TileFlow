"""CPU-only checks for the optional tile context padding (① ``context_pad``).

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_context_pad.py

A padded tile is cut larger than its strip, generated, then cut back to the
strip box before PSD writing and merging.  These checks lock both steps: the
crop of the parts/composite, and the overlap position the guide uses when the
tile window is larger than the strip.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import nodes as stlu  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def part(size, color, xyxy=None):
    img = np.zeros((size, size, 4), np.uint8)
    img[..., :3] = color
    img[..., 3] = 255
    return {
        "img": img,
        "depth": np.full((size, size), 128, np.uint8),
        "xyxy": list(xyxy or [0, 0, size, size]),
        "depth_median": 0.5,
        "tag": "face",
    }


def main():
    print("[context: crop a padded result back to the strip]")
    input_box = (100, 100, 300, 300)      # what the model saw (200x200)
    content_box = (120, 120, 280, 280)    # the strip (160x160)
    parts = {"tag2pinfo": {"face": part(200, (255, 0, 0), [0, 0, 200, 200])},
             "frame_size": (200, 200)}
    rgba = np.zeros((200, 200, 4), np.uint8)
    rgba[..., :3] = (0, 0, 255)
    rgba[..., 3] = 255
    composite = {"status": "ok", "rgba": rgba}

    out, comp = stlu._crop_tile_parts(parts, composite, content_box, input_box)
    check(out["frame_size"] == (160, 160),
          "frame size becomes the strip box (%s)" % (out["frame_size"],))
    face = out["tag2pinfo"]["face"]
    check(np.asarray(face["img"]).shape == (160, 160, 4),
          "part image is cut to the strip (%s)" % (np.asarray(face["img"]).shape,))
    check(np.asarray(face["depth"]).shape == (160, 160),
          "depth is cut with the image (%s)" % (np.asarray(face["depth"]).shape,))
    check(face["xyxy"] == [0, 0, 160, 160],
          "xyxy follows the strip canvas (%s)" % (face["xyxy"],))
    check(np.asarray(comp["rgba"]).shape == (160, 160, 4),
          "composite is cut too (%s)" % (np.asarray(comp["rgba"]).shape,))
    check(comp["status"] == "ok", "other composite fields survive")

    print("[context: a model-space frame is cut proportionally]")
    # The model renders a 200x200 window at 64x64.  The 160 px strip starts
    # 20 px into the window, i.e. 64 * 20/200 = 6 px, and ends at 58 px.
    # Cutting with source pixels instead would ask for a 160 px slice of a
    # 64 px array and silently keep nothing.
    parts64 = {"tag2pinfo": {"face": part(64, (255, 0, 0), [0, 0, 64, 64])},
               "frame_size": (64, 64)}
    rgba64 = np.zeros((64, 64, 4), np.uint8)
    rgba64[..., 3] = 255
    out64, comp64 = stlu._crop_tile_parts(
        parts64, {"status": "ok", "rgba": rgba64},
        (120, 120, 280, 280), (100, 100, 300, 300))
    check(out64["frame_size"] == (52, 52),
          "frame size follows the model space (%s)" % (out64["frame_size"],))
    check(np.asarray(out64["tag2pinfo"]["face"]["img"]).shape == (52, 52, 4),
          "part image is cut in model space (%s)"
          % (np.asarray(out64["tag2pinfo"]["face"]["img"]).shape,))
    check(out64["tag2pinfo"]["face"]["xyxy"] == [0, 0, 52, 52],
          "xyxy follows the model-space cut (%s)" % (out64["tag2pinfo"]["face"]["xyxy"],))
    check(np.asarray(comp64["rgba"]).shape == (52, 52, 4),
          "composite is cut in model space (%s)" % (np.asarray(comp64["rgba"]).shape,))

    print("[context: no padding leaves the parts untouched]")
    same = {"tag2pinfo": {"face": part(64, (1, 2, 3))},
            "frame_size": (64, 64)}
    kept, comp2 = stlu._crop_tile_parts(same, composite, (0, 0, 64, 64),
                                        (0, 0, 64, 64))
    check(kept is same and comp2 is composite,
          "an unpadded tile is returned as-is")

    print("[context: guide pastes the neighbour at the strip position]")
    # Tile 1 is fed a 1024x1024 window; its strip is only the top-left 512.
    # Tile 0 (content 512..1024 x 0..512) was generated and must land on the
    # window pixels 512..1024 x 0..512 -- not shifted by the padding.
    tile = torch.from_numpy(np.full((1, 1024, 1024, 3), 0.0, np.float32))
    neighbour = np.zeros((512, 512, 4), np.uint8)
    neighbour[..., :3] = (255, 0, 0)
    neighbour[..., 3] = 255
    guide = {"enabled": True, "strength": 100, "feather": 0, "tolerance": 0}
    coords = [(512, 0, 1024, 512), (0, 0, 512, 512)]
    guided = stlu._apply_tile_guide(
        tile, 1, coords, {0: neighbour}, guide, input_box=(0, 0, 1024, 1024))
    px = lambda x, y: (guided[0, y, x].numpy() * 255.0).round().astype(int).tolist()
    check(px(768, 256)[0] > 200, "the overlap receives the neighbour (%s)" % (px(768, 256),))
    check(px(256, 256)[0] < 40, "outside the overlap stays the source (%s)" % (px(256, 256),))

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
