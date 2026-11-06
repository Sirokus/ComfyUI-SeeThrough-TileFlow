"""CPU-only checks for the cross-tile seam ownership (no GPU, no server).

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_tile_seam.py
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
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import psd_reader  # noqa: E402
import psd_writer  # noqa: E402
import tile_psd  # noqa: E402

_failures = []
CANVAS_W, CANVAS_H = 1751, 1024
TILE0 = [0, 0, 1024, 1024]
TILE1 = [727, 0, 1751, 1024]
SEAM = (TILE0[2] + TILE1[0]) / 2.0  # 875.5


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def fill(name, size, left, top, color):
    pixels = np.zeros((size[1], size[0], 4), np.uint8)
    pixels[..., :3] = color
    pixels[..., 3] = 255
    return {"name": name, "pixels": pixels, "left": int(left), "top": int(top)}


def build_run(tmp):
    """Two tiles that both draw a face across the overlap, differently.

    Tile 0 sees the left part of the face and invents the rest (red); tile 1
    sees the whole face (blue).  This is the shape that used to average into
    a wide ghost band.
    """
    run = os.path.join(tmp, "wave")
    os.makedirs(run, exist_ok=True)
    tile0_layers = [
        fill("face", (324, 600), 700, 200, (255, 0, 0)),   # ends at x = 1024
        fill("hair", (200, 100), 600, 600, (0, 255, 0)),
        # Only tile 0 draws this: the re-composed crop invented it at its own
        # border.  It must be cut at the seam like everything else.
        fill("junk", (324, 40), 700, 100, (10, 10, 10)),
    ]
    tile1_layers = [
        fill("face", (356, 600), 0, 200, (0, 0, 255)),     # source x 727..1083
        fill("hair", (200, 100), 0, 600, (0, 255, 0)),
    ]
    psd_writer.write_psd(os.path.join(run, "tile_0.psd"), tile0_layers,
                         TILE0[2] - TILE0[0], TILE0[3] - TILE0[1], "rle")
    psd_writer.write_psd(os.path.join(run, "tile_1.psd"), tile1_layers,
                         TILE1[2] - TILE1[0], TILE1[3] - TILE1[1], "rle")
    manifest = {
        "source": [CANVAS_W, CANVAS_H],
        "overlap": 297,
        "tiles": [
            {"index": 0, "file": "tile_0.psd", "tile": TILE0,
             "layers": [{"name": "face", "depth_median": 0.5},
                        {"name": "hair", "depth_median": 0.9}]},
            {"index": 1, "file": "tile_1.psd", "tile": TILE1,
             "layers": [{"name": "face", "depth_median": 0.5},
                        {"name": "hair", "depth_median": 0.9}]},
        ],
    }
    with open(os.path.join(run, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return run, manifest


def layer_of(doc, name):
    return next(layer for layer in doc["layers"] if layer["name"] == name)


def sample(layer, x, y):
    px = layer["pixels"]
    lx, ly = int(x) - int(layer["left"]), int(y) - int(layer["top"])
    if lx < 0 or ly < 0 or lx >= px.shape[1] or ly >= px.shape[0]:
        return (0, 0, 0, 0)
    return tuple(int(v) for v in px[ly, lx])


def main():
    tmp = tempfile.mkdtemp(prefix="stlu_seam_")
    run, manifest = build_run(tmp)

    print("[seam: one tile owns each pixel]")
    info = tile_psd.merge_tiles(
        run, manifest, out_name="flat", group_layers=False,
        include_depth=False, include_original=False, seam_feather=32,
    )
    merged = psd_reader.read_psd(os.path.join(run, info["file"]))
    face = layer_of(merged, "face")
    left = sample(face, 800, 400)
    right = sample(face, 950, 400)
    check(left[0] > 240 and left[2] < 15,
          "left of the seam keeps tile 0 (%s)" % (left,))
    check(right[2] > 240 and right[0] < 15,
          "right of the seam takes tile 1 (%s)" % (right,))
    # Purple pixels only exist inside the ramp; anything wider would mean the
    # two renderings are averaged again across the whole overlap.
    purple = []
    for x in range(600, 1100):
        r, g, b, a = sample(face, x, 400)
        if a > 128 and r > 60 and b > 60:
            purple.append(x)
    width = (max(purple) - min(purple) + 1) if purple else 0
    centre = (max(purple) + min(purple)) / 2.0 if purple else 0.0
    check(width <= 36 and abs(centre - SEAM) <= 4,
          "ghost band stays inside the 32 px handover (width %d, centre %.1f)"
          % (width, centre))
    check(sample(face, 860, 400)[2] < 20 and
          sample(face, 890, 400)[2] > 60,
          "the handover is centred on the seam")

    print("[seam: a part only one tile draws is still cut at the seam]")
    hair = layer_of(merged, "hair")
    check(sample(hair, 750, 650)[1] > 200,
          "the owning tile keeps the part (%s)" % (sample(hair, 750, 650),))
    check(sample(hair, 850, 650)[3] == 0,
          "the non-owning tile's copy is dropped (%s)" % (sample(hair, 850, 650),))
    junk = layer_of(merged, "junk")
    check(sample(junk, 750, 120)[3] > 200,
          "invented content stays on its own side (%s)" % (sample(junk, 750, 120),))
    check(sample(junk, 950, 120)[3] == 0,
          "invented content is removed past the seam (%s)" % (sample(junk, 950, 120),))
    near = sample(junk, 882, 120)[3]
    check(near < 100,
          "the flat fade past the seam is at most half opacity (%d)" % near)

    print("[seam: group mode keeps both copies whole]")
    info = tile_psd.merge_tiles(
        run, manifest, out_name="grouped", group_layers=True,
        include_depth=False, include_original=False, seam_feather=32,
    )
    grouped = psd_reader.read_psd(os.path.join(run, info["file"]))
    first = next(layer for layer in grouped["layers"] if layer["name"] == "face_0")
    second = next(layer for layer in grouped["layers"] if layer["name"] == "face_1")
    check(sample(first, 800, 400)[0] > 200,
          "tile 0 copy keeps its red left side (%s)" % (sample(first, 800, 400),))
    check(sample(first, 950, 400)[0] > 200 and sample(first, 950, 400)[3] == 255,
          "tile 0 copy is NOT cut past the seam (%s)" % (sample(first, 950, 400),))
    check(sample(second, 950, 400)[2] > 200,
          "tile 1 copy keeps its blue right side (%s)" % (sample(second, 950, 400),))
    check(sample(second, 800, 400)[2] > 200 and sample(second, 800, 400)[3] == 255,
          "tile 1 copy is NOT cut before the seam (%s)" % (sample(second, 800, 400),))
    check(int(first["right"]) - int(first["left"]) == 324
          and int(second["right"]) - int(second["left"]) == 356,
          "both copies keep their full width")
    # A part only one tile drew stays a plain, uncut layer.
    junk = layer_of(grouped, "junk")
    check(sample(junk, 750, 120)[3] == 255 and sample(junk, 880, 120)[3] == 255,
          "a one-tile part is not cut at the seam (%s / %s)"
          % (sample(junk, 750, 120), sample(junk, 880, 120)))

    print("[seam: zero feather is a hard cut]")
    info = tile_psd.merge_tiles(
        run, manifest, out_name="hard", group_layers=False,
        include_depth=False, include_original=False, seam_feather=0,
    )
    hard = psd_reader.read_psd(os.path.join(run, info["file"]))
    hard_face = layer_of(hard, "face")
    check(sample(hard_face, 874, 400)[0] > 240 and
          sample(hard_face, 877, 400)[2] > 240,
          "the switch happens on the bisector")

    print("[seam: a single tile is unchanged]")
    info = tile_psd.merge_tiles(
        run, manifest, indices=[0], out_name="single", group_layers=False,
        include_depth=False, include_original=False)
    single = psd_reader.read_psd(os.path.join(run, info["file"]))
    solo = layer_of(single, "face")
    check(sample(solo, 900, 400)[3] == 255 and sample(solo, 700, 400)[3] == 255,
          "one tile merges with full alpha everywhere")

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
