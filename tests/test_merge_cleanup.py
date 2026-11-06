"""CPU-only checks for merge-time cleanup of tiled SeeThrough output.

Covers two failures that appear only after a grid split:

* a tile whose crop is mostly empty invents skin in the background; ④ drops
  that whole component when the untouched source says it is background;
* a tile that sees one ear keeps SeeThrough's unsuffixed ``ears`` tag, and two
  such tiles would merge both ears into one layer; ④ assigns the side from the
  face centre instead.

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_merge_cleanup.py

No GPU, server or network.
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


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def fill(name, size, left, top, color, alpha=255):
    pixels = np.zeros((size[1], size[0], 4), np.uint8)
    pixels[..., :3] = color
    pixels[..., 3] = alpha
    return {"name": name, "pixels": pixels, "left": int(left), "top": int(top)}


def write_run(tmp, name, tiles, source, overlap=0, original=None):
    """``tiles``: [(index, box, [layers], [manifest layers])]."""
    run = os.path.join(tmp, name)
    os.makedirs(run, exist_ok=True)
    entries = []
    for index, box, layers, manifest_layers in tiles:
        filename = "tile_%02d.psd" % index
        psd_writer.write_psd(os.path.join(run, filename), layers,
                             box[2] - box[0], box[3] - box[1], "rle")
        entries.append({
            "index": int(index),
            "file": filename,
            "canvas": [box[2] - box[0], box[3] - box[1]],
            "tile": [int(v) for v in box],
            "layers": manifest_layers,
        })
    if original is not None:
        from PIL import Image
        Image.fromarray(original, "RGB").save(os.path.join(run, "original.png"))
    manifest = {
        "source": [int(source[0]), int(source[1])],
        "tile_size": [entries[0]["tile"][2] - entries[0]["tile"][0],
                      entries[0]["tile"][3] - entries[0]["tile"][1]],
        "overlap": int(overlap),
        "tiles": entries,
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


def test_background_phantom(tmp):
    print("[cleanup: an invented background blob is dropped]")
    size = 1024
    source = np.full((size, size, 3), 255, np.uint8)
    source[200:800, 400:700] = (200, 40, 40)  # the real subject
    # Main face on the subject; a beige blob on plain white source.
    layers = [
        fill("face", (300, 600), 400, 200, (240, 210, 190)),
        fill("face", (70, 70), 100, 150, (228, 219, 204)),
        # White clothing over a white background: source and layer agree, so
        # this must NOT be mistaken for a phantom.
        fill("topwear", (80, 80), 40, 300, (255, 255, 255)),
    ]
    run, manifest = write_run(tmp, "phantom", [
        (0, [0, 0, size, size], layers,
         [{"name": "face", "depth_median": 0.5},
          {"name": "topwear", "depth_median": 0.9}]),
    ], (size, size), original=source)

    info = tile_psd.merge_tiles(
        run, manifest, out_name="clean", group_layers=False,
        include_depth=False, include_original=False, seam_feather=0,
    )
    merged = psd_reader.read_psd(os.path.join(run, info["file"]))
    face = layer_of(merged, "face")
    check(sample(face, 130, 180)[3] == 0,
          "the blob on white background is removed (%s)" % (sample(face, 130, 180),))
    check(sample(face, 500, 400)[3] == 255,
          "the real face keeps full alpha (%s)" % (sample(face, 500, 400),))
    clothing = layer_of(merged, "topwear")
    check(sample(clothing, 70, 330)[3] == 255,
          "white clothing matching the source is kept (%s)"
          % (sample(clothing, 70, 330),))


def test_symmetric_sides(tmp):
    print("[cleanup: one-ear tiles are named by the face centre]")
    run, manifest = write_run(tmp, "sides", [
        (0, [0, 0, 512, 512], [
            fill("nose", (16, 18), 488, 240, (220, 180, 170)),
            fill("ears", (40, 70), 300, 200, (230, 190, 170)),
        ], [{"name": "nose", "left": 488, "right": 504,
             "depth_median": 0.5},
            {"name": "ears", "left": 300, "right": 340,
             "depth_median": 0.6}]),
        (1, [512, 0, 1024, 512], [
            fill("ears", (40, 70), 188, 200, (230, 190, 170)),
        ], [{"name": "ears", "left": 188, "right": 228,
             "depth_median": 0.6}]),
    ], (1024, 512), overlap=0)

    info = tile_psd.merge_tiles(
        run, manifest, out_name="sides", group_layers=False,
        include_depth=False, include_original=False, seam_feather=0,
    )
    merged = psd_reader.read_psd(os.path.join(run, info["file"]))
    names = {layer["name"] for layer in merged["layers"]}
    check("ears" not in names, "no unsuffixed ears layer survives (%s)" % sorted(names))
    check("ears-r" in names and "ears-l" in names,
          "the two single-ear copies become ears-r and ears-l")
    check(sample(layer_of(merged, "ears-r"), 320, 230)[3] == 255,
          "the smaller-x ear keeps its pixels")
    check(sample(layer_of(merged, "ears-l"), 710, 230)[3] == 255,
          "the larger-x ear keeps its pixels")


def test_connected_ears_split(tmp):
    print("[cleanup: a connected pair of ears is split at the face centre]")
    run, manifest = write_run(tmp, "connected", [
        (0, [0, 0, 1024, 512], [
            fill("nose", (24, 18), 500, 240, (220, 180, 170)),
            fill("ears", (400, 70), 300, 200, (230, 190, 170)),
        ], [{"name": "nose", "left": 500, "right": 524,
             "depth_median": 0.5},
            {"name": "ears", "left": 300, "right": 700,
             "depth_median": 0.6}]),
    ], (1024, 512), overlap=0)

    info = tile_psd.merge_tiles(
        run, manifest, out_name="connected", group_layers=False,
        include_depth=False, include_original=False, seam_feather=0,
    )
    merged = psd_reader.read_psd(os.path.join(run, info["file"]))
    names = {layer["name"] for layer in merged["layers"]}
    check("ears" not in names, "the connected blob is not left as one part")
    check("ears-r" in names and "ears-l" in names,
          "the blob is split into ears-r and ears-l")


def main():
    tmp = tempfile.mkdtemp(prefix="stlu_cleanup_")
    test_background_phantom(tmp)
    test_symmetric_sides(tmp)
    test_connected_ears_split(tmp)

    print()
    if _failures:
        print("FAILED %d check(s):" % len(_failures))
        for item in _failures:
            print("  - " + item)
        raise SystemExit(1)
    print("ALL-OK")


if __name__ == "__main__":
    main()
