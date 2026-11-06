"""CPU-only checks for "a part one tile saw whole is not cut at a seam".

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_part_complete.py

No GPU, server or network: synthetic tile PSDs are merged on the CPU.
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


def fill(name, width, height, left, top, color):
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[..., :3] = color
    pixels[..., 3] = 255
    return {"name": name, "pixels": pixels, "left": int(left), "top": int(top)}


def write_run(tmp, name, tiles, source, overlap):
    """``tiles``: [(index, box, [layers])] -> (run_dir, manifest)."""
    run = os.path.join(tmp, name)
    os.makedirs(run, exist_ok=True)
    entries = []
    for index, box, layers in tiles:
        filename = "tile_%02d.psd" % index
        psd_writer.write_psd(os.path.join(run, filename), layers,
                             box[2] - box[0], box[3] - box[1], "rle")
        entries.append({
            "index": int(index),
            "file": filename,
            "tile": [int(v) for v in box],
            "layers": [{"name": layer["name"], "depth_median": 0.5}
                       for layer in layers],
        })
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


def main():
    tmp = tempfile.mkdtemp(prefix="stlu_complete_")
    A = [0, 0, 1024, 1024]
    B = [512, 0, 1536, 1024]
    C = [1024, 0, 2048, 1024]
    # The face is 724..1324.  A sees its left half (cut at its right border),
    # C sees its right half (cut at its left border), B sees it whole.
    run, manifest = write_run(tmp, "three", [
        (0, A, [fill("face", 300, 400, 724, 300, (255, 0, 0))]),
        (1, B, [fill("face", 600, 400, 212, 300, (0, 0, 255))]),
        (2, C, [fill("face", 300, 400, 0, 300, (0, 255, 0))]),
    ], (2048, 1024), 512)

    print("[complete: the tile that saw the whole face owns it]")
    info = tile_psd.merge_tiles(
        run, manifest, out_name="flat", group_layers=False,
        include_depth=False, include_original=False, seam_feather=32,
    )
    merged = psd_reader.read_psd(os.path.join(run, info["file"]))
    face = layer_of(merged, "face")
    left = sample(face, 730, 400)
    middle = sample(face, 1000, 400)
    right = sample(face, 1310, 400)
    check(left == (0, 0, 255, 255), "the face starts with tile 1's copy (%s)" % (left,))
    check(middle == (0, 0, 255, 255), "the face centre stays tile 1's copy")
    check(right == (0, 0, 255, 255), "the face ends with tile 1's copy (%s)" % (right,))
    check(face["left"] <= 724 and face["left"] + face["pixels"].shape[1] >= 1324,
          "the merged face covers the whole 724..1324 span (%d..%d)"
          % (face["left"], face["left"] + face["pixels"].shape[1]))
    reds = int(((face["pixels"][..., 0] > 200) & (face["pixels"][..., 2] < 40)).sum())
    greens = int(((face["pixels"][..., 1] > 200) & (face["pixels"][..., 0] < 40)).sum())
    check(reds == 0 and greens == 0,
          "no half-face colour survives (red %d, green %d)" % (reds, greens))

    print("[group: every copy survives, including the complete one]")
    info = tile_psd.merge_tiles(
        run, manifest, out_name="grouped", group_layers=True,
        include_depth=False, include_original=False, seam_feather=32,
    )
    grouped = psd_reader.read_psd(os.path.join(run, info["file"]))
    names = [layer["name"] for layer in grouped["layers"]]
    check({"face_0", "face_1", "face_2"} <= set(names),
          "all three face copies are kept (%s)" % names)
    widths = {}
    for name in ("face_0", "face_1", "face_2"):
        layer = layer_of(grouped, name)
        widths[name] = int(layer["right"]) - int(layer["left"])
    check(widths == {"face_0": 300, "face_1": 600, "face_2": 300},
          "none of the copies is cut (%s)" % (widths,))
    whole = layer_of(grouped, "face_1")
    check(sample(whole, 730, 400) == (0, 0, 255, 255)
          and sample(whole, 1310, 400) == (0, 0, 255, 255),
          "the whole copy still spans both edges")
    left = layer_of(grouped, "face_0")
    check(sample(left, 1000, 400)[0] > 200,
          "tile 0's red copy is not cut at the seam (%s)" % (sample(left, 1000, 400),))

    print("[complete: a fragment does not swallow a larger part]")
    A2 = [0, 0, 1024, 1024]
    B2 = [512, 0, 1536, 1024]
    run2, manifest2 = write_run(tmp, "fragment", [
        (0, A2, [fill("hair", 100, 100, 300, 600, (0, 0, 255))]),
        (1, B2, [fill("hair", 488, 100, 0, 600, (255, 0, 0))]),
    ], (1536, 1024), 512)
    info = tile_psd.merge_tiles(
        run2, manifest2, out_name="flat", group_layers=False,
        include_depth=False, include_original=False, seam_feather=32,
    )
    merged2 = psd_reader.read_psd(os.path.join(run2, info["file"]))
    hair = layer_of(merged2, "hair")
    check(sample(hair, 350, 650)[3] == 255,
          "the small complete copy still contributes its own pixels (%s)"
          % (sample(hair, 350, 650),))
    check(sample(hair, 900, 650)[3] == 255,
          "the larger copy is not dropped as a fragment (%s)"
          % (sample(hair, 900, 650),))

    print("[complete: a part that only one tile has is untouched]")
    run3, manifest3 = write_run(tmp, "single", [
        (0, A2, [fill("mouth", 80, 40, 460, 500, (10, 200, 10))]),
    ], (1024, 1024), 0)
    info = tile_psd.merge_tiles(
        run3, manifest3, out_name="solo", group_layers=False,
        include_depth=False, include_original=False)
    solo = psd_reader.read_psd(os.path.join(run3, info["file"]))
    mouth = layer_of(solo, "mouth")
    check(sample(mouth, 500, 520)[3] == 255, "a one-tile part keeps full alpha")

    print()
    if _failures:
        print("FAILED %d check(s):" % len(_failures))
        for item in _failures:
            print("  - " + item)
        raise SystemExit(1)
    print("ALL-OK")


if __name__ == "__main__":
    main()
