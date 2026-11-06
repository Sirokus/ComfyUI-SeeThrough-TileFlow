"""CPU-only checks for the two merge output styles.

``group_layers=True`` is the no-crop version: every tile's copy of a part is
kept whole and copies with the same name are collected in one PSD folder, so
a bad generation can be repaired without re-running diffusion.

``group_layers=False`` is the direct-merge version: copies are cut at the
ownership seam and blended into one layer per part.

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_source_copies.py

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
CANVAS_W, CANVAS_H = 1751, 1024
TILE0 = [0, 0, 1024, 1024]
TILE1 = [727, 0, 1751, 1024]


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


def group_pairs(layers):
    """``(end_index, open_index, header)`` for every folder record."""
    pending = []
    pairs = []
    for index, layer in enumerate(layers):
        kind = layer.get("kind")
        if kind == "group_end":
            pending.append(index)
        elif kind == "group_open":
            if pending:
                pairs.append((pending.pop(), index, layer))
    return pairs


def folder_children(layers, name):
    for end, open_index, header in group_pairs(layers):
        if header["name"] == name:
            return layers[end + 1:open_index], header
    return None, None


def sample(layer, x, y):
    px = layer["pixels"]
    lx, ly = int(x) - int(layer["left"]), int(y) - int(layer["top"])
    if lx < 0 or ly < 0 or lx >= px.shape[1] or ly >= px.shape[0]:
        return (0, 0, 0, 0)
    return tuple(int(v) for v in px[ly, lx])


def build_two_tiles(tmp):
    """Face across the seam, eye seen whole by tile 0, hair in both."""
    run, manifest = write_run(tmp, "wave", [
        (0, TILE0, [
            fill("face", (324, 600), 700, 200, (255, 0, 0)),
            fill("hat", (200, 100), 600, 600, (0, 255, 0)),
            # 700..860 stays clear of tile 0's shared right border, while
            # tile 1 only sees the 727..787 slice of it.
            fill("eye", (160, 80), 700, 400, (0, 255, 255)),
        ], [{"name": "face", "depth_median": 0.5},
            {"name": "hat", "depth_median": 0.9},
            {"name": "eye", "depth_median": 0.4}]),
        (1, TILE1, [
            fill("face", (356, 600), 0, 200, (0, 0, 255)),
            fill("hat", (200, 100), 0, 600, (0, 255, 0)),
            fill("eye", (60, 80), 0, 400, (255, 255, 0)),
        ], [{"name": "face", "depth_median": 0.5},
            {"name": "hat", "depth_median": 0.9},
            {"name": "eye", "depth_median": 0.4}]),
    ], (CANVAS_W, CANVAS_H), overlap=297)
    return run, manifest


def merge(run, manifest, out_name, group_layers=True):
    return tile_psd.merge_tiles(
        run, manifest, out_name=out_name, group_layers=group_layers,
        include_depth=False, include_original=False, seam_feather=32)


def test_group_structure(tmp):
    print("[group: same names become visible folders]")
    run, manifest = build_two_tiles(tmp)
    info = merge(run, manifest, "grouped.psd", group_layers=True)
    doc = psd_reader.read_psd(os.path.join(run, info["file"]))

    check(set(info.get("groups") or []) == {"face", "hat", "eye"},
          "merge reports one folder per multi-copy part (%s)" % (info.get("groups"),))
    for name in ("face", "hat", "eye"):
        children, header = folder_children(doc["layers"], name)
        check(header is not None, "%s is a folder" % name)
        if header is None:
            continue
        check(not header["hidden"], "%s folder is visible" % name)
        names = [layer["name"] for layer in children if layer["kind"] == "layer"]
        check(names == [name + "_0", name + "_1"],
              "%s keeps one layer per tile (%s)" % (name, names))
        check(all(not layer["hidden"] for layer in children
                  if layer["kind"] == "layer"),
              "%s copies are visible" % name)


def test_uncropped_copies(tmp):
    print("[group: each copy keeps its full pixels]")
    run, manifest = build_two_tiles(tmp)
    info = merge(run, manifest, "uncut.psd", group_layers=True)
    doc = psd_reader.read_psd(os.path.join(run, info["file"]))

    face_children, _ = folder_children(doc["layers"], "face")
    left = next(layer for layer in face_children if layer["name"] == "face_0")
    right = next(layer for layer in face_children if layer["name"] == "face_1")
    check(int(left["right"]) - int(left["left"]) == 324,
          "tile 0 face width is untouched (%d px)"
          % (int(left["right"]) - int(left["left"])))
    check(int(right["right"]) - int(right["left"]) == 356,
          "tile 1 face width is untouched (%d px)"
          % (int(right["right"]) - int(right["left"])))
    check(sample(left, 950, 400)[0] > 240 and sample(left, 950, 400)[2] < 15,
          "tile 0 keeps the red half past the seam (%s)" % (sample(left, 950, 400),))
    check(sample(right, 750, 400)[2] > 240 and sample(right, 750, 400)[0] < 15,
          "tile 1 keeps the blue half before the seam (%s)" % (sample(right, 750, 400),))

    # The complete-part rule must not drop tile 1's narrower eye copy here.
    eye_children, _ = folder_children(doc["layers"], "eye")
    eye_names = [layer["name"] for layer in eye_children if layer["kind"] == "layer"]
    check(eye_names == ["eye_0", "eye_1"],
          "both eye copies are kept (%s)" % (eye_names,))
    tile1_eye = next(layer for layer in eye_children if layer["name"] == "eye_1")
    check(sample(tile1_eye, 750, 420)[:3] == (255, 255, 0),
          "the dropped tile 1 eye keeps its own pixels (%s)"
          % (sample(tile1_eye, 750, 420),))


def test_single_copy_stays_layer(tmp):
    print("[group: a part with one copy stays a plain layer]")
    run, manifest = write_run(tmp, "solo", [
        (0, [0, 0, 1024, 1024], [
            fill("mouth", (80, 40), 460, 500, (10, 200, 10)),
        ], [{"name": "mouth", "depth_median": 0.2}]),
    ], (1024, 1024), overlap=0)
    info = merge(run, manifest, "solo.psd", group_layers=True)
    doc = psd_reader.read_psd(os.path.join(run, info["file"]))
    check(not info.get("groups"),
          "no folder for a lone part (%s)" % (info.get("groups"),))
    mouth = next(layer for layer in doc["layers"]
                 if layer["name"] == "mouth" and layer["kind"] == "layer")
    check(sample(mouth, 500, 520)[3] == 255, "the lone part keeps full alpha")


def test_flat_still_cuts(tmp):
    print("[direct: flat merge still cuts at the seam]")
    run, manifest = build_two_tiles(tmp)
    info = merge(run, manifest, "flat.psd", group_layers=False)
    doc = psd_reader.read_psd(os.path.join(run, info["file"]))
    faces = [layer for layer in doc["layers"]
             if layer["name"] == "face" and layer["kind"] == "layer"]
    check(len(faces) == 1, "the flat merge writes one face layer")
    face = faces[0]
    check(sample(face, 750, 400)[0] > 150,
          "left of the seam is tile 0's red (%s)" % (sample(face, 750, 400),))
    check(sample(face, 950, 400)[2] > 150,
          "right of the seam is tile 1's blue (%s)" % (sample(face, 950, 400),))


def test_phantom_cut_in_group(tmp):
    print("[group: the invented blob is cut in the uncropped packaging too]")
    size = 1024
    source = np.full((size, size, 3), 255, np.uint8)
    source[200:800, 400:700] = (200, 40, 40)
    run, manifest = write_run(tmp, "phantom", [
        (0, [0, 0, size, size], [
            fill("face", (300, 600), 400, 200, (240, 210, 190)),
            fill("face", (70, 70), 100, 150, (228, 219, 204)),
        ], [{"name": "face", "depth_median": 0.5}]),
    ], (size, size), original=source)

    # Folder mode keeps whole copies per tile, but a copy that carries an
    # invented backdrop is not a useful reference: the phantom trim applies
    # here too, so the blob must be gone from every face copy.
    grouped = merge(run, manifest, "phantom_group.psd", group_layers=True)
    doc = psd_reader.read_psd(os.path.join(run, grouped["file"]))
    face_layers = [layer for layer in doc["layers"]
                   if layer["kind"] == "layer"
                   and str(layer["name"]).split("_")[0] == "face"]
    check(bool(face_layers), "the subject face survives the group merge")
    check(all(sample(layer, 130, 180)[3] == 0 for layer in face_layers),
          "no face copy keeps the invented blob (%s)"
          % ([sample(layer, 130, 180) for layer in face_layers],))

    flat = merge(run, manifest, "phantom_flat.psd", group_layers=False)
    doc = psd_reader.read_psd(os.path.join(run, flat["file"]))
    face = next(layer for layer in doc["layers"]
                if layer["name"] == "face" and layer["kind"] == "layer")
    check(sample(face, 130, 180)[3] == 0,
          "the direct merge still removes the phantom blob")


def main():
    tmp = tempfile.mkdtemp(prefix="stlu_sources_")
    test_group_structure(tmp)
    test_uncropped_copies(tmp)
    test_single_copy_stays_layer(tmp)
    test_flat_still_cuts(tmp)
    test_phantom_cut_in_group(tmp)

    print()
    if _failures:
        print("FAILED %d check(s):" % len(_failures))
        for item in _failures:
            print("  - " + item)
        raise SystemExit(1)
    print("ALL-OK")


if __name__ == "__main__":
    main()
