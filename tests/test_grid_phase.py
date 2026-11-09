"""CPU-only checks for the grid phase (offset) and its cache identity.

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_grid_phase.py

No GPU, server or network: only ``tile_psd.grid_coords`` / ``grid_document``
and the small manifest helpers are exercised.
"""

from __future__ import annotations

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

import psd_writer  # noqa: E402
import tile_psd  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def legacy_grid(width, height, tile_w, tile_h, overlap):
    """The pre-offset algorithm, kept here as the compatibility oracle."""
    coords = []
    stride_x = round(max(tile_w * 0.25, tile_w - overlap))
    stride_y = round(max(tile_h * 0.25, tile_h - overlap))
    y = 0
    while y < height:
        x = 0
        y_end = min(y + tile_h, height)
        y_start = max(0, y_end - tile_h)
        while x < width:
            x_end = min(x + tile_w, width)
            x_start = max(0, x_end - tile_w)
            coords.append((x_start, y_start, x_end, y_end))
            if x_end >= width:
                break
            x += stride_x
        if y_end >= height:
            break
        y += stride_y
    return coords


def covers_1d(intervals, length):
    end = 0
    for start, stop in sorted(intervals):
        if start > end:
            return False
        end = max(end, stop)
    return end >= length


def covers(coords, width, height):
    rows = {}
    for x0, y0, x1, y1 in coords:
        rows.setdefault(y0, []).append((x0, x1))
    if not rows:
        return False
    for intervals in rows.values():
        if not covers_1d(intervals, width):
            return False
    return covers_1d([(c[1], c[3]) for c in coords], height)


def main():
    print("[grid: offset 0 is byte-identical to the old grid]")
    cases = [
        (1300, 900, 1024, 1024, 128),
        (2048, 1024, 1024, 1024, 128),
        (2048, 2048, 1024, 1024, 0),
        (3000, 2000, 1024, 1024, 128),
        (800, 600, 1024, 1024, 128),
        (1024, 1024, 1024, 1024, 128),
        (1300, 700, 512, 512, 64),
    ]
    same = True
    for width, height, tw, th, ov in cases:
        got = tile_psd.grid_coords(width, height, tw, th, ov)
        want = legacy_grid(width, height, tw, th, ov)
        same = same and got == want
    check(same, "phase 0 matches the mirrored comfy_extras algorithm")

    print("[grid: a phase keeps every tile full-size and the canvas covered]")
    for offset in (0, 1, 64, 256, 512, 700, 894):
        for width in (1300, 1800, 2048, 3000):
            coords = tile_psd.grid_coords(width, 1024, 1024, 1024, 128,
                                           offset, offset)
            ok = covers(coords, width, 1024)
            for x0, y0, x1, y1 in coords:
                ok = ok and 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= 1024
                ok = ok and (x1 - x0) == min(1024, width)
                ok = ok and (y1 - y0) == min(1024, 1024)
            if not ok:
                check(False, "offset %d / width %d breaks coverage or tile size"
                      % (offset, width))
                break
    else:
        check(True, "every tested phase covers the canvas with full tiles")

    print("[grid: the phase can centre a subject in one tile]")
    width, tile_w, overlap = 2048, 1024, 128
    stride = round(max(tile_w * 0.25, tile_w - overlap))
    face_center, face_half = 1024, 300
    phase = ((face_center - tile_w // 2) % stride + stride) % stride
    coords = tile_psd.grid_coords(width, 1024, tile_w, 1024, overlap, phase, 0)
    holder = [c for c in coords
              if c[0] <= face_center - face_half and c[2] >= face_center + face_half]
    check(bool(holder), "phase %d gives a tile that contains the whole face (%s)"
          % (phase, holder[:1]))
    check(any(abs((c[0] + c[2]) // 2 - face_center) <= 1 for c in coords),
          "the face centre lands on a tile centre")

    print("[grid: the document and the cache remember the phase]")
    doc = tile_psd.grid_document((2048, 1024), (1024, 1024), 128,
                                 offset_x=phase, offset_y=0)
    check(doc.get("offset") == [phase, 0], "grid_document stores the phase")
    check(doc["coords"] == [[int(v) for v in c] for c in coords],
          "stored coords are the phased grid")
    manifest = {
        "tiles": [{"index": 0, "file": "tile_00.psd",
                   "tile": list(doc["coords"][0])}],
        "source": [2048, 1024],
        "tile_size": [1024, 1024],
        "overlap": 128,
        "offset": [0, 0],
        "source_fingerprint": "fp",
    }
    doc["source"]["fingerprint"] = "fp"
    check("偏移" in tile_psd.cache_mismatch_reason(manifest, doc),
          "a folder from offset 0 is cleared for a phased grid")
    manifest["offset"] = [phase, 0]
    check(tile_psd.cache_mismatch_reason(manifest, doc) == "",
          "the same phase is reusable")
    rebuilt = tile_psd._coords_from_manifest(manifest)
    check(rebuilt == doc["coords"], "a manifest without a grid_doc rebuilds the phase")

    print("[grid: write_tiles uses the document's boxes]")
    tmp = tempfile.mkdtemp(prefix="stlu_phase_")
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[..., 3] = 255
    parts = {"frame_size": (64, 64),
             "tag2pinfo": {"face": {"img": rgba, "xyxy": [0, 0, 64, 64],
                                    "depth_median": 0.5}}}
    wanted = [int(v) for v in doc["coords"][1]]
    written = tile_psd.write_tiles(
        [parts], tmp, filename_prefix="tile", merge_parts=True,
        alpha_threshold=8, source_size=(2048, 1024),
        tile_size=(1024, 1024), overlap=128, tile_index_map=[1],
        run_dir_abs=tmp, write_manifest=False,
        tile_coords=doc["coords"], grid_offset=doc["offset"])
    check(written["tiles"][0]["tile"] == wanted,
          "the tile is placed at the phased box (%s)" % written["tiles"][0]["tile"])
    check(written.get("offset") == [phase, 0], "the manifest carries the phase")

    print()
    if _failures:
        print("FAILED %d check(s):" % len(_failures))
        for item in _failures:
            print("  - " + item)
        raise SystemExit(1)
    print("ALL-OK")


if __name__ == "__main__":
    main()
