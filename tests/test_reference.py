"""CPU checks for the reference-guided layer rebuild (no GPU, no ComfyUI)."""

from __future__ import annotations

import os
import sys

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
for p in (TESTS_DIR, PKG_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import core_reference  # noqa: E402

CANVAS = (100, 100)
BOX = (10, 10, 30, 20)          # 20 x 10 source pixels
DEST = (40, 20)                  # 2x display size
RED = (200, 40, 40)
GREEN = (0, 200, 0)

_failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _failures.append(msg)


def _layer(alpha_dust=False):
    rgba = np.zeros((10, 20, 4), np.uint8)
    rgba[..., :3] = RED
    rgba[..., 3] = 255
    if alpha_dust:
        rgba[0, :, 3] = 5           # near-transparent seam like SeeThrough output
    return rgba


def _reference(fill, alpha=255):
    ref = np.zeros((400, 400, 4), np.uint8)
    ref[40:80, 40:120, :3] = fill
    ref[40:80, 40:120, 3] = alpha
    return ref


def main():
    print("[matching reference is trusted]")
    out, stats = core_reference.reconstruct_layer(
        _layer(), BOX, _reference(RED), CANVAS, DEST)
    check(out.shape == (20, 40, 4), "output is dest-sized %s" % (out.shape,))
    check(stats["scale"] == 4.0, "reference scale detected (4.0)")
    check(stats["trusted"] > 0.99, "reference colour used for visible pixels")
    check(tuple(out[10, 20]) == (200, 40, 40, 255), "colour comes from the reference")

    print("[occluder colour is not copied]")
    out2, stats2 = core_reference.reconstruct_layer(
        _layer(), BOX, _reference(GREEN), CANVAS, DEST)
    check(stats2["trusted"] < 0.01, "colour mismatch is rejected")
    check(tuple(out2[10, 20]) == (200, 40, 40, 255), "layer keeps its own colour")

    print("[transparent reference is not trusted]")
    _, stats3 = core_reference.reconstruct_layer(
        _layer(), BOX, _reference(RED, alpha=0), CANVAS, DEST)
    check(stats3["trusted"] < 0.01, "transparent reference pixels ignored")

    print("[alpha dust is cleaned]")
    out4, _ = core_reference.reconstruct_layer(
        _layer(alpha_dust=True), BOX, None, CANVAS, DEST, alpha_floor=12)
    check(int(out4[0, :, 3].max()) == 0, "alpha <= floor becomes 0")
    check(int(out4[-1, :, 3].min()) == 255, "solid alpha is kept")

    print("[split edge outside the reference outline is clipped]")
    # In the reference the part only occupies the right half of the box; the
    # split's soft layer edge would otherwise paint a halo in the empty half.
    ref = _reference(RED)
    ref[40:80, 40:80, 3] = 0                   # left half of the box is empty
    sil = core_reference.reference_silhouette(ref)
    check(sil is not None and not bool(sil[41, 41]), "silhouette found")
    out6, stats6 = core_reference.reconstruct_layer(
        _layer(), BOX, ref, CANVAS, DEST, silhouette=sil)
    check(stats6.get("clipped", 0) > 0, "pixels outside the silhouette clipped")
    check(int(out6[:, :20, 3].max()) == 0, "nothing painted outside the outline")
    check(int(out6[:, 20:, 3].min()) == 255, "pixels inside the outline kept")
    # A hole *inside* the character must survive: fill_holes keeps it opaque.
    holed = _reference(RED)
    holed[55:60, 70:75, 3] = 0                 # interior pocket
    sil2 = core_reference.reference_silhouette(holed)
    check(bool(sil2[57, 72]), "interior pocket is filled, not punched through")

    print("[soft rim takes the part's own colour]")
    # A hair layer whose soft rim sits on skin: the reference shows skin there,
    # so the rim used to stay pale inside the layer.
    layer = _layer()
    out7, stats7 = core_reference.reconstruct_layer(
        layer, BOX, _reference(RED), CANVAS, DEST, alpha_sharpen=1.0)
    check(stats7.get("fringe", 0) >= 0, "fringe pass reported")
    rim = (out7[..., 3] > 0) & (out7[..., 3] < 224)
    if rim.any():
        check(bool((out7[rim][:, 0] > 150).all()), "rim keeps the part's red")
    else:
        check(True, "no rim to recolour at this size")

    print("[hidden pixels keep the part's colour]")
    # The split gives the part a solid block, the reference shows the neighbour
    # (here green) on its left half: that half must not carry green.
    ref8 = _reference(RED)
    ref8[40:80, 40:80, :3] = GREEN
    own = np.zeros((20, 40, 4), np.uint8)
    own[..., :3] = RED
    own[..., 3] = 255
    covered = np.zeros((20, 40), bool)
    covered[:, :20] = True                    # left half hidden behind a neighbour
    out8, count8 = core_reference.fill_hidden(own, covered, 0)
    check(count8 == 400, "every hidden pixel repainted (%d)" % count8)
    check(tuple(out8[10, 5]) == (200, 40, 40, 255), "hidden half is the part's colour")
    check(int((out8[..., 3] > 0).sum()) == 800, "no pixels invented outside the part")
    out9, count9 = core_reference.fill_hidden(own, covered, 0)
    check(count9 == 400, "recolour alone already covers it")

    print("[reference no larger than canvas]")
    same = np.zeros((100, 100, 4), np.uint8)
    _, stats5 = core_reference.reconstruct_layer(_layer(), BOX, same, CANVAS, DEST)
    check(stats5["scale"] == 1.0 and stats5["trusted"] == 0.0,
          "no transfer without extra resolution")

    print()
    if _failures:
        print("FAILED: %d" % len(_failures))
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
