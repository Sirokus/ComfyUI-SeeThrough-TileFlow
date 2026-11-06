"""CPU checks for the 4K part masks (no GPU, no ComfyUI)."""

from __future__ import annotations

import os
import sys

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
for p in (TESTS_DIR, PKG_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import core_mask4k as M  # noqa: E402

CANVAS = (100, 100)
_failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _failures.append(msg)


def main():
    print("[a real alpha decides inside]")
    rgba = np.zeros((60, 60, 4), np.uint8)
    rgba[20:40, 20:40] = (200, 30, 40, 255)
    inside = M.character_inside(rgba)
    check(bool(inside[30, 30]) and not bool(inside[5, 5]), "alpha > 8 is inside")

    print("[flattened white background is outside]")
    rgb = np.full((80, 80, 3), 255, np.uint8)
    rgb[30:50, 30:50] = (200, 30, 40)
    inside = M.character_inside(rgb)
    check(bool(inside[40, 40]) and not bool(inside[5, 5]),
          "border white is not the character")

    print("[a crop with no background is entirely inside]")
    crop = np.zeros((40, 40, 3), np.uint8)
    crop[..., :2] = (40, 30)
    check(bool(M.character_inside(crop).all()), "no white border -> all inside")

    print("[solid split alpha is clipped to the reference silhouette]")
    ref = np.zeros((400, 400, 4), np.uint8)
    ref[100:220, 100:220] = (200, 30, 40, 255)
    alpha = np.full((40, 40), 255, np.uint8)            # box (20,20,60,60) @ 4x
    result = M.mask_for_layer("face", alpha, (20, 20, 60, 60), ref, CANVAS)
    mask, stats = result["mask"], result["stats"]
    check(stats["kind"] == "part", "a plain layer uses the part path (%s)" % stats["kind"])
    check(mask.shape == (160, 160), "mask is at reference scale %s" % (mask.shape,))
    # The window starts at reference (80, 80); the part is 20..140 inside it.
    check(bool(mask[70, 70]), "inside the reference silhouette kept")
    check(not bool(mask[5, 5]) and not bool(mask[150, 150]),
          "nothing painted outside the reference silhouette")

    print("[contour snapping moves the mask onto the reference edge]")
    ref2 = np.zeros((400, 400, 4), np.uint8)
    ref2[..., 3] = 255
    ref2[..., :3] = 120
    ref2[100:300, 100:180, :3] = (200, 30, 40)          # part at ref 100..180
    alpha2 = np.zeros((100, 100), np.uint8)             # box (0,0,100,100) @ 4x
    alpha2[25:75, 26:46] = 255                          # 4 px off to the right
    plain = M.part_mask_at_reference(alpha2, (0, 0, 100, 100), ref2, CANVAS)
    cols = np.nonzero(plain["mask"].any(axis=0))[0]
    left_plain = int(cols.min()) if cols.size else -1
    check(left_plain >= 103, "without snapping the split's offset stays (%d)" % left_plain)
    snapped = M.part_mask_at_reference(alpha2, (0, 0, 100, 100), ref2, CANVAS,
                                       snap_px=6)
    cols = np.nonzero(snapped["mask"].any(axis=0))[0]
    left = int(cols.min()) if cols.size else -1
    check(abs(left - 100) <= 2, "snapped to the drawn edge (%d vs 100)" % left)

    print("[iris becomes a full circle]")
    h = w = 120
    guide = np.zeros((h, w, 4), np.uint8)
    guide[..., 3] = 255
    guide[..., :3] = (245, 245, 245)
    import cv2
    # The array is RGBA; the scalar is written channel-for-channel, so the
    # alpha has to be given explicitly (a 3-tuple would leave it at 0).
    cv2.circle(guide, (60, 60), 32, (200, 30, 40, 255), -1)
    cv2.circle(guide, (60, 60), 12, (10, 10, 10, 255), -1)
    guide[0:45, :, :3] = (10, 10, 10)                   # the lash hides the top
    yy, xx = np.mgrid[0:h, 0:w]
    prior = (np.hypot(xx - 60, yy - 60) <= 34).astype(np.float32)
    iris = M.complete_iris_mask(guide, prior)
    check(iris is not None, "circle fitted")
    if iris is not None:
        cx, cy, r = iris["circle"]
        check(abs(cx - 60) <= 1.5 and abs(cy - 60) <= 1.5,
              "centre at the iris (%0.1f, %0.1f)" % (cx, cy))
        check(abs(r - 32) <= 1.5, "radius matches the visible arc (%0.1f)" % r)
        mask = iris["mask"]
        check(bool(mask[60, 60]), "pupil is inside the part")
        check(bool(mask[35, 60]), "the hidden top is filled (full circle)")
        check(not bool(mask[60, 5]), "nothing invented outside the iris")

    print("[lash/brow strokes stay near the prior]")
    crop = np.full((60, 100, 4), 120, np.uint8)
    crop[..., 3] = 255
    crop[20:24, 10:70, :3] = (20, 10, 10)               # the stroke
    crop[2:8, 2:8, :3] = (10, 10, 10)                   # a far-away dark blob
    line = np.zeros((60, 100), np.float32)
    line[20:24, 10:70] = 1.0
    stroke = M.stroke_mask(crop, line, luma_max=100.0, near_px=5,
                           close_px=2, prior_threshold=0.3)
    check(stroke is not None and bool(stroke[21, 40]), "the stroke is kept")
    check(stroke is not None and not bool(stroke[4, 4]),
          "a dark blob away from the prior is not")

    print()
    if _failures:
        print("FAILED: %d" % len(_failures))
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
