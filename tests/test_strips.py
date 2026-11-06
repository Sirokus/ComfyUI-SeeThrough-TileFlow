"""CPU-only checks for the square subject-strip planner.

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_strips.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import tile_strips  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def synthetic(width=1751, height=4096, speck=True):
    """A flat-background character-like rectangle, including the 152 px speck
    that the user's Test image has at the bottom right."""
    image = np.full((height, width, 3), 255, np.uint8)
    image[43:3990, 34:1696] = (200, 40, 40)
    if speck:
        # Touch the right edge like the user's 152 px corner speck: a speck
        # surrounded by background is a hole and the background detector fills
        # it, while an edge-touching speck remains a foreground component.
        image[4000:4020, 1731:1751] = (200, 40, 40)
    return image


def boxes_inside(coords, width, height):
    return all(
        x0 >= 0 and y0 >= 0 and x1 <= width and y1 <= height and
        (x1 - x0) == (y1 - y0)
        for x0, y0, x1, y1 in coords
    )


def overlaps(coords):
    return [
        coords[i][3] - coords[i + 1][1]
        for i in range(len(coords) - 1)
    ]


def test_detect_and_plan():
    print("[subject detection and square strips]")
    rgb = synthetic()
    subject = tile_strips.detect_subject(rgb)
    check(subject["source"] == "background", "flat white background detected")
    check(subject["bbox"] == [34, 43, 1696, 3990],
          "largest component bbox is the subject (%s)" % subject["bbox"])
    check(subject["ignored_pixels"] >= 400,
          "the 20x20 speck is ignored (%d px)" % subject["ignored_pixels"])

    plan = tile_strips.plan_from_rgb(
        rgb, overlap=256, max_side=2048, max_resolution=768, margin=32)
    check(plan["ok"], "planner succeeds")
    check(plan["side"] == 1726, "square side covers the 1662 px subject + margins (%s)"
          % plan["side"])
    check(len(plan["coords"]) == 3, "three strips cover the 3947 px subject (%s)"
          % plan["coords"])
    check(plan["resolution"] == 768, "resolution capped at 768")
    check(1.73 < plan["density"] < 2.30,
          "density is about 2.25 source px/mask px (%.2f)" % plan["density"])
    check(plan["overlap"] >= 256, "actual overlap is at least the requested 256 (%s)"
          % plan["overlap"])
    check(boxes_inside(plan["coords"], 1751, 4096),
          "every strip is square and inside the canvas")
    check(all(o > 0 for o in overlaps(plan["coords"])),
          "consecutive strips overlap")
    check(plan["coords"][0][1] <= 43 and plan["coords"][-1][3] >= 3990,
          "the stack covers the subject from top to bottom")
    check(plan["coords"][0][1] <= 43 and plan["coords"][0][3] >= 43 + 500,
          "the first strip contains the whole head band")
    check(plan["coords"][0][0] <= 34 and plan["coords"][0][2] >= 1696,
          "the first strip spans the subject width")

    again = tile_strips.plan_from_rgb(
        rgb, overlap=256, max_side=2048, max_resolution=768, margin=32)
    check(again["coords"] == plan["coords"] and again["plan_id"] == plan["plan_id"],
          "the plan is deterministic")


def test_plan_from_subject():
    print("[planning from a cached subject]")
    rgb = synthetic()
    subject = tile_strips.detect_subject(rgb)
    direct = tile_strips.plan_from_rgb(
        rgb, overlap=256, max_side=2048, max_resolution=768, margin=32)
    cached = tile_strips.plan_from_subject(
        rgb.shape[1], rgb.shape[0], subject,
        overlap=256, max_side=2048, max_resolution=768, margin=32)
    check(cached["coords"] == direct["coords"] and
          cached["plan_id"] == direct["plan_id"],
          "a cached subject yields the identical plan")
    check(cached["subject_detection"]["bbox"] == subject["bbox"],
          "the plan still reports the detected bbox")


def test_document():
    print("[document shape consumed by ③/④]")
    rgb = synthetic()
    plan = tile_strips.plan_from_rgb(
        rgb, overlap=256, max_side=2048, max_resolution=768, margin=32)
    doc = tile_strips.strips_document(
        plan, rgb.shape[1], rgb.shape[0], name="grid", created="now")
    check(doc["mode"] == tile_strips.SUBJECT_MODE, "doc mode is subject_strips")
    check(doc["tile_size"] == [1726, 1726], "doc tile size is square")
    check(doc["count"] == 3 and doc["rows"] == 3 and doc["columns"] == 1,
          "doc is a 3x1 stack")
    check(doc["coords"] == plan["coords"], "coords match the plan")
    check(doc["tiles"][0]["box"] == doc["coords"][0],
          "tile entries carry the exact region box")
    check(doc["partition_plan"]["plan_id"] == plan["plan_id"],
          "partition plan carries the plan id")
    check(len(doc["partition_plan"]["regions"]) == 3,
          "partition plan lists every region")
    public = tile_strips.public_plan(plan, rgb.shape[1], rgb.shape[0])
    check(public["ok"] and public["regions"] == plan["coords"],
          "public plan returns the same regions")
    check("foreground" not in public and "foreground" not in public.get("subject", {}),
          "public plan omits the full-size foreground array")


def test_caps_and_one_region():
    print("[caps: side, resolution and one-region case]")
    rgb = synthetic()
    grow = tile_strips.plan_from_rgb(
        rgb, overlap=256, max_side=1024, max_resolution=768, margin=32)
    check(grow["ok"] and grow["side"] == 1726,
          "a target side below the subject grows to cover its full width (%s)"
          % grow.get("side"))
    check(any("自动放大" in warning for warning in grow["warnings"]),
          "the automatic growth is reported")

    # A square cannot cover a subject wider than the canvas' short side.
    impossible = tile_strips.plan_subject_strips(
        1000, 500, {"bbox": [10, 10, 990, 490], "warnings": []},
        overlap=256, max_side=2048, max_resolution=768, margin=32)
    check(not impossible["ok"] and impossible.get("fallback") == "uniform",
          "a subject wider than the short side falls back, never splits vertically")
    check("不会竖直" in (impossible.get("error") or ""),
          "the error explains the no-vertical-cut rule")

    squeeze = tile_strips.plan_subject_strips(
        1751, 1690, {"bbox": [34, 43, 1696, 1650], "warnings": []},
        overlap=256, max_side=2048, max_resolution=768, margin=32)
    check(squeeze["ok"] and squeeze["side"] == 1690,
          "a narrow canvas squeezes the margin but keeps the full subject width")
    check(any("边距" in warning for warning in squeeze["warnings"]),
          "the squeezed margin is reported")

    low = tile_strips.plan_from_rgb(
        rgb, overlap=256, max_side=2048, max_resolution=512, margin=32)
    check(low["ok"] and low["resolution"] == 512,
          "resolution cap is respected (%s)" % low.get("resolution"))
    check(any("分辨率" in warning for warning in low["warnings"]),
          "the capped resolution is reported")

    small = np.full((512, 512, 3), 255, np.uint8)
    small[156:356, 156:356] = (200, 40, 40)
    one = tile_strips.plan_from_rgb(
        small, overlap=128, max_side=2048, max_resolution=768, margin=32)
    check(one["ok"] and len(one["coords"]) == 1,
          "a subject that fits one square uses one region (%s)" % one.get("coords"))
    check(one["coords"][0] == [156, 156, 420, 420] or
          boxes_inside(one["coords"], 512, 512),
          "the one region is a square inside the canvas")


def test_fallback_and_alpha():
    print("[background fallback and explicit alpha]")
    noise = np.random.RandomState(0).randint(0, 256, (256, 256, 3), dtype=np.uint8)
    subject = tile_strips.detect_subject(noise)
    check(subject["source"] == "full", "busy background falls back to the full frame")
    check(any("整幅图" in warning for warning in subject["warnings"]),
          "the full-frame fallback is visible")
    full = tile_strips.plan_from_rgb(
        noise, overlap=32, max_side=512, max_resolution=512, margin=0)
    check(full["ok"] and len(full["coords"]) == 1,
          "the full-frame fallback still plans a square region")

    rgb = np.full((256, 256, 3), 10, np.uint8)
    alpha = np.zeros((256, 256), np.uint8)
    alpha[60:200, 40:180] = 255
    subj = tile_strips.detect_subject(rgb, alpha=alpha)
    check(subj["source"] == "alpha", "explicit alpha takes precedence")
    check(subj["bbox"] == [40, 60, 180, 200], "alpha bbox is used (%s)" % subj["bbox"])


def main():
    test_detect_and_plan()
    test_plan_from_subject()
    test_document()
    test_caps_and_one_region()
    test_fallback_and_alpha()
    print()
    if _failures:
        print("FAILED %d check(s):" % len(_failures))
        for item in _failures:
            print("  - " + item)
        return 1
    print("ALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
