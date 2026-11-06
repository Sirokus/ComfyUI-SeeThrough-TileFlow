"""CPU-only checks for the user-placed square window planner.

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_region_plan.py
"""

from __future__ import annotations

import json
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (COMFY_ROOT, PKG_DIR, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import region_plan  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def expect_error(fn, message):
    try:
        fn()
    except ValueError:
        check(True, message)
        return
    check(False, message + " (no ValueError)")


def test_parse_round_and_clamp():
    print("[parse: round, clamp and follow-global]")
    warnings = []
    regions = region_plan.parse_custom_regions(
        json.dumps([
            {"x": -30, "y": 10, "size": 1000, "res": 0, "name": "头"},
            {"x": 5000, "y": 5000, "size": 100, "res": 700},
        ]), 1751, 4096, warnings)
    check(len(regions) == 2, "two windows survive")
    first, second = regions
    check(first["size"] == 1024, "size 1000 rounds to 1024 (%s)" % first["size"])
    check(first["x"] == 0, "negative x clamps to 0 (%s)" % first["x"])
    check(first["res"] == 0, "res 0 keeps follow-global")
    check(first["name"] == "头", "display name survives")
    check(second["x"] == 1751 - 128, "oversized x clamps inside the canvas (%s)" % second["x"])
    check(second["y"] == 4096 - 128, "oversized y clamps inside the canvas (%s)" % second["y"])
    check(second["size"] == 128, "size 100 rounds to 128 (%s)" % second["size"])
    check(second["res"] == 704, "res 700 rounds to 64 grid (%s)" % second["res"])


def test_parse_rejects_and_dedupes():
    print("[parse: dedupe and invalid input]")
    warnings = []
    regions = region_plan.parse_custom_regions(
        json.dumps([
            {"x": 100, "y": 100, "size": 512},
            {"x": 100, "y": 100, "size": 512},   # exact duplicate
            {"size": "nope"},                     # not numeric
            "garbage",                            # not a dict
        ]), 1024, 1024, warnings)
    check(len(regions) == 1, "only the first copy survives")
    check(len(warnings) == 3, "each skipped row gets a warning (%d)" % len(warnings))
    expect_error(lambda: region_plan.parse_custom_regions("[]", 1024, 1024),
                 "an empty layout raises instead of falling back")
    expect_error(lambda: region_plan.parse_custom_regions("{not json", 1024, 1024),
                 "broken JSON raises")
    expect_error(lambda: region_plan.parse_custom_regions(
        json.dumps([{"x": 0, "y": 0, "size": 64}]), 32, 32),
        "a canvas smaller than the minimum side raises")


def test_resolution():
    print("[resolution: follow global and override]")
    check(region_plan.region_resolution({"res": 0}, 768) == 768,
          "res 0 follows the global value")
    check(region_plan.region_resolution({"res": 1024}, 768) == 1024,
          "an explicit override wins")
    check(region_plan.region_resolution({"res": 4096}, 768) == 2048,
          "an override is clamped to the model maximum")
    check(region_plan.region_resolution({"res": 1}, 768) == 512,
          "a too-small override clamps to the model minimum")


def test_plan_key():
    print("[plan_key: stable, geometry-only]")
    a = [{"x": 10, "y": 20, "size": 512, "res": 0, "name": "head"}]
    b = [{"x": 10, "y": 20, "size": 512, "res": 0, "name": "renamed"}]
    check(region_plan.plan_key(a, 1024, 1024) == region_plan.plan_key(b, 1024, 1024),
          "renaming a window does not change the key")
    check(region_plan.plan_key(a, 1024, 1024) != region_plan.plan_key(
        [{"x": 11, "y": 20, "size": 512, "res": 0}], 1024, 1024),
        "moving a window changes the key")
    check(region_plan.plan_key(a, 1024, 1024) != region_plan.plan_key(
        [{"x": 10, "y": 20, "size": 512, "res": 1024}], 1024, 1024),
        "a per-window resolution change changes the key")
    check(region_plan.plan_key(a, 1024, 1024) != region_plan.plan_key(
        a, 1025, 1024), "a different source size changes the key")
    swapped = [{"x": 200, "y": 20, "size": 512, "res": 0},
               {"x": 10, "y": 20, "size": 512, "res": 0}]
    check(region_plan.plan_key(swapped, 1024, 1024) != region_plan.plan_key(
        list(reversed(swapped)), 1024, 1024),
        "reordering windows changes the key (order drives the guide)")


def test_document():
    print("[document: shape consumed by ③/④]")
    regions = [
        {"x": 100, "y": 40, "size": 1536, "res": 0, "name": "头"},
        {"x": 100, "y": 900, "size": 1024, "res": 1024, "name": "身"},
    ]
    doc = region_plan.custom_document(regions, 1751, 4096,
                                      global_resolution=768, overlap=128)
    check(doc["mode"] == region_plan.CUSTOM_MODE, "mode is custom_regions")
    check(doc["kind"] == region_plan.GRID_KIND, "kind matches the tile grid contract")
    check(doc["count"] == 2 and len(doc["coords"]) == 2, "one box per window")
    check(doc["coords"][0] == [100, 40, 1636, 1576], "coords are x0,y0,x1,y1")
    check(doc["tile_size"] == [1536, 1536], "compat tile_size is the largest side")
    plan = doc["partition_plan"]
    check(plan["plan_id"] == region_plan.plan_key(regions, 1751, 4096),
          "plan_id is the shared geometry key")
    check(plan["order"] == [0, 1], "order is the array order")
    check([r["resolution"] for r in plan["regions"]] == [768, 1024],
          "each region carries its effective resolution")
    check(plan["regions"][0]["res_follow"] is True,
          "res 0 is marked as follow-global")
    check(plan["regions"][1]["res_follow"] is False,
          "an explicit override is not follow-global")
    check(doc["tiles"][1]["box"] == [100, 900, 1124, 1924],
          "compat tiles mirror the coords")
    check(region_plan.order_from_document(doc, 2) == [0, 1],
          "order_from_document reads the plan")


def test_summaries():
    print("[summaries: editor windows and plan regions]")
    text = region_plan.summaries(
        [{"size": 1536, "res": 0}, {"side": 1024, "res": 1024}], 512)
    check(text == "0:1536@512 1:1024@1024",
          "sizes come from size or side (%s)" % text)


def test_uniform_windows():
    print("[uniform_windows: import from an existing partition]")
    squares = region_plan.uniform_windows(
        [[0, 0, 1024, 1024], [727, 0, 1751, 1024], [1148, 1024, 1751, 2048]],
        width=1751, height=2048)
    check(len(squares) == 3, "three windows imported")
    check(squares[0] == {"x": 0, "y": 0, "size": 1024, "res": 0, "name": ""},
          "an edge tile keeps its square and origin")
    check(squares[1]["x"] == 727 and squares[1]["size"] == 1024,
          "a full-size square keeps its origin")
    wide = region_plan.uniform_windows(
        [[0, 0, 1751, 1024]], width=1751, height=2048)[0]
    check(wide["size"] == 1024 and wide["y"] == 0,
          "a wide box keeps the inscribed square, centred")
    check(wide["x"] == (1751 - 1024) // 2,
          "the inscribed square is horizontally centred (%s)" % wide["x"])
    duplicates = region_plan.uniform_windows(
        [[0, 0, 1024, 1024], [0, 0, 1024, 1024]], width=1024, height=1024)
    check(len(duplicates) == 1, "identical boxes collapse to one window")
    tiny = region_plan.uniform_windows(
        [[0, 0, 32, 32]], width=1024, height=1024)
    check(tiny == [], "a box below the minimum side is skipped")


def main():
    test_parse_round_and_clamp()
    test_parse_rejects_and_dedupes()
    test_resolution()
    test_plan_key()
    test_document()
    test_summaries()
    test_uniform_windows()
    print()
    if _failures:
        print("%d checks failed" % len(_failures))
        return 1
    print("all region-plan checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
