"""CPU-only checks that a broken grid_doc fails loudly.

A corrupt document used to degrade into "run without guidance", which looks
like success while producing the wrong picture.  Every consumer must raise.
"""

from __future__ import annotations

import json
import os
import sys

import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import nodes as stlu  # noqa: E402
import nodes_shared  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


class DummySeeThrough:
    """Stand-in for the upstream SeeThrough node classes."""

    def generate(self, *args, **kwargs):
        raise AssertionError("must not reach the model")

    def process(self, *args, **kwargs):
        raise AssertionError("must not reach the model")


def expect_value_error(label, fn, needle):
    try:
        fn()
    except ValueError as exc:
        text = str(exc)
        check(needle in text, "%s raises ValueError naming %s" % (label, needle))
    except Exception as exc:  # noqa: BLE001
        check(False, "%s raised %s instead of ValueError" % (label, type(exc).__name__))
    else:
        check(False, "%s did not raise" % label)


def main():
    print("[stage 3 rejects a corrupt grid_doc]")
    expect_value_error(
        "STLU_TilePSD",
        lambda: stlu.STLU_TilePSD().execute(
            parts={"tag2pinfo": {}, "frame_size": (64, 64)},
            index="0", grid_doc="not json"),
        "STLU_TilePSD")

    print("[stage 3b rejects a corrupt grid_doc]")
    expect_value_error(
        "STLU_WaveIndex",
        lambda: stlu.STLU_WaveIndex().execute(
            tiles=[], original=torch.zeros(1, 64, 64, 3), grid_doc="not json"),
        "STLU_WaveIndex")

    print("[stage 2 rejects a corrupt grid_doc before running the model]")
    original = nodes_shared._seethrough_node_class
    nodes_shared._seethrough_node_class = lambda name: DummySeeThrough
    try:
        expect_value_error(
            "STLU_GuidedLayers (corrupt)",
            lambda: stlu.STLU_GuidedLayers().execute(
                tiles=[torch.zeros(1, 64, 64, 3)], indices=["0"],
                layerdiff_model=[object()], depth_model=[object()],
                grid_doc="not json"),
            "STLU ②")

        empty_doc = json.dumps({"coords": [], "out": {"dir": ""}})
        expect_value_error(
            "STLU_GuidedLayers (no coords)",
            lambda: stlu.STLU_GuidedLayers().execute(
                tiles=[torch.zeros(1, 64, 64, 3)], indices=["0"],
                layerdiff_model=[object()], depth_model=[object()],
                grid_doc=empty_doc),
            "没有块坐标")
    finally:
        nodes_shared._seethrough_node_class = original

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
