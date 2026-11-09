"""CPU-only round-trip checks for per-layer blend / opacity / folder records.

Writes a small layered PSD through ``psd_writer.write_psd`` and decodes it
back with the strict reader in this folder.  Confirms that a finished PSD can
be read, modified and written again without losing its folder structure,
blend keys or opacities -- the PSD-upscale flow depends on that.
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

import psd_reader  # noqa: E402
import psd_writer  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def solid(width, height, rgba):
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[..., :] = rgba
    return pixels


def build_layers():
    """Bottom -> top: body (100 %), a folder with two members, eyes above."""
    return [
        {"name": "body", "left": 0, "top": 0,
         "pixels": solid(32, 32, (200, 40, 40, 255))},
        {"type": "group", "name": "服装", "blend": b"norm", "opacity": 200,
         "children": [
             {"name": "inner", "left": 8, "top": 8,
              "pixels": solid(8, 8, (10, 200, 10, 255))},
             {"name": "outer", "left": 4, "top": 4, "hidden": True,
              "blend": b"mul ", "opacity": 128,
              "pixels": solid(12, 12, (10, 10, 200, 255))},
         ]},
        {"name": "eyes", "left": 16, "top": 16,
         "blend": b"scrn", "opacity": 64,
         "pixels": solid(8, 8, (250, 250, 10, 255))},
    ]


def name_map(layers):
    return {layer["name"]: layer for layer in layers}


def test_round_trip(tmp):
    print("[psd writer: blend, opacity and folder records survive a round trip]")
    path = os.path.join(tmp, "meta.psd")
    psd_writer.write_psd(path, build_layers(), 32, 32)
    doc = psd_reader.read_psd(path)
    check([doc["width"], doc["height"]] == [32, 32], "canvas size is kept")
    layers = doc["layers"]
    by_name = name_map(layers)

    check({"body", "服装", "inner", "outer", "eyes"} <= set(by_name) and
          "</Layer group>" in by_name,
          "every layer and folder is present (%s)" % sorted(by_name))
    check(by_name["body"]["blend"] == b"norm" and by_name["body"]["opacity"] == 255,
          "a plain layer keeps normal / 255")
    check(by_name["eyes"]["blend"] == b"scrn" and by_name["eyes"]["opacity"] == 64,
          "an explicit blend and opacity are written (%r/%r)"
          % (by_name["eyes"]["blend"], by_name["eyes"]["opacity"]))
    check(by_name["outer"]["blend"] == b"mul " and by_name["outer"]["opacity"] == 128,
          "a nested layer keeps its own blend and opacity")
    check(by_name["outer"]["hidden"], "the hidden member stays hidden")
    check(by_name["服装"]["kind"] == "group_open",
          "the folder header is a group_open record")
    check(by_name["服装"]["opacity"] == 200, "folder opacity is kept")

    order = [layer["name"] for layer in layers]
    check(order == ["body", "</Layer group>", "inner", "outer", "服装", "eyes"],
          "record order is bottom -> top with the divider pair around children (%s)"
          % order)
    closing = by_name["</Layer group>"]
    check(closing["kind"] == "group_end" and closing["blend"] == b"pass",
          "the closing divider is an end marker with pass blend")
    check(by_name["inner"]["opacity"] == 255 and not by_name["inner"]["hidden"],
          "a sibling without metadata stays plain")
    return path


def test_composite_opacity(path):
    print("[psd writer: the merged preview honours layer opacity]")
    doc = psd_reader.read_psd(path, with_composite=True)
    composite = doc.get("composite")
    check(composite is not None, "a merged composite is present")
    if composite is None:
        return
    # body (200, 40, 40) with eyes (250, 250, 10) at 64/255 on top -> the
    # eye patch is a quarter-strength blend, not the full eye colour.
    inside = composite[16 + 4, 16 + 4]
    check(abs(int(inside[0]) - 213) <= 2 and abs(int(inside[1]) - 93) <= 2 and
          abs(int(inside[2]) - 32) <= 2,
          "the eye patch is the 64/255 blend of red and screen-white (%s)"
          % (tuple(int(v) for v in inside),))
    outside = composite[1, 1]
    check(tuple(int(v) for v in outside[:3]) == (200, 40, 40) and outside[3] == 255,
          "the body alone is its own colour (%s)"
          % (tuple(int(v) for v in outside),))
    # The hidden, multiply-blended "outer" must not darken the canvas.
    check(tuple(int(v) for v in composite[5, 5][:3]) == (200, 40, 40),
          "the hidden member does not paint (%s)"
          % (tuple(int(v) for v in composite[5, 5]),))


def test_rewrite_keeps_meta(tmp):
    print("[psd writer: read -> write does not drop blend/opacity/folders]")
    source = os.path.join(tmp, "meta.psd")
    doc = psd_reader.read_psd(source)
    rewritten = []
    for layer in doc["layers"]:
        if layer["kind"] == "group_end":
            rewritten.append({"type": psd_writer.GROUP_END,
                              "name": layer["name"]})
        elif layer["kind"] == "group_open":
            rewritten.append({"type": psd_writer.GROUP_START,
                              "name": layer["name"],
                              "blend": layer["blend"],
                              "opacity": layer["opacity"],
                              "hidden": layer["hidden"]})
        else:
            item = {"name": layer["name"], "left": layer["left"],
                    "top": layer["top"], "pixels": layer["pixels"],
                    "blend": layer["blend"], "opacity": layer["opacity"]}
            if layer["hidden"]:
                item["hidden"] = True
            rewritten.append(item)

    path = os.path.join(tmp, "rewritten.psd")
    psd_writer.write_psd(path, rewritten, 32, 32)
    again = name_map(psd_reader.read_psd(path)["layers"])
    original = name_map(doc["layers"])
    check(set(again) == set(original), "the layer set is unchanged")
    for name in sorted(original):
        check(again[name]["blend"] == original[name]["blend"] and
              again[name]["opacity"] == original[name]["opacity"] and
              again[name]["hidden"] == original[name]["hidden"] and
              again[name]["kind"] == original[name]["kind"],
              "%s keeps blend/opacity/hidden/kind" % name)


def test_defaults():
    print("[psd writer: garbage metadata falls back to normal / opaque]")
    check(psd_writer._blend_bytes(None) == psd_writer.BLEND_NORMAL,
          "None becomes normal")
    check(psd_writer._blend_bytes("mul ") == b"mul ", "a four-char key is kept")
    check(psd_writer._blend_bytes("toolong") == b"tool", "long keys are cut to four bytes")
    check(psd_writer._opacity_byte(999) == 255 and
          psd_writer._opacity_byte(-5) == 0,
          "opacity is clamped")
    check(psd_writer._opacity_byte("x") == 255, "garbage opacity is opaque")


def main():
    with tempfile.TemporaryDirectory(prefix="stlu_psdmeta_") as tmp:
        path = test_round_trip(tmp)
        test_composite_opacity(path)
        test_rewrite_keeps_meta(tmp)
        test_defaults()
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
