"""CPU-only checks for the PSD -> edit -> PSD upscale helpers.

Builds a small layered PSD, plans the upscale, writes crops, fakes the model
output and assembles the result again -- no torch, no GPU.

    python custom_nodes/ComfyUI-SeeThrough-TileFlow/tests/test_psd_upscale.py
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

import core_io  # noqa: E402
import psd_reader  # noqa: E402
import psd_upscale  # noqa: E402
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


def build_psd(path):
    layers = [
        {"name": "body", "left": 0, "top": 0,
         "pixels": solid(64, 64, (200, 40, 40, 255))},
        {"type": "group", "name": "服装", "blend": b"norm", "opacity": 200,
         "children": [
             {"name": "inner", "left": 16, "top": 16,
              "pixels": solid(8, 8, (10, 200, 10, 255))},
             {"name": "outer", "left": 8, "top": 8, "hidden": True,
              "blend": b"mul ", "opacity": 128,
              "pixels": solid(16, 16, (10, 10, 200, 255))},
         ]},
        {"name": "eyes", "left": 40, "top": 40,
         "blend": b"scrn", "opacity": 64,
         "pixels": solid(8, 8, (250, 250, 10, 255))},
    ]
    psd_writer.write_psd(path, layers, 64, 64)
    return path


def test_plan(tmp):
    print("[psd upscale: plan keeps folders, blend, opacity and per-layer sizes]")
    psd_path = build_psd(os.path.join(tmp, "piece.psd"))
    plan = psd_upscale.prepare_plan(psd_path, os.path.join(tmp, "run"), factor=2.0,
                                    max_side=4096, select="all")
    check(plan["mode"] == psd_upscale.MODE, "plan mode is psd_upscale")
    check(plan["canvas"] == [128, 128], "canvas follows the global factor (%s)" % (plan["canvas"],))
    check(plan["count"] == 4, "four real layers are selected (%s)" % plan["count"])
    order = [(layer["kind"], layer["name"]) for layer in plan["layers"]]
    check(order == [("layer", "body"), ("group_end", "</Layer group>"),
                    ("layer", "inner"), ("layer", "outer"),
                    ("group_open", "服装"), ("layer", "eyes")],
          "record order is kept bottom -> top (%s)" % (order,))
    by_name = {layer["name"]: layer for layer in plan["layers"]}
    check(by_name["eyes"]["blend"] == "scrn" and by_name["eyes"]["opacity"] == 64,
          "blend and opacity travel with the layer")
    check(by_name["outer"]["hidden"], "the hidden member stays hidden")
    check(by_name["服装"]["opacity"] == 200 and by_name["服装"]["kind"] == "group_open",
          "folder header keeps its opacity")
    check(by_name["body"]["target"] == [128, 128] and not by_name["body"]["capped"],
          "a 64 px layer becomes 128 px at 2x")
    check(by_name["eyes"]["target"] == [16, 16], "an 8 px layer becomes 16 px")
    check(os.path.isfile(by_name["body"]["file"]) and
          core_io.read_rgba(by_name["body"]["file"]).shape == (64, 64, 4),
          "every layer is extracted to its own PNG")
    check(os.path.isfile(psd_upscale.plan_path(os.path.join(tmp, "run"))),
          "psd_plan.json is written")
    loaded = psd_upscale.load_plan(os.path.join(tmp, "run", "psd_plan.json"))
    check(loaded["count"] == 4 and loaded["canvas"] == [128, 128],
          "the plan round-trips through JSON")
    return psd_path, plan


def test_capped(tmp):
    print("[psd upscale: the per-layer pixel cap lowers the factor and reports]")
    psd_path = build_psd(os.path.join(tmp, "capped.psd"))
    plan = psd_upscale.prepare_plan(psd_path, os.path.join(tmp, "cap_run"),
                                    factor=2.0, max_side=100, select="all")
    body = {layer["name"]: layer for layer in plan["layers"]}["body"]
    check(body["target"] == [100, 100] and body["capped"],
          "a 64 px layer is capped at 100 px (%s)" % (body["target"],))
    check("上限" in psd_upscale.capped_report(plan),
          "the report names the capped layers (%s)" % psd_upscale.capped_report(plan))


def test_work_cap():
    print("[psd upscale: the work canvas never exceeds work_max_side]")
    check(psd_upscale.cap_work_size((3827, 4719), 2048) == (1661, 2048),
          "the longest side is clamped and the aspect kept (%s)"
          % (psd_upscale.cap_work_size((3827, 4719), 2048),))
    check(psd_upscale.cap_work_size((600, 400), 2048) == (600, 400),
          "a small crop is left alone")
    check(psd_upscale.cap_work_size((4000, 2000), 0) == (4000, 2000),
          "0 disables the cap")
    check(psd_upscale.cap_work_size((2048, 300), 2048) == (2048, 300),
          "a crop exactly at the limit is kept")


def test_selection():
    print("[psd upscale: layer selection syntax]")
    names = ["body", "inner", "outer", "eyes"]
    check(psd_upscale.parse_selection("all", names) == names, "all keeps everything")
    check(psd_upscale.parse_selection("", names) == names, "empty means all")
    check(psd_upscale.parse_selection("none", names) == [], "none selects nothing")
    check(psd_upscale.parse_selection("eyes,body", names) == ["body", "eyes"],
          "comma list runs in catalogue order (%s)" % psd_upscale.parse_selection("eyes,body", names))
    check(psd_upscale.parse_selection("all,-outer", names) == ["body", "inner", "eyes"],
          "exclusions drop a layer")
    check(psd_upscale.parse_selection("EYE", names) == ["eyes"],
          "matching is case-insensitive and substring based")
    check(psd_upscale.parse_selection("nonsense", names) == names,
          "an unrecognised value falls back to all")


def test_crops(tmp, plan):
    print("[psd upscale: crops are straight RGBA with a colour-extended context]")
    jobs = psd_upscale.write_crops(plan, margin_ratio=0.1)
    check(len(jobs) == 4, "one crop per selected layer (%s)" % len(jobs))
    by_name = {job["name"]: job for job in jobs}
    eyes = by_name["eyes"]
    x0, y0, x1, y1 = eyes["crop_box"]
    check(x0 <= 40 and y0 <= 40 and x1 >= 48 and y1 >= 48,
          "the crop includes a margin around the part (%s)" % (eyes["crop_box"],))
    crop = core_io.read_rgba(eyes["crop_file"])
    check(tuple(crop.shape[:2]) == (y1 - y0, x1 - x0), "the crop PNG has the crop size")
    check(crop[..., 3].max() == 255, "the part itself is opaque")
    corner = crop[0, 0]
    check(corner[3] == 0, "the context is transparent (%s)" % (tuple(int(v) for v in corner),))
    check(tuple(int(v) for v in corner[:3]) == (250, 250, 10),
          "the context colour is the layer's own, not white (%s)"
          % (tuple(int(v) for v in corner),))
    check(eyes["target"] == [16, 16], "the crop job carries the target size")
    return jobs, by_name


def test_assemble(tmp, plan, jobs):
    print("[psd upscale: assembly restores the layered PSD]")
    # Fake the model: paint every upscaled layer green at its target size.
    for job in jobs:
        core_io.write_rgba(job["out_file"], solid(job["target"][0], job["target"][1],
                                                  (0, 255, 0, 255)))
    # Leave one layer unedited to exercise the Lanczos fallback.
    os.remove({job["name"]: job for job in jobs}["inner"]["out_file"])
    out_psd = os.path.join(tmp, "upscaled.psd")
    report = psd_upscale.assemble_psd(plan, out_psd)
    check(report["canvas"] == [128, 128], "the PSD canvas is the upscaled one")
    check("inner" in report["fallback"], "the unedited layer is reported as fallback")

    doc = psd_reader.read_psd(out_psd, with_composite=True)
    check([doc["width"], doc["height"]] == [128, 128], "written PSD size matches")
    by_name = {layer["name"]: layer for layer in doc["layers"]}
    check({"body", "服装", "inner", "outer", "eyes", "</Layer group>"} <= set(by_name),
          "every layer and folder is written (%s)" % sorted(by_name))
    check(by_name["body"]["blend"] == b"norm" and by_name["body"]["opacity"] == 255,
          "body stays normal / 255")
    check(by_name["eyes"]["blend"] == b"scrn" and by_name["eyes"]["opacity"] == 64,
          "eyes keep scrn / 64")
    check(by_name["outer"]["blend"] == b"mul " and by_name["outer"]["hidden"],
          "the hidden multiply member is kept")
    check(by_name["服装"]["kind"] == "group_open" and by_name["服装"]["opacity"] == 200,
          "the folder header keeps its kind and opacity")
    check(by_name["</Layer group>"]["kind"] == "group_end",
          "the closing divider is an end record")
    check([by_name[n]["left"] for n in ("body", "eyes")] == [0, 80],
          "boxes are scaled to the canvas (%s)" % [by_name[n]["left"] for n in ("body", "eyes")])
    check(by_name["eyes"]["right"] - by_name["eyes"]["left"] == 16,
          "the eye box is 16 px after 2x")

    composite = doc["composite"]
    check(tuple(int(v) for v in composite[2, 2][:3]) == (0, 255, 0),
          "the edited body paints green (%s)" % (tuple(int(v) for v in composite[2, 2]),))
    # The unedited layer keeps its own colour through the Lanczos fallback.
    check(tuple(int(v) for v in composite[40, 40][:3]) == (10, 200, 10),
          "the unedited inner layer keeps its colour (%s)"
          % (tuple(int(v) for v in composite[40, 40]),))
    return out_psd


def test_preview(plan):
    print("[psd upscale: the preview is a downscaled merge of the same layers]")
    preview = psd_upscale.preview_rgba(plan, max_side=64)
    check(tuple(preview.shape[:2]) == (64, 64), "the preview fits max_side (%s)"
          % (tuple(preview.shape[:2]),))
    check(tuple(int(v) for v in preview[1, 1][:3]) == (0, 255, 0),
          "the body is still green in the preview")


def test_restore():
    print("[psd upscale: the edited crop is cut back with straight alpha]")
    check(psd_upscale.work_size((48, 48), (8, 8), (16, 16)) == (96, 96),
          "the crop is fed at the layer's upscaled scale")
    crop_box = [0, 0, 48, 48]
    box = [0, 0, 48, 48]
    size = (48, 48)
    alpha = np.zeros((48, 48), np.uint8)
    alpha[8:40, 8:40] = 255
    edited = np.full((48, 48, 3), 255, np.uint8)
    edited[8:40, 8:40] = (10, 20, 30)
    layer = psd_upscale.restore_layer(edited, alpha, (48, 48), crop_box, box,
                                      size, size)
    check(layer.shape == (48, 48, 4), "the restored layer has the target size")
    check(tuple(int(v) for v in layer[20, 20]) == (10, 20, 30, 255),
          "the opaque interior keeps the model colour (%s)"
          % (tuple(int(v) for v in layer[20, 20]),))
    check(layer[0, 0, 3] == 0, "the crop context is transparent again")

    # Straight alpha: a semi-transparent source pixel keeps the model colour
    # exactly as it came; there is nothing to un-composite.
    alpha[20, 20] = 128
    edited[20, 20] = (10, 20, 30)
    layer = psd_upscale.restore_layer(edited, alpha, (48, 48), crop_box, box,
                                      size, size)
    got = tuple(int(v) for v in layer[20, 20, :3])
    check(got == (10, 20, 30),
          "the crop colour passes straight through (%s)" % (got,))
    check(layer[20, 20, 3] == 128, "the edge keeps its alpha")

    # The layer sits at an offset inside a cropped region.
    offset = psd_upscale.restore_layer(
        edited, alpha, (48, 48), [24, 24, 72, 72], [40, 40, 48, 48],
        (8, 8), (16, 16))
    check(offset.shape == (16, 16, 4) and 100 < offset[8, 8, 3] < 200,
          "an offset layer is placed at its own size (%s)" % (offset.shape,))


def test_sigmas():
    print("[psd upscale: denoise scales the sigma schedule]")
    check(psd_upscale.scale_sigmas("1.0, 0.7, 0.3, 0.0", 0.45) ==
          "0.45, 0.315, 0.135, 0", "0.45 keeps 45 % of every sigma")
    check(psd_upscale.scale_sigmas("", 0.45) == "", "an empty schedule stays empty")
    check(psd_upscale.scale_sigmas("junk", 0.45) == "junk", "garbage is returned unchanged")


def test_name(plan):
    print("[psd upscale: the merged PSD name falls back widget -> plan -> stem]")
    base = dict(plan)
    base["psd"] = "/somewhere/1024_3_Tiles_edited.psd"
    base["settings"] = dict(plan.get("settings") or {})
    base["settings"].pop("psd_name", None)
    check(psd_upscale.resolve_psd_name(base) == "1024_3_Tiles_edited_upscaled",
          "the stem plus _upscaled is the last resort")

    with_plan = dict(base, settings={"psd_name": "我的角色"})
    check(psd_upscale.resolve_psd_name(with_plan) == "我的角色",
          "①'s psd_name is used when the widget is empty")
    check(psd_upscale.resolve_psd_name(with_plan, "  ") == "我的角色",
          "a blank widget still falls through to the plan")

    check(psd_upscale.resolve_psd_name(with_plan, "final") == "final",
          "the ③ widget wins over the plan")
    check(psd_upscale.resolve_psd_name(with_plan, "a/b:c") == "a_b_c",
          "illegal filename characters are sanitised (%s)"
          % psd_upscale.resolve_psd_name(with_plan, "a/b:c"))


def main():
    with tempfile.TemporaryDirectory(prefix="stlu_psdup_") as tmp:
        _psd, plan = test_plan(tmp)
        test_capped(tmp)
        test_work_cap()
        test_selection()
        jobs, _by_name = test_crops(tmp, plan)
        test_assemble(tmp, plan, jobs)
        test_preview(plan)
        test_restore()
        test_sigmas()
        test_name(plan)
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
