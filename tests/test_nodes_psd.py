"""CPU-only checks for the PSD-upscale nodes (no model, no GPU).

Builds the plan the way ``STLU_PsdUpscaleMin`` does, then runs the per-part
nodes with the output root pointed at a temp dir instead of ``output/``:

    python custom_nodes/ComfyUI-SeeThrough-TileFlow/tests/test_nodes_psd.py
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

import core_io  # noqa: E402
import nodes_psd  # noqa: E402
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
         "pixels": solid(32, 32, (200, 40, 40, 255))},
        {"type": "group", "name": "服装", "blend": b"norm", "opacity": 200,
         "children": [
             {"name": "inner", "left": 8, "top": 8,
              "pixels": solid(8, 8, (10, 200, 10, 255))},
         ]},
        {"name": "eyes", "left": 16, "top": 12,
         "blend": b"scrn", "opacity": 64,
         "pixels": solid(8, 8, (250, 250, 10, 255))},
    ]
    psd_writer.write_psd(path, layers, 32, 32)
    return path


def build_plan(tmp, psd_name, out_dir, ref_path=None):
    """Run the same prepare sequence as ``STLU_PsdUpscaleMin.execute``."""
    psd_path = build_psd(os.path.join(tmp, psd_name))
    run_abs = core_io.out_dir_abs(tmp, out_dir)
    plan = psd_upscale.prepare_plan(psd_path, run_abs, factor=2.0,
                                    max_side=4096, select="all")
    jobs = psd_upscale.write_crops(plan, margin_ratio=0.15)
    plan["crops"] = jobs
    if ref_path:
        psd_upscale.write_references(plan, ref_path)
    psd_upscale.write_plan(run_abs, plan)
    return psd_upscale.plan_path(run_abs), plan


def test_prepare(tmp):
    print("[nodes_psd: prepare writes crops and plan]")
    plan_json, plan = build_plan(tmp, "piece.psd", "PsdUpscaleTest")
    check(os.path.isfile(plan_json), "a plan.json path is returned (%s)" % plan_json)
    check(plan["canvas"] == [64, 64] and plan["count"] == 3,
          "the plan reports the 2x canvas and the layer count "
          "(%s / %s)" % (plan["canvas"], plan["count"]))
    check(isinstance(psd_upscale.capped_report(plan), str),
          "the plan carries the cap report")

    plan = json.loads(open(plan_json, encoding="utf-8").read())
    check(len(plan.get("crops") or []) == 3,
          "the crop jobs are stored in the plan (%d)" % len(plan.get("crops") or []))
    for job in plan["crops"]:
        check(os.path.isfile(job["crop_file"]),
              "crop %r exists" % job["name"])
    return plan_json, plan


def test_prepare_reference(tmp):
    print("[nodes_psd: prepare cuts a reference block per layer]")
    # Same frame as the PSD (32x32), drawn at 4x: 128x128 solid blue.
    ref_path = os.path.join(tmp, "ref4x.png")
    core_io.write_rgba(ref_path, solid(128, 128, (0, 0, 255, 255)))
    plan_json, _plan = build_plan(tmp, "piece_ref.psd", "PsdUpscaleRef",
                                  ref_path=ref_path)

    plan = json.loads(open(plan_json, encoding="utf-8").read())
    ref = plan.get("reference") or {}
    check(ref.get("size") == [128, 128], "the reference size is recorded (%s)" % ref)
    check(ref.get("scale") == [4.0, 4.0], "the axis scale is recorded (%s)" % ref)
    check(ref.get("count") == 3, "every crop got a reference block (%s)" % ref.get("count"))

    plan = json.loads(open(plan_json, encoding="utf-8").read())
    for job in plan["crops"]:
        ref_file = job.get("ref_file") or ""
        check(os.path.isfile(ref_file), "reference %r exists" % job["name"])
        pixels = core_io.read_rgba(ref_file)
        rx0, ry0, rx1, ry1 = job["ref_box"]
        x0, y0, x1, y1 = job["crop_box"]
        check([rx0, ry0, rx1, ry1] == [x0 * 4, y0 * 4, x1 * 4, y1 * 4],
              "the ref box is the crop box at 4x (%s)" % (job["ref_box"],))
        check(list(pixels.shape[:2]) == [ry1 - ry0, rx1 - rx0],
              "the reference block is native resolution (%s)" % (pixels.shape,))
        check(int(pixels[..., 3].min()) == 255,
              "the reference block is fully opaque")
        got = tuple(int(v) for v in pixels[pixels.shape[0] // 2, pixels.shape[1] // 2])
        check(got == (0, 0, 255, 255),
              "the reference colour survives the white flatten (%s)" % (got,))
    # No reference selected: the plan must not gain ref files.
    plan_json, _plan = build_plan(tmp, "piece_ref.psd", "PsdUpscaleNoRef")
    plan = json.loads(open(plan_json, encoding="utf-8").read())
    check("reference" not in plan, "a run without a reference records none")
    check(all(not j.get("ref_file") for j in plan["crops"]),
          "a run without a reference writes no ref files")


def test_load_crop(plan):
    print("[nodes_psd: LoadCrop hands the model RGB plus 1 - alpha]")
    job = {j["name"]: j for j in plan["crops"]}["eyes"]
    work_w, work_h = 2 * (job["crop_box"][2] - job["crop_box"][0]), \
        2 * (job["crop_box"][3] - job["crop_box"][1])
    image, mask = nodes_psd.STLU_LoadCrop().execute(
        job["crop_file"], width=work_w, height=work_h)
    check(tuple(image.shape) == (1, work_h, work_w, 3),
          "the image is resized to the work size (%s)" % (tuple(image.shape),))
    check(tuple(mask.shape) == (1, work_h, work_w),
          "the mask matches the image (%s)" % (tuple(mask.shape),))
    check(float(image.min()) >= 0.0 and float(image.max()) <= 1.0,
          "the image is normalised to 0..1")
    corner = float(mask[0, 0, 0])
    check(abs(corner - 1.0) < 1e-3,
          "the white crop context is fully masked (%s)" % corner)
    # Sample the layer's own centre, not the crop's: MIN_MARGIN can grow a
    # small crop to the whole canvas, so the crop centre may sit on the layer
    # border where a resampled alpha is legitimately half-covered.
    layer = {l["name"]: l for l in plan["layers"]}["eyes"]
    box = layer["box"]
    k = work_w / float(job["crop_size"][0])
    cx = int(round(((box[0] + box[2]) / 2.0 - job["crop_box"][0]) * k))
    cy = int(round(((box[1] + box[3]) / 2.0 - job["crop_box"][1]) * k))
    mid = mask[0, cy, cx]
    check(float(mid) < 1e-3, "the part centre is not masked (%s)" % float(mid))
    return job, image, mask


def test_cut(tmp, plan, job, image, mask):
    print("[nodes_psd: CutToLayer writes the target-sized layer]")
    layer = {l["name"]: l for l in plan["layers"]}["eyes"]
    geometry = {
        "crop_size": job["crop_size"],
        "crop_box": job["crop_box"],
        "box": layer["box"],
        "size": layer["size"],
        "target": layer["target"],
        "out_file": layer["out_file"],
        "file": layer["file"],
    }
    # Fake the model: it returned the same picture (a perfect edit).
    out_file, layer_img = nodes_psd.STLU_CutToLayer().execute(
        image, mask, json.dumps(geometry))
    check(out_file == layer["out_file"], "the layer is written to its plan path")
    pixels = core_io.read_rgba(out_file)
    check(list(pixels.shape[:2]) == [layer["target"][1], layer["target"][0]],
          "the PNG has the target size (%s)" % (pixels.shape,))
    check(tuple(layer_img.shape) == (1, layer["target"][1], layer["target"][0], 3),
          "the layer IMAGE output matches the target (%s)"
          % (tuple(layer_img.shape),))
    cy, cx = pixels.shape[0] // 2, pixels.shape[1] // 2
    got = tuple(int(v) for v in pixels[cy, cx])
    check(all(abs(g - e) <= 12 for g, e in zip(got, (250, 250, 10, 255))),
          "the untouched centre keeps the source colour (%s)" % (got,))
    check(pixels[0, 0, 3] == 255,
          "the crop context never leaks into the layer alpha")
    return out_file


def test_missing_geometry(image, mask):
    print("[nodes_psd: CutToLayer rejects an incomplete geometry]")
    try:
        nodes_psd.STLU_CutToLayer().execute(image, mask, '{"crop_size": [4, 4]}')
    except ValueError as exc:
        check("缺少字段" in str(exc), "the error names the missing fields (%s)" % exc)
    else:
        check(False, "an incomplete geometry must raise")


def test_assemble(plan_json, plan):
    print("[nodes_psd: Assemble rebuilds the PSD from the plan]")
    # CutToLayer only wrote "eyes"; give the other layers a stand-in edit so
    # the report has no fallbacks.
    for layer in plan["layers"]:
        if layer.get("kind") != "layer" or not layer.get("out_file"):
            continue
        if os.path.isfile(layer["out_file"]):
            continue
        core_io.write_rgba(layer["out_file"], solid(
            layer["target"][0], layer["target"][1], (0, 255, 0, 255)))
    psd_path, report_json = nodes_psd.STLU_PsdAssemble().execute(
        plan_json, trigger=None, psd_filename="out")
    check(os.path.isfile(psd_path), "the PSD is written (%s)" % psd_path)
    report = json.loads(report_json)
    check(report["canvas"] == [64, 64], "the canvas follows the plan (%s)"
          % (report["canvas"],))
    check(report["upscaled"] and not report["fallback"],
          "every edited layer is used (%s / %s)"
          % (report["upscaled"], report["fallback"]))


def main():
    with tempfile.TemporaryDirectory(prefix="stlu_nodespsd_") as tmp:
        original = nodes_psd._output_root
        nodes_psd._output_root = lambda: tmp
        try:
            plan_json, plan = test_prepare(tmp)
            test_prepare_reference(tmp)
            job, image, mask = test_load_crop(plan)
            test_cut(tmp, plan, job, image, mask)
            test_missing_geometry(image, mask)
            test_assemble(plan_json, plan)
        finally:
            nodes_psd._output_root = original
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
