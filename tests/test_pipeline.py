"""End-to-end dry run of Plan -> LoadLayer -> SaveLayer -> AssemblePSD.

No GPU and no ComfyUI server: the "upscaled" images are produced by simply
resizing the source layers, which is enough to validate file naming, the
canvas mapping and the PSD writer.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from PIL import Image

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for p in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import core_io  # noqa: E402
import core_plan  # noqa: E402
import nodes as stlu  # noqa: E402
import nodes_shared  # noqa: E402
import psd_reader  # noqa: E402

SOURCE_OUTPUT = os.path.join(COMFY_ROOT, "output")


def _find_layers_json():
    preferred = os.path.join(
        SOURCE_OUTPUT, "SeeThrough", "seethrough_20261031_144549_2c644a74_layers.json"
    )
    if os.path.isfile(preferred):
        return preferred
    return core_plan.find_latest_layers_json(SOURCE_OUTPUT) or preferred


LAYERS_JSON = _find_layers_json()

_failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _failures.append(msg)


def main():
    if not os.path.isfile(LAYERS_JSON):
        print("sample layers.json missing")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        nodes_shared._output_root = lambda: tmp  # keep the real output folder clean

        plan_text, plan_path, total, report_text = stlu.STLU_Plan().execute(
            layers_json=LAYERS_JSON,
            select="eyebrow-l,handwear-r,face",
            exclude="",
            size_rules='{"*": {"long_side": 256}, "face": {"long_side": 512}}',
            out_dir="SeeThrough/UpScale",
            canvas_size=512,
        )
        check(total == 3, "plan selected 3 layers")
        check(os.path.isfile(plan_path), "plan file written")
        plan = core_plan.plan_from_text(plan_text)
        selected = [l for l in plan["layers"] if l["selected"]]
        check(len(selected) == 3, "plan has 3 selected entries")

        planner = stlu.STLU_LoadLayer()
        for layer in selected:
            image, mask, name, width, height, target_path = planner.execute(
                plan=plan_path, index=layer["z"], reference=None
            )
            check(name == layer["name"], "loader name %r" % name)
            check((width, height) == (layer["target"]["w"], layer["target"]["h"]),
                  "loader size %r -> %dx%d" % (name, width, height))
            check(tuple(image.shape[:2]) == (1, layer["file"] and image.shape[1]),
                  "loader image tensor shape")

        # Simulate the model output: stretch the source art to the target size.
        saver = stlu.STLU_SaveLayer()
        produced = []
        for layer in selected:
            image, mask, name, width, height, _ = planner.execute(
                plan=plan_path, index=layer["z"], reference=None
            )
            import torch
            src = Image.fromarray(core_io.read_rgba(layer["file"]), "RGBA")
            up = src.resize((width, height), Image.Resampling.LANCZOS)
            up_t = torch.from_numpy(np.array(up, dtype=np.float32) / 255.0)[None]
            saved_path, _ = saver.execute(
                images=up_t, mask=mask, name=name, out_dir="SeeThrough/UpScale"
            )
            check(os.path.isfile(saved_path), "saved %r" % name)
            produced.append(saved_path)

        assembled = stlu.STLU_AssemblePSD().execute(
            plan=plan_text, out_dir="SeeThrough/UpScale", deps="done", psd_filename="smoke"
        )
        psd_path, layers_json_path, report, preview = assembled["result"]
        check(os.path.isfile(psd_path), "PSD written: %s" % psd_path)
        check(os.path.isfile(layers_json_path), "layers.json written")
        check(tuple(preview.shape) == (1, 512, 512, 3), "preview tensor shape")

        ui = (assembled.get("ui") or {}).get("stlu_psd") or [{}]
        ui = ui[0]
        check(str(ui.get("filename", "")).endswith(".psd"), "UI payload names the PSD")
        check(ui.get("size", 0) > 0, "UI payload reports the file size")
        check(ui.get("preview") is not None, "UI payload has a preview image")
        if ui.get("preview"):
            preview_path = os.path.join(
                tmp, ui["preview"]["subfolder"], ui["preview"]["filename"]
            )
            check(os.path.isfile(preview_path), "preview PNG written next to the PSD")

        doc = psd_reader.read_psd(psd_path)
        expected = [l["name"] for l in plan["layers"] if not (l["empty"] and not l["selected"])]
        check(doc["width"] == 512 and doc["height"] == 512, "PSD is 512x512")
        check(len(doc["layers"]) == len(expected),
              "PSD has every non-empty layer (%d/%d)" % (len(doc["layers"]), len(plan["layers"])))
        names = [l["name"] for l in doc["layers"]]
        check("face" in names, "face layer present")

        # The PSD writer emits bottom -> top, plan order is far -> near.
        check(names == expected, "PSD layer order matches plan")

        # The preview is not empty.
        arr = preview[0].numpy()
        check(arr.std() > 0.01, "preview has content")

        print("  psd size: %.1f MB" % (os.path.getsize(psd_path) / 1e6))

    print()
    if _failures:
        print("FAILED: %d" % len(_failures))
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
