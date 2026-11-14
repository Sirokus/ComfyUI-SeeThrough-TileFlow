"""Offline tests for the image-edit part extraction.

Run with the ComfyUI embedded python (no GPU, no server):

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_edit_parts.py
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
for _p in (PKG_DIR, COMFY_ROOT, TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import core_io  # noqa: E402
import edit_expand  # noqa: E402
import edit_parts  # noqa: E402
import psd_reader  # noqa: E402

TEMPLATE = edit_parts.default_template_path()


def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------

def test_catalogue():
    names = edit_parts.PART_NAMES
    _ok(len(names) == len(set(names)), "duplicate part names")
    for expected in ("back hair", "front hair", "face", "irides-l", "eyewhite-r",
                     "eyelash-l", "handwear-r", "footwear", "topwear"):
        _ok(expected in names, "missing part %r" % expected)
    # SeeThrough order: first entry is the bottom-most PSD layer.
    depths = [edit_parts.PART_DEPTH[n] for n in names]
    _ok(depths == sorted(depths, reverse=True), "part order is not far -> near")
    print("catalogue ok (%d parts, far -> near)" % len(names))


def test_select():
    _ok(edit_parts.parse_select("all") == edit_parts.PART_NAMES, "all must expand")
    _ok(edit_parts.parse_select("") == edit_parts.PART_NAMES, "empty must expand")
    picked = edit_parts.parse_select("face, mouth, eyewhite-l")
    _ok(picked == ["face", "mouth", "eyewhite-l"], "explicit select: %r" % picked)
    picked = edit_parts.parse_select("all -wings -tail")
    _ok("wings" not in picked and "tail" not in picked, "minus syntax failed")
    _ok(len(picked) == len(edit_parts.PART_NAMES) - 2, "minus removed too much")
    _ok(edit_parts.parse_select("nonsense") == edit_parts.PART_NAMES,
        "unknown names must not silently drop parts")
    # The panel writes comma-separated whole names; "back hair" has a space.
    picked = edit_parts.parse_select("mouth, back hair, face")
    _ok(picked == ["back hair", "face", "mouth"],
        "comma names + catalogue order: %r" % picked)
    _ok(edit_parts.parse_select("none") == [], "none must select nothing")
    _ok(len(edit_parts.parse_select("all -handwear-l -handwear-r -wings -tail")) == 25,
        "minus with dashes in names failed")
    print("select ok")


def test_catalogue_rows():
    table = edit_parts.catalogue()
    _ok(len(table) == len(edit_parts.PART_NAMES), "catalogue size")
    for row in table:
        _ok(row["name"] and row["label"] and row["group"], "catalogue row: %r" % row)
        _ok(row["group"] in dict(edit_parts.GROUPS), "unknown group %r" % row["group"])
    print("catalogue rows ok")


def test_prompt():
    prompt = edit_parts.build_prompt("irides-l")
    _ok("left eye" in prompt, "part description missing")
    _ok("<image1>" in prompt, "image reference missing")
    _ok("real alpha channel" in prompt and "transparent" in prompt,
        "prompt must ask for a native alpha channel")
    _ok("solid colour" in prompt, "prompt must forbid a painted backdrop")
    _ok("#FF00FF" not in prompt and "green" not in prompt.lower(),
        "no key colour may leak into the prompt")
    print("prompt ok")


# ---------------------------------------------------------------------------
# layers.json + PSD
# ---------------------------------------------------------------------------

def test_layers_and_psd():
    tmp = tempfile.mkdtemp(prefix="stlu_edit_")
    try:
        out_rel = "SeeThrough/_test_edit"
        out_abs = os.path.join(tmp, out_rel)
        os.makedirs(out_abs, exist_ok=True)
        size = 64

        def write(name, colour, alpha):
            arr = np.zeros((size, size, 4), np.uint8)
            arr[..., :3] = colour
            arr[16:48, 16:48, 3] = alpha   # a part, not a full-frame result
            core_io.write_rgba(os.path.join(out_abs, name + ".png"), arr)

        write("wings", (0, 0, 0), 0)          # empty -> skipped
        write("face", (40, 120, 200), 255)
        write("front hair", (200, 60, 60), 255)

        entries = [
            edit_parts.layer_entry("wings", out_rel, "wings.png", 0, 0, size, size),
            edit_parts.layer_entry("face", out_rel, "face.png", 0, 0, size, size),
            edit_parts.layer_entry("front hair", out_rel, "front hair.png",
                                   0, 0, size, size),
        ]
        doc = edit_parts.layers_json_doc("editparts", "20260101_000000_test",
                                         entries, size, size)
        _ok([e["name"] for e in doc["layers"]] == ["wings", "face", "front hair"],
            "layers.json order must stay far -> near")
        _ok(set(doc.keys()) >= {"prefix", "timestamp", "layers", "width", "height"},
            "layers.json must keep the SeeThrough keys")

        layers_path = os.path.join(tmp, "layers.json")
        edit_parts.write_layers_json(layers_path, doc)
        again = edit_parts.read_layers_json(layers_path)
        _ok(len(again["layers"]) == 3, "read back failed")

        psd_path = os.path.join(tmp, "out.psd")
        info = edit_parts.assemble_psd(layers_path, tmp, psd_path)
        _ok(info["layers"] == ["face", "front hair"],
            "empty layer not skipped: %r" % info["layers"])
        _ok(any("wings" in n for n in info["notes"]), "skip note missing")

        psd = psd_reader.read_psd(psd_path)
        _ok(psd["width"] == size and psd["height"] == size, "canvas size wrong")
        names = [layer.get("name") for layer in psd["layers"]]
        _ok(names == ["face", "front hair"],
            "PSD order must be bottom -> top: %r" % names)
        alpha = psd["layers"][0]["pixels"][..., 3]
        _ok(int(alpha[32, 32]) == 255, "PSD layer alpha wrong")
        _ok(int(alpha[0, 0]) == 0, "PSD layer must keep transparent corners")
        print("layers/psd ok")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# subgraph expansion
# ---------------------------------------------------------------------------

def test_expand():
    _ok(os.path.isfile(TEMPLATE), "template missing: %s" % TEMPLATE)
    parts = [
        {"name": "face", "prompt": edit_parts.build_prompt("face")},
        {"name": "front hair", "prompt": edit_parts.build_prompt("front hair")},
    ]
    external = {
        "reference": ["10", 0],
        "model": ["11", 0],
        "clip": ["12", 0],
        "vae": ["13", 0],
    }
    expand, psd, saves = edit_expand.build_expand(
        workflow_path=TEMPLATE,
        parts=parts,
        external=external,
        seed=100,
        out_dir_rel="SeeThrough/_test_edit",
        layers_json_path=os.path.join(COMFY_ROOT, "output", "_test_layers.json"),
        psd_name="test",
        negative_prompt="", cfg=1.0, resolution=0, scheduler="euler",
        sigmas="1.0, 0.5, 0", width=512, height=512,
    )

    classes = [n["class_type"] for n in expand.values()]
    for cls in ("STLU_SaveLayer", "STLU_PartsPSD",
                "STLU_Gate", "STLU_Barrier", "TextEncodeQwenImage21",
                "SamplerCustomAdvanced", "ImageToMask", "InvertMask"):
        _ok(cls in classes, "expansion lost %s" % cls)
    for banned in ("TextGenerate", "BatchImagesNode", "PreviewAny",
                   "UNETLoader", "CLIPLoader", "VAELoader",
                   "STLU_ChromaToAlpha", "STLU_KeyCut"):
        _ok(banned not in classes, "expansion must drop %s" % banned)

    _ok(classes.count("STLU_SaveLayer") == 2, "one save per part")
    _ok(classes.count("STLU_PartsPSD") == 1, "one PSD at the end")
    _ok(all(n["inputs"].get("keep_input_alpha") is True
            for n in expand.values()
            if n["class_type"] == "STLU_SaveLayer"),
        "the layer keeps the model's own alpha")

    encoders = [n for n in expand.values()
                if n["class_type"] == "TextEncodeQwenImage21"]
    prompts = [e["inputs"].get("prompt") for e in encoders]
    _ok(len(prompts) == 2 and "bare face skin" in prompts[0]
        and "front hair" in prompts[1], "per-part prompts wrong: %r" % prompts)
    _ok(all(e["inputs"].get("clip") == ["12", 0] for e in encoders),
        "encoder clip must come from the batch node")

    noises = [n for n in expand.values() if n["class_type"] == "RandomNoise"]
    seeds = sorted(n["inputs"].get("noise_seed") for n in noises)
    _ok(seeds == [100, 101], "seeds must be distinct: %r" % seeds)

    guiders = [n for n in expand.values() if n["class_type"] == "CFGGuider"]
    _ok(all(g["inputs"].get("model") == ["11", 0] for g in guiders),
        "guider model must come from the batch node")

    saves_list = [n for n in expand.values() if n["class_type"] == "STLU_SaveLayer"]
    names = sorted(n["inputs"].get("name") for n in saves_list)
    _ok(names == ["face", "front hair"], "save names wrong: %r" % names)

    # Every wire must be a link ([node_id, slot]); a Node object left in an
    # input is not a link, so the executor runs that node before its
    # upstream -- exactly how the PSD once ran before any sampling.
    for node_id, info in expand.items():
        for name, value in (info.get("inputs") or {}).items():
            _ok(not hasattr(value, "class_type") or isinstance(value, str),
                "non-link Node on %s.%s: %r" % (node_id, name, value))

    psd = next(n for n in expand.values() if n["class_type"] == "STLU_PartsPSD")
    trigger = psd["inputs"].get("trigger")
    _ok(isinstance(trigger, list) and len(trigger) == 2
        and trigger[0] in expand, "PSD trigger must be a link: %r" % (trigger,))
    _ok(expand[trigger[0]]["class_type"] == "STLU_Barrier",
        "PSD must wait on the save barrier")
    print("expand ok (%d nodes)" % len(expand))


def test_expand_crop_reference():
    _ok(os.path.isfile(TEMPLATE), "template missing: %s" % TEMPLATE)
    geometry = {
        "crop_size": [64, 64], "crop_box": [0, 0, 64, 64],
        "box": [8, 8, 56, 56], "size": [64, 64], "target": [128, 128],
        "out_file": "layer.png", "file": "source.png",
    }
    base = {
        "crop_file": "crop.png", "width": 64, "height": 64,
        "resolution": 64, "geometry": geometry,
    }
    parts = [
        dict(base, name="with ref", prompt="with ref", ref_file="ref.png"),
        dict(base, name="no ref", prompt="no ref", ref_file=""),
    ]
    external = {"model": ["11", 0], "clip": ["12", 0], "vae": ["13", 0]}
    expand, _psd, _saves = edit_expand.build_expand(
        workflow_path=TEMPLATE, parts=parts, external=external, seed=7,
        out_dir_rel="SeeThrough/_test_crop", layers_json_path="",
        psd_name="", psd_plan="plan.json", crop_mode=True,
        negative_prompt="", cfg=1.0, resolution=0, scheduler="euler",
        sigmas="1.0, 0.5, 0", width=0, height=0,
    )

    classes = [n["class_type"] for n in expand.values()]
    _ok(classes.count("STLU_CutToLayer") == 2, "one cut per crop part")
    _ok("STLU_AlphaCut" not in classes,
        "the model's own alpha is used, no post cut")
    _ok("STLU_ChromaToAlpha" not in classes and "STLU_KeyCut" not in classes,
        "crop mode must not key anything")

    loads = [n for n in expand.values() if n["class_type"] == "STLU_LoadCrop"]
    _ok(len(loads) == 3, "two crops + one reference expected, got %d" % len(loads))
    ref_loads = [n for n in loads if n["inputs"].get("file") == "ref.png"]
    _ok(len(ref_loads) == 1, "the reference block is read once")
    _ok(ref_loads[0]["inputs"].get("width") == 0
        and ref_loads[0]["inputs"].get("height") == 0,
        "the reference keeps its native size for the encoder")

    encoders = [n for n in expand.values()
                if n["class_type"] == "TextEncodeQwenImage21"]
    _ok(len(encoders) == 2, "one encoder per part")
    by_prompt = {e["inputs"].get("prompt"): e for e in encoders}
    image_1 = by_prompt["with ref"]["inputs"].get("images.image_1")
    image_2 = by_prompt["with ref"]["inputs"].get("images.image_2")
    _ok(isinstance(image_1, list) and image_1[0] in expand
        and expand[image_1[0]]["inputs"].get("file") == "crop.png",
        "image_1 must be the crop: %r" % (image_1,))
    _ok(isinstance(image_2, list) and image_2[0] in expand
        and expand[image_2[0]]["inputs"].get("file") == "ref.png",
        "image_2 must be the reference block: %r" % (image_2,))
    _ok(by_prompt["no ref"]["inputs"].get("images.image_2") is None,
        "no reference selected means no image_2 wire")
    print("crop reference expand ok (%d nodes)" % len(expand))


if __name__ == "__main__":
    test_catalogue()
    test_catalogue_rows()
    test_select()
    test_prompt()
    test_layers_and_psd()
    test_expand()
    test_expand_crop_reference()
    print("\nall edit-part tests passed")
