"""Offline tests for the SeeThrough layer-upscale package.

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_offline.py

No GPU, no ComfyUI server and no network are touched.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, TESTS_DIR)

import core_compose  # noqa: E402
import core_io  # noqa: E402
import core_plan  # noqa: E402
import psd_reader  # noqa: E402
import psd_writer  # noqa: E402

OUTPUT_ROOT = os.path.abspath(os.path.join(PKG_DIR, "..", "..", "output"))


def _find_layers_json():
    preferred = os.path.join(
        OUTPUT_ROOT, "SeeThrough", "seethrough_20261031_144549_2c644a74_layers.json"
    )
    if os.path.isfile(preferred):
        return preferred
    return core_plan.find_latest_layers_json(OUTPUT_ROOT) or preferred


LAYERS_JSON = _find_layers_json()

_failures = []


def check(cond, message):
    if cond:
        print("  ok   %s" % message)
    else:
        print("  FAIL %s" % message)
        _failures.append(message)


def test_safe_names():
    print("[safe names]")
    check(core_plan.safe_name("front hair") == "front hair", "spaces kept")
    check(core_plan.safe_name("a/b:c*?") == "a_b_c__", "illegal chars replaced")
    check(core_plan.safe_name("CON") == "_CON", "reserved device name escaped")
    check(core_plan.safe_name("..") == "_", "dot-dot escaped")
    check(core_plan.safe_name("眼睛") == "眼睛", "CJK kept")


def test_plan():
    print("[plan]")
    if not os.path.isfile(LAYERS_JSON):
        check(False, "sample layers.json exists (%s)" % LAYERS_JSON)
        return
    rules = (
        '{"*": {"long_side": 512},'
        ' "face": {"long_side": 1024},'
        ' "eyebrow-*": 256}'
    )
    with tempfile.TemporaryDirectory() as tmp:
        plan = core_plan.build_plan(
            LAYERS_JSON,
            tmp,
            select="all",
            exclude="",
            size_rules_text=rules,
            out_dir="SeeThrough/UpScale",
            canvas_size=1024,
            skip_empty=False,
        )
        check(plan["total"] == len(plan["layers"]), "all layers selected")

        plan_default = core_plan.build_plan(
            LAYERS_JSON, tmp, select="all", size_rules_text="",
            out_dir="SeeThrough/UpScale0", canvas_size=1024,
        )
        check(plan_default["total"] < len(plan_default["layers"]),
              "empty layers skipped by default (%d/%d)" % (
                  plan_default["total"], len(plan_default["layers"])))
        check("tail" in plan_default["empty_skipped"], "dust-only layer detected as empty")
        # A faint but compact part (the nose shadow) must survive: judging by
        # the fraction of the *canvas* used to drop it, exposing the pale fill
        # of the face layer underneath.
        check("nose" not in plan_default["empty_skipped"], "small compact part is kept")
        check("face" not in plan_default["empty_skipped"], "face is not empty")
        dep = [l["depth_median"] for l in plan["layers"]]
        check(dep == sorted(dep, reverse=True), "z order is far -> near")
        by_name = {l["name"]: l for l in plan["layers"]}
        check(by_name["face"]["target"] == {"w": 1024, "h": 1024}, "face size override")
        check(by_name["eyebrow-l"]["target"]["w"] == 256, "eyebrow long side is 256")
        check(by_name["eyebrow-l"]["target"]["h"] % 8 == 0, "target is multiple of 8")
        check(by_name["face"]["display_box"] == [0, 0, 1024, 1024], "display box scaled")

        plan2 = core_plan.build_plan(
            LAYERS_JSON, tmp, select="face, front hair", size_rules_text="",
            out_dir="SeeThrough/UpScale2", canvas_size=1024,
        )
        check(plan2["selected"] == ["front hair", "face"] or
              sorted(plan2["selected"]) == ["face", "front hair"], "select by name")
        check(plan2["total"] == 2, "two layers selected")

        plan3 = core_plan.build_plan(
            LAYERS_JSON, tmp, select="all", exclude="*wear,face",
            size_rules_text="", out_dir="SeeThrough/UpScale3", canvas_size=1024,
        )
        names = set(plan3["selected"])
        check("face" not in names and "topwear" not in names, "exclude works")
        check("front hair" in names, "unexcluded layers kept")

        # Round trip through text.
        text = core_plan.plan_to_text(plan)
        again = core_plan.plan_from_text(text)
        check(again["total"] == plan["total"], "plan text round trip")


def test_auto_out_dir():
    print("[auto out dir]")
    root = "D:/out" if os.name != "nt" else "D:\\out"
    check(core_plan.normalize_out_dir("") == "", "blank is auto")
    check(core_plan.normalize_out_dir("SeeThrough/UpScale") == "", "legacy default is auto")
    check(core_plan.normalize_out_dir("a/b") == "a/b", "explicit path kept")
    check(core_plan.auto_out_dir(os.path.join(root, "SeeThrough", "角色A"), root)
          == "SeeThrough/角色A/UpScale", "follows the split folder")
    check(core_plan.auto_out_dir(root, root) == "UpScale", "split at the root")
    outside = "C:\\elsewhere" if os.name == "nt" else os.path.join(os.sep, "elsewhere")
    check(core_plan.auto_out_dir(outside, root) == "UpScale", "outside root falls back")
    check(core_plan.auto_out_dir(None, root) == "UpScale", "unknown source falls back")
    check(core_plan.resolve_out_dir("custom/dir", os.path.join(root, "x"), root) == "custom/dir",
          "explicit out_dir wins")
    check(core_plan.resolve_out_dir("", os.path.join(root, "x"), root) == "x/UpScale",
          "blank resolves against the source")

    if not os.path.isfile(LAYERS_JSON):
        check(False, "sample layers.json exists (%s)" % LAYERS_JSON)
        return
    src_dir = os.path.dirname(os.path.abspath(LAYERS_JSON))
    rel = os.path.relpath(src_dir, OUTPUT_ROOT).replace("\\", "/")
    with tempfile.TemporaryDirectory() as tmp:
        # Copy the sample split into a fresh output root so the derived folder
        # is created somewhere temporary.
        sub = os.path.join(tmp, rel.replace("/", os.sep))
        os.makedirs(sub, exist_ok=True)
        for name in os.listdir(src_dir):
            if name.endswith(".png") or name.endswith(".json"):
                shutil.copy2(os.path.join(src_dir, name), os.path.join(sub, name))
        plan = core_plan.build_plan(os.path.join(sub, os.path.basename(LAYERS_JSON)), tmp,
                                    select="face", size_rules_text="", canvas_size=1024)
        want = rel + "/UpScale"
        check(plan["out_dir"] == want, "plan writes into %s" % plan["out_dir"])
        check(plan["out_dir_auto"] is True, "flagged as auto")
        check(os.path.isdir(os.path.join(tmp, want.replace("/", os.sep))), "folder created")
        face = [l for l in plan["layers"] if l["name"] == "face"][0]
        check(face["target_file"] == want + "/face.png", "files sit in the same folder")


def test_directory_source():
    print("[directory source]")
    doc = core_plan.resolve_layers_source("SeeThrough", OUTPUT_ROOT)
    check(os.path.isfile(doc["path"]), "folder resolves to a layers.json")
    check(doc["path"].endswith("_layers.json"), "resolved name looks right")
    check(len(doc["layers"]) == 29, "29 layers found")
    check(not doc.get("synthetic"), "real layers.json used")

    with tempfile.TemporaryDirectory() as tmp:
        for name, size in (("a", (64, 32)), ("b", (128, 64)), ("a_depth", (64, 32))):
            arr = np.full((size[1], size[0], 4), 255, np.uint8)
            core_io.write_rgba(os.path.join(tmp, name + ".png"), arr)
        synth = core_plan.synthesize_doc_from_pngs(tmp)
        names = [l["name"] for l in synth["layers"]]
        check(names == ["a", "b"], "depth PNGs excluded, sorted: %s" % names)
        check(synth["width"] == 128 and synth["height"] == 64, "synthetic canvas size")
        check(synth.get("synthetic"), "marked synthetic")

    # An assembled _psdinfo.json must never be picked up as a source.
    with tempfile.TemporaryDirectory() as tmp:
        sub = os.path.join(tmp, "SeeThrough")
        os.makedirs(sub)
        fake = os.path.join(sub, "x_psdinfo.json")
        with open(fake, "w", encoding="utf-8") as f:
            f.write('{"upscaled": true, "layers": []}')
        check(core_plan.find_layers_json_in_dir(sub) is None, "psdinfo ignored")


def test_psd_roundtrip():
    print("[psd roundtrip]")
    rng = np.random.default_rng(1234)
    width = height = 96
    # bottom layer: solid blue square
    bottom = np.zeros((40, 50, 4), np.uint8)
    bottom[..., 0] = 20
    bottom[..., 2] = 200
    bottom[..., 3] = 255
    # top layer: red with a soft alpha ramp
    top = np.zeros((30, 30, 4), np.uint8)
    top[..., 0] = 230
    top[..., 3] = np.linspace(0, 255, 30, dtype=np.uint8)[None, :].repeat(30, 0)
    layers = [
        {"name": "bottom", "left": 5, "top": 10, "pixels": bottom},
        {"name": "上 层", "left": 20, "top": 30, "pixels": top},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for compression in ("rle", "raw"):
            path = os.path.join(tmp, "roundtrip_%s.psd" % compression)
            psd_writer.write_psd(path, layers, width, height, compression)
            check(os.path.getsize(path) > 0, "%s file written" % compression)

            doc = psd_reader.read_psd(path)
            check(doc["width"] == width and doc["height"] == height, "%s canvas size" % compression)
            check(len(doc["layers"]) == 2, "%s layer count" % compression)
            names = [l["name"] for l in doc["layers"]]
            check(names == ["bottom", "上 层"], "%s unicode layer names" % compression)

            l0 = doc["layers"][0]
            check((l0["left"], l0["top"]) == (5, 10), "%s layer offset" % compression)
            check(np.array_equal(l0["pixels"], bottom), "%s bottom pixels identical" % compression)

            l1 = doc["layers"][1]
            check(np.array_equal(l1["pixels"], top), "%s top pixels identical" % compression)

            # PIL must be able to *decode* the merged composite.  ``load()`` is
            # the part that actually walks the RLE row tables; opening alone
            # only reads the header.
            from PIL import Image
            with Image.open(path) as im:
                check(im.size == (width, height), "%s Pillow reads size" % compression)
                check(im.mode in ("RGBA", "RGB"), "%s Pillow mode %s" % (compression, im.mode))
                im.load()
                got = np.array(im.convert("RGBA"))
            expected = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            for layer in layers:
                expected.alpha_composite(
                    Image.fromarray(layer["pixels"], "RGBA"),
                    (layer["left"], layer["top"]),
                )
            expected_arr = np.array(expected)
            check(
                np.array_equal(got, expected_arr),
                "%s merged composite matches layers" % compression,
            )

    # PackBits literal runs must never reach 129 bytes: the 0x80 control byte
    # then decodes as a no-op and desynchronises the whole row.
    row = bytes((i * 37) % 256 for i in range(127)) + b"\x07\x07" + bytes(range(200))
    encoded = psd_writer.packbits_encode(row)
    check(psd_reader.unpackbits(encoded, len(row)) == row, "packbits literal-run boundary")


def test_hidden_groups():
    print("[psd hidden layers and folders]")
    width = height = 64

    def solid(size, left, top, color):
        px = np.zeros((size, size, 4), np.uint8)
        px[..., :3] = color
        px[..., 3] = 255
        return {"name": "x", "left": left, "top": top, "pixels": px}

    visible = solid(40, 0, 0, (0, 200, 0))
    hidden_layer = solid(20, 0, 0, (200, 0, 0))
    hidden_layer["name"] = "hidden_red"
    hidden_layer["hidden"] = True
    child = solid(10, 10, 10, (0, 0, 200))
    child["name"] = "child_blue"
    folder = {"type": "group", "name": "repair", "hidden": True,
              "children": [child]}
    layers = [visible, folder, hidden_layer]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "hidden.psd")
        psd_writer.write_psd(path, layers, width, height, "rle")
        doc = psd_reader.read_psd(path)
        names = [l["name"] for l in doc["layers"]]
        check("repair" in names and "child_blue" in names,
              "folder and child survive (%s)" % names)
        header = next(l for l in doc["layers"]
                      if l["name"] == "repair" and l["kind"] == "group_open")
        check(header["hidden"] is True, "a hidden folder stays hidden")
        red = next(l for l in doc["layers"] if l["name"] == "hidden_red")
        check(red["hidden"] is True and red["kind"] == "layer",
              "a hidden layer stays hidden")
        blue = next(l for l in doc["layers"] if l["name"] == "child_blue")
        check(blue["hidden"] is False,
              "the child's own flag is visible; its folder hides it")

        comp = doc["composite"]
        check(tuple(comp[5, 5, :3]) == (0, 200, 0),
              "hidden layers are absent from the composite (%s)"
              % (tuple(comp[5, 5, :3]),))
        check(comp[..., 3].max() == 255 and comp[50, 50, 3] == 0,
              "the composite keeps the visible layer only")

        from PIL import Image
        with Image.open(path) as im:
            im.load()
            pillow = np.array(im.convert("RGBA"))
        expected = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        expected.alpha_composite(
            Image.fromarray(visible["pixels"], "RGBA"), (0, 0))
        check(np.array_equal(pillow, np.array(expected)),
              "Pillow's merged image matches the visible layers")


def test_compose_with_real_assets():
    print("[compose with real SeeThrough assets]")
    if not os.path.isfile(LAYERS_JSON):
        check(False, "sample layers.json exists")
        return
    with tempfile.TemporaryDirectory() as tmp:
        plan = core_plan.build_plan(
            LAYERS_JSON, tmp, select="all", size_rules_text="",
            out_dir="SeeThrough/UpScale", canvas_size=512,
        )
        # Create a fake "upscaled" output for two small layers by copying the source.
        for name in ("eyebrow-l", "eyebrow-r"):
            layer = next(l for l in plan["layers"] if l["name"] == name)
            dst = os.path.join(tmp, layer["target_file"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            pix = core_io.read_rgba(layer["file"])
            core_io.write_rgba(dst, pix)

        canvas, psd_layers, notes = core_compose.build_canvas(plan, tmp)
        check(canvas.shape == (512, 512, 4), "canvas shape")
        check(canvas[..., 3].max() > 0, "canvas has visible pixels")
        check(len(psd_layers) > 10, "many layers assembled (%d)" % len(psd_layers))
        check(any(l["upscaled"] for l in psd_layers), "upscaled layer detected")

        path = os.path.join(tmp, "assembled.psd")
        psd_writer.write_psd(path, psd_layers, 512, 512, "rle")
        doc = psd_reader.read_psd(path)
        check(doc["width"] == 512, "assembled psd size")
        check(len(doc["layers"]) == len(psd_layers), "assembled psd layer count")
        print("  notes: %d" % len(notes))


def test_full_frame_mapping():
    print("[full frame vs cropped mapping]")
    with tempfile.TemporaryDirectory() as tmp:
        # Full-canvas source: art in a corner, stray noise-free.
        src = np.zeros((100, 100, 4), np.uint8)
        src[10:40, 10:40] = [200, 30, 10, 255]
        src_path = os.path.join(tmp, "src.png")
        core_io.write_rgba(src_path, src)

        # Model output: part occupies only a sub-region of its frame.
        up = np.zeros((200, 200, 4), np.uint8)
        up[20:80, 20:80] = [210, 40, 20, 255]
        up_path = os.path.join(tmp, "up.png")
        core_io.write_rgba(up_path, up)

        pixels, notes = core_compose.prepare_layer_pixels(
            src_path, up_path, 100, 100, 8, full_frame=True
        )
        check(pixels.shape[:2] == (100, 100), "full-frame keeps the whole frame")
        check(pixels[10, 10, 3] > 0, "full-frame content stays in place")

        pixels2, _ = core_compose.prepare_layer_pixels(
            src_path, up_path, 100, 100, 8, full_frame=False
        )
        check(pixels2.shape[:2] == (100, 100), "cropped layer resized into its box")
        check(pixels2[0, 0, 3] > 0, "cropped layer fills its box")


def test_compose_tensors():
    print("[tensor helpers]")
    import torch
    t = torch.zeros((1, 4, 5, 3), dtype=torch.float32)
    t[..., 0] = 1.0
    rgba = core_io.tensor_to_rgba(t)
    check(rgba.shape == (4, 5, 4), "tensor -> rgba shape")
    check(rgba[0, 0, 0] == 255 and rgba[0, 0, 3] == 255, "tensor rgb/alpha")

    t4 = torch.zeros((1, 2, 2, 4), dtype=torch.float32)
    t4[..., 3] = 0.5
    rgba4 = core_io.tensor_to_rgba(t4)
    check(rgba4[0, 0, 3] == 128, "4-channel alpha kept")

    mask = torch.ones((1, 3, 3), dtype=torch.float32) * 0.25
    alpha = core_io.mask_to_alpha(mask)
    check(alpha[0, 0] == 191, "mask 1=transparent conversion")


def main():
    test_safe_names()
    test_plan()
    test_auto_out_dir()
    test_directory_source()
    test_psd_roundtrip()
    test_hidden_groups()
    test_compose_with_real_assets()
    test_full_frame_mapping()
    test_compose_tensors()
    print()
    if _failures:
        print("FAILED: %d" % len(_failures))
        for f in _failures:
            print("  - %s" % f)
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
