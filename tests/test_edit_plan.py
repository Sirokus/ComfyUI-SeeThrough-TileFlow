"""CPU-only checks for the PSD per-layer edit workflow (no model, no GPU).

Covers the planning layer (``edit_plan``) and the expansion branch that turns a
crop-mode part into a ``STLU_CutToLayer`` tail (the model's own alpha):

    python custom_nodes/ComfyUI-SeeThrough-TileFlow/tests/test_edit_plan.py
"""

from __future__ import annotations

import json
import os
import shutil
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
import edit_expand  # noqa: E402
import edit_parts  # noqa: E402
import edit_plan  # noqa: E402
import psd_upscale  # noqa: E402
import psd_writer  # noqa: E402

TEMPLATE = edit_parts.default_template_path()

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
        {"name": "back hair", "left": 0, "top": 0,
         "pixels": solid(32, 32, (60, 40, 120, 255))},
        {"name": "body", "left": 0, "top": 8,
         "pixels": solid(32, 24, (200, 40, 40, 255))},
        {"name": "face", "left": 8, "top": 8,
         "pixels": solid(16, 16, (250, 220, 190, 255))},
    ]
    psd_writer.write_psd(path, layers, 32, 32)
    return path


def test_parse_tasks():
    rows = edit_plan.parse_tasks(
        "# a comment\n"
        "face | 换成微笑 | | lock\n"
        "hairpin | 加一个发卡 | | 原生 | 4,4,10,10 | face\n"
        "eye | 改成红色 | | 绿幕\n"
    )
    check(len(rows) == 3, "three text rows parsed")
    check(rows[0]["silhouette"] == edit_plan.SILHOUETTE_LOCK, "lock alias")
    check(rows[1]["silhouette"] == edit_plan.SILHOUETTE_ALPHA, "alpha alias")
    check(rows[2]["silhouette"] == edit_plan.SILHOUETTE_ALPHA,
          "a legacy key mode falls back to alpha")
    check(rows[1]["box"] == [4, 4, 14, 14],
          "x,y,w,h box becomes l,t,r,b: %r" % (rows[1]["box"],))
    check(rows[1]["insert"] == "face", "insert column kept")

    rows = edit_plan.parse_tasks(
        "hairpin | 加发卡 | a.png ; b.png | key\n")
    check(rows[0]["ref_files"] == ["a.png", "b.png"],
          "several refs split on ';': %r" % (rows[0]["ref_files"],))
    check(rows[0]["ref_file"] == "a.png", "first ref kept as ref_file")

    data = json.loads(json.dumps({"tasks": [{
        "name": "body", "instruction": "换成红色外套",
        "silhouette": "绿幕", "box": {"x": 1, "y": 2, "w": 8, "h": 9},
        "ref_files": ["p1.png", "p2.png"],
    }]}))
    rows = edit_plan.parse_tasks(json.dumps(data))
    check(len(rows) == 1 and rows[0]["prompt"] == "换成红色外套",
          "JSON task: instruction -> prompt")
    check(rows[0]["ref_files"] == ["p1.png", "p2.png"],
          "JSON ref_files list kept")
    check(rows[0]["box"] == [1, 2, 9, 11], "dict box converted")


def test_insert_index():
    layers = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    check(edit_plan._insert_index(layers, "") == 3, "empty insert -> top")
    check(edit_plan._insert_index(layers, "top") == 3, "top -> last")
    check(edit_plan._insert_index(layers, "bottom") == 0, "bottom -> first")
    check(edit_plan._insert_index(layers, "b") == 2, "after the named layer")
    check(edit_plan._insert_index(layers, "zzz") == 3, "unknown name -> top")


def test_restore_crop():
    # A 4x4 layer box inside a 8x8 crop that sits at (2, 6) on the canvas.
    # The crop is straight RGBA: alpha is the silhouette, colours pass through.
    crop = np.zeros((8, 8, 4), np.uint8)
    crop[3:5, 3:5] = (200, 10, 10, 255)  # the part
    out = edit_plan.restore_crop(
        crop, crop_size=[8, 8], crop_box=[2, 6, 10, 14],
        box=[4, 8, 8, 12], size=[4, 4], target=[4, 4],
    )
    check(out.shape == (4, 4, 4), "restored at the target size")
    # box(4,8) is 2px right / 2px down of the crop origin -> part at (1,1).
    check(out[1, 1, 3] == 255, "part survives the paste")
    check(out[0, 0, 3] == 0, "empty crop stays transparent")
    check(int(out[1, 1, 0]) >= 180, "colours pass through unchanged")


def test_full_plan():
    tmp = tempfile.mkdtemp(prefix="stlu_editplan_")
    try:
        psd = build_psd(os.path.join(tmp, "source.psd"))
        out_abs = os.path.join(tmp, "SeeThrough", "PsdEdit", "source")
        refs = []
        for i in range(2):
            ref_path = os.path.join(tmp, "pattern%d.png" % i)
            core_io.write_rgba(ref_path, solid(8, 8, (20 * i, 200, 90, 255)))
            refs.append(ref_path)
        tasks = edit_plan.parse_tasks(
            "face | 微笑 | | lock\n"
            "body | 红色外套 | | alpha\n"
            "hairpin | 加一个发卡 | %s;%s | alpha | 2,2,12,12 | face\n"
            % (refs[0], refs[1])
        )
        plan = edit_plan.prepare_edit_plan(psd, out_abs, tasks)

        names = [layer["name"] for layer in plan["layers"]]
        check("hairpin" in names, "new layer inserted: %r" % names)
        check(names.index("hairpin") == names.index("face") + 1,
              "hairpin sits right above face: %r" % names)
        check(plan["count"] == 3, "three selected layers")

        hairpin = next(l for l in plan["layers"] if l["name"] == "hairpin")
        check(hairpin.get("new") is True, "new layer flagged")
        src = core_io.read_rgba(hairpin["file"])
        check(src[..., 3].max() == 0, "new-layer source PNG is transparent")
        crop = core_io.read_rgba(hairpin["crop_file"])
        check(crop[..., 3].max() == 0,
              "new layer paints on a transparent canvas (no key colour)")

        jobs = {job["stem"]: job for job in plan["crops"]}
        check(len(plan["crops"]) == 3, "one crop per task")
        face = next(l for l in plan["layers"] if l["name"] == "face")
        check("key" not in jobs[face["stem"]],
              "crop jobs carry no key settings")

        check(hairpin["edit"]["ref_files"] == refs,
              "task refs stored in order: %r" % (hairpin["edit"]["ref_files"],))
        check(jobs[hairpin["stem"]]["ref_files"] == refs,
              "crop job carries the task refs")
        check(jobs[face["stem"]]["ref_files"] == [],
              "a task without refs has none")
        check(plan["max_task_refs"] == edit_plan.MAX_TASK_REFS,
              "no global ref -> full range of task ref slots")

        modes = {t["name"]: t["silhouette"] for t in plan["tasks"]}
        check(modes["body"] == "alpha" and modes["face"] == "lock",
              "per-task silhouette stored")

        for layer in (face, hairpin):
            text = layer["edit"]["prompt"].lower()
            check("alpha" in text and "transparent" in text,
                  "prompt asks for a real alpha channel: %r" % (text[:60],))

        info = psd_upscale.assemble_psd(plan, os.path.join(out_abs, "out.psd"))
        check(sorted(info["upscaled"]) == ["back hair", "body", "face", "hairpin"]
              or "hairpin" in info["upscaled"] + info["fallback"],
              "assemble knows about the new layer: %r" % (info,))
        print("full plan ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_crop_expand():
    check(os.path.isfile(TEMPLATE), "template missing: %s" % TEMPLATE)
    geometry = {
        "crop_size": [64, 64], "crop_box": [0, 0, 64, 64],
        "box": [8, 8, 56, 56], "size": [64, 64], "target": [64, 64],
        "out_file": "layer.png", "file": "source.png",
    }
    base = {
        "crop_file": "crop.png", "width": 64, "height": 64,
        "resolution": 64, "geometry": geometry,
    }
    parts = [
        dict(base, name="existing", prompt="existing", ref_file="global.png",
             ref_files=["p1.png", "p2.png"], silhouette="alpha"),
        dict(base, name="new part", prompt="new part", ref_file="",
             ref_files=["p3.png"], silhouette="alpha"),
        dict(base, name="locked", prompt="locked", ref_file="",
             ref_files=[], silhouette="lock"),
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
    check(classes.count("STLU_CutToLayer") == 3, "one cut per crop part")
    check("STLU_KeyCut" not in classes, "crop mode never keys by hand")
    check("STLU_AlphaCut" not in classes, "the model's own alpha is used")
    check("STLU_ChromaToAlpha" not in classes, "no colour-key node survives")

    cuts = [n for n in expand.values() if n["class_type"] == "STLU_CutToLayer"]
    check(all(isinstance(n["inputs"].get("geometry"), str) for n in cuts),
          "every cut carries the crop geometry")
    check(all(n["inputs"].get("alpha") is None for n in cuts),
          "the cut reads the model's alpha, not a separate mask")
    check(not any("key" in str(k).lower() for n in expand.values()
                  for k in (n.get("inputs") or {})),
          "no key settings reach the graph")

    stray = [(node_id, name) for node_id, node in expand.items()
             for name, value in (node.get("inputs") or {}).items()
             if hasattr(value, "class_type")]
    check(not stray, "every wire is a link: %r" % (stray[:3],))

    def encoder_inputs(prefix):
        for node_id, node in expand.items():
            if prefix + "_" in node_id and \
                    node["class_type"] == "TextEncodeQwenImage21":
                return node["inputs"]
        return {}

    def ref_file_of(inputs, slot):
        link = inputs.get("images.image_%d" % slot)
        return expand[link[0]]["inputs"].get("file") if link else None

    ep0 = encoder_inputs("ep0")
    check(ref_file_of(ep0, 2) == "global.png",
          "global reference owns <image2>")
    check(ref_file_of(ep0, 3) == "p1.png" and ref_file_of(ep0, 4) == "p2.png",
          "task refs continue at <image3>..: %r"
          % ([ref_file_of(ep0, n) for n in (2, 3, 4)],))
    check("images.image_5" not in ep0, "unused slots are not wired")

    ep1 = encoder_inputs("ep1")
    check(ref_file_of(ep1, 2) == "p3.png",
          "without a global ref the task refs start at <image2> (前移)")
    check("images.image_3" not in ep1, "no second slot for one ref")

    ep2 = encoder_inputs("ep2")
    check("images.image_2" not in ep2 and "images.image_3" not in ep2,
          "no refs -> no image_2.. slots")
    print("crop expand ok (%d nodes)" % len(expand))


def test_skip_existing():
    tmp = tempfile.mkdtemp(prefix="stlu_editskip_")
    try:
        psd = build_psd(os.path.join(tmp, "source.psd"))
        out_abs = os.path.join(tmp, "run")
        tasks = edit_plan.parse_tasks(
            "face | 微笑 | | lock\n"
            "body | 红色外套 | | key | | | 1\n")
        check(tasks[0]["force"] is False, "text force column defaults off")
        check(tasks[1]["force"] is True, "text 7th column forces a rerun")

        plan = edit_plan.prepare_edit_plan(psd, out_abs, tasks)
        status = {t["name"]: t["exists"] for t in plan["tasks"]}
        check(status == {"face": False, "body": False},
              "no PNG yet -> nothing exists: %r" % (status,))

        face = next(l for l in plan["layers"] if l["name"] == "face")
        core_io.write_rgba(face["out_file"], solid(4, 4, (1, 2, 3, 255)))
        check(edit_plan._layer_generated(face) is True,
              "a PNG on disk counts as generated")

        pending, skipped = edit_plan.split_pending(edit_plan.edit_layers(plan))
        check([l["name"] for l in skipped] == ["face"], "face is skipped")
        check([l["name"] for l in pending] == ["body"], "body still pending")

        face["edit"]["force"] = True
        pending, _skipped = edit_plan.split_pending(edit_plan.edit_layers(plan))
        check("face" in [l["name"] for l in pending],
              "强制重跑 overrides the skip")

        rows = edit_plan.parse_tasks(json.dumps({"tasks": [
            {"name": "face", "prompt": "x", "force_rerun": True},
            {"name": "body", "prompt": "y", "force": "yes"},
        ]}))
        check(rows[0]["force"] is True and rows[1]["force"] is True,
              "JSON force / force_rerun accepted")
        print("skip existing ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("[edit_plan: task parsing]")
    test_parse_tasks()
    print("[edit_plan: insertion order]")
    test_insert_index()
    print("[edit_plan: straight crop restore]")
    test_restore_crop()
    print("[edit_plan: full plan]")
    test_full_plan()
    print("[edit_plan: skip already generated layers]")
    test_skip_existing()
    print("[edit_expand: alpha crop expansion]")
    test_crop_expand()
    if _failures:
        print("\n%d FAILED" % len(_failures))
        for message in _failures:
            print("  - " + message)
        raise SystemExit(1)
    print("\nALL-OK")


if __name__ == "__main__":
    main()
