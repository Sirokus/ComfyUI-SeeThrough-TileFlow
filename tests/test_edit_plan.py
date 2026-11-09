"""CPU-only checks for the PSD per-layer edit workflow (no model, no GPU).

Covers the planning layer (``edit_plan``) and the expansion branch that turns a
``silhouette="key"`` task into a ``STLU_KeyCut`` tail:

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

TEMPLATE = os.path.join(
    COMFY_ROOT, "user", "default", "workflows", "Comfy-My-Workflow", "图片",
    "Live2D拆层", edit_parts.TEMPLATE_NAME,
)

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
        "hairpin | 加一个发卡 | | key | 4,4,10,10 | face\n"
    )
    check(len(rows) == 2, "two text rows parsed")
    check(rows[0]["silhouette"] == edit_plan.SILHOUETTE_LOCK, "lock alias")
    check(rows[1]["silhouette"] == edit_plan.SILHOUETTE_KEY, "key alias")
    check(rows[1]["box"] == [4, 4, 14, 14],
          "x,y,w,h box becomes l,t,r,b: %r" % (rows[1]["box"],))
    check(rows[1]["insert"] == "face", "insert column kept")

    data = json.loads(json.dumps({"tasks": [{
        "name": "body", "instruction": "换成红色外套",
        "silhouette": "绿幕", "box": {"x": 1, "y": 2, "w": 8, "h": 9},
    }]}))
    rows = edit_plan.parse_tasks(json.dumps(data))
    check(len(rows) == 1 and rows[0]["prompt"] == "换成红色外套",
          "JSON task: instruction -> prompt")
    check(rows[0]["box"] == [1, 2, 9, 11], "dict box converted")


def test_insert_index():
    layers = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    check(edit_plan._insert_index(layers, "") == 3, "empty insert -> top")
    check(edit_plan._insert_index(layers, "top") == 3, "top -> last")
    check(edit_plan._insert_index(layers, "bottom") == 0, "bottom -> first")
    check(edit_plan._insert_index(layers, "b") == 2, "after the named layer")
    check(edit_plan._insert_index(layers, "zzz") == 3, "unknown name -> top")


def test_restore_keyed():
    # A 4x4 layer box inside a 8x8 crop that sits at (2, 6) on the canvas.
    # The crop is already keyed: only the part survives, the rest is clear.
    crop = np.zeros((8, 8, 4), np.uint8)
    crop[3:5, 3:5] = (200, 10, 10, 255)  # the part
    out = edit_plan.restore_keyed(
        crop, crop_size=[8, 8], crop_box=[2, 6, 10, 14],
        box=[4, 8, 8, 12], size=[4, 4], target=[4, 4],
    )
    check(out.shape == (4, 4, 4), "restored at the target size")
    # box(4,8) is 2px right / 2px down of the crop origin -> part at (1,1).
    check(out[1, 1, 3] == 255, "part survives the paste")
    check(out[0, 0, 3] == 0, "empty crop stays transparent")
    check(int(out[1, 1, 0]) >= 180, "colours are not white-un-composited")


def test_full_plan():
    tmp = tempfile.mkdtemp(prefix="stlu_editplan_")
    try:
        psd = build_psd(os.path.join(tmp, "source.psd"))
        out_abs = os.path.join(tmp, "SeeThrough", "PsdEdit", "source")
        tasks = edit_plan.parse_tasks(
            "face | 微笑 | | lock\n"
            "body | 红色外套 | | key\n"
            "hairpin | 加一个发卡 | | key | 2,2,12,12 | face\n"
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
        _key_id, key_rgb = edit_parts.resolve_key(plan["key"])
        check(tuple(int(v) for v in crop[0, 0, :3]) == tuple(key_rgb),
              "new layer paints on the key colour: %r" % (crop[0, 0, :3],))

        jobs = {job["stem"]: job for job in plan["crops"]}
        check(len(plan["crops"]) == 3, "one crop per task")
        check(jobs[hairpin["stem"]]["key_reference"] is False,
              "a new layer's crop is not a key reference")
        face = next(l for l in plan["layers"] if l["name"] == "face")
        check(jobs[face["stem"]]["key_reference"] is True,
              "an existing layer protects its own colours")

        modes = {t["name"]: t["silhouette"] for t in plan["tasks"]}
        check(modes["body"] == "key" and modes["face"] == "lock",
              "per-task silhouette stored")

        face_prompt = face["edit"]["prompt"]
        check("00FF00" not in face_prompt.upper(),
              "a locked task keeps its literal prompt")
        body = next(l for l in plan["layers"] if l["name"] == "body")
        check("00FF00" in body["edit"]["prompt"].upper(),
              "key task prompt names the key colour")

        info = psd_upscale.assemble_psd(plan, os.path.join(out_abs, "out.psd"))
        check(sorted(info["upscaled"]) == ["back hair", "body", "face", "hairpin"]
              or "hairpin" in info["upscaled"] + info["fallback"],
              "assemble knows about the new layer: %r" % (info,))
        print("full plan ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_key_expand():
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
        dict(base, name="existing", prompt="existing", ref_file="",
             silhouette="key", key_reference=True),
        dict(base, name="new part", prompt="new part", ref_file="",
             silhouette="key", key_reference=False),
        dict(base, name="locked", prompt="locked", ref_file="",
             silhouette="lock", key_reference=True),
    ]
    external = {"model": ["11", 0], "clip": ["12", 0], "vae": ["13", 0]}
    expand, _psd, _saves = edit_expand.build_expand(
        workflow_path=TEMPLATE, parts=parts, external=external, seed=7,
        out_dir_rel="SeeThrough/_test_key", layers_json_path="",
        psd_name="", psd_plan="plan.json", crop_mode=True,
        key="green", tolerance=12.0, softness=6.0, protect=True, despill=0.65,
        negative_prompt="", cfg=1.0, resolution=0, scheduler="euler",
        sigmas="1.0, 0.5, 0", width=0, height=0,
    )

    classes = [n["class_type"] for n in expand.values()]
    check(classes.count("STLU_KeyCut") == 2, "one key cut per keyed part")
    check(classes.count("STLU_CutToLayer") == 1, "the locked part is cut")
    check("STLU_ChromaToAlpha" not in classes, "crop mode never chroma-keys")

    keycuts = [n for n in expand.values() if n["class_type"] == "STLU_KeyCut"]
    with_ref = [n for n in keycuts if n["inputs"].get("reference") is not None]
    check(len(with_ref) == 1, "only the existing layer gets a reference")
    check(with_ref[0]["inputs"]["reference"][0] in expand,
          "the reference is a crop load link")
    check(all(n["inputs"].get("key_color") == "green" for n in keycuts),
          "key colour wired through")

    stray = [(node_id, name) for node_id, node in expand.items()
             for name, value in (node.get("inputs") or {}).items()
             if hasattr(value, "class_type")]
    check(not stray, "every wire is a link: %r" % (stray[:3],))
    print("key expand ok (%d nodes)" % len(expand))


def main():
    print("[edit_plan: task parsing]")
    test_parse_tasks()
    print("[edit_plan: insertion order]")
    test_insert_index()
    print("[edit_plan: keyed restore]")
    test_restore_keyed()
    print("[edit_plan: full plan]")
    test_full_plan()
    print("[edit_expand: keyed crop expansion]")
    test_key_expand()
    if _failures:
        print("\n%d FAILED" % len(_failures))
        for message in _failures:
            print("  - " + message)
        raise SystemExit(1)
    print("\nALL-OK")


if __name__ == "__main__":
    main()
