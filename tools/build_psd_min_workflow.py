"""Regenerate the minimal PSD-upscale workflow shipped in the workflow tree.

    python custom_nodes/ComfyUI-SeeThrough-TileFlow/tools/build_psd_min_workflow.py

RUN THIS ONLY WHEN the node schema changes.  It rebuilds the file from
``TEMPLATE_VALUES`` and therefore RESETS the values a session has tuned
(``select``, seed, prompt, paths) -- the checked workflow carries the user's
current selection, which a plain rerun would wipe.  For a node-only addition
prefer editing the saved JSON directly.

Mirrors the psd-edit flow: a loader, a user-input prepare node, and an
execute-only node.  Widget values are derived from each node's
``INPUT_TYPES()``, so adding a widget in ``nodes_psd_min.py`` keeps this
workflow aligned.  The frontend restores ``widgets_values`` BY POSITION: a
hand-written array that drifts by one slot silently shifts every later value
onto the wrong widget (that is how ``select`` once ended up holding the
prompt).  ComfyUI also appends ``control_after_generate`` after ``seed`` at
runtime, so the seed is followed by ``"fixed"`` here to match the real order.
"""

from __future__ import annotations

import json
import os
import sys

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from nodes_psd_min import STLU_PsdUpscaleMin, STLU_PsdUpscalePrepare  # noqa: E402

OUT = os.path.join(
    COMFY_ROOT, "user", "default", "workflows", "Comfy-My-Workflow", "图片",
    "Live2D拆层", "seethrough-psd-upscale.json",
)

UNET = "qwen_image_2.1_turbo_int8_convrot.safetensors"
CLIP = "qwen3vl_8b_int8_convrot.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"
LINKS = ("MODEL", "CLIP", "VAE")
# Template-only overrides: the saved graph keeps the current session's PSD,
# reference, output location and 4x factor, while the node defaults stay at
# psd_upscale.DEFAULT_*.
TEMPLATE_VALUES = {
    "psd_path": r"D:\Development\AI\ComfyUI\ComfyUI\output\SeeThrough\我的自设\1024_3_Tiles_edited.psd",
    "reference_path": r"\\Z4PROPLUS-5U7K\sata11-189XXXX5105\Media\AI\图片\我的自设\游戏少女\1-初版\Design_4K_Slim.png",
    "factor": 4.0,
    "seed": 686801554,
    "out_dir": "SeeThrough/我的自设",
    "sub_dir": "PsdUpscale",
}
DEFAULT_TEMPLATE = "qwen_image_edit_2.1"

NOTE = (
    "### PSD 放大（精简版）\n\n"
    "**用法**\n\n"
    "1. 加载器已填好常用的 UNET / CLIP / VAE，换模型时只改三个下拉框。\n"
    "2. 『放大准备』点『选择成品 PSD』，勾选要放大的图层（默认全选），调好倍率与单层上限；"
    "运行后把 `plan_json` 交给『放大执行』。\n"
    "3. 『参考原图』可选：选一张与 PSD 同构图的高分辨率原图，发丝与边缘细节就照真实像素修；"
    "不选则只靠模型补细节。\n"
    "4. 轮廓直接沿用原图层：裁剪图以带透明通道的 RGBA 送进模型，模型放大后返回自己的透明通道，"
    "不再做任何二次切割。\n"
    "5. 种子只在这里填，每层种子 = 种子 + 层序号，逐层递增。同一张 PSD 用同一种子结果可复现。\n"
    "6. 『输出位置』可改根目录（output 下的相对路径）、子目录和 PSD 名称；不填名称则用 `<原文件名>_upscaled`。"
    "默认在 `output/SeeThrough/PsdUpscale/<PSD 名>/`，完成后可在面板里打开目录。\n"
    "7. 『负向提示词』只在 CFG > 1 时生效：默认 CFG 1 最快，填了负向结果也不会变；"
    "要压制某种效果就把 CFG 调到 2-4，代价是每步算两遍、速度约翻倍，太高会过饱和。\n"
    "**固定参数**：denoise 1.0、8 步 sigmas、euler；cfg 默认 1，可调。\n\n"
    "默认提示词已允许修复 AI 瑕疵（噪点、锯齿、模糊边、断裂发丝），并明确要求输出带透明通道、"
    "背景完全透明（开头结尾各强调一次），降低模型输出白底卡片的概率；"
    "裁剪图以带透明通道的 RGBA 送进模型，轮廓因此沿用原图层、由模型放大后原样返回。"
    "需要改这些行为，直接说需求，不在工作流里加控件。"
)


def widget(name, type_, shape=None):
    if isinstance(type_, (list, tuple)):
        # Combos save as the generic COMBO type in the UI format.
        type_ = "COMBO"
    item = {"localized_name": name, "name": name, "type": type_,
            "widget": {"name": name}, "link": None}
    if shape is not None:
        item["shape"] = shape
    return item


def link_in(name, type_, link):
    return {"localized_name": name, "name": name, "type": type_, "link": link}


def _widgets_from(required, skip_links=True, control_seed=True):
    """Widget order exactly as the frontend builds it."""
    values, named, inputs = [], {}, []
    for name, spec in required.items():
        type_ = spec[0]
        if skip_links and type_ in LINKS:
            continue
        opts = spec[1] if len(spec) > 1 else {}
        default = TEMPLATE_VALUES.get(name, opts.get("default"))
        values.append(default)
        named[name] = default
        inputs.append(widget(name, type_, 7 if opts.get("multiline") else None))
        if control_seed and name == "seed":
            # ComfyUI auto-adds control_after_generate right after ``seed``.
            values.append("fixed")
            named["control_after_generate"] = "fixed"
    return values, named, inputs


def prepare_widgets():
    return _widgets_from(STLU_PsdUpscalePrepare.INPUT_TYPES()["required"])


def min_widgets():
    """plan_json is linked but STILL keeps its widgets_values slot.

    ComfyUI restores ``widgets_values`` by position and includes a widget that
    has been converted to a (linked) input -- the psd-edit ``STLU_LayerEdit``
    node proves it: its linked ``plan_json`` occupies slot 0 ahead of
    ``edit_template``.  So only the true link types (MODEL/CLIP/VAE) are
    skipped here; ``plan_json`` keeps its default (``""``) slot.
    """
    required = STLU_PsdUpscaleMin.INPUT_TYPES()["required"]
    values, named = [], {}
    for name, spec in required.items():
        type_ = spec[0]
        if type_ in LINKS:
            continue
        opts = spec[1] if len(spec) > 1 else {}
        default = TEMPLATE_VALUES.get(name, opts.get("default"))
        values.append(default)
        named[name] = default
    return values, named


def build():
    prep_values, prep_named, prep_inputs = prepare_widgets()
    min_values, min_named = min_widgets()
    return {
        "id": "b7d1e4a2-0000-4000-8000-000000000104",
        "revision": 0,
        "last_node_id": 4,
        "last_link_id": 4,
        "nodes": [
            {
                "id": 1,
                "type": "STLU_QwenEditLoader",
                "pos": [0, 0],
                "size": [380, 170],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [widget("unet_name", "COMBO"),
                           widget("clip_name", "COMBO"),
                           widget("vae_name", "COMBO")],
                "outputs": [
                    {"localized_name": "MODEL", "name": "MODEL", "type": "MODEL",
                     "links": [1]},
                    {"localized_name": "CLIP", "name": "CLIP", "type": "CLIP",
                     "links": [2]},
                    {"localized_name": "VAE", "name": "VAE", "type": "VAE",
                     "links": [3]},
                ],
                "properties": {"Node name for S&R": "STLU_QwenEditLoader"},
                "widgets_values": [UNET, CLIP, VAE],
                "widgets_values_named": {"unet_name": UNET, "clip_name": CLIP,
                                         "vae_name": VAE},
            },
            {
                "id": 2,
                "type": "STLU_PsdUpscalePrepare",
                "pos": [440, 0],
                "size": [560, 760],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": prep_inputs,
                "outputs": [
                    {"localized_name": "plan_json", "name": "plan_json",
                     "type": "STRING", "links": [4]},
                    {"localized_name": "info", "name": "info",
                     "type": "STRING", "links": []},
                ],
                "title": "PSD 放大准备",
                "properties": {"Node name for S&R": "STLU_PsdUpscalePrepare"},
                "widgets_values": prep_values,
                "widgets_values_named": prep_named,
            },
            {
                "id": 3,
                "type": "STLU_PsdUpscaleMin",
                "pos": [1060, 0],
                "size": [420, 210],
                "flags": {},
                "order": 2,
                "mode": 0,
                "inputs": [
                    link_in("model", "MODEL", 1),
                    link_in("clip", "CLIP", 2),
                    link_in("vae", "VAE", 3),
                    {"localized_name": "plan_json", "name": "plan_json",
                     "type": "STRING", "widget": {"name": "plan_json"},
                     "link": 4},
                    widget("edit_template", "COMBO"),
                ],
                "outputs": [
                    {"localized_name": "psd_path", "name": "psd_path",
                     "type": "STRING", "links": []},
                    {"localized_name": "info", "name": "info",
                     "type": "STRING", "links": []},
                ],
                "title": "PSD 放大执行",
                "properties": {"Node name for S&R": "STLU_PsdUpscaleMin"},
                "widgets_values": min_values,
                "widgets_values_named": min_named,
            },
            {
                "id": 4,
                "type": "MarkdownNote",
                "pos": [-384, 320],
                "size": [760, 340],
                "flags": {},
                "order": 3,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "title": "使用说明",
                "properties": {"Node name for S&R": "MarkdownNote"},
                "widgets_values": [NOTE],
                "widgets_values_named": {"text": NOTE},
                "color": "#432",
                "bgcolor": "#653",
            },
        ],
        "links": [
            [1, 1, 0, 3, 0, "MODEL"],
            [2, 1, 1, 3, 1, "CLIP"],
            [3, 1, 2, 3, 2, "VAE"],
            [4, 2, 0, 3, 3, "STRING"],
        ],
        "groups": [
            {"id": 1, "title": "① 模型加载（Qwen Image Edit）",
             "bounding": [-30, -60, 440, 300], "color": "#4a3f5a", "flags": {}},
            {"id": 2, "title": "② 输入（放大准备）",
             "bounding": [410, -60, 620, 820], "color": "#3f5a3f",
             "flags": {}},
            {"id": 3, "title": "③ 放大执行",
             "bounding": [1040, -60, 470, 300], "color": "#3f4a5a",
             "flags": {}},
        ],
        "config": {},
        "extra": {"frontendVersion": "1.55.18",
                  "ds": {"scale": 1.0, "offset": [200, 200]}},
        "version": 0.4,
    }


def main():
    doc = build()
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    prep, mn = doc["nodes"][1], doc["nodes"][2]
    print("written:", OUT)
    print("prepare widgets:", len(prep["widgets_values"]))
    print("min widgets:", len(mn["widgets_values"]),
          json.dumps(mn["widgets_values"], ensure_ascii=False))


if __name__ == "__main__":
    main()
