"""One-off generator for the ``Live2D层级二次放大-批量.json`` workflow.

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tools/build_batch_workflow.py

It reads the existing single-image workflow (read only) to copy the model
loader settings, then writes a new batch workflow next to it.  The original
file is never modified.
"""

from __future__ import annotations

import json
import os
import sys

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for _p in (PKG_DIR, COMFY_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nodes as stlu_nodes  # noqa: E402

SOURCE_WORKFLOW = os.path.join(
    COMFY_ROOT, "user", "default", "workflows", "Comfy-My-Workflow", "图片",
    "Live2D层级二次放大.json",
)
OUTPUT_WORKFLOW = os.path.join(
    COMFY_ROOT, "user", "default", "workflows", "Comfy-My-Workflow", "图片",
    "Live2D层级二次放大-批量.json",
)

DEFAULT_SIZE_RULES = stlu_nodes.DEFAULT_SIZE_RULES
DEFAULT_FOCUS_PROMPT = stlu_nodes.core_plan.DEFAULT_FOCUS_PROMPT

NOTE_TEXT = """## Live2D 分层批量二次放大 + 自动 PSD

**日常就三步：**

1. 在 `1. 选层 / 分辨率 / 运行` 面板里确认拆层文件夹（点 `浏览…`），
   勾选这次要 **AI 重绘** 的部件。没勾的部件仍会用原图放进 PSD，不耗时。
   分辨率默认跟随 PSD 画布（源图 1024 → 画布 2048 就是 2 倍）；
   个别部件在行内改长边，或用顶部 `已勾选：批量长边…` 一次设置。
2. 面板里的 `参考图` 一行选择 / 上传本次的全身参考图。
3. 点面板右下角 `▶ 开始放大`（或右上角运行）。运行时面板会显示逐层进度。

全部完成后 `3. 合成 PSD` 卡片里直接显示结果，可一键 `打开 PSD`。

- `预览合成` 是纯 CPU 离线合成，不跑 GPU，可先检查构图和当前选择。
- 空层（这个角色没有的翅膀 / 尾巴 / 鼻孔等）默认不勾选，不浪费采样。
- 左侧模型加载器一般不用改；`2. 批量放大` 会把每层克隆原工作流
  `Live2D层级二次放大.json` 里的子图模板，改了 prompt / 采样器会自动生效。
- 4096 画布很吃显存（这台机器有掉电/PCIe 记录），按需只对个别部件开。"""


def load_source():
    with open(SOURCE_WORKFLOW, "r", encoding="utf-8") as f:
        doc = json.load(f)
    subgraph = doc["definitions"]["subgraphs"][0]
    internal = {n["id"]: n for n in subgraph["nodes"]}
    return doc, subgraph, internal


def widget_values(node, count):
    values = list(node.get("widgets_values") or [])
    return values[:count]


def loader_widgets(internal, node_id, count, fallback):
    node = internal.get(node_id)
    if not node:
        return list(fallback)
    values = widget_values(node, count)
    return values if len(values) == count else list(fallback)


def build():
    doc, subgraph, internal = load_source()
    nodes = []
    links = []
    last_link = [100]

    def new_link(origin_id, origin_slot, target_id, target_slot, link_type):
        last_link[0] += 1
        links.append([last_link[0], origin_id, origin_slot, target_id, target_slot, link_type])
        return last_link[0]

    def input_entry(name, type_, link=None, widget=False, shape=None):
        entry = {"localized_name": name, "name": name, "type": type_, "link": link}
        if widget:
            entry["widget"] = {"name": name}
        if shape is not None:
            entry["shape"] = shape
        return entry

    def output_entry(name, type_, links_=None):
        return {"localized_name": name, "name": name, "type": type_, "links": links_}

    def add_node(node_id, class_type, pos, size, inputs, outputs, widgets, order,
                 title=None, widgets_named=None):
        properties = {"Node name for S&R": class_type}
        node = {
            "id": node_id,
            "type": class_type,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "properties": properties,
            "widgets_values": widgets,
        }
        if title:
            node["title"] = title
        if widgets_named:
            node["widgets_values_named"] = widgets_named
        nodes.append(node)
        return node

    # --- 1. Plan -----------------------------------------------------------
    plan_inputs = [
        input_entry("layers_json", "STRING", widget=True),
        input_entry("select", "STRING", widget=True),
        input_entry("exclude", "STRING", widget=True),
        input_entry("size_rules", "STRING", widget=True),
        input_entry("out_dir", "STRING", widget=True),
        input_entry("canvas_size", "INT", widget=True),
        input_entry("skip_empty", "BOOLEAN", widget=True),
        input_entry("empty_coverage", "FLOAT", widget=True),
        input_entry("focus_crop", "BOOLEAN", widget=True),
        input_entry("focus_margin", "INT", widget=True),
        input_entry("focus_context", "COMBO", widget=True),
        input_entry("focus_prompt", "STRING", widget=True),
    ]
    plan_outputs = [
        output_entry("plan", "STRING", []),
        output_entry("plan_path", "STRING", []),
        output_entry("total", "INT", []),
        output_entry("report", "STRING", []),
    ]
    plan_node = add_node(
        1001, "STLU_Plan", (-2200, -300), (780, 900),
        plan_inputs, plan_outputs,
        [
            "SeeThrough",
            "all",
            "",
            DEFAULT_SIZE_RULES,
            "",  # 留空 = 拆层文件夹下的 UpScale
            2048,
            True,
            0.0005,
            True,
            32,
            "backdrop",
            DEFAULT_FOCUS_PROMPT,
        ],
        0,
        title="1. 选层 / 分辨率 / 运行",
    )

    # --- 2. model loaders (copied from the subgraph) -----------------------
    unet = add_node(
        1002, "UNETLoader", (-1400, 300), (330, 82),
        [input_entry("unet_name", "COMBO", widget=True),
         input_entry("weight_dtype", "COMBO", widget=True)],
        [output_entry("MODEL", "MODEL", [])],
        loader_widgets(internal, 451, 2, ["qwen_image_2.1_turbo_int8_convrot.safetensors", "default"]),
        1,
    )
    cache = add_node(
        1003, "QwenImage21Cache", (-1020, 300), (300, 82),
        [input_entry("model", "MODEL"),
         input_entry("device", "COMBO", widget=True),
         input_entry("dtype", "COMBO", widget=True)],
        [output_entry("MODEL", "MODEL", [])],
        loader_widgets(internal, 469, 2, ["auto", "default"]),
        2,
    )
    clip = add_node(
        1004, "CLIPLoader", (-1400, 420), (330, 106),
        [input_entry("clip_name", "COMBO", widget=True),
         input_entry("type", "COMBO", widget=True),
         input_entry("device", "COMBO", widget=True)],
        [output_entry("CLIP", "CLIP", [])],
        loader_widgets(internal, 453, 3, ["qwen3vl_8b_int8_convrot.safetensors", "qwen_image", "default"]),
        3,
    )
    pe_clip = add_node(
        1005, "CLIPLoader", (-1400, 560), (330, 106),
        [input_entry("clip_name", "COMBO", widget=True),
         input_entry("type", "COMBO", widget=True),
         input_entry("device", "COMBO", widget=True)],
        [output_entry("CLIP", "CLIP", [])],
        loader_widgets(internal, 477, 3, ["qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors", "qwen_image", "default"]),
        4,
        title="CLIP (prompt enhancer)",
    )
    vae = add_node(
        1006, "VAELoader", (-1020, 420), (300, 60),
        [input_entry("vae_name", "COMBO", widget=True)],
        [output_entry("VAE", "VAE", [])],
        loader_widgets(internal, 454, 1, ["qwen_image_2.1_vae_bf16.safetensors"]),
        5,
    )
    ref = add_node(
        1007, "LoadImage", (-2200, 630), (330, 340),
        [input_entry("image", "COMBO", widget=True),
         input_entry("upload", "IMAGEUPLOAD", widget=True)],
        [output_entry("IMAGE", "IMAGE", []), output_entry("MASK", "MASK", None)],
        ["Qwen_image_2.1_00029_Improve_1K.png", "image"],
        6,
        title="全身参考图（每次运行选择）",
    )

    # --- 3. batch ----------------------------------------------------------
    batch_inputs = [
        input_entry("plan", "STRING", widget=True),
        input_entry("reference", "IMAGE"),
        input_entry("model", "MODEL"),
        input_entry("clip", "CLIP"),
        input_entry("vae", "VAE"),
        input_entry("plan_path", "STRING", widget=True),
        input_entry("pe_clip", "CLIP"),
        input_entry("seed", "INT", widget=True),
        input_entry("workflow_json", "STRING", widget=True),
    ]
    batch_outputs = [
        output_entry("last_saved", "STRING", []),
        output_entry("report", "STRING", []),
    ]
    batch_node = add_node(
        1008, "STLU_UpscaleBatch", (-600, 180), (420, 280),
        batch_inputs, batch_outputs,
        ["", "", 42, SOURCE_WORKFLOW],
        7,
        title="2. 批量放大（自动）",
    )

    # --- 4. PSD ------------------------------------------------------------
    psd_inputs = [
        input_entry("plan", "STRING", widget=True),
        input_entry("out_dir", "STRING", widget=True),
        input_entry("deps", "*"),
        input_entry("psd_filename", "STRING", widget=True),
        input_entry("fallback", "COMBO", widget=True),
        input_entry("write_layers_json", "BOOLEAN", widget=True),
        input_entry("alpha_threshold", "INT", widget=True),
    ]
    psd_outputs = [
        output_entry("psd_path", "STRING", []),
        output_entry("layers_json_path", "STRING", []),
        output_entry("report", "STRING", []),
        output_entry("preview", "IMAGE", []),
    ]
    psd_node = add_node(
        1009, "STLU_AssemblePSD", (-600, 500), (420, 380),
        psd_inputs, psd_outputs,
        ["", "", "", "both", True, 8],
        8,
        title="3. 合成 PSD（自动）",
    )

    # --- 5. compare + note -------------------------------------------------
    compare = add_node(
        1010, "ImageCompare", (60, 180), (420, 480),
        [input_entry("image_a", "IMAGE"), input_entry("image_b", "IMAGE"),
         input_entry("compare_view", "IMAGECOMPARE", widget=True)],
        [],
        [],
        9,
        title="原图 / PSD 预览",
    )
    add_node(
        1011, "MarkdownNote", (-1430, -300), (750, 500),
        [], [], [NOTE_TEXT], 10, title="说明",
    )

    # --- links -------------------------------------------------------------
    new_link(1002, 0, 1003, 0, "MODEL")
    new_link(1003, 0, 1008, 2, "MODEL")
    new_link(1004, 0, 1008, 3, "CLIP")
    new_link(1005, 0, 1008, 6, "CLIP")
    new_link(1006, 0, 1008, 4, "VAE")
    new_link(1007, 0, 1008, 1, "IMAGE")
    new_link(1007, 0, 1010, 0, "IMAGE")
    new_link(1001, 0, 1008, 0, "STRING")
    new_link(1001, 1, 1008, 5, "STRING")
    new_link(1001, 0, 1009, 0, "STRING")
    new_link(1008, 0, 1009, 2, "STRING")
    new_link(1009, 3, 1010, 1, "IMAGE")

    # Mirror the link table onto node inputs/outputs (the frontend expects it).
    by_id = {n["id"]: n for n in nodes}
    for link_id, origin, origin_slot, target, target_slot, link_type in links:
        if target in by_id and target_slot < len(by_id[target]["inputs"]):
            by_id[target]["inputs"][target_slot]["link"] = link_id
        if origin in by_id and origin_slot < len(by_id[origin]["outputs"]):
            slot = by_id[origin]["outputs"][origin_slot]
            if slot.get("links") is None:
                slot["links"] = []
            slot["links"].append(link_id)

    workflow = {
        "id": "a0b1c2d3-0000-4000-8000-000000000001",
        "revision": 0,
        "last_node_id": 1011,
        "last_link_id": last_link[0],
        "nodes": nodes,
        "links": links,
        "groups": [{
            "id": 1,
            "title": "模型加载器（一般不用改）",
            "bounding": [-1430, 260, 750, 470],
            "color": "#3f5a3f",
            "font_size": 20,
            "flags": {},
        }],
        "definitions": {"subgraphs": []},
        "config": {},
        "extra": {
            "frontendVersion": "1.55.14",
            # Frame the Plan panel on load (screen = (canvas + offset) * scale).
            "ds": {"scale": 0.65, "offset": [2261, 346]},
        },
        "version": 0.4,
    }
    with open(OUTPUT_WORKFLOW, "w", encoding="utf-8") as f:
        json.dump(workflow, f, ensure_ascii=False, indent=2)
    print("wrote %s (%d nodes, %d links)" % (OUTPUT_WORKFLOW, len(nodes), len(links)))
    return OUTPUT_WORKFLOW


if __name__ == "__main__":
    build()
