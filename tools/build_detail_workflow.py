"""Generate the CPU-only "参考图细节还原" batch workflow.

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tools/build_detail_workflow.py

Unlike ``build_batch_workflow.py`` this workflow has **no model loaders**: it
runs STLU_Plan -> STLU_DetailBatch -> STLU_AssemblePSD, so nothing is loaded
onto the GPU.  It rebuilds every layer from a higher-resolution full-body
reference of the same artwork (see ``core_reference.py``).
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

WORKFLOW_DIR = os.path.join(
    COMFY_ROOT, "user", "default", "workflows", "Comfy-My-Workflow", "图片",
    "Live2D拆层",
)
BATCH_WORKFLOW = os.path.join(WORKFLOW_DIR, "Live2D层级二次放大-批量.json")
OUTPUT_WORKFLOW = os.path.join(WORKFLOW_DIR, "Live2D层级细节还原-批量.json")
# 就放在 AI 结果的 UpScale 目录里，找的时候不用再猜；子目录避免同名图层互相覆盖。
DETAIL_OUT_DIR = "SeeThrough/我的自设/UpScale/细节还原"

NOTE_TEXT = """## Live2D 分层「参考图细节还原」+ 自动 PSD（纯 CPU，不跑 GPU）

**适用**：参考图是同一张图的更高分辨率版本（例如全身参考 4096，拆层 1024）。

它不跑扩散模型：每层的 alpha 取自拆层，颜色从高分辨率参考图的对应位置 1:1 取；
线条、阴影、瞳孔反光都是原画本身的像素，不是模型猜出来的。

1. 在 `1. 选层 / 分辨率 / 运行` 面板里确认拆层文件夹；`参考图` 一行选 4K 原图。
2. PSD 画布默认 4096 = 参考图原生分辨率（1024 拆层 ×4，最大真实细节）。
3. 点 `▶ 开始放大`。所有非空层都会重建，几十秒内完成，不占显存。

- **轮廓**：alpha 直接来自拆层；`alpha_sharpen`（默认 1.6）只把 0.5 等值线附近的
  过渡收窄，不移动轮廓位置。放大倍率越高越该开大（4× 建议 1.4–2.0）。
- **脏色**：透明区的残留颜色不会渗进边缘（预乘 alpha 重采样）；参考图只有在
  本层 alpha 之内、且不是「另一种高饱和颜色」时才被采信（挡住本层的上层部件不会串色）。
- **color_tolerance**：调小=更保守（只信本色）；调大=把更多邻近色也算作本层。
- **alpha_floor**：清掉拆层留下的极淡噪点（默认 8；再高可能削到最外圈羽化边）。
- 输出在 `UpScale/细节还原/`：`seethrough_detail.psd` + 每层 PNG + 预览图
  （与 AI 放大结果分开存放，互不覆盖）。
"""


def load_batch():
    with open(BATCH_WORKFLOW, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    base = load_batch()
    by_type = {}
    for node in base["nodes"]:
        by_type.setdefault(node["type"], []).append(node)

    plan = json.loads(json.dumps(by_type["STLU_Plan"][0]))
    load = json.loads(json.dumps(by_type["LoadImage"][0]))
    assemble = json.loads(json.dumps(by_type["STLU_AssemblePSD"][0]))
    compare = json.loads(json.dumps(by_type["ImageCompare"][0]))
    note = json.loads(json.dumps(by_type["MarkdownNote"][0]))

    # Detail workflow: select everything, ignore the AI size rules.
    plan["widgets_values"] = [
        "SeeThrough/我的自设", "all", "", "{}", DETAIL_OUT_DIR, 4096,
        True, 0.0005, True, 32, "reference",
        stlu_nodes.core_plan.DEFAULT_FOCUS_PROMPT,
    ]
    plan["widgets_values_named"] = {
        "layers_json": "SeeThrough/我的自设",
        "select": "all",
        "exclude": "",
        "size_rules": "{}",
        "out_dir": DETAIL_OUT_DIR,
        "canvas_size": 4096,
        "skip_empty": True,
        "empty_coverage": 0.0005,
        "focus_crop": True,
        "focus_margin": 32,
        "focus_context": "reference",
        "focus_prompt": stlu_nodes.core_plan.DEFAULT_FOCUS_PROMPT,
    }
    plan["outputs"][0]["links"] = [101, 103]   # plan -> DetailBatch, Assemble
    plan["outputs"][1]["links"] = [102]        # plan_path -> DetailBatch
    plan["outputs"][2]["links"] = []
    plan["outputs"][3]["links"] = []

    load["outputs"][0]["links"] = [104, 105]   # -> DetailBatch, ImageCompare
    load["outputs"][1]["links"] = []
    load["widgets_values"] = ["Qwen_image_2.1_00029_mouse.png", "image"]
    load["widgets_values_named"] = {
        "image": "Qwen_image_2.1_00029_mouse.png", "upload": "image",
    }

    detail = {
        "id": 1008,
        "type": "STLU_DetailBatch",
        "pos": [-600, 180],
        "size": [420, 240],
        "flags": {},
        "order": 8,
        "mode": 0,
        "inputs": [
            {"localized_name": "plan", "name": "plan", "type": "STRING",
             "widget": {"name": "plan"}, "link": 101},
            {"localized_name": "reference", "name": "reference", "type": "IMAGE",
             "link": 104},
            {"localized_name": "plan_path", "name": "plan_path", "shape": 7,
             "type": "STRING", "widget": {"name": "plan_path"}, "link": 102},
            {"localized_name": "only_selected", "name": "only_selected", "shape": 7,
             "type": "BOOLEAN", "widget": {"name": "only_selected"}, "link": None},
            {"localized_name": "color_tolerance", "name": "color_tolerance", "shape": 7,
             "type": "FLOAT", "widget": {"name": "color_tolerance"}, "link": None},
            {"localized_name": "alpha_sharpen", "name": "alpha_sharpen", "shape": 7,
             "type": "FLOAT", "widget": {"name": "alpha_sharpen"}, "link": None},
            {"localized_name": "alpha_floor", "name": "alpha_floor", "shape": 7,
             "type": "INT", "widget": {"name": "alpha_floor"}, "link": None},
        ],
        "outputs": [
            {"localized_name": "last_saved", "name": "last_saved", "type": "STRING",
             "links": [106]},
            {"localized_name": "report", "name": "report", "type": "STRING",
             "links": []},
        ],
        "properties": {"Node name for S&R": "STLU_DetailBatch"},
        "title": "2. 参考图细节还原（CPU，自动）",
        "widgets_values": ["", "", False, 26.0, 1.6, 8],
        "widgets_values_named": {
            "plan": "", "plan_path": "", "only_selected": False,
            "color_tolerance": 26.0, "alpha_sharpen": 1.6, "alpha_floor": 8,
        },
    }

    # AssemblePSD input order is deps, plan, out_dir, ...
    assemble["inputs"][0]["link"] = 106     # deps <- DetailBatch
    assemble["inputs"][1]["link"] = 103     # plan
    assemble["inputs"][2]["link"] = None    # out_dir: use the plan's
    assemble["outputs"][3]["links"] = [107]  # preview -> ImageCompare
    assemble["order"] = 9
    # Do not clobber the AI batch's <prefix>_upscaled.psd in the same folder.
    assemble["widgets_values"][2] = "seethrough_detail"
    assemble["widgets_values_named"]["psd_filename"] = "seethrough_detail"

    compare["inputs"][0]["link"] = 105      # image_a <- reference
    compare["inputs"][1]["link"] = 107      # image_b <- preview
    compare["order"] = 10

    note["widgets_values"] = [NOTE_TEXT]
    note["widgets_values_named"] = {"text": NOTE_TEXT}
    note["title"] = "说明"

    nodes = [plan, load, detail, assemble, compare, note]
    links = [
        [101, 1001, 0, 1008, 0, "STRING"],
        [102, 1001, 1, 1008, 2, "STRING"],
        [103, 1001, 0, 1009, 1, "STRING"],
        [104, 1007, 0, 1008, 1, "IMAGE"],
        [105, 1007, 0, 1010, 0, "IMAGE"],
        [106, 1008, 0, 1009, 0, "*"],
        [107, 1009, 3, 1010, 1, "IMAGE"],
    ]

    doc = {
        "id": "b7d2e4f1-0000-4000-8000-000000000002",
        "revision": 0,
        "last_node_id": 1011,
        "last_link_id": 107,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"frontendVersion": "1.55.14"},
        "version": 0.4,
    }
    with open(OUTPUT_WORKFLOW, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    print("wrote", OUTPUT_WORKFLOW)


if __name__ == "__main__":
    main()
