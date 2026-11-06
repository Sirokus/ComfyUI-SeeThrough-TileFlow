"""Generate the CPU-only "4K 部件遮罩" workflow.

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tools/build_mask4k_workflow.py

The workflow is STLU_Plan -> STLU_Mask4KBatch -> PreviewImage.  Nothing is
loaded onto the GPU: it writes one complete 4K outline per layer (the shape the
redraw stage is allowed to change) plus ``masks.json`` and a preview sheet.
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
BASE_WORKFLOW = os.path.join(WORKFLOW_DIR, "Live2D层级二次放大-批量.json")
OUTPUT_WORKFLOW = os.path.join(WORKFLOW_DIR, "Live2D-4K遮罩-批量.json")
MASK_OUT_DIR = "SeeThrough/我的自设/UpScale"

NOTE_TEXT = """## Live2D 分层「4K 部件遮罩」（纯 CPU，不跑 GPU）

这一步只解决一件事：**每个部件在 4K 上的准确、完整轮廓**。它是下一步
「遮罩内重绘」的约束——模型只允许改遮罩里的像素。

1. 确认拆层文件夹 + 选 4K 参考图（与细节还原用的是同一张）。
2. `select` 保持 `all`：所有非空层都出遮罩。
3. 点运行，约 1 分钟；显存占用 0。

输出在 `UpScale/4K_masks/`：每层一张黑白 PNG（白=部件）+ `masks.json`
（记录 kind / 尺寸 / 边缘误差 / 虹膜圆心半径）+ 预览表（节点输出）。

**遮罩怎么来的**
- `part`：1K alpha 用 4K 参考做**引导滤波**上采样，再按 `snap_px` 把轮廓点
  吸附到参考图的实时边缘（0 = 不吸附，4 = 默认，最多移动 4 个 4K 像素）。
- `iris`：只取参考图里**可见的那段虹膜弧**，拟合成整圆——被睫毛盖住的上半
  段也包含在遮罩里（`masks.json` 里的 `circle` 就是圆心/半径）。
- `eye`：眼白 = 可见眼白 + 虹膜的外包凸形，所以遮罩是完整的眼球。
- `lash` / `brow`：以拆层的 alpha 为邻域，取参考图里的深色笔画。
- 眉毛按名字自动走 `brow`；如果某个部件没被识别，可把 `mode` 手动改成对应类型。

下一步（尚未接通）：在这些遮罩里做局部重绘，再用同一张遮罩把结果贴回。
"""


def main():
    with open(BASE_WORKFLOW, "r", encoding="utf-8") as f:
        base = json.load(f)
    by_type = {}
    for node in base["nodes"]:
        by_type.setdefault(node["type"], []).append(node)

    plan = json.loads(json.dumps(by_type["STLU_Plan"][0]))
    load = json.loads(json.dumps(by_type["LoadImage"][0]))
    preview_src = by_type.get("ImageCompare") or by_type.get("PreviewImage")
    if preview_src:
        preview = json.loads(json.dumps(preview_src[0]))
    else:
        preview = None
    note = json.loads(json.dumps(by_type["MarkdownNote"][0]))

    plan["widgets_values"] = [
        "SeeThrough/我的自设", "all", "", "{}", MASK_OUT_DIR, 4096,
        True, 0.0005, True, 32, "reference",
        stlu_nodes.core_plan.DEFAULT_FOCUS_PROMPT,
    ]
    plan["widgets_values_named"] = {
        "layers_json": "SeeThrough/我的自设",
        "select": "all",
        "exclude": "",
        "size_rules": "{}",
        "out_dir": MASK_OUT_DIR,
        "canvas_size": 4096,
        "skip_empty": True,
        "empty_coverage": 0.0005,
        "focus_crop": True,
        "focus_margin": 32,
        "focus_context": "reference",
        "focus_prompt": stlu_nodes.core_plan.DEFAULT_FOCUS_PROMPT,
    }
    plan["outputs"][0]["links"] = [101]
    plan["outputs"][1]["links"] = []
    plan["outputs"][2]["links"] = []
    plan["outputs"][3]["links"] = []

    load["outputs"][0]["links"] = [102]
    load["outputs"][1]["links"] = []
    load["widgets_values"] = ["Qwen_image_2.1_00029_mouse.png", "image"]
    load["widgets_values_named"] = {
        "image": "Qwen_image_2.1_00029_mouse.png", "upload": "image",
    }

    mask_node = {
        "id": 1008,
        "type": "STLU_Mask4KBatch",
        "pos": [-600, 180],
        "size": [440, 300],
        "flags": {},
        "order": 8,
        "mode": 0,
        "inputs": [
            {"localized_name": "plan", "name": "plan", "type": "STRING",
             "widget": {"name": "plan"}, "link": 101},
            {"localized_name": "reference", "name": "reference", "type": "IMAGE",
             "link": 102},
            {"localized_name": "mode", "name": "mode", "type": "COMBO",
             "widget": {"name": "mode"}, "link": None},
            {"localized_name": "radius", "name": "radius", "type": "INT",
             "widget": {"name": "radius"}, "link": None},
            {"localized_name": "eps", "name": "eps", "type": "FLOAT",
             "widget": {"name": "eps"}, "link": None},
            {"localized_name": "threshold", "name": "threshold", "type": "FLOAT",
             "widget": {"name": "threshold"}, "link": None},
            {"localized_name": "snap_px", "name": "snap_px", "type": "INT",
             "widget": {"name": "snap_px"}, "link": None},
            {"localized_name": "out_dir", "name": "out_dir", "type": "STRING",
             "widget": {"name": "out_dir"}, "link": None},
            {"localized_name": "only_selected", "name": "only_selected",
             "type": "BOOLEAN", "widget": {"name": "only_selected"}, "link": None},
            {"localized_name": "sheet_width", "name": "sheet_width", "type": "INT",
             "widget": {"name": "sheet_width"}, "link": None},
        ],
        "outputs": [
            {"localized_name": "manifest_path", "name": "manifest_path",
             "type": "STRING", "links": []},
            {"localized_name": "preview_sheet", "name": "preview_sheet",
             "type": "IMAGE", "links": [103]},
        ],
        "properties": {"Node name for S&R": "STLU_Mask4KBatch"},
        "title": "2. 4K 部件遮罩（CPU，自动）",
        "widgets_values": ["", "auto", 8, 0.0002, 0.5, 4, "", False, 1800],
        "widgets_values_named": {
            "plan": "", "mode": "auto", "radius": 8, "eps": 0.0002,
            "threshold": 0.5, "snap_px": 4, "out_dir": "",
            "only_selected": False, "sheet_width": 1800,
        },
    }

    # Preview the sheet inside ComfyUI (the masks themselves are on disk).
    preview_node = {
        "id": 1010,
        "type": "PreviewImage",
        "pos": [-120, 180],
        "size": [420, 380],
        "flags": {},
        "order": 9,
        "mode": 0,
        "inputs": [
            {"localized_name": "images", "name": "images", "type": "IMAGE",
             "link": 103},
        ],
        "outputs": [],
        "properties": {"Node name for S&R": "PreviewImage"},
        "title": "3. 遮罩预览（每格一个部件）",
        "widgets_values": [],
    }

    note["widgets_values"] = [NOTE_TEXT]
    note["widgets_values_named"] = {"text": NOTE_TEXT}
    note["title"] = "说明"

    nodes = [plan, load, mask_node, preview_node, note]
    links = [
        [101, 1001, 0, 1008, 0, "STRING"],
        [102, 1007, 0, 1008, 1, "IMAGE"],
        [103, 1008, 1, 1010, 0, "IMAGE"],
    ]
    doc = {
        "id": "b7d2e4f1-0000-4000-8000-000000000003",
        "revision": 0,
        "last_node_id": 1010,
        "last_link_id": 103,
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
