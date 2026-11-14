"""Regenerate the image-edit subgraph template shipped inside the node pack.

    python custom_nodes/ComfyUI-SeeThrough-TileFlow/tools/build_edit_workflow.py

Writes ``comfy/.../ComfyUI-SeeThrough-TileFlow/templates/Live2D图编辑抠层.json``
-- a copy of the user's single-image Qwen edit workflow.  It is the subgraph
template every edit node clones (``STLU_EditPartsBatch``, ``STLU_LayerEdit``);
it does not live in the workflow tree.
"""

from __future__ import annotations

import json
import os
import sys

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

import edit_parts  # noqa: E402

DEFAULT_SIGMAS = (
    "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, "
    "0.414568, 0"
)

IMAGE_DIR = os.path.join(
    COMFY_ROOT, "user", "default", "workflows", "Comfy-My-Workflow", "图片",
)
SOURCE = os.path.join(IMAGE_DIR, "Qwen2.1图编辑.json")
TEMPLATE_OUT = edit_parts.default_template_path()

TEMPLATE_NOTE = """## Live2D 图编辑子图模板（节点包自带）

这是 `ComfyUI-SeeThrough-TileFlow` 节点包里的编辑子图模板，不在工作流树里。
批量抠层、逐层编辑两个节点都会克隆这里的 Qwen Image Edit 子图，
每个部件跑一次。改这里的采样器 / 步数 / 模型，这些节点会自动生效。

注意：这些节点会**跳过提示词增强器（PE）**，直接用自己的提示词，
因为增强器会改写“只抠这个部件”的指令，而且要多加载一个 9B 文本模型。
"""

def _load_source():
    with open(SOURCE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_template(doc):
    out = json.loads(json.dumps(doc))
    out["id"] = "b7d1e4a2-0000-4000-8000-000000000101"
    subgraph_id = out["definitions"]["subgraphs"][0]["id"]

    # Point the template instance at a generic extraction prompt and turn
    # the prompt enhancer off (the batch clone never uses it either).
    for node in out.get("nodes", []):
        if node.get("type") == subgraph_id:
            values = list(node.get("widgets_values") or [])
            if len(values) > 5:
                values[1] = edit_parts.build_prompt("mouth")
                values[2] = False            # refine_prompt (PE) off
                values[3] = False            # thinking off
                values[5] = ""
                values[6] = 1.0              # cfg
                values[11] = 42
                node["widgets_values"] = values
            break

    note_id = max(int(n.get("id", 0)) for n in out.get("nodes", [])) + 1
    out.setdefault("nodes", []).append({
        "id": note_id,
        "type": "MarkdownNote",
        "pos": [-2200, -900],
        "size": [760, 460],
        "flags": {},
        "order": 99,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {},
        "widgets_values": [TEMPLATE_NOTE],
        "title": "说明：模板 / 单部件调试",
    })
    out["last_node_id"] = note_id
    return out


def main():
    if not os.path.isfile(SOURCE):
        raise SystemExit("source workflow not found: %s" % SOURCE)
    doc = _load_source()

    template = build_template(doc)
    os.makedirs(os.path.dirname(TEMPLATE_OUT), exist_ok=True)
    with open(TEMPLATE_OUT, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print("wrote %s" % TEMPLATE_OUT)


if __name__ == "__main__":
    main()
