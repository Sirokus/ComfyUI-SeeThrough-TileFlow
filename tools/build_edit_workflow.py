"""Generate the image-edit Live2D part workflows.

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tools/build_edit_workflow.py

Writes two files next to the existing Live2D workflows:

* ``Live2D图编辑抠层.json`` -- a copy of the user's single-image Qwen edit
  workflow.  It doubles as the subgraph template the batch node clones, and
  as a place to debug one part at a time.
* ``Live2D图编辑抠层-批量.json`` -- the batch workflow: reference image +
  model loaders + ``STLU_EditPartsBatch``.  One run extracts every part and
  writes the SeeThrough-compatible PSD.

Both files are only written here; nothing in the existing workflows is
modified.
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
WORKFLOW_DIR = os.path.join(IMAGE_DIR, "Live2D拆层")
SOURCE = os.path.join(IMAGE_DIR, "Qwen2.1图编辑.json")
TEMPLATE_OUT = os.path.join(WORKFLOW_DIR, edit_parts.TEMPLATE_NAME)
BATCH_OUT = os.path.join(WORKFLOW_DIR, "Live2D图编辑抠层-批量.json")

TEMPLATE_NOTE = """## Live2D 图编辑抠层（模板 / 单部件调试）

这个文件有两个用途：

1. **批量抠层的子图模板** —— `Live2D图编辑抠层-批量.json` 里的批量节点会克隆
   这里的 Qwen Image Edit 子图，每个部件跑一次。改这里的采样器 / 步数 / 模型，
   批量运行会自动生效。
2. **单部件调试** —— 直接在这里跑，看某一个部件的提示词效果。

注意：批量节点会**跳过提示词增强器（PE）**，直接用节点的提��词，
因为增强器会改写“只抠这个部件”的指令，而且要多加载一个 9B 文本模型。
"""

BATCH_NOTE = """## Live2D 图编辑抠层 · 批量（一次跑完 → PSD）

**用法：**

1. 左上 `参考图` 选择角色全身立绘（透明 PNG 或纯色背景都行）。
2. 中间节点的 **部件选择面板**：直接勾要抠的部件（可搜索、全选、清空）。
   默认全选；角色没有的部件（翅膀/尾巴等）保持勾着也没关系，会自动判空跳过。
   面板会同步到 `select` 字符串，想手写也可以：`all -wings -tail`，或
   `face, back hair, mouth`（带空格的部件名用逗号分隔）。
3. 选 `key_color`（抠图用的纯色底）。默认绿色；角色有绿发/绿瞳时换洋红，
   有大量粉色衣物时换蓝色。提示词会跟着变。
4. 点运行。所有部件按顺序生成、自动抠透明、自动写 PSD。

**输出：**
`output/SeeThrough/EditParts/` 下有每个部件 PNG、一份 `*_layers.json`
（与 SeeThrough 完全同格式）和最终 PSD。PSD 图层顺序 = SeeThrough 的标准
前后顺序（后发在最底、前发在最上）。

**29 个标准部件（与 SeeThrough 输出同名）：**

```
neckwear tail wings objects eyewear earwear
back hair headwear legwear handwear-r footwear handwear-l
neck bottomwear topwear
eyebrow-r ears-r ears-l eyebrow-l
face nose mouth
eyewhite-r eyewhite-l eyelash-l irides-r irides-l eyelash-r
front hair
```

**说明：**
- 不再使用 SeeThrough 的分层模型，也不需要二次放大；画布 = 参考图尺寸。
- 角色没有的部件（翅膀/尾巴/眼镜等）会输出整张纯色底，自动判定为空层并跳过。
- 万一某个部件整张图都被保留（覆盖率 > 90%），会判为失败并跳过，报告里会写出来。
- 每部件一次采样，部件越多越慢；这是“一次批量”的代价，中途不要关窗口。
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
                values[1] = edit_parts.build_prompt("mouth", "green")
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


def _input(name, type_, link=None, widget=False):
    entry = {"localized_name": name, "name": name, "type": type_, "link": link}
    if widget:
        entry["widget"] = {"name": name}
    return entry


def _output(name, type_, links=None):
    return {"localized_name": name, "name": name, "type": type_, "links": links}


def build_batch():
    nodes, links = [], []
    last_link = [200]

    def new_link(origin, origin_slot, target, target_slot, link_type):
        last_link[0] += 1
        links.append([last_link[0], origin, origin_slot, target, target_slot,
                      link_type])
        return last_link[0]

    def add(node_id, class_type, pos, size, inputs, outputs, widgets, order,
            title=None, group=None):
        node = {
            "id": node_id, "type": class_type, "pos": list(pos),
            "size": list(size), "flags": {}, "order": order, "mode": 0,
            "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": class_type},
            "widgets_values": widgets,
        }
        if title:
            node["title"] = title
        if group is not None:
            node["group"] = group
        nodes.append(node)
        return node

    # --- reference image ---------------------------------------------------
    add(
        2001, "LoadImage", (-1900, 120), (340, 360),
        [_input("image", "COMBO", widget=True),
         _input("upload", "IMAGEUPLOAD", widget=True)],
        [_output("IMAGE", "IMAGE", []), _output("MASK", "MASK", None)],
        ["Qwen_image_2.1_00029_Improve_1K.png", "image"], 0,
        title="参考图（角色全身立绘）",
    )

    # --- model loaders -----------------------------------------------------
    add(
        2002, "UNETLoader", (-1330, 120), (360, 82),
        [_input("unet_name", "COMBO", widget=True),
         _input("weight_dtype", "COMBO", widget=True)],
        [_output("MODEL", "MODEL", [])],
        ["qwen_image_2.1_turbo_int8_convrot.safetensors", "default"], 1,
    )
    add(
        2003, "QwenImage21Cache", (-920, 120), (320, 82),
        [_input("model", "MODEL"), _input("device", "COMBO", widget=True),
         _input("dtype", "COMBO", widget=True)],
        [_output("MODEL", "MODEL", [])],
        ["auto", "default"], 2,
    )
    add(
        2004, "CLIPLoader", (-1330, 250), (360, 106),
        [_input("clip_name", "COMBO", widget=True),
         _input("type", "COMBO", widget=True),
         _input("device", "COMBO", widget=True)],
        [_output("CLIP", "CLIP", [])],
        ["qwen3vl_8b_int8_convrot.safetensors", "qwen_image", "default"], 3,
    )
    add(
        2005, "VAELoader", (-920, 250), (320, 60),
        [_input("vae_name", "COMBO", widget=True)],
        [_output("VAE", "VAE", [])],
        ["qwen_image_2.1_vae_bf16.safetensors"], 4,
    )

    # --- batch extract -----------------------------------------------------
    batch_inputs = [
        _input("reference", "IMAGE"),
        _input("model", "MODEL"),
        _input("clip", "CLIP"),
        _input("vae", "VAE"),
        _input("select", "STRING", widget=True),
        _input("key_color", "COMBO", widget=True),
        _input("out_dir", "STRING", widget=True),
        _input("workflow_json", "STRING", widget=True),
        _input("seed", "INT", widget=True),
        _input("tolerance", "FLOAT", widget=True),
        _input("softness", "FLOAT", widget=True),
        _input("protect", "BOOLEAN", widget=True),
        _input("despill", "FLOAT", widget=True),
        _input("negative_prompt", "STRING", widget=True),
        _input("cfg", "FLOAT", widget=True),
        _input("resolution", "INT", widget=True),
        _input("scheduler", "STRING", widget=True),
        _input("sigmas", "STRING", widget=True),
        _input("psd_filename", "STRING", widget=True),
    ]
    add(
        2006, "STLU_EditPartsBatch", (-450, 120), (660, 1000),
        batch_inputs,
        [_output("layers_json", "STRING", []),
         _output("psd_path", "STRING", []),
         _output("report", "STRING", [])],
        [
            "all", "green",
            "SeeThrough/EditParts",
            "",              # workflow_json 留空 = 自动找模板
            42,
            "fixed",         # seed 后面的 control_after_generate 控件
            12.0, 6.0, True, 0.65,
            "",
            1.0, 0, "euler", DEFAULT_SIGMAS, "",
        ],
        5,
        title="批量抠部件（一次跑完全部 → PSD）",
    )

    # --- report viewer -----------------------------------------------------
    add(
        2007, "PreviewAny", (230, 120), (420, 300),
        [_input("source", "*")],
        [],
        [], 6, title="运行报告",
    )

    # --- note --------------------------------------------------------------
    add(
        2008, "MarkdownNote", (-1900, -520), (900, 560),
        [], [], [BATCH_NOTE], 7, title="说明",
    )

    # --- links -------------------------------------------------------------
    new_link(2001, 0, 2006, 0, "IMAGE")
    new_link(2002, 0, 2003, 0, "MODEL")
    new_link(2003, 0, 2006, 1, "MODEL")
    new_link(2004, 0, 2006, 2, "CLIP")
    new_link(2005, 0, 2006, 3, "VAE")
    new_link(2006, 2, 2007, 0, "*")

    by_id = {n["id"]: n for n in nodes}
    for link_id, origin, origin_slot, target, target_slot, _type in links:
        if target in by_id and target_slot < len(by_id[target]["inputs"]):
            by_id[target]["inputs"][target_slot]["link"] = link_id
        if origin in by_id and origin_slot < len(by_id[origin]["outputs"]):
            slot = by_id[origin]["outputs"][origin_slot]
            if slot.get("links") is None:
                slot["links"] = []
            slot["links"].append(link_id)

    return {
        "id": "b7d1e4a2-0000-4000-8000-000000000102",
        "revision": 0,
        "last_node_id": 2008,
        "last_link_id": last_link[0],
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "模型加载器（一般不用改）",
                "bounding": [-1400, 60, 830, 320],
                "color": "#3f5a3f",
                "font_size": 20,
                "flags": {},
            },
            {
                "id": 2,
                "title": "输出：SeeThrough/EditParts",
                "bounding": [200, 60, 480, 260],
                "color": "#4a3f5a",
                "font_size": 20,
                "flags": {},
            },
        ],
        "definitions": {"subgraphs": []},
        "config": {},
        "extra": {"frontendVersion": "1.55.14"},
        "version": 0.4,
    }


def main():
    if not os.path.isfile(SOURCE):
        raise SystemExit("source workflow not found: %s" % SOURCE)
    doc = _load_source()

    template = build_template(doc)
    with open(TEMPLATE_OUT, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print("wrote %s" % TEMPLATE_OUT)

    batch = build_batch()
    # Keep a stale template around?  The batch node only needs the subgraph
    # definition, and the template file above is the canonical source.
    with open(BATCH_OUT, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
    print("wrote %s" % BATCH_OUT)

    # The GUI relies on input order matching the widget order; check the
    # things we can verify without importing ComfyUI.  The frontend adds one
    # extra "control_after_generate" value after a seed widget, so the saved
    # list is widgets + 1.
    node = next(n for n in batch["nodes"] if n["type"] == "STLU_EditPartsBatch")
    widgets = [i["name"] for i in node["inputs"]
               if i["name"] not in ("reference", "model", "clip", "vae")]
    assert "seed" in widgets, "seed widget missing"
    assert len(node["widgets_values"]) == len(widgets) + 1, (
        "%d widget inputs but %d values (expected %d = widgets + "
        "control_after_generate)" % (len(widgets), len(node["widgets_values"]),
                                     len(widgets) + 1))
    seed_index = widgets.index("seed")
    assert node["widgets_values"][seed_index + 1] in (
        "fixed", "randomize", "increment", "decrement"), (
        "seed control_after_generate value is %r"
        % (node["widgets_values"][seed_index + 1],))
    print("ok")


if __name__ == "__main__":
    main()
