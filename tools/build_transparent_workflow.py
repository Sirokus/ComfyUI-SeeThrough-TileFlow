"""Build (or repair) the transparent variant of the guided tile workflow.

``seethrough-四步分块流水线-引导.json`` is the current one; it hides stage ①
in ``STLU_PipelineInput``, the loop in ``STLU_GuidedLayers`` and stage ③ in
two monolith nodes.  This tool writes a sibling file,
``seethrough-四步分块流水线-引导-透明版.json``, that shows the same pipeline as
four subgraphs built from the fine-grained nodes in ``nodes_transparent``:

    ① 计划与切块   PlanGrid -> PickTiles -> PlanContext -> PrepareRun -> CropTiles
    ② 顺序拆层     SeqTileLoop (the one loop that must stay Python)
    ③ 存盘         RestoreLocks -> WriteTilePSD / WriteTileImages -> SaveOriginal
                   -> WriteManifest
    ④ 合并         MergeTiles (one indivisible algorithm, kept whole)

Modes::

    python tools/build_transparent_workflow.py <workflow_dir>            # build / repair
    python tools/build_transparent_workflow.py <workflow_dir> --force    # ignore saved values
    python tools/build_transparent_workflow.py <workflow_dir> --validate # contract check only

The default build keeps the widget values a user tuned on the settings node
(and the preview's max_side), so running the tool twice is idempotent.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

TARGET_FILE = "seethrough-tile-flow.json"

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

import nodes as stlu  # noqa: E402

MAPPINGS = stlu.NODE_CLASS_MAPPINGS

# Registered by ComfyUI itself or by the SeeThrough extension, not by this
# pack; the contract check only verifies the pack's own classes.
EXTERNAL_TYPES = {
    "LoadImage", "MarkdownNote", "Note", "PreviewImage",
    "SeeThrough_LoadLayerDiffModel", "SeeThrough_LoadDepthModel",
    "SeeThrough_GenerateLayers", "SeeThrough_GenerateDepth",
    "SeeThrough_PostProcess", "KSampler",
}

CUSTOM_TYPES = {"STLU_SETTINGS", "STLU_PLAN", "STLU_TILE", "STLU_TILE_IMAGES",
                "STLU_MANIFEST", "STLU_MERGE", "STLU_GENERATED"}


def _uuid(tag: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "stlu/transparent/" + tag))


SG_PLAN = _uuid("plan")
SG_LOOP = _uuid("loop")
SG_SAVE = _uuid("save")
SG_MERGE = _uuid("merge")


# ---------------------------------------------------------------------------
# node / link construction straight from the registered classes
# ---------------------------------------------------------------------------

def _spec_type(spec):
    kind = spec[0]
    if isinstance(kind, (list, tuple)):
        return "COMBO"
    return str(kind)


def _spec_default(spec):
    kind = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(kind, (list, tuple)):
        return opts.get("default", kind[0] if kind else "")
    if "default" in opts:
        return opts["default"]
    return {"INT": 0, "FLOAT": 0.0, "BOOLEAN": False, "STRING": ""}.get(
        str(kind), "")


def _entries(cls):
    """Every INPUT_TYPES entry in declaration order, widgets and sockets."""
    inputs = cls.INPUT_TYPES()
    rows = []
    for group in ("required", "optional"):
        for name, spec in (inputs.get(group) or {}).items():
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            rows.append({
                "name": name,
                "type": _spec_type(spec),
                "default": _spec_default(spec),
                "widget": not opts.get("forceInput")
                and _spec_type(spec) not in CUSTOM_TYPES,
                "list_input": bool(opts.get("forceInput")) or False,
                "required": group == "required",
            })
    return rows


def make_node(node_id, class_name, pos, size=None, title=None, colors=None):
    cls = MAPPINGS[class_name]
    rows = _entries(cls)
    inputs = []
    widgets = []
    for row in rows:
        item = {"localized_name": row["name"], "name": row["name"],
                "type": row["type"], "link": None}
        if row["widget"]:
            item["shape"] = 7
            item["widget"] = {"name": row["name"]}
            widgets.append(row["default"])
        inputs.append(item)
    types = list(getattr(cls, "RETURN_TYPES", ()))
    names = list(getattr(cls, "RETURN_NAMES", None) or [])
    is_list = list(getattr(cls, "OUTPUT_IS_LIST", ()))
    outputs = []
    for slot, type_name in enumerate(types):
        outputs.append({
            "localized_name": names[slot] if slot < len(names) else type_name,
            "name": names[slot] if slot < len(names) else type_name,
            "type": type_name, "links": [],
            **({"shape": 6} if slot < len(is_list) and is_list[slot] else {}),
        })
    node = {
        "id": int(node_id), "type": class_name, "pos": list(pos),
        "size": list(size or (320, 140)), "flags": {}, "order": 0,
        "mode": 0, "inputs": inputs, "outputs": outputs,
        "properties": {"Node name for S&R": class_name},
        "widgets_values": widgets,
        "widgets_values_named": {
            row["name"]: row["default"] for row in rows if row["widget"]},
    }
    if title:
        node["title"] = title
    if colors:
        node["color"], node["bgcolor"] = colors
    return node


def inner_link(link_id, src, src_slot, dst, dst_slot, type_name):
    return {"id": int(link_id), "origin_id": int(src), "origin_slot": int(src_slot),
            "target_id": int(dst), "target_slot": int(dst_slot),
            "type": type_name}


# Subgraph bodies are written with local node/link ids (1..n) so each builder
# stays readable, then lifted into a global id range.  ComfyUI keeps ONE id
# space for the whole document, subgraph interiors included: reusing 1..5 in
# four subgraphs makes the frontend's link store collide and wires snap onto
# the wrong node.  The ranges below keep them apart and are stable, so a
# rebuild is still byte-identical.
SG_NODE_BASE = {"plan": 1000, "loop": 1100, "save": 1200, "merge": 1300}
SG_LINK_BASE = {"plan": 1000, "loop": 1100, "save": 1200, "merge": 1300}


def _renumber(subgraph, node_base, link_base):
    """Move local ids into the global namespace, refs included."""
    node_map = {node["id"]: node_base + i
                for i, node in enumerate(subgraph["nodes"])}
    link_map = {link["id"]: link_base + i
                for i, link in enumerate(subgraph["links"])}
    for node in subgraph["nodes"]:
        node["id"] = node_map[node["id"]]
    for link in subgraph["links"]:
        link["id"] = link_map[link["id"]]
        if link["origin_id"] != -10:
            link["origin_id"] = node_map[link["origin_id"]]
        if link["target_id"] != -20:
            link["target_id"] = node_map[link["target_id"]]


def _wire_inner_nodes(subgraph):
    """Fill each inner node's ``inputs[].link`` / ``outputs[].links``.

    The frontend fills these when it resaves a file, but a freshly built
    document that omits them draws every inner node as unconnected until the
    user touches it.
    """
    by_id = {node["id"]: node for node in subgraph["nodes"]}
    for node in subgraph["nodes"]:
        for item in node.get("inputs") or []:
            item["link"] = None
        for item in node.get("outputs") or []:
            item["links"] = []
    for link in subgraph["links"]:
        source = by_id.get(link["origin_id"])
        if source is not None:
            source["outputs"][link["origin_slot"]]["links"].append(link["id"])
        target = by_id.get(link["target_id"])
        if target is not None:
            target["inputs"][link["target_slot"]]["link"] = link["id"]


def _fill_link_ids(subgraph):
    """Derive every promoted slot's ``linkIds`` from the inner links.

    The frontend rebuilds the subgraph border from these lists.  They are
    never hand-written: a stale array is exactly what made the transparent
    file draw four subgraphs whose slots pointed at the wrong wires.
    """
    into, out_of = {}, {}
    for link in subgraph["links"]:
        if link["origin_id"] == -10:
            into.setdefault(link["origin_slot"], []).append(link["id"])
        if link["target_id"] == -20:
            out_of.setdefault(link["target_slot"], []).append(link["id"])
    for slot, item in enumerate(subgraph["inputs"]):
        item["linkIds"] = sorted(into.get(slot, []))
    for slot, item in enumerate(subgraph["outputs"]):
        item["linkIds"] = sorted(out_of.get(slot, []))
    return subgraph


def _finish_subgraph(subgraph, key):
    """Renumber, wire the inner nodes and derive the promoted slots."""
    _renumber(subgraph, SG_NODE_BASE[key], SG_LINK_BASE[key])
    _wire_inner_nodes(subgraph)
    _fill_link_ids(subgraph)
    return subgraph


# ---------------------------------------------------------------------------
# subgraph bodies
# ---------------------------------------------------------------------------

def build_plan_subgraph():
    """① PlanGrid -> PickTiles -> PlanContext -> PrepareRun -> CropTiles."""
    nodes = [
        make_node(1, "STLU_PlanGrid", (40, 40), (360, 130),
                  "①-1 规划网格（纯函数）"),
        make_node(2, "STLU_PickTiles", (440, 40), (300, 90),
                  "①-2 选块（读了面板的格子勾选）"),
        make_node(3, "STLU_PlanContext", (780, 40), (300, 90),
                  "①-3 上下文外扩（context_pad）"),
        make_node(4, "STLU_PrepareRun", (1120, 40), (320, 110),
                  "①-4 准备运行（唯一会写盘/清目录的一步）"),
        make_node(5, "STLU_CropTiles", (1480, 40), (320, 120),
                  "①-5 裁剪（tiles 在外层预览）"),
    ]
    # linkIds 全部由 _fill_link_ids 从 links 推导，不手写。
    inputs = [
        {"id": _uuid("plan/in/image"), "name": "image", "type": "IMAGE",
         "label": "image"},
        {"id": _uuid("plan/in/settings"), "name": "settings",
         "type": "STLU_SETTINGS", "label": "settings"},
        {"id": _uuid("plan/in/layerdiff"), "name": "layerdiff_model",
         "type": "SEETHROUGH_LAYERDIFF_MODEL", "label": "layerdiff_model"},
        {"id": _uuid("plan/in/depth"), "name": "depth_model",
         "type": "SEETHROUGH_DEPTH_MODEL", "label": "depth_model"},
    ]
    outputs = [
        {"id": _uuid("plan/out/plan"), "name": "plan", "type": "STLU_PLAN",
         "label": "plan"},
        {"id": _uuid("plan/out/tiles"), "name": "tiles", "type": "IMAGE",
         "label": "tiles"},
        {"id": _uuid("plan/out/indices"), "name": "indices", "type": "STRING",
         "label": "indices"},
    ]
    links = [
        inner_link(1, -10, 0, 1, 0, "IMAGE"),
        inner_link(2, -10, 1, 1, 1, "STLU_SETTINGS"),
        inner_link(3, -10, 2, 1, 2, "SEETHROUGH_LAYERDIFF_MODEL"),
        inner_link(4, -10, 3, 1, 3, "SEETHROUGH_DEPTH_MODEL"),
        inner_link(5, 1, 0, 2, 0, "STLU_PLAN"),
        inner_link(6, 2, 0, 3, 0, "STLU_PLAN"),
        inner_link(7, 3, 0, 4, 0, "STLU_PLAN"),
        inner_link(8, -10, 0, 4, 1, "IMAGE"),
        inner_link(9, 4, 0, 5, 1, "STLU_PLAN"),
        inner_link(10, -10, 0, 5, 0, "IMAGE"),
        # subgraph outputs
        inner_link(11, 5, 0, -20, 1, "IMAGE"),      # tiles
        inner_link(12, 5, 1, -20, 2, "STRING"),     # indices
        inner_link(13, 4, 0, -20, 0, "STLU_PLAN"),  # plan
    ]
    sg = {
        "id": SG_PLAN, "version": 1, "revision": 0,
        "state": {"lastGroupId": 0, "lastNodeId": 5, "lastLinkId": 13,
                  "lastRerouteId": 0},
        "config": {},
        "name": "① 计划与切块",
        "inputNode": {"id": -10, "bounding": [0, -60, 160, 60]},
        "outputNode": {"id": -20, "bounding": [2140, -60, 160, 60]},
        "inputs": _place(inputs, 120, -40), "outputs": _place(outputs, 2200, -40),
        "widgets": [],
        "nodes": nodes, "links": links, "groups": [],
        "extra": {},
        "category": "", "description": (
            "① 的全部计算：规划网格 → 选块 → 外扩 → 指纹/清目录 → 裁剪。"
            "参数来自外层的 STLU_PipelineSettings，节点图上一步一格。"),
    }
    return _finish_subgraph(sg, "plan")


def build_loop_subgraph():
    nodes = [make_node(1, "STLU_SeqTileLoop", (40, 40), (360, 200),
                       "② 顺序拆层循环（唯一必须用 Python 的循环）")]
    inputs = [
        {"id": _uuid("loop/in/tiles"), "name": "tiles", "type": "IMAGE",
         "label": "tiles"},
        {"id": _uuid("loop/in/indices"), "name": "indices", "type": "STRING",
         "label": "indices"},
        {"id": _uuid("loop/in/plan"), "name": "plan", "type": "STLU_PLAN",
         "label": "plan"},
        {"id": _uuid("loop/in/layerdiff"), "name": "layerdiff_model",
         "type": "SEETHROUGH_LAYERDIFF_MODEL", "label": "layerdiff_model"},
        {"id": _uuid("loop/in/depth"), "name": "depth_model",
         "type": "SEETHROUGH_DEPTH_MODEL", "label": "depth_model"},
    ]
    outputs = [
        {"id": _uuid("loop/out/parts"), "name": "parts",
         "type": "SEETHROUGH_PARTS", "label": "parts"},
        {"id": _uuid("loop/out/indices"), "name": "indices", "type": "STRING",
         "label": "indices"},
        {"id": _uuid("loop/out/guided"), "name": "guided_inputs",
         "type": "IMAGE", "label": "guided_inputs"},
        {"id": _uuid("loop/out/composites"), "name": "composites",
         "type": "IMAGE", "label": "composites"},
    ]
    links = [
        inner_link(1, -10, 0, 1, 0, "IMAGE"),
        inner_link(2, -10, 1, 1, 1, "STRING"),
        inner_link(3, -10, 2, 1, 2, "STLU_PLAN"),
        inner_link(4, -10, 3, 1, 3, "SEETHROUGH_LAYERDIFF_MODEL"),
        inner_link(5, -10, 4, 1, 4, "SEETHROUGH_DEPTH_MODEL"),
        inner_link(6, 1, 0, -20, 0, "SEETHROUGH_PARTS"),
        inner_link(7, 1, 1, -20, 1, "STRING"),
        inner_link(8, 1, 2, -20, 2, "IMAGE"),
        inner_link(9, 1, 3, -20, 3, "IMAGE"),
    ]
    sg = {
        "id": SG_LOOP, "version": 1, "revision": 0,
        "state": {"lastGroupId": 0, "lastNodeId": 1, "lastLinkId": 9,
                  "lastRerouteId": 0},
        "config": {},
        "name": "② 顺序拆层（循环保留 Python）",
        "inputNode": {"id": -10, "bounding": [0, -80, 160, 60]},
        "outputNode": {"id": -20, "bounding": [460, -80, 160, 60]},
        "inputs": _place(inputs, 120, -60), "outputs": _place(outputs, 520, -60),
        "widgets": [],
        "nodes": nodes, "links": links, "groups": [],
        "extra": {},
        "category": "", "description": (
            "② 唯一无法用无环节点图表达的部分：第 N 块要看到第 N-1 块生成的像素。"
            "循环内部每一步（回填 / 三步链 / 重试 / 裁回）都是独立函数，"
            "guided_inputs 和 composites 两个输出已在外层接了预览。"),
    }
    return _finish_subgraph(sg, "loop")


def build_save_subgraph():
    nodes = [
        make_node(1, "STLU_RestoreLocks", (40, 140), (300, 110),
                  "③-1 恢复锁定部件（只读）"),
        make_node(2, "STLU_WriteTilePSD", (380, 40), (320, 110),
                  "③-2 写单块 PSD"),
        make_node(3, "STLU_WriteTileImages", (380, 200), (320, 110),
                  "③-3 写单块图层图"),
        make_node(4, "STLU_SaveOriginal", (40, 320), (300, 100),
                  "③-4 存原图"),
        make_node(5, "STLU_WriteManifest", (760, 120), (340, 160),
                  "③-5 写清单（manifest + images + 折入旧块）"),
    ]
    inputs = [
        {"id": _uuid("save/in/parts"), "name": "parts",
         "type": "SEETHROUGH_PARTS", "label": "parts"},
        {"id": _uuid("save/in/indices"), "name": "indices", "type": "STRING",
         "label": "indices"},
        {"id": _uuid("save/in/plan"), "name": "plan", "type": "STLU_PLAN",
         "label": "plan"},
        {"id": _uuid("save/in/original"), "name": "original", "type": "IMAGE",
         "label": "original"},
    ]
    outputs = [
        {"id": _uuid("save/out/manifest"), "name": "manifest",
         "type": "STLU_MANIFEST", "label": "manifest"},
        {"id": _uuid("save/out/merge"), "name": "merge_doc",
         "type": "STLU_MERGE", "label": "merge_doc"},
    ]
    links = [
        inner_link(1, -10, 0, 1, 0, "SEETHROUGH_PARTS"),
        inner_link(2, -10, 1, 1, 2, "STRING"),
        inner_link(3, -10, 2, 1, 1, "STLU_PLAN"),
        inner_link(4, 1, 0, 2, 0, "SEETHROUGH_PARTS"),
        inner_link(5, -10, 1, 2, 2, "STRING"),
        inner_link(6, -10, 2, 2, 1, "STLU_PLAN"),
        inner_link(7, 1, 0, 3, 0, "SEETHROUGH_PARTS"),
        inner_link(8, -10, 1, 3, 2, "STRING"),
        inner_link(9, -10, 2, 3, 1, "STLU_PLAN"),
        inner_link(10, -10, 3, 4, 0, "IMAGE"),
        inner_link(11, -10, 2, 4, 1, "STLU_PLAN"),
        inner_link(12, 2, 0, 5, 0, "STLU_TILE"),
        inner_link(13, 3, 0, 5, 1, "STLU_TILE_IMAGES"),
        inner_link(14, 4, 0, 5, 2, "STRING"),
        inner_link(15, -10, 2, 5, 3, "STLU_PLAN"),
        inner_link(16, 5, 0, -20, 0, "STLU_MANIFEST"),
        inner_link(17, 5, 1, -20, 1, "STLU_MERGE"),
    ]
    sg = {
        "id": SG_SAVE, "version": 1, "revision": 0,
        "state": {"lastGroupId": 0, "lastNodeId": 5, "lastLinkId": 17,
                  "lastRerouteId": 0},
        "config": {},
        "name": "③ 存盘",
        "inputNode": {"id": -10, "bounding": [0, -110, 160, 60]},
        "outputNode": {"id": -20, "bounding": [1260, -110, 160, 60]},
        "inputs": _place(inputs, 120, -90), "outputs": _place(outputs, 1320, -90),
        "widgets": [],
        "nodes": nodes, "links": links, "groups": [],
        "extra": {},
        "category": "", "description": (
            "③ 拆成五个可见步骤：恢复锁定（只读）→ 写 PSD → 写分层 PNG → "
            "存原图 → 写清单。写清单会把同目录上一轮已经完成的块折进来，"
            "并恢复还有锁定快照的孤儿块。"),
    }
    return _finish_subgraph(sg, "save")


def build_merge_subgraph():
    nodes = [make_node(1, "STLU_MergeTiles", (40, 40), (360, 150),
                       "④ 合并（接缝归属 + 分层组装的整块算法）")]
    inputs = [
        {"id": _uuid("merge/in/doc"), "name": "merge_doc",
         "type": "STLU_MERGE", "label": "merge_doc"},
    ]
    outputs = [
        {"id": _uuid("merge/out/path"), "name": "psd_path", "type": "STRING",
         "label": "psd_path"},
        {"id": _uuid("merge/out/report"), "name": "report",
         "type": "STLU_MERGE", "label": "report"},
    ]
    links = [
        inner_link(1, -10, 0, 1, 0, "STLU_MERGE"),
        inner_link(2, 1, 0, -20, 0, "STRING"),
        inner_link(3, 1, 1, -20, 1, "STLU_MERGE"),
    ]
    sg = {
        "id": SG_MERGE, "version": 1, "revision": 0,
        "state": {"lastGroupId": 0, "lastNodeId": 1, "lastLinkId": 3,
                  "lastRerouteId": 0},
        "config": {},
        "name": "④ 合并",
        "inputNode": {"id": -10, "bounding": [0, -80, 160, 60]},
        "outputNode": {"id": -20, "bounding": [460, -80, 160, 60]},
        "inputs": _place(inputs, 120, -60), "outputs": _place(outputs, 520, -60),
        "widgets": [],
        "nodes": nodes, "links": links, "groups": [],
        "extra": {},
        "category": "", "description": (
            "④ 的接缝归属 / 缩放 / 整部件例外 / 分层组装是一整块算法，不再拆；"
            "report 输出列出它实际做了什么。"),
    }
    return _finish_subgraph(sg, "merge")


# ---------------------------------------------------------------------------
# the top-level graph
# ---------------------------------------------------------------------------

NOTE = """# SeeThrough 四步分块流水线（透明版）

这一版把原来的「盲盒节点」全部摊开：每个阶段都是一个子图，子图里每一步都是一个可见节点，
点开就能看到内部发生了什么。旧文件 `seethrough-四步分块流水线-引导.json` 保持原样，随时可回退。

## 只改这里：①-0 用户输入（参数包）
面板和全部参数都在这一个节点上，但它自己什么都不算，只把参数打包成 `settings` 往下传。

- **① 计划与切块**：规划网格 → 选块 → 上下文外扩 → 准备运行 → 裁剪。
  其中只有「准备运行」会写盘（写 generation.json、参数变了清空旧结果）。
- **② 顺序拆层**：唯一必须保留 Python 的循环——第 N 块要看到第 N-1 块生成的像素。
  `guided_inputs` / `composites` 两个输出接在外层预览上，能直接看每块实际吃了什么。
- **③ 存盘**：恢复锁定部件（只读）→ 写单块 PSD → 写单块分层 PNG → 存原图 → 写清单。
- **④ 合并**：接缝归属 / 缩放 / 整部件例外 / 分层组装是一整块算法，保留为单节点；
  `report` 输出列出它做了什么。
- **⑤ 预览**：合并 PSD 的全身总览 + 按块 / 按部位分组。② 的 `guided_inputs`
  和 `indices` 接在这一个节点上，「按块」每一行多一张「输入」——引导回填后
  真正送进模型的那张图（只有本次跑过的块才有）。

## 用法
1. 在 LoadImage 选图，在 ①-0 面板勾选要跑的块（未勾选不跑 GPU）。
2. 按 **Queue**，①②③④ 自动接力。
3. 结果在 `output/<根目录>/<分类?>/`。

## 调试入口
- 生成中途不再弹预览：每块实际喂给模型的输入就在 ⑤ 面板「按块」的「输入」列，
  合成结果就是同一行的「合并」图和拆分图层。
- 想手动重跑合并：④ 子图单独 Queue 即可（输入 merge_doc 来自 ③）。
"""


def _place(items, x, y, step=40):
    for slot, item in enumerate(items):
        item.setdefault("pos", [x, y + slot * step])
    return items


def build_document(existing=None):
    nodes = []
    links = []
    next_id = [1]
    next_link = [1]

    def add(node):
        node["id"] = next_id[0]
        next_id[0] += 1
        nodes.append(node)
        return node

    def connect(src, src_slot, dst, dst_slot, type_name):
        link_id = next_link[0]
        next_link[0] += 1
        links.append([link_id, src["id"], src_slot, dst["id"], dst_slot,
                      type_name])
        out = src["outputs"][src_slot]
        out.setdefault("links", [])
        out["links"].append(link_id)
        dst["inputs"][dst_slot]["link"] = link_id
        return link_id

    load = add({
        "type": "LoadImage", "pos": [40, 60], "size": [340, 360],
        "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            {"localized_name": "图像", "name": "image", "type": "COMBO",
             "widget": {"name": "image"}, "link": None},
            {"localized_name": "选择文件上传", "name": "upload",
             "type": "IMAGEUPLOAD", "widget": {"name": "upload"}, "link": None},
        ],
        "outputs": [
            {"localized_name": "图像", "name": "IMAGE", "type": "IMAGE",
             "links": []},
            {"localized_name": "遮罩", "name": "MASK", "type": "MASK",
             "links": []},
        ],
        "title": "原图（点这里选图）",
        "properties": {"Node name for S&R": "LoadImage"},
        "widgets_values": ["example.png", "image"],
        "widgets_values_named": {"image": "example.png", "upload": "image"},
    })

    settings = add(make_node(0, "STLU_PipelineSettings", (40, 480),
                             (560, 1400), "①-0 用户输入（只改这里）",
                             ("#1f3b2f", "#2f5b45")))

    layer_model = add({
        "type": "SeeThrough_LoadLayerDiffModel", "pos": [660, 60],
        "size": [330, 170], "flags": {}, "order": 0, "mode": 0,
        "inputs": [],
        "outputs": [{"localized_name": "layerdiff_model",
                     "name": "layerdiff_model",
                     "type": "SEETHROUGH_LAYERDIFF_MODEL", "links": []}],
        "title": "拆层模型（默认即可）",
        "properties": {"Node name for S&R": "SeeThrough_LoadLayerDiffModel",
                       "stlu_locked": True,
                       "stlu_show_widgets": ["model"]},
        "widgets_values": ["seethroughv0.0.2_layerdiff3d_nf4", "", "", "nf4",
                           False, False, False],
    })
    depth_model = add({
        "type": "SeeThrough_LoadDepthModel", "pos": [660, 260],
        "size": [330, 160], "flags": {}, "order": 0, "mode": 0,
        "inputs": [],
        "outputs": [{"localized_name": "depth_model", "name": "depth_model",
                     "type": "SEETHROUGH_DEPTH_MODEL", "links": []}],
        "title": "深度模型（默认即可）",
        "properties": {"Node name for S&R": "SeeThrough_LoadDepthModel",
                       "stlu_locked": True,
                       "stlu_show_widgets": ["model"]},
        "widgets_values": ["seethroughv0.0.1_marigold_nf4", "nf4", False,
                           False, False],
    })

    def instance(subgraph, pos, size, title, colors=None):
        ins = [{"name": item["name"], "localized_name": item["name"],
                "label": item.get("label") or item["name"],
                "type": item["type"], "link": None}
               for item in subgraph["inputs"]]
        outs = [{"name": item["name"], "localized_name": item["name"],
                 "label": item.get("label") or item["name"],
                 "type": item["type"], "links": []}
                for item in subgraph["outputs"]]
        node = add({
            "type": subgraph["id"], "pos": list(pos), "size": list(size),
            "flags": {}, "order": 0, "mode": 0, "inputs": ins,
            "outputs": outs,
            "properties": {"proxyWidgets": []},
            "widgets_values": [], "title": title,
        })
        if colors:
            node["color"], node["bgcolor"] = colors
        return node

    sg_plan = build_plan_subgraph()
    sg_loop = build_loop_subgraph()
    sg_save = build_save_subgraph()
    sg_merge = build_merge_subgraph()

    plan = instance(sg_plan, (660, 520), (440, 280),
                    "① 计划与切块（点开看内部）", ("#3f789e", "#2a4a5e"))
    loop = instance(sg_loop, (1180, 520), (440, 280),
                    "② 顺序拆层（循环保留 Python）", ("#3f789e", "#2a4a5e"))
    save = instance(sg_save, (660, 880), (440, 260),
                    "③ 存盘（点开看内部）", ("#b58b2a", "#6b5219"))
    # ③ → ④ 竖直摆放：合并紧跟它的输入下方，不要横向并排。
    merge = instance(sg_merge, (660, 1200), (440, 220),
                     "④ 合并（点开看内部）", ("#a1309b", "#5e1c5a"))

    # ⑤ owns every picture now: the merged PSD, its per-tile / per-part
    # groups, and ②'s per-tile model inputs.  Separate PreviewImage nodes
    # during the run only flooded the queue with images nobody asked for.
    preview = add({
        "type": "STLU_PSDPreview", "pos": [1180, 1140], "size": [900, 700],
        "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            {"localized_name": "psd_path", "name": "psd_path",
             "type": "STRING", "link": None},
            {"localized_name": "max_side", "name": "max_side", "type": "INT",
             "shape": 7, "widget": {"name": "max_side"}, "link": None},
            {"localized_name": "show_parts", "name": "show_parts",
             "type": "BOOLEAN", "shape": 7,
             "widget": {"name": "show_parts"}, "link": None},
            {"localized_name": "guide_images", "name": "guide_images",
             "type": "IMAGE", "link": None},
            {"localized_name": "guide_indices", "name": "guide_indices",
             "type": "STRING", "link": None},
        ],
        "outputs": [{"localized_name": "preview", "name": "preview",
                     "type": "IMAGE", "links": []}],
        "title": "⑤ 预览合并 PSD",
        "properties": {"Node name for S&R": "STLU_PSDPreview"},
        "widgets_values": [1600, True],
        "widgets_values_named": {"max_side": 1600, "show_parts": True},
    })

    # ---- wiring -----------------------------------------------------------
    connect(load, 0, settings, 0, "IMAGE")
    connect(load, 0, plan, 0, "IMAGE")
    connect(settings, 0, plan, 1, "STLU_SETTINGS")
    connect(layer_model, 0, plan, 2, "SEETHROUGH_LAYERDIFF_MODEL")
    connect(depth_model, 0, plan, 3, "SEETHROUGH_DEPTH_MODEL")

    connect(plan, 1, loop, 0, "IMAGE")
    connect(plan, 2, loop, 1, "STRING")
    connect(plan, 0, loop, 2, "STLU_PLAN")
    connect(layer_model, 0, loop, 3, "SEETHROUGH_LAYERDIFF_MODEL")
    connect(depth_model, 0, loop, 4, "SEETHROUGH_DEPTH_MODEL")

    connect(loop, 0, save, 0, "SEETHROUGH_PARTS")
    connect(loop, 1, save, 1, "STRING")
    connect(plan, 0, save, 2, "STLU_PLAN")
    connect(load, 0, save, 3, "IMAGE")

    connect(save, 1, merge, 0, "STLU_MERGE")
    connect(merge, 0, preview, 0, "STRING")
    # ②'s per-tile model inputs go straight to the panel's tile rows: the
    # "what did the seam models actually see" check lives where the results
    # are read, not in a preview that flashes by mid-run.
    connect(loop, 2, preview, 3, "IMAGE")
    connect(loop, 1, preview, 4, "STRING")

    note = add({
        "type": "MarkdownNote", "pos": [-560, 60], "size": [540, 900],
        "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [],
        "title": "透明版说明",
        "properties": {"Node name for S&R": "MarkdownNote"},
        "widgets_values": [NOTE],
        "widgets_values_named": {"text": NOTE},
        "color": "#432", "bgcolor": "#653",
    })

    for order, node in enumerate(nodes):
        node["order"] = order

    # The document keeps ONE id space; the frontend writes the global maximum
    # into every subgraph's state, so a node created inside any of them can
    # never collide with another's.
    last_node_id = max([node["id"] for node in nodes] +
                       [node["id"] for sg in (sg_plan, sg_loop, sg_save,
                                              sg_merge)
                        for node in sg["nodes"]])
    last_link_id = max([link[0] for link in links] +
                       [link["id"] for sg in (sg_plan, sg_loop, sg_save,
                                              sg_merge)
                        for link in sg["links"]])
    for sg in (sg_plan, sg_loop, sg_save, sg_merge):
        sg["state"]["lastNodeId"] = last_node_id
        sg["state"]["lastLinkId"] = last_link_id

    doc = {
        "revision": 0,
        "last_node_id": last_node_id,
        "last_link_id": last_link_id,
        "nodes": nodes,
        "links": links,
        "groups": [
            {"id": 1, "title": "① 用户输入（只改这里）",
             "bounding": [0, 0, 640, 1960], "color": "#4a7a5a", "flags": {}},
            {"id": 2, "title": "模型（默认即可）",
             "bounding": [640, 20, 380, 420], "color": "#444",
             "flags": {}},
            {"id": 3, "title": "① 计划与切块",
             "bounding": [640, 480, 480, 340], "color": "#3f789e",
             "flags": {}},
            {"id": 4, "title": "② 顺序拆层（循环保留 Python）",
             "bounding": [1160, 480, 480, 340], "color": "#3f789e",
             "flags": {}},
            {"id": 5, "title": "③ 存盘",
             "bounding": [640, 840, 480, 320], "color": "#b58b2a",
             "flags": {}},
            {"id": 6, "title": "④ 合并",
             "bounding": [640, 1160, 480, 280], "color": "#a1309b",
             "flags": {}},
            {"id": 8, "title": "⑤ 预览合并 PSD",
             "bounding": [1160, 1100, 940, 780], "color": "#444",
             "flags": {}},
        ],
        "config": {},
        "extra": {"ds": {"scale": 0.9, "offset": [60, 40]},
                  "frontendVersion": "1.55.18"},
        "version": 0.4,
        "definitions": {"subgraphs": [sg_plan, sg_loop, sg_save, sg_merge]},
    }

    if existing:
        _carry_over(doc, existing)
    return doc


def _carry_over(doc, existing):
    """Keep hand-tuned widget values (settings + preview + LoadImage)."""
    old = {}
    for node in existing.get("nodes") or []:
        old.setdefault(node.get("type"), node)
    new_settings = settings = next(
        (n for n in doc["nodes"] if n["type"] == "STLU_PipelineSettings"), None)
    if settings is not None and "STLU_PipelineSettings" in old:
        named = old["STLU_PipelineSettings"].get("widgets_values_named") or {}
        for index, name in enumerate(new_settings["widgets_values_named"]):
            if name in named:
                new_settings["widgets_values"][index] = named[name]
                new_settings["widgets_values_named"][name] = named[name]
    preview = next((n for n in doc["nodes"]
                    if n["type"] == "STLU_PSDPreview"), None)
    if preview is not None and "STLU_PSDPreview" in old:
        # stlu_last_psd is what lets the panel show the previous merge right
        # after reopening the workflow; a rebuild must not forget it.
        old_last = (old["STLU_PSDPreview"].get("properties") or {}).get(
            "stlu_last_psd")
        if old_last:
            preview.setdefault("properties", {})["stlu_last_psd"] = old_last
        named = old["STLU_PSDPreview"].get("widgets_values_named") or {}
        for index, name in enumerate(preview["widgets_values_named"]):
            if name in named:
                preview["widgets_values"][index] = named[name]
                preview["widgets_values_named"][name] = named[name]
    load = next((n for n in doc["nodes"] if n["type"] == "LoadImage"), None)
    if load is not None and "LoadImage" in old:
        named = old["LoadImage"].get("widgets_values_named") or {}
        if named.get("image"):
            load["widgets_values"][0] = named["image"]
            load["widgets_values_named"]["image"] = named["image"]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate(path):
    problems = []
    if not os.path.isfile(path):
        return ["%s: missing" % path]
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)

    subgraphs = {sg["id"]: sg for sg in
                 (doc.get("definitions") or {}).get("subgraphs") or []}
    if len(subgraphs) != 4:
        problems.append("expected 4 subgraphs, found %d" % len(subgraphs))

    def check_graph(nodes, links, where):
        by_id = {node["id"]: node for node in nodes}
        for node in nodes:
            type_name = node.get("type")
            if type_name in subgraphs or type_name in EXTERNAL_TYPES:
                continue
            if type_name not in MAPPINGS:
                problems.append("%s: node %s type %r is not registered"
                                % (where, node["id"], type_name))
        for link in links:
            if isinstance(link, dict):
                link_id = link.get("id")
                src = link.get("origin_id")
                src_slot = link.get("origin_slot")
                dst = link.get("target_id")
                dst_slot = link.get("target_slot")
                type_name = link.get("type")
            else:
                link_id, src, src_slot, dst, dst_slot, type_name = link[:6]
            if dst in (-10, -20) or src in (-10, -20):
                continue
            source = by_id.get(src)
            target = by_id.get(dst)
            if source is None or target is None:
                problems.append("%s: link %s touches missing node (%s -> %s)"
                                % (where, link_id, src, dst))
                continue
            outs = source.get("outputs") or []
            ins = target.get("inputs") or []
            if not 0 <= int(src_slot) < len(outs):
                problems.append("%s: link %s source slot out of range"
                                % (where, link_id))
                continue
            if not 0 <= int(dst_slot) < len(ins):
                problems.append("%s: link %s target slot out of range"
                                % (where, link_id))
                continue
            got = outs[int(src_slot)].get("type")
            want = ins[int(dst_slot)].get("type")
            if got != want:
                problems.append("%s: link %s type %r -> %r"
                                % (where, link_id, got, want))
            if type_name not in (got, want):
                problems.append("%s: link %s declares %r but connects %r/%r"
                                % (where, link_id, type_name, got, want))

    check_graph(doc.get("nodes") or [], doc.get("links") or [], "top level")
    instances = [n for n in doc["nodes"] if n.get("type") in subgraphs]
    if len(instances) != 4:
        problems.append("expected 4 subgraph instances, found %d"
                        % len(instances))
    for sg in subgraphs.values():
        check_graph(sg.get("nodes") or [], sg.get("links") or [],
                    "subgraph %r" % sg.get("name"))
        # Every promoted output must exist on the inner node it names.
        by_id = {node["id"]: node for node in sg.get("nodes") or []}
        for link in sg.get("links") or []:
            if link.get("target_id") != -20:
                continue
            source = by_id.get(link.get("origin_id"))
            slot = int(link.get("origin_slot", -1))
            if source is None or not 0 <= slot < len(source["outputs"]):
                problems.append("subgraph %r: output link %s has no source"
                                % (sg.get("name"), link.get("id")))
        problems.extend(_check_promoted(sg))
        problems.extend(_check_inner_nodes(sg))
    problems.extend(_check_ids(doc))
    return problems


def _check_promoted(sg):
    """A promotion slot's linkIds MUST be the links that actually reach it.

    The frontend trusts this list instead of rescanning the wires, so a stale
    array silently re-points a subgraph border slot at another wire.
    """
    problems = []
    into, out_of = {}, {}
    for link in sg.get("links") or []:
        if link.get("origin_id") == -10:
            into.setdefault(int(link.get("origin_slot", -1)), []).append(link)
        if link.get("target_id") == -20:
            out_of.setdefault(int(link.get("target_slot", -1)), []).append(link)
    for slot, item in enumerate(sg.get("inputs") or []):
        want = sorted(link["id"] for link in into.get(slot, []))
        if sorted(item.get("linkIds") or []) != want:
            problems.append("subgraph %r: input %r linkIds %s != %s"
                            % (sg.get("name"), item.get("name"),
                               item.get("linkIds"), want))
        problems.extend(_check_link_types(sg, "input", item, into.get(slot, [])))
    for slot, item in enumerate(sg.get("outputs") or []):
        want = sorted(link["id"] for link in out_of.get(slot, []))
        if sorted(item.get("linkIds") or []) != want:
            problems.append("subgraph %r: output %r linkIds %s != %s"
                            % (sg.get("name"), item.get("name"),
                               item.get("linkIds"), want))
        problems.extend(_check_link_types(sg, "output", item, out_of.get(slot, [])))
    return problems


def _check_link_types(sg, side, item, links):
    """The wire feeding a promoted slot must carry the slot's own type.

    linkIds alone only prove *which* wire is promoted; if a builder swaps two
    wires the list stays populated but the subgraph instance then exposes a
    slot whose real type differs from its label — the frontend reports that as
    "已连接的节点输入输出类型不兼容" at the outer link.
    """
    problems = []
    for link in links:
        if link.get("type") != item.get("type"):
            problems.append(
                "subgraph %r: %s %r is %s but its wire %s carries %s"
                % (sg.get("name"), side, item.get("name"), item.get("type"),
                   link.get("id"), link.get("type")))
    return problems


def _check_inner_nodes(sg):
    """The inner nodes' own link fields must match the subgraph links."""
    problems = []
    by_id = {node["id"]: node for node in sg.get("nodes") or []}
    out_links, in_links = {}, {}
    for link in sg.get("links") or []:
        out_links.setdefault((link.get("origin_id"),
                              int(link.get("origin_slot", -1))),
                             []).append(link["id"])
        in_links[(link.get("target_id"),
                  int(link.get("target_slot", -1)))] = link["id"]
    for node in sg.get("nodes") or []:
        for slot, item in enumerate(node.get("outputs") or []):
            want = sorted(out_links.get((node["id"], slot), []))
            if sorted(item.get("links") or []) != want:
                problems.append("subgraph %r: node %s output %r links %s != %s"
                                % (sg.get("name"), node["id"],
                                   item.get("name"), item.get("links"), want))
        for slot, item in enumerate(node.get("inputs") or []):
            want = in_links.get((node["id"], slot))
            if item.get("link") != want:
                problems.append("subgraph %r: node %s input %r link %s != %s"
                                % (sg.get("name"), node["id"],
                                   item.get("name"), item.get("link"), want))
    return problems


def _check_ids(doc):
    """Node and link ids share one namespace across the whole document."""
    problems = []
    node_ids, link_ids = [], []
    for node in doc.get("nodes") or []:
        node_ids.append(node["id"])
    for link in doc.get("links") or []:
        link_ids.append(link[0] if isinstance(link, list) else link.get("id"))
    for sg in (doc.get("definitions") or {}).get("subgraphs") or []:
        for node in sg.get("nodes") or []:
            node_ids.append(node["id"])
        for link in sg.get("links") or []:
            link_ids.append(link.get("id"))
    dup_nodes = sorted({i for i in node_ids if node_ids.count(i) > 1})
    dup_links = sorted({i for i in link_ids if link_ids.count(i) > 1})
    if dup_nodes:
        problems.append("node ids are not unique: %s" % dup_nodes)
    if dup_links:
        problems.append("link ids are not unique: %s" % dup_links)
    highest_node = max(node_ids, default=0)
    highest_link = max(link_ids, default=0)
    if doc.get("last_node_id") != highest_node:
        problems.append("last_node_id %s != %d"
                        % (doc.get("last_node_id"), highest_node))
    if doc.get("last_link_id") != highest_link:
        problems.append("last_link_id %s != %d"
                        % (doc.get("last_link_id"), highest_link))
    return problems


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build(workflow_dir, force=False):
    path = os.path.join(workflow_dir, TARGET_FILE)
    existing = None
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
    doc = build_document(None if force else existing)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2)
    return path


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    workflow_dir = argv[0]
    flags = set(argv[1:])
    unknown = flags - {"--force", "--validate"}
    if unknown:
        print("unknown flag(s):", " ".join(sorted(unknown)))
        return 2
    path = os.path.join(workflow_dir, TARGET_FILE)
    if "--validate" in flags:
        problems = validate(path)
        if problems:
            print("INVALID %s" % path)
            for problem in problems:
                print("  -", problem)
            return 1
        print("OK %s" % path)
        return 0
    path = build(workflow_dir, force="--force" in flags)
    problems = validate(path)
    if problems:
        print("wrote %s but the contract check found:" % path)
        for problem in problems:
            print("  -", problem)
        return 1
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
