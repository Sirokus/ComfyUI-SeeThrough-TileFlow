"""Build (or repair) the guided variant of the four-step tile workflow.

``seethrough-四步分块流水线.json`` is the plain pipeline;
``seethrough-四步分块流水线-引导.json`` replaces the per-node SeeThrough chain
(GenerateLayers -> GenerateDepth -> PostProcess) with one
``STLU_GuidedLayers`` node that runs the same three steps tile by tile, so a
finished tile can be pasted into the next tile's overlap before it is
generated.

Modes (``workflow_dir`` points at the folder holding both files):

    python tools/build_guided_workflow.py <dir>             # rebuild, keep hand-tuned values
    python tools/build_guided_workflow.py <dir> --force     # rebuild from the plain file
    python tools/build_guided_workflow.py <dir> --fix-plain # repair the plain file only
    python tools/build_guided_workflow.py <dir> --validate  # contract check, no write

The default rebuild reads ``widgets_values_named`` (and node position, size,
colors) from the existing guided file, so a seed / strip setting / path the
user tuned survives.  ``--force`` throws those away.  The guide note is
replaced, not appended, so running the tool twice is idempotent.
"""

from __future__ import annotations

import json
import os
import sys

GUIDED = "STLU_GuidedLayers"
REMOVE = {24, 23, 20}  # GenerateLayers, GenerateDepth, PostProcess
GUIDE_NODE_ID = 60

PLAIN_FILE = "seethrough-四步分块流水线.json"
GUIDED_FILE = "seethrough-四步分块流水线-引导.json"

GUIDE_MARKER = "## 分块引导（引导版专属）"

# ① outputs, in RETURN_TYPES order
PIPE_OUT = {"tiles": 0, "indices": 1, "grid_doc": 2, "original": 3,
            "resolution": 4, "num_inference_steps": 5, "depth_resolution": 6,
            "seed": 7, "tblr_split": 8, "use_lama": 9}
# ② guided inputs, in INPUT_TYPES order
GUIDE_IN = {"tiles": 0, "indices": 1, "layerdiff_model": 2, "depth_model": 3,
            "grid_doc": 4, "random_seed": 5, "resolution": 6,
            "num_inference_steps": 7, "depth_resolution": 8,
            "tblr_split": 9, "use_lama": 10}

GUIDE_NOTE = (
    "\n## 分块引导（引导版专属）\n"
    "- **用已生成相邻块的重叠区做引导**：顺序逐块拆层，每块生成前把先完成的块"
    "在重叠区的像素回填进来，再接 SeeThrough。\n"
    "- **同目录缓存**：文件夹里之前已经生成好的块也会被读入当引导源；只勾选一块"
    "重跑时，左右/上下已完成的邻居仍能参与。\n"
    "- **羽化**：重叠区边缘的渐变像素；越大过渡越软、平均回填力度越低。\n"
    "- **强度%**：默认 40，只做轻推；100 = 用上一块的生成结果完全覆盖原图，"
    "会把它补全/脑补的内容（半张脸、衣摆、被头发盖住的眼睛）一起带进来。\n"
    "- **与原图容差**：只回填生成结果与原图平均色差不超过该值的像素，避免把补全"
    "内容带进下一块；默认 16，0 = 不过滤（危险）。\n"
    "- **接缝归属（合并）**：每个像素只归离自己块边更远的一块，跟它在那里画没画"
    "东西无关；落败块在接缝处被裁掉。单独成组时硬切，平板合并按「接缝过渡」淡出。"
    "相邻块各自补全出来的边会被丢弃，不再平均成一条重影。\n"
    "- **同名图层单独成组（不裁剪）**：勾选时每块的原件按部位收进同名文件夹，"
    "不做接缝裁切，便于生成出错后直接修；不勾时按接缝归属裁切，输出直出合并版。\n"
    "- **输出位置**：结果直接写入 ① 里选的文件夹，不再每次新建波次子目录；"
    "⑤ 预览节点会显示合并后的 PSD。\n"
    "- 参数在 ① 里改；②③④ 节点的 widget 已锁定。\n"
)

# Widget order of ① / ④ / ⑤, with the defaults from their INPUT_TYPES.
# Workflow files store widget values positionally, so every widget the node
# defines needs an entry -- including ones an older save predates.  Rebuilding
# from this table (instead of truncating by the number of widget-input
# records) keeps the order correct when a workflow is saved by an older
# version.
PIPE_ORDER = [
    ("tile_width", 1024), ("tile_height", 1024), ("overlap", 128),
    ("select", "all"), ("resolution", 512), ("num_inference_steps", 30),
    ("depth_resolution", -1), ("random_seed", 0), ("tblr_split", True),
    ("use_lama", True), ("alpha_threshold", 8),
    ("out_dir", "SeeThrough/我的自设/"), ("group", ""),
    ("filename_prefix", "tile"), ("merge_name", "merged"),
    ("group_layers", True), ("include_depth", False),
    ("include_original", False), ("guide_context", True),
    ("guide_feather", 32), ("guide_strength", 40), ("guide_tolerance", 16),
    ("seam_feather", 32), ("grid_offset_x", 0),
    ("grid_offset_y", 0), ("partition_mode", "均匀网格"),
    ("strip_max_side", 2048), ("strip_max_resolution", 768),
    ("strip_margin", 32), ("select_plan", ""), ("context_pad", 0),
    ("custom_regions", "[]"),
]
PIPE_TYPES = {
    "select": "STRING", "tblr_split": "BOOLEAN", "use_lama": "BOOLEAN",
    "out_dir": "STRING", "group": "STRING", "filename_prefix": "STRING",
    "merge_name": "STRING", "group_layers": "BOOLEAN",
    "include_depth": "BOOLEAN", "include_original": "BOOLEAN",
    "guide_context": "BOOLEAN",
    "partition_mode": "COMBO", "select_plan": "STRING",
    "context_pad": "INT", "custom_regions": "STRING",
}
MERGE_ORDER = [
    ("out_dir", "SeeThrough/Tiles"), ("name", "merged"),
    ("group_layers", True), ("include_depth", False),
    ("include_original", False), ("run_dir", ""), ("seam_feather", 32),
]
MERGE_TYPES = {
    "out_dir": "STRING", "name": "STRING", "group_layers": "BOOLEAN",
    "include_depth": "BOOLEAN", "include_original": "BOOLEAN",
    "run_dir": "STRING",
}
PREVIEW_ORDER = [("max_side", 1600), ("show_parts", True)]
PREVIEW_TYPES = {"show_parts": "BOOLEAN"}

GUIDE_WIDGET_FIELDS = ["guide_context", "guide_feather", "guide_strength",
                       "guide_tolerance"]


def _order(entries):
    return [name for name, _default in entries]


def _types(entries, overrides):
    return {name: overrides.get(name, "INT") for name, _default in entries}


def _named_values(node, order):
    """The node's widget values as a name -> value dict."""
    named = node.get("widgets_values_named")
    if isinstance(named, dict) and named:
        return dict(named)
    values = list(node.get("widgets_values") or [])
    return {name: values[i] for i, (name, _default) in enumerate(order)
            if i < len(values)}


def _rebuild_values(node, order, overrides):
    """Rewrite ``widgets_values`` in the node's real widget order."""
    named = _named_values(node, order)
    for name, default in order:
        named.setdefault(name, default)
    named.update(overrides or {})
    node["widgets_values_named"] = named
    node["widgets_values"] = [named.get(name, default)
                              for name, default in order]


def _widget_input(name, type_name):
    return {"localized_name": name, "name": name, "shape": 7,
            "type": type_name, "widget": {"name": name}, "link": None}


def _ensure_widget_inputs(node, order, type_overrides):
    """Add input records for widgets an older save did not know about."""
    have = {item.get("name") for item in node.get("inputs") or []
            if item.get("widget")}
    types = _types(order, type_overrides)
    for name, _default in order:
        if name not in have:
            node.setdefault("inputs", []).append(_widget_input(name, types[name]))


def _slot(name, type_name, link=None, shape=None):
    item = {"localized_name": name, "name": name, "type": type_name, "link": link}
    if shape is not None:
        item["shape"] = shape
    return item


def _out(name, type_name, links):
    return {"localized_name": name, "name": name, "type": type_name,
            "links": list(links)}


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save(path, doc):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2)


def _nodes_by_id(doc):
    return {node["id"]: node for node in doc["nodes"]}


def _carry_over_node(old, new, order):
    """Keep one existing node's hand-tuned widgets, layout and properties."""
    if old is None or new is None:
        return
    _rebuild_values(new, order, _named_values(old, order))
    for key in ("pos", "size", "color", "bgcolor"):
        if key in old:
            new[key] = old[key]
    props = {k: v for k, v in (old.get("properties") or {}).items()
             if k not in ("Node name for S&R", "stlu_hidden_widgets")}
    if props:
        new.setdefault("properties", {}).update(props)


def _replace_note(doc):
    for node in doc["nodes"]:
        if node.get("type") != "MarkdownNote":
            continue
        text = (node.get("widgets_values") or [""])[0]
        marker = text.find(GUIDE_MARKER)
        if marker >= 0:
            # The marker line starts after the preceding newline.
            start = text.rfind("\n", 0, marker)
            text = text[:start if start >= 0 else marker]
        text = text.rstrip() + GUIDE_NOTE
        node["widgets_values"][0] = text
        if node.get("widgets_values_named"):
            node["widgets_values_named"]["text"] = text


def fix_plain(workflow_dir: str) -> str:
    """Repair the plain workflow's widget records; no structural change."""
    path = os.path.join(workflow_dir, PLAIN_FILE)
    doc = _load(path)
    nodes = _nodes_by_id(doc)

    pipe = nodes[50]
    _rebuild_values(pipe, PIPE_ORDER, {})
    _ensure_widget_inputs(pipe, PIPE_ORDER, PIPE_TYPES)
    pipe.setdefault("properties", {})["stlu_hidden_widgets"] = list(GUIDE_WIDGET_FIELDS)

    merge_node = nodes[34]
    _rebuild_values(merge_node, MERGE_ORDER, {"run_dir": ""})
    _ensure_widget_inputs(merge_node, MERGE_ORDER, MERGE_TYPES)

    preview = nodes[47]
    _rebuild_values(preview, PREVIEW_ORDER, {})
    _ensure_widget_inputs(preview, PREVIEW_ORDER, PREVIEW_TYPES)

    _save(path, doc)
    return path


def build(workflow_dir: str, force: bool = False) -> str:
    source = os.path.join(workflow_dir, PLAIN_FILE)
    target = os.path.join(workflow_dir, GUIDED_FILE)
    doc = _load(source)
    existing = None if force else (_load(target) if os.path.isfile(target) else None)

    nodes = _nodes_by_id(doc)

    # Drop the batch chain and every link that touched it.
    doc["nodes"] = [node for node in doc["nodes"] if node["id"] not in REMOVE]
    # 15/16 became the guided node's outputs; 45 is rewired below.
    stale = {15, 16}
    kept_links = [link for link in doc["links"]
                  if link[1] not in REMOVE and link[3] not in REMOVE
                  and link[0] not in stale]

    pipe = nodes[50]
    tile_psd = nodes[45]
    layer_model = nodes[28]
    depth_model = nodes[29]

    # ① keeps its existing input order (sockets included: links 101/102 land
    # on fixed slot numbers).  Only records an older save did not know about
    # are appended, by name -- never reordered.  The guided workflow uses the
    # guide widgets, so it must not inherit the plain file's hidden list.
    pipe.setdefault("properties", {}).pop("stlu_hidden_widgets", None)
    _rebuild_values(pipe, PIPE_ORDER, {})
    _ensure_widget_inputs(pipe, PIPE_ORDER, PIPE_TYPES)

    guide_inputs = [
        _slot("tiles", "IMAGE", 21),
        _slot("indices", "STRING", 22),
        _slot("layerdiff_model", "SEETHROUGH_LAYERDIFF_MODEL", 23),
        _slot("depth_model", "SEETHROUGH_DEPTH_MODEL", 24),
        _slot("grid_doc", "STRING", 25, shape=7),
        _slot("random_seed", "INT", 26),
        _slot("resolution", "INT", 27),
        _slot("num_inference_steps", "INT", 28),
        _slot("depth_resolution", "INT", 29),
        _slot("tblr_split", "BOOLEAN", 30),
        _slot("use_lama", "BOOLEAN", 31),
    ]
    for item in guide_inputs:
        if item["name"] in ("random_seed", "resolution", "num_inference_steps",
                            "depth_resolution", "tblr_split", "use_lama"):
            item["widget"] = {"name": item["name"]}

    guide = {
        "id": GUIDE_NODE_ID,
        "type": GUIDED,
        "pos": [738.35791907821, 309.4736282897432],
        "size": [420, 330],
        "flags": {},
        "order": 5,
        "mode": 0,
        "inputs": guide_inputs,
        "outputs": [
            _out("parts", "SEETHROUGH_PARTS", [32]),
            _out("indices", "STRING", [33]),
        ],
        "title": "② 顺序拆层（带重叠引导，不用改）",
        "properties": {"Node name for S&R": GUIDED, "stlu_locked": True},
        "widgets_values": [42, 512, 30, -1, True, True],
        "widgets_values_named": {
            "random_seed": 42, "resolution": 512, "num_inference_steps": 30,
            "depth_resolution": -1, "tblr_split": True, "use_lama": True,
        },
        "color": "#333",
        "bgcolor": "#222",
    }
    doc["nodes"].append(guide)

    new_links = [
        [21, 50, PIPE_OUT["tiles"], GUIDE_NODE_ID, GUIDE_IN["tiles"], "IMAGE"],
        [22, 50, PIPE_OUT["indices"], GUIDE_NODE_ID, GUIDE_IN["indices"], "STRING"],
        [23, 28, 0, GUIDE_NODE_ID, GUIDE_IN["layerdiff_model"], "SEETHROUGH_LAYERDIFF_MODEL"],
        [24, 29, 0, GUIDE_NODE_ID, GUIDE_IN["depth_model"], "SEETHROUGH_DEPTH_MODEL"],
        [25, 50, PIPE_OUT["grid_doc"], GUIDE_NODE_ID, GUIDE_IN["grid_doc"], "STRING"],
        [26, 50, PIPE_OUT["seed"], GUIDE_NODE_ID, GUIDE_IN["random_seed"], "INT"],
        [27, 50, PIPE_OUT["resolution"], GUIDE_NODE_ID, GUIDE_IN["resolution"], "INT"],
        [28, 50, PIPE_OUT["num_inference_steps"], GUIDE_NODE_ID,
         GUIDE_IN["num_inference_steps"], "INT"],
        [29, 50, PIPE_OUT["depth_resolution"], GUIDE_NODE_ID,
         GUIDE_IN["depth_resolution"], "INT"],
        [30, 50, PIPE_OUT["tblr_split"], GUIDE_NODE_ID, GUIDE_IN["tblr_split"], "BOOLEAN"],
        [31, 50, PIPE_OUT["use_lama"], GUIDE_NODE_ID, GUIDE_IN["use_lama"], "BOOLEAN"],
        [32, GUIDE_NODE_ID, 0, 45, 0, "SEETHROUGH_PARTS"],
        [33, GUIDE_NODE_ID, 1, 45, 1, "STRING"],
        [34, 50, PIPE_OUT["grid_doc"], 45, 2, "STRING"],
    ]
    doc["links"] = kept_links + new_links
    doc["last_node_id"] = max(doc["last_node_id"], GUIDE_NODE_ID)
    doc["last_link_id"] = max(doc["last_link_id"], 34)

    # Rewire the outputs that used to feed the deleted chain.
    pipe_outputs = {out["name"]: out for out in pipe["outputs"]}
    for name, ids in {
        "tiles": [21], "indices": [22], "grid_doc": [25, 34, 19], "original": [18],
        "resolution": [27], "num_inference_steps": [28],
        "depth_resolution": [29], "seed": [26], "tblr_split": [30],
        "use_lama": [31],
    }.items():
        if name in pipe_outputs:
            pipe_outputs[name]["links"] = ids
    layer_model["outputs"][0]["links"] = [23]
    depth_model["outputs"][0]["links"] = [24]

    # ③ now reads the guided node instead of the deleted PostProcess.
    tile_inputs = {item["name"]: item for item in tile_psd["inputs"]}
    tile_inputs["parts"]["link"] = 32
    tile_inputs["index"]["link"] = 33
    tile_inputs["grid_doc"]["link"] = 34

    # ④ keeps its widgets in the same order; the merge-only seam width is
    # appended so older saves stay aligned.
    merge_node = nodes.get(34)
    if merge_node is not None:
        merge_node["inputs"] = [item for item in merge_node.get("inputs") or []
                                if item.get("name") != "seam_feather"]
        _rebuild_values(merge_node, MERGE_ORDER, {})
        _ensure_widget_inputs(
            merge_node,
            [(name, default) for name, default in MERGE_ORDER
             if name != "seam_feather"], MERGE_TYPES)
        merge_node["inputs"].append(_widget_input("seam_feather", "INT"))

    # ⑤ is untouched structurally, but its widget list must stay complete and
    # a hand-tuned max_side / show_parts survives the rebuild.
    preview = nodes.get(47)
    if preview is not None:
        _rebuild_values(preview, PREVIEW_ORDER, {})
        _ensure_widget_inputs(preview, PREVIEW_ORDER, PREVIEW_TYPES)

    if existing is not None:
        before = _nodes_by_id(existing)
        after = _nodes_by_id(doc)
        for node_id, order in ((50, PIPE_ORDER), (34, MERGE_ORDER),
                               (47, PREVIEW_ORDER)):
            _carry_over_node(before.get(node_id), after.get(node_id), order)

    for node in doc["nodes"]:
        if node.get("id") in (28, 29):
            # Model pickers stay visible; everything else stays locked.
            node.setdefault("properties", {})["stlu_show_widgets"] = ["model"]

    _replace_note(doc)

    _save(target, doc)
    return target


def validate(workflow_dir: str) -> list:
    """Return a list of contract problems (empty means the files are sane)."""
    problems = []
    docs = {}
    for name in (PLAIN_FILE, GUIDED_FILE):
        path = os.path.join(workflow_dir, name)
        if not os.path.isfile(path):
            problems.append("%s: missing" % name)
            continue
        doc = _load(path)
        docs[name] = doc
        ids = [node["id"] for node in doc["nodes"]]
        if len(ids) != len(set(ids)):
            problems.append("%s: duplicate node ids" % name)
        nodes = _nodes_by_id(doc)
        for link in doc.get("links") or []:
            link_id, src, src_slot, dst, dst_slot, type_name = link[:6]
            if src not in nodes:
                problems.append("%s: link %s starts at missing node %s"
                                % (name, link_id, src))
                continue
            if dst not in nodes:
                problems.append("%s: link %s ends at missing node %s"
                                % (name, link_id, dst))
                continue
            outputs = nodes[src].get("outputs") or []
            inputs = nodes[dst].get("inputs") or []
            if not 0 <= src_slot < len(outputs):
                problems.append("%s: link %s source slot %s out of range"
                                % (name, link_id, src_slot))
                continue
            if not 0 <= dst_slot < len(inputs):
                problems.append("%s: link %s target slot %s out of range"
                                % (name, link_id, dst_slot))
                continue
            out_type = outputs[src_slot].get("type")
            in_type = inputs[dst_slot].get("type")
            if out_type not in (None, "*") and in_type not in (None, "*") \
                    and in_type != "COMBO" and out_type != in_type:
                problems.append(
                    "%s: link %s type %r does not match %r -> %r"
                    % (name, link_id, type_name, out_type, in_type))
        for node in doc["nodes"]:
            if not str(node.get("type") or "").startswith("STLU_"):
                continue  # upstream nodes use widget lists we do not own
            widget_inputs = [item for item in node.get("inputs") or []
                             if item.get("widget")]
            values = node.get("widgets_values") or []
            if widget_inputs and len(values) != len(widget_inputs):
                problems.append(
                    "%s: node %s has %d widget values for %d widget inputs"
                    % (name, node["id"], len(values), len(widget_inputs)))
            named = node.get("widgets_values_named")
            if isinstance(named, dict) and widget_inputs:
                missing = [item["name"] for item in widget_inputs
                           if item["name"] not in named]
                if missing:
                    problems.append(
                        "%s: node %s named values missing %s"
                        % (name, node["id"], ", ".join(missing)))

    plain = docs.get(PLAIN_FILE)
    if plain is not None:
        nodes = _nodes_by_id(plain)
        merge_node = nodes.get(34) or {}
        if str(_named_values(merge_node, MERGE_ORDER).get("run_dir") or ""):
            problems.append("%s: ④ run_dir should be empty" % PLAIN_FILE)
        preview = nodes.get(47) or {}
        preview_values = _named_values(preview, PREVIEW_ORDER)
        if "show_parts" not in preview_values:
            problems.append("%s: ⑤ widget list lacks show_parts" % PLAIN_FILE)
        hidden = (nodes.get(50) or {}).get("properties", {}).get(
            "stlu_hidden_widgets") or []
        for field in GUIDE_WIDGET_FIELDS:
            if field not in hidden:
                problems.append("%s: ① should hide %s" % (PLAIN_FILE, field))

    guided = docs.get(GUIDED_FILE)
    if guided is not None:
        nodes = _nodes_by_id(guided)
        if GUIDE_NODE_ID not in nodes:
            problems.append("%s: missing ② guided node" % GUIDED_FILE)
        note_hits = sum(1 for node in guided["nodes"]
                        if node.get("type") == "MarkdownNote"
                        and GUIDE_MARKER in str((node.get("widgets_values") or [""])[0]))
        if note_hits > 1:
            problems.append("%s: guide note appended more than once" % GUIDED_FILE)
    return problems


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    workflow_dir = argv[1]
    flags = set(argv[2:])
    if flags - {"--force", "--validate", "--fix-plain"}:
        raise SystemExit("unknown flags: %s" % " ".join(sorted(flags - {
            "--force", "--validate", "--fix-plain"})))

    if "--validate" in flags:
        problems = validate(workflow_dir)
        if problems:
            print("INVALID (%d)" % len(problems))
            for item in problems:
                print("  -", item)
            return 1
        print("OK: both workflows satisfy the contract")
        return 0

    if "--fix-plain" in flags:
        print(fix_plain(workflow_dir))
        problems = validate(workflow_dir)
        for item in problems:
            print("  -", item)
        return 1 if problems else 0

    target = build(workflow_dir, force="--force" in flags)
    print(target)
    problems = validate(workflow_dir)
    for item in problems:
        print("  -", item)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
