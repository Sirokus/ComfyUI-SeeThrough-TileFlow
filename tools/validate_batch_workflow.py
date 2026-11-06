"""Static validation for the generated batch workflow (no ComfyUI needed)."""

from __future__ import annotations

import json
import os
import sys

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for p in (PKG_DIR, COMFY_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import nodes as stlu  # noqa: E402

WORKFLOW_NAME = "Live2D层级二次放大-批量.json"


def _find_workflow():
    """Locate the batch workflow wherever the user filed it."""
    root = os.path.join(COMFY_ROOT, "user", "default", "workflows")
    for dirpath, _dirnames, filenames in os.walk(root):
        if WORKFLOW_NAME in filenames:
            return os.path.join(dirpath, WORKFLOW_NAME)
    return os.path.join(root, WORKFLOW_NAME)


WORKFLOW = _find_workflow()

EXPECTED = {
    1001: "STLU_Plan",
    1002: "UNETLoader",
    1003: "QwenImage21Cache",
    1004: "CLIPLoader",
    1005: "CLIPLoader",
    1006: "VAELoader",
    1007: "LoadImage",
    1008: "STLU_UpscaleBatch",
    1009: "STLU_AssemblePSD",
    1010: "ImageCompare",
    1011: "MarkdownNote",
}

# Widget count per STLU node, from INPUT_TYPES.
WIDGET_COUNTS = {
    "STLU_Plan": 12,
    "STLU_UpscaleBatch": 4,
    "STLU_AssemblePSD": 6,
}

_failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _failures.append(msg)


def main():
    with open(WORKFLOW, "r", encoding="utf-8") as f:
        wf = json.load(f)
    nodes = {n["id"]: n for n in wf["nodes"]}
    check(set(nodes) == set(EXPECTED), "node ids match")
    for nid, cls in EXPECTED.items():
        check(nodes.get(nid, {}).get("type") == cls, "node %d is %s" % (nid, cls))

    # Widget counts.  The frontend may append extra values (seed control,
    # promoted subgraph widgets), so require at least one value per widget
    # input instead of an exact count.
    for nid, node in nodes.items():
        cls = node["type"]
        if cls in WIDGET_COUNTS:
            widget_inputs = [i for i in (node.get("inputs") or []) if i.get("widget")]
            values = node.get("widgets_values") or []
            check(len(values) >= len(widget_inputs),
                  "%s widgets_values covers widget inputs (%d/%d)"
                  % (cls, len(values), len(widget_inputs)))

    # Every link endpoint is valid and types agree.
    links = wf.get("links", [])
    check(len(links) == len({l[0] for l in links}), "link ids unique")
    for link in links:
        lid, origin, oslot, target, tslot, ltype = link
        ok = origin in nodes and target in nodes
        check(ok, "link %d endpoints exist" % lid)
        if not ok:
            continue
        out_slots = nodes[origin].get("outputs") or []
        in_slots = nodes[target].get("inputs") or []
        check(oslot < len(out_slots), "link %d origin slot %d" % (lid, oslot))
        check(tslot < len(in_slots), "link %d target slot %d" % (lid, tslot))
        if oslot < len(out_slots) and tslot < len(in_slots):
            out_type = out_slots[oslot].get("type")
            in_type = in_slots[tslot].get("type")
            check(out_type == in_type or in_type == "*",
                  "link %d type %s -> %s" % (lid, out_type, in_type))

    # Mirror fields agree with the link table.
    for link in links:
        lid, origin, oslot, target, tslot, _ = link
        check(nodes[target]["inputs"][tslot].get("link") == lid,
              "link %d mirrored on target input" % lid)
        check(lid in (nodes[origin]["outputs"][oslot].get("links") or []),
              "link %d mirrored on origin output" % lid)

    # The intended wiring, looked up by name: the frontend re-orders a node's
    # input slots when the workflow is saved, so indexes are not stable.
    def linked(node, name):
        for inp in node.get("inputs") or []:
            if inp.get("name") == name:
                return inp.get("link") is not None
        return False

    batch = nodes[1008]
    for name in ("plan", "reference", "model", "clip", "vae", "pe_clip"):
        check(linked(batch, name), "batch.%s linked" % name)
    check(linked(nodes[1009], "deps"), "psd.deps linked (execution order)")

    # Node class names referenced by the workflow exist in this install's schema
    # for our own nodes, and are known ComfyUI node types for the rest.
    known = set(stlu.NODE_CLASS_MAPPINGS) | {
        "UNETLoader", "QwenImage21Cache", "CLIPLoader", "VAELoader",
        "LoadImage", "ImageCompare", "MarkdownNote",
    }
    for node in wf["nodes"]:
        check(node["type"] in known, "type known: %s" % node["type"])

    print()
    if _failures:
        print("FAILED: %d" % len(_failures))
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
