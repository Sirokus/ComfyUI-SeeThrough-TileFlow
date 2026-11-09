"""CPU-only contract checks for the transparent (subgraph) workflow.

Builds the file into a temporary directory with
``tools/build_transparent_workflow.py`` and locks the structure the frontend
and the backend both rely on:

* four subgraphs, four instances, one settings node with every panel widget,
* every inner node type registered by this pack,
* the contract check passes, and rebuilding is byte-identical (idempotent).

The real workflow file, when present, must pass the same contract.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import nodes as stlu  # noqa: E402

WORKFLOW_DIR = os.path.join(
    COMFY_ROOT, "user", "default", "workflows", "Comfy-My-Workflow",
    "图片", "Live2D拆层")

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def load_builder():
    path = os.path.join(PKG_DIR, "tools", "build_transparent_workflow.py")
    spec = importlib.util.spec_from_file_location("build_transparent_workflow",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    builder = load_builder()
    with tempfile.TemporaryDirectory() as tmp:
        path = builder.build(tmp)
        check(os.path.isfile(path), "the builder writes the workflow")
        problems = builder.validate(path)
        check(not problems, "the contract check passes (%s)"
              % ("; ".join(problems[:3]) or "no problems"))

        first = open(path, "rb").read()
        builder.build(tmp)
        second = open(path, "rb").read()
        check(first == second, "rebuilding twice is byte-identical")

        doc = json.loads(first.decode("utf-8"))
        subgraphs = doc["definitions"]["subgraphs"]
        check(len(subgraphs) == 4, "four subgraphs (%d)" % len(subgraphs))
        ids = {sg["id"] for sg in subgraphs}
        instances = [node for node in doc["nodes"]
                     if node.get("type") in ids]
        check(len(instances) == 4, "four subgraph instances (%d)"
              % len(instances))

        inner_types = {node["type"] for sg in subgraphs
                       for node in sg["nodes"]}
        missing = sorted(t for t in inner_types
                         if t not in stlu.NODE_CLASS_MAPPINGS)
        check(not missing, "every inner node type is registered (%s)"
              % (missing or "none missing"))

        settings = next(node for node in doc["nodes"]
                        if node["type"] == "STLU_PipelineSettings")
        panel_widgets = {"select", "select_plan", "custom_regions",
                         "partition_mode", "tile_width", "tile_height",
                         "overlap", "resolution", "random_seed",
                         "strip_max_side", "strip_max_resolution",
                         "strip_margin", "grid_offset_x", "grid_offset_y",
                         "context_pad"}
        names = set(settings.get("widgets_values_named") or {})
        check(panel_widgets <= names, "the settings node carries every widget "
              "the grid panel reads (%s)" % sorted(panel_widgets - names))
        check(len(settings["widgets_values"]) == len(names),
              "widgets_values matches widgets_values_named")

        # The panel needs a link back to LoadImage to draw the preview.
        load = next(node for node in doc["nodes"]
                    if node["type"] == "LoadImage")
        check(any(item.get("name") == "image" and item.get("link")
                  for item in settings["inputs"]),
              "the settings node is linked to LoadImage")
        image_links = [link for link in doc["links"]
                       if link[1] == load["id"]] if isinstance(
                           doc["links"][0], list) else [
            link for link in doc["links"] if link.get("origin_id") == load["id"]]
        check(len(image_links) >= 2,
              "LoadImage feeds both the settings node and the plan subgraph")

        # A PreviewImage mid-run only flooded the queue with images nobody
        # asked for, so the transparent file must not carry one.
        previews = [node for node in doc["nodes"]
                    if node["type"] == "PreviewImage"]
        check(not previews,
              "no mid-run PreviewImage nodes (%d)" % len(previews))
        psd_preview = next(node for node in doc["nodes"]
                           if node["type"] == "STLU_PSDPreview")
        connected = {psd_preview["inputs"][link[4]].get("name")
                     for link in doc["links"] if isinstance(link, list)
                     and link[3] == psd_preview["id"]}
        check(connected == {"psd_path"},
              "the ⑤ panel takes only ④'s merged PSD (%s)"
              % (sorted(s for s in connected if s) or "nothing"))

    real = os.path.join(WORKFLOW_DIR, builder.TARGET_FILE)
    subgraphs = []
    if os.path.isfile(real):
        with open(real, encoding="utf-8") as fh:
            subgraphs = (json.load(fh).get("definitions") or {}).get("subgraphs") or []
    if subgraphs:
        problems = builder.validate(real)
        check(not problems, "the shipped workflow passes the contract (%s)"
              % ("; ".join(problems[:3]) or "no problems"))
    else:
        print("  skip the shipped workflow is not the split-step variant")

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
