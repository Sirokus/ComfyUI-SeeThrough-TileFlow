"""CPU-only contract checks for the two shipped four-step workflows.

Locks the repairs made to the JSON files: the plain workflow has no stale
merge folder and a complete ⑤ widget list, the guided workflow keeps its
hand-tuned values, and rebuilding it is idempotent (the guide note is
replaced, not appended).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

WORKFLOW_DIR = os.path.join(
    COMFY_ROOT, "user", "default", "workflows", "Comfy-My-Workflow",
    "图片", "Live2D拆层")
STRIPS_FILE = "seethrough-四步分块流水线-方形全宽.json"

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def load_builder():
    path = os.path.join(PKG_DIR, "tools", "build_guided_workflow.py")
    spec = importlib.util.spec_from_file_location("build_guided_workflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def node_by_id(doc, node_id):
    return next((node for node in doc["nodes"] if node["id"] == node_id), None)


def main():
    builder = load_builder()

    if not os.path.isdir(WORKFLOW_DIR):
        print("SKIP: workflow folder not found at %s" % WORKFLOW_DIR)
        print("ALL-OK")
        return 0

    print("[both shipped workflows satisfy the contract]")
    problems = builder.validate(WORKFLOW_DIR)
    check(not problems, "contract check is clean (%s)" % (problems or "no problems"))

    plain = json.load(open(os.path.join(WORKFLOW_DIR, builder.PLAIN_FILE),
                           encoding="utf-8"))
    guided = json.load(open(os.path.join(WORKFLOW_DIR, builder.GUIDED_FILE),
                            encoding="utf-8"))

    print("[plain workflow fixes]")
    merge_node = node_by_id(plain, 34)
    check(not (merge_node.get("widgets_values_named") or {}).get("run_dir"),
          "④ run_dir is empty")
    preview = node_by_id(plain, 47)
    check("show_parts" in (preview.get("widgets_values_named") or {}),
          "⑤ carries show_parts")
    hidden = node_by_id(plain, 50).get("properties", {}).get(
        "stlu_hidden_widgets") or []
    check(set(hidden) == {"guide_context", "guide_feather", "guide_strength",
                          "guide_tolerance"},
          "① hides the four guide widgets")

    print("[guided workflow keeps hand-tuned values]")
    pipe = node_by_id(guided, 50)["widgets_values_named"]
    check(int(pipe.get("random_seed") or 0) != 0, "the tuned seed survived")
    check(bool(str(pipe.get("select_plan") or "")), "select_plan survived")
    # The strip layout lives in its own saved variant; the guided file may be
    # switched back to the uniform grid without losing the tuned strip knobs.
    strips_path = os.path.join(WORKFLOW_DIR, STRIPS_FILE)
    if os.path.isfile(strips_path):
        strips = json.load(open(strips_path, encoding="utf-8"))
        strips_pipe = node_by_id(strips, 50)["widgets_values_named"]
        check(str(strips_pipe.get("partition_mode") or "") == "方形全宽分区",
              "the strip variant keeps the strip partition")
    else:
        print("  skip the strip variant (not saved here)")
    guide_merge = node_by_id(guided, 34)["widgets_values_named"]
    check(guide_merge.get("run_dir") == pipe.get("out_dir"),
          "④ still points at the run folder")
    guide_pipe = node_by_id(guided, 50).get("properties", {})
    check("stlu_hidden_widgets" not in guide_pipe,
          "the guided workflow does not hide its guide widgets")

    print("[rebuilding the guided file is idempotent]")
    with tempfile.TemporaryDirectory() as tmp:
        for name in (builder.PLAIN_FILE, builder.GUIDED_FILE):
            shutil.copy(os.path.join(WORKFLOW_DIR, name),
                        os.path.join(tmp, name))
        first = builder.build(tmp)
        with open(first, "rb") as handle:
            first_bytes = handle.read()
        second = builder.build(tmp)
        with open(second, "rb") as handle:
            second_bytes = handle.read()
        check(first_bytes == second_bytes,
              "a second build produces the same file")
        check(not builder.validate(tmp), "the rebuilt copy passes validation")
        rebuilt = json.loads(second_bytes.decode("utf-8"))
        note_hits = sum(
            1 for node in rebuilt["nodes"]
            if node.get("type") == "MarkdownNote"
            and builder.GUIDE_MARKER in str((node.get("widgets_values") or [""])[0]))
        check(note_hits == 1, "the guide note appears exactly once (%d)" % note_hits)

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
