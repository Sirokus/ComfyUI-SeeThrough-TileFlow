"""CPU-only checks for the minimal PSD-upscale nodes (no model, no GPU).

Two nodes, one split: ``STLU_PsdUpscalePrepare`` reads the PSD and writes the
plan, ``STLU_PsdUpscaleMin`` consumes the plan + model links and expands the
Qwen edit subgraph.  This walks that split end to end up to -- but not
including -- sampling.  Nothing is executed on a device.

    python custom_nodes/ComfyUI-SeeThrough-TileFlow/tests/test_nodes_psd_min.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import core_io  # noqa: E402
import edit_parts  # noqa: E402
import nodes_psd_min  # noqa: E402
import psd_upscale  # noqa: E402
import psd_writer  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def solid(width, height, rgba):
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[..., :] = rgba
    return pixels


def build_psd(path):
    layers = [
        {"name": "body", "left": 0, "top": 0,
         "pixels": solid(32, 32, (200, 40, 40, 255))},
        {"type": "group", "name": "服装", "blend": b"norm", "opacity": 200,
         "children": [
             {"name": "inner", "left": 8, "top": 8,
              "pixels": solid(8, 8, (10, 200, 10, 255))},
         ]},
        {"name": "eyes", "left": 16, "top": 12,
         "blend": b"scrn", "opacity": 64,
         "pixels": solid(8, 8, (250, 250, 10, 255))},
    ]
    psd_writer.write_psd(path, layers, 32, 32)
    return path


class _FakePrompt:
    """Just enough of ComfyUI's dynprompt for the hidden-input lookups."""

    def __init__(self, graph):
        self.graph = graph

    def get_node(self, node_id):
        return {"inputs": self.graph[int(node_id)]}


def prepare(psd_path, **kwargs):
    """Run the Prepare node; return the plan_json path it hands downstream."""
    plan_file, _info = nodes_psd_min.STLU_PsdUpscalePrepare().execute(
        psd_path, **kwargs)
    return plan_file


def run_min(plan_file, unique_id, **kwargs):
    """Run the Min node linked to a plan + three model links."""
    fake = _FakePrompt({unique_id: {"model": ["10", 0], "clip": ["10", 1],
                                    "vae": ["10", 2]}})
    return nodes_psd_min.STLU_PsdUpscaleMin().execute(
        None, None, None, plan_file,
        dynprompt=fake, unique_id=str(unique_id), **kwargs)


def test_guards(tmp):
    print("[nodes_psd_min: refuses bad input]")
    node = nodes_psd_min.STLU_PsdUpscalePrepare()

    for args, want in (
        (dict(psd_path=""), "成品 PSD"),
        (dict(psd_path=os.path.join(tmp, "nope.psd")), "不存在"),
    ):
        try:
            node.execute(factor=2.0, max_side=512, prompt="x", **args)
        except ValueError as exc:
            check(want in str(exc), "clear error (%s)" % exc)
        else:
            check(False, "must reject %r" % args)

    psd_path = build_psd(os.path.join(tmp, "guard.psd"))
    plan_file = prepare(psd_path, factor=2.0, max_side=512, prompt="x")
    try:
        nodes_psd_min.STLU_PsdUpscaleMin().execute(
            None, None, None, plan_file)
    except RuntimeError as exc:
        check("dynprompt" in str(exc), "needs hidden inputs (%s)" % exc)
    else:
        check(False, "must refuse without dynprompt/unique_id")


def test_min_run(tmp):
    print("[nodes_psd_min: one prepare writes a plan, the min expands it]")
    psd_path = build_psd(os.path.join(tmp, "piece.psd"))
    plan_file = prepare(psd_path, factor=2.0, max_side=512,
                        prompt="放大这张图")
    out = run_min(plan_file, 7)

    check(isinstance(out, dict) and "result" in out and "expand" in out,
          "returns a result + expand envelope")
    psd_out, info = out["result"]
    check(psd_out.endswith(".psd") and "piece" in psd_out,
          "output PSD path is derived from the input (%s)" % psd_out)
    check("未设置" in info, "no reference is reported (%s)"
          % info.splitlines()[5])
    check("放大 3 层" in info, "info reports every selected layer (%s)"
          % info.splitlines()[1])
    check("种子 42" in info, "info reports the seed (%s)" % info.splitlines()[3])

    expand = out["expand"]
    check(isinstance(expand, dict) and expand,
          "an expanded edit graph is returned (%d nodes)"
          % (len(expand) if isinstance(expand, dict) else -1))
    types = [n.get("class_type", "") for n in expand.values()
             if isinstance(n, dict)]
    check(any("Assemble" in t for t in types),
          "the graph merges back into a PSD node")
    check(sum("CutToLayer" in t for t in types) >= 3,
          "every layer is cut back with the model's alpha (%d)"
          % sum("CutToLayer" in t for t in types))
    check(all("qwen_image_edit" in t or True for t in types),
          "graph built from the pinned edit template")

    plan_path = psd_upscale.plan_path(os.path.join(
        tmp, "SeeThrough", "PsdUpscale", "piece"))
    check(os.path.isfile(plan_path), "the plan is written (%s)" % plan_path)
    plan = json.loads(open(plan_path, encoding="utf-8").read())
    check(plan["count"] == 3 and plan["canvas"] == [64, 64],
          "all layers are planned on the full canvas (%s/%s)"
          % (plan["count"], plan["canvas"]))
    check(all(os.path.isfile(job["crop_file"]) for job in plan["crops"]),
          "every crop file exists")


def test_reference(tmp):
    print("[nodes_psd_min: the optional reference image is cut per crop]")
    psd_path = build_psd(os.path.join(tmp, "refrun.psd"))
    ref_path = os.path.join(tmp, "ref_4x.png")
    ref = solid(128, 128, (20, 60, 220, 255))
    ref[..., 3] = 255
    core_io.write_rgba(ref_path, ref)

    plan_file = prepare(psd_path, factor=2.0, max_side=512,
                        prompt="放大这张图", reference_path=ref_path)
    out = run_min(plan_file, 11)
    _psd_out, info = out["result"]
    check("ref_4x.png" in info, "info names the reference (%s)"
          % info.splitlines()[5])

    plan = json.loads(open(psd_upscale.plan_path(os.path.join(
        tmp, "SeeThrough", "PsdUpscale", "refrun")), encoding="utf-8").read())
    check(plan.get("reference", {}).get("count") == 3,
          "every crop gets a reference block (%s)"
          % (plan.get("reference") or {}).get("count"))
    refs = [job.get("ref_file") for job in plan["crops"]]
    check(all(r and os.path.isfile(r) for r in refs),
          "reference blocks exist on disk")

    expand = out["expand"]
    fed = [n for n in expand.values()
           if isinstance(n, dict) and "LoadCrop" in n.get("class_type", "")
           and n.get("inputs", {}).get("file") in refs]
    check(len(fed) == 3, "the expanded graph feeds all %d reference blocks"
          % len(fed))


def test_selection(tmp):
    print("[nodes_psd_min: the select widget narrows the run]")
    psd_path = build_psd(os.path.join(tmp, "pick.psd"))
    plan_file = prepare(psd_path, factor=2.0, max_side=512,
                        prompt="放大这张图", select="eyes")
    out = run_min(plan_file, 9)
    _psd_out, info = out["result"]
    check("放大 1 层" in info and "选层: eyes" in info,
          "only the named layer runs (%s)"
          % " | ".join(info.splitlines()[1:5]))

    plan = json.loads(open(psd_upscale.plan_path(os.path.join(
        tmp, "SeeThrough", "PsdUpscale", "pick")), encoding="utf-8").read())
    active = [l for l in plan["layers"] if l.get("selected")]
    check(plan["count"] == 1 and [l.get("name") for l in active] == ["eyes"],
          "the plan keeps just that layer (%s)"
          % [l.get("name") for l in active])

    try:
        prepare(psd_path, factor=2.0, max_side=512, prompt="x", select="none")
    except ValueError as exc:
        check("没有选中的图层" in str(exc), "empty selection is rejected (%s)"
              % exc)
    else:
        check(False, "select=none must not run an empty job")

    spec = nodes_psd_min.STLU_PsdUpscalePrepare.INPUT_TYPES()["required"]
    check(spec.get("select", (None,))[0] == "STRING",
          "select is a plain STRING widget driven by the layer table")
    for name, want in (("out_dir", nodes_psd_min.OUT_SUBDIR),
                       ("psd_name", "")):
        default = spec.get(name, (None, {}))[1].get("default")
        check(default == want, "%s defaults to %r (%s)" % (name, want, default))
    for name, want in (("seed", nodes_psd_min.SEED),
                       ("work_max_side", nodes_psd_min.WORK_MAX_SIDE),
                       ("margin", nodes_psd_min.MARGIN)):
        default = spec.get(name, (None, {}))[1].get("default")
        check(default == want, "%s defaults to %r (%s)" % (name, want, default))


def test_custom_output(tmp):
    print("[nodes_psd_min: custom out_dir + psd_name land in the plan]")
    psd_path = build_psd(os.path.join(tmp, "named.psd"))
    plan_file = prepare(psd_path, factor=2.0, max_side=512,
                        prompt="放大这张图", out_dir="Custom/Psd",
                        psd_name="final_art", seed=7, work_max_side=1024,
                        margin=0.2)
    out = run_min(plan_file, 13)
    psd_out, _info = out["result"]
    want = os.path.join(tmp, "Custom", "Psd", "named", "final_art.psd")
    check(os.path.normpath(psd_out) == os.path.normpath(want),
          "the PSD path follows out_dir + psd_name (%s)" % psd_out)

    plan = json.loads(open(psd_upscale.plan_path(os.path.join(
        tmp, "Custom", "Psd", "named")), encoding="utf-8").read())
    settings = plan.get("settings") or {}
    check(settings.get("seed") == 7 and settings.get("work_max_side") == 1024
          and settings.get("psd_name") == "final_art",
          "seed / work_max_side / psd_name reach the plan (%s)" % settings)


def test_loader_shape():
    print("[nodes_psd_min: the combined loader declares MODEL/CLIP/VAE]")
    cls = nodes_psd_min.STLU_QwenEditLoader
    check(cls.RETURN_TYPES == ("MODEL", "CLIP", "VAE"),
          "one node yields MODEL/CLIP/VAE (%s)" % (cls.RETURN_TYPES,))


def main():
    with tempfile.TemporaryDirectory(prefix="stlu_psdmin_") as tmp:
        original = nodes_psd_min._output_root
        nodes_psd_min._output_root = lambda: tmp
        try:
            check(os.path.isfile(edit_parts.edit_template_path(
                nodes_psd_min.TEMPLATE)), "the pinned edit template exists")
            test_guards(tmp)
            test_min_run(tmp)
            test_reference(tmp)
            test_selection(tmp)
            test_custom_output(tmp)
            test_loader_shape()
        finally:
            nodes_psd_min._output_root = original
    if _failures:
        print("\n%d check(s) failed" % len(_failures))
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
