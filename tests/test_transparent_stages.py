"""CPU-only equivalence checks for the transparent (fine-grained) nodes.

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-TileFlow/tests/test_transparent_stages.py

The old ``STLU_PipelineInput`` did stage ① in one shot; the transparent graph
does plan -> pick -> context -> prepare -> crop as five visible nodes.  This
file proves the two produce byte-identical tiles, indices and plan documents
for every partition mode, so the new workflow cannot drift from the old one.

No GPU, no server and no network are touched: every file lands in a temporary
folder.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile

import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import nodes as stlu  # noqa: E402
import nodes_shared  # noqa: E402
import tile_psd  # noqa: E402

_failures = []

CASES = (
    ("uniform grid", {
        "tile_width": 1024, "tile_height": 1024, "overlap": 128,
        "partition_mode": "均匀网格",
    }),
    ("uniform grid + context_pad", {
        "tile_width": 1024, "tile_height": 1024, "overlap": 128,
        "partition_mode": "均匀网格", "context_pad": 64,
    }),
    ("subject strips", {
        "tile_width": 1024, "tile_height": 1024, "overlap": 128,
        "partition_mode": "方形全宽分区",
        "strip_max_side": 2048, "strip_max_resolution": 768, "strip_margin": 32,
    }),
    ("custom regions", {
        "partition_mode": "自定义分区",
        "custom_regions": json.dumps([
            {"x": 0, "y": 40, "size": 768, "res": 0, "name": "上"},
            {"x": 300, "y": 700, "size": 512, "res": 0, "name": "头"},
        ], ensure_ascii=False),
    }),
)

BASE = {
    "select": "all", "resolution": 512, "num_inference_steps": 30,
    "depth_resolution": -1, "random_seed": 12345, "tblr_split": True,
    "use_lama": True, "alpha_threshold": 8, "out_dir": "SeeThrough/TransparentTest",
    "group": "", "filename_prefix": "tile", "merge_name": "merged",
    "group_layers": True, "include_depth": False, "include_original": False,
    "guide_context": True, "guide_feather": 32, "guide_strength": 40,
    "guide_tolerance": 16, "seam_feather": 32,
}


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def _strip_created(doc):
    """The run timestamp is the only field two runs may legitimately differ in."""
    doc = dict(doc or {})
    doc.pop("created", None)
    # ``reset`` reports what changed against the folder's old fingerprint; two
    # runs starting from different folder states legitimately differ here.
    doc.pop("reset", None)
    for key in ("partition_plan", "out"):
        if isinstance(doc.get(key), dict):
            doc[key] = _strip_created(doc[key])
    return doc


def run_old(image, overrides):
    kwargs = dict(BASE)
    kwargs.update(overrides)
    return stlu.STLU_PipelineInput().execute(image=image, **kwargs)


def _result(out):
    return out["result"] if isinstance(out, dict) else out


def run_new(image, overrides):
    kwargs = dict(BASE)
    kwargs.update(overrides)
    settings = _result(stlu.STLU_PipelineSettings().execute(**kwargs))[0]
    chain = _result(stlu.STLU_PlanGrid().execute(image=image,
                                                 settings=settings))[0]
    chain = _result(stlu.STLU_PickTiles().execute(plan=chain))[0]
    chain = _result(stlu.STLU_PlanContext().execute(plan=chain))[0]
    chain = _result(stlu.STLU_PrepareRun().execute(plan=chain, image=image))[0]
    tiles, indices = _result(stlu.STLU_CropTiles().execute(image=image,
                                                           plan=chain))
    return tiles, indices, chain


def main():
    image = torch.rand(1, 900, 1300, 3)
    with tempfile.TemporaryDirectory() as tmp:
        original_root = nodes_shared._output_root
        nodes_shared._output_root = lambda: tmp
        try:
            for label, overrides in CASES:
                print("[%s]" % label)
                old = run_old(image, overrides)
                old_tiles, old_indices, old_doc = old["result"][:3]
                old_plan = tile_psd.load_grid_ref(old_doc)
                new_tiles, new_indices, new_plan = run_new(image, overrides)

                check(len(new_tiles) == len(old_tiles) and len(new_tiles) > 0,
                      "same tile count (%d)" % len(new_tiles))
                check(new_indices == old_indices, "same grid indices")
                same = all(torch.allclose(a, b)
                           for a, b in zip(new_tiles, old_tiles))
                check(same, "the cropped pixels are identical")
                check(_strip_created(json.loads(json.dumps(new_plan["doc"]))) ==
                      _strip_created(old_plan),
                      "the plan document is identical (coords / out / generation)")
                check(new_plan["run"]["fingerprint"] == old_plan["out"]["fingerprint"],
                      "the fingerprint matches")
                check(os.path.isdir(os.path.join(tmp, new_plan["run"]["rel"])),
                      "prepare_run created the run folder")
        finally:
            nodes_shared._output_root = original_root

    print("[pure guide / crop-back steps]")
    with tempfile.TemporaryDirectory() as tmp:
        original_root = nodes_shared._output_root
        nodes_shared._output_root = lambda: tmp
        try:
            settings = _result(stlu.STLU_PipelineSettings().execute(**dict(BASE)))[0]
            plan = _result(stlu.STLU_PlanGrid().execute(image=image,
                                                        settings=settings))[0]
            plan = _result(stlu.STLU_PickTiles().execute(plan=plan))[0]
            check(isinstance(stlu.STLU_SeqTileLoop(), stlu.STLU_SeqTileLoop),
                  "the loop node instantiates")
            guided = _result(stlu.STLU_GuideTile().execute(
                tile=image, index="0", plan=plan, generated=None))[0]
            check(tuple(guided.shape) == (1, 900, 1300, 3),
                  "guide with nothing generated returns the tile unchanged")
            parts, composite = _result(stlu.STLU_CropBack().execute(
                parts={"tag2pinfo": {}}, composite=image, index="0",
                plan=plan))
            check(parts == {"tag2pinfo": {}} and composite is image,
                  "crop-back without context_pad is a no-op")
        finally:
            nodes_shared._output_root = original_root

    print("[registration]")
    for name in ("STLU_PipelineSettings", "STLU_PlanGrid", "STLU_PickTiles",
                 "STLU_PlanContext", "STLU_PrepareRun", "STLU_CropTiles",
                 "STLU_GuideTile", "STLU_SeqTileLoop", "STLU_CropBack",
                 "STLU_WriteTilePSD",
                 "STLU_WriteTileImages", "STLU_SaveOriginal",
                 "STLU_WriteManifest", "STLU_MergeTiles"):
        check(name in stlu.NODE_CLASS_MAPPINGS, "%s is registered" % name)

    print("[signatures]")
    for name, cls in stlu.NODE_CLASS_MAPPINGS.items():
        if not name.startswith("STLU_") or cls.__module__ != "nodes_transparent":
            continue
        inputs = cls.INPUT_TYPES()
        keys = list((inputs.get("required") or {}).keys()) + \
            list((inputs.get("optional") or {}).keys())
        params = set(inspect.signature(cls.execute).parameters)
        missing = [key for key in keys if key not in params]
        check(not missing, "%s execute() takes every input %s" % (name, missing))

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
