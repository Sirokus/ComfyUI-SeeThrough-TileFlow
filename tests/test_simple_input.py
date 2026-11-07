"""CPU-only checks for the simplified ① node.

``STLU_SimpleInput`` must be the same planner as ``STLU_PipelineInput`` -- the
widgets it drops are filled back in by ``normalize_settings`` from the shared
defaults, so geometry, selection, the generation fingerprint and the
``grid_doc`` contract all have to come out identical.  Nothing here touches a
model: only the planner runs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (COMFY_ROOT, PKG_DIR, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import nodes as stlu  # noqa: E402
import nodes_pipeline  # noqa: E402
import nodes_shared  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def widget_names(cls):
    inputs = {**cls.INPUT_TYPES().get("required", {}),
              **cls.INPUT_TYPES().get("optional", {})}
    return [name for name, spec in inputs.items()
            if isinstance(spec, tuple) and len(spec) > 1
            and isinstance(spec[1], dict) and "forceInput" not in spec[1]]


def main():
    print("[the simple node only subtracts widgets]")
    drop = set(nodes_pipeline.SIMPLE_INPUT_DROP)
    full = widget_names(stlu.STLU_PipelineInput)
    simple = widget_names(stlu.STLU_SimpleInput)
    check(simple == [name for name in full if name not in drop],
          "the simple widget list is the full one minus the drop (%d vs %d)"
          % (len(simple), len(full)))
    check(set(full) - set(simple) == drop, "nothing else is dropped")
    check(isinstance(stlu.STLU_SimpleInput, type)
          and issubclass(stlu.STLU_SimpleInput, stlu.STLU_PipelineInput),
          "the simple node derives from the full one")

    print("[both nodes plan the same run]")
    image = torch.zeros(1, 256, 256, 4)
    common = dict(tile_width=64, tile_height=64, overlap=16, select="all",
                  resolution=512, num_inference_steps=4, depth_resolution=-1,
                  random_seed=7, out_dir="SeeThrough/SimpleParity",
                  group="", filename_prefix="tile", merge_name="merged",
                  group_layers=True, include_depth=False,
                  include_original=False, tblr_split=True, use_lama=True,
                  alpha_threshold=8, seam_feather=32, grid_offset_x=0,
                  grid_offset_y=0, partition_mode="均匀网格",
                  strip_max_side=2048, strip_margin=32, select_plan="",
                  custom_regions="[]")

    with tempfile.TemporaryDirectory() as tmp:
        original_root = nodes_shared._output_root
        nodes_shared._output_root = lambda: tmp
        try:
            full_out = stlu.STLU_PipelineInput().execute(image=image, **common)
            simple_out = stlu.STLU_SimpleInput().execute(image=image, **common)
        finally:
            nodes_shared._output_root = original_root

    full_doc = json.loads(full_out["result"][2])
    simple_doc = json.loads(simple_out["result"][2])
    volatile = ("created", "generation_id")
    check({k: v for k, v in full_doc.items() if k not in volatile}
          == {k: v for k, v in simple_doc.items() if k not in volatile},
          "the whole grid_doc is identical (dropped widgets took the shared "
          "defaults)")
    check(full_doc["generation"] == simple_doc["generation"],
          "the generation fingerprint is identical, so a shared output "
          "folder is not cleared")
    check(simple_doc["coords"] == full_doc["coords"],
          "tile coordinates are identical")

    full_tiles, full_indices = full_out["result"][0], full_out["result"][1]
    simple_tiles, simple_indices = simple_out["result"][0], simple_out["result"][1]
    check(len(simple_tiles) == len(full_tiles) == 25,
          "the same tiles are cropped (%d vs %d)"
          % (len(simple_tiles), len(full_tiles)))
    check(simple_indices == full_indices, "the tile indices are identical")
    check(all(torch.equal(a, b) for a, b in zip(full_tiles, simple_tiles)),
          "every cropped tile is pixel-identical")
    check(full_out["result"][4:] == simple_out["result"][4:],
          "the scalar fan-out is identical")

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
