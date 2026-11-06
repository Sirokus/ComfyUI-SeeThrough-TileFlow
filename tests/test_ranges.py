"""CPU-only checks that widget ranges match the upstream SeeThrough nodes.

ComfyUI-See-through accepts resolution 512-2048 (step 64), seed 0..2**32-1
and depth resolution -1..2048.  Anything wider is rejected (or clamped) in
the model, so the pipeline must not offer it in the first place.
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
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import nodes as stlu  # noqa: E402
import nodes_shared  # noqa: E402
from stlu import limits  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def widget(spec):
    name, (kind, options) = spec
    return options


def main():
    print("[limits mirror the upstream node]")
    check((limits.RES_MIN, limits.RES_MAX, limits.RES_STEP) == (512, 2048, 64),
          "resolution is 512-2048 step 64")
    check((limits.DEPTH_MIN, limits.DEPTH_MAX) == (-1, 2048),
          "depth resolution is -1..2048")
    check(limits.SEED_MAX == 2 ** 32 - 1, "seed tops out at 2**32-1")
    check((limits.STEPS_MIN, limits.STEPS_MAX) == (1, 100),
          "steps stay 1-100")

    print("[both nodes expose the same ranges]")
    for cls, label in ((stlu.STLU_PipelineInput, "①"),
                       (stlu.STLU_GuidedLayers, "②")):
        fields = {**cls.INPUT_TYPES().get("required", {}),
                  **cls.INPUT_TYPES().get("optional", {})}
        res = widget(("resolution", fields["resolution"]))
        seed = widget(("random_seed", fields["random_seed"]))
        depth = widget(("depth_resolution", fields["depth_resolution"]))
        check((res["min"], res["max"]) == (limits.RES_MIN, limits.RES_MAX),
              "%s resolution widget matches limits" % label)
        check((seed["min"], seed["max"]) == (limits.SEED_MIN, limits.SEED_MAX),
              "%s seed widget matches limits" % label)
        check((depth["min"], depth["max"]) == (limits.DEPTH_MIN, limits.DEPTH_MAX),
              "%s depth widget matches limits" % label)

    print("[out-of-range values fail at run time]")
    image = torch.zeros(1, 128, 128, 3)

    def run(**overrides):
        kwargs = dict(image=image, tile_width=64, tile_height=64, overlap=0,
                      select="all", resolution=512, num_inference_steps=1,
                      depth_resolution=-1, random_seed=7,
                      out_dir="SeeThrough/RangeTest")
        kwargs.update(overrides)
        return stlu.STLU_PipelineInput().execute(**kwargs)

    for label, overrides in (
            ("resolution 4096", {"resolution": 4096}),
            ("resolution 256", {"resolution": 256}),
            ("seed 2**32", {"random_seed": 2 ** 32}),
            ("depth 4096", {"depth_resolution": 4096}),
            ("steps 0", {"num_inference_steps": 0})):
        try:
            run(**overrides)
            check(False, "%s should raise" % label)
        except ValueError as exc:
            check("超出" in str(exc) and "STLU_PipelineInput" in str(exc),
                  "%s is rejected with a Chinese range error" % label)

    print("[in-range depth still passes through]")
    with tempfile.TemporaryDirectory() as tmp:
        original_root = nodes_shared._output_root
        nodes_shared._output_root = lambda: tmp
        try:
            out = run(image=torch.zeros(1, 256, 256, 3),
                      tile_width=256, tile_height=256, overlap=0,
                      resolution=512, depth_resolution=768)
        finally:
            nodes_shared._output_root = original_root
        doc = json.loads(out["result"][2])
        check(doc["generation"]["depth_resolution"] == 768,
              "a legal depth value is kept as asked")

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
