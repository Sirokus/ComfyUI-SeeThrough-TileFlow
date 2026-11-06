"""CPU-only checks for the sequential tile guide (no GPU, no server).

Run with the ComfyUI embedded python:

    python custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/test_tile_guide.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import nodes as stlu  # noqa: E402
import nodes_shared  # noqa: E402
import region_plan  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def solid_part(value, x0=0, y0=0, size=512, alpha=255):
    pixels = np.zeros((size, size, 4), np.uint8)
    pixels[..., :3] = value
    pixels[..., 3] = alpha
    return {
        "img": pixels,
        "depth": np.full((size, size), 128, np.uint8),
        "xyxy": [x0, y0, x0 + size, y0 + size],
        "depth_median": 0.5,
        "tag": "face",
    }


def tile_of(color, width=1024, height=1024):
    arr = np.zeros((1, height, width, 3), np.float32)
    arr[:] = np.asarray(color, np.float32) / 255.0
    return torch.from_numpy(arr)


def pixel(tile, x, y):
    return (tile[0, y, x].numpy() * 255.0).round().astype(int).tolist()


def main():
    print("[guide: feather mask]")
    mask = stlu._guide_feather_mask(100, 100, 10)
    check(abs(float(mask[50, 50]) - 1.0) < 1e-6, "centre is fully guided")
    check(float(mask[0, 50]) < 1e-6, "edge fades to zero")
    check(mask.shape == (100, 100), "mask matches the overlap")
    flat = stlu._guide_feather_mask(10, 10, 0)
    check(np.allclose(flat, 1.0), "feather 0 means a hard fill")

    print("[guide: composite one tile]")
    parts = {"tag2pinfo": {"face": solid_part((255, 0, 0))},
             "frame_size": (512, 512)}
    comp = stlu._guide_composite(parts)
    check(comp is not None and comp.shape == (512, 512, 4),
          "composite is the tile frame, RGBA")
    check(tuple(comp[256, 256].tolist()) == (255, 0, 0, 255),
          "composite keeps the part colour and alpha")

    print("[guide: paste previous tile into the overlap]")
    # 1536x1024 source: tile 0 at x=0..1024, tile 1 at x=512..1536, overlap 512.
    coords = [(0, 0, 1024, 1024), (512, 0, 1536, 1024)]
    generated = {0: comp}
    guide = {"enabled": True, "feather": 32, "strength": 100, "tolerance": 0}
    blue = tile_of((0, 0, 255))
    out = stlu._apply_tile_guide(blue, 1, coords, generated, guide)

    centre = pixel(out, 300, 512)          # inside the overlap
    outside = pixel(out, 800, 512)         # right of the overlap
    check(centre[0] > 200 and centre[2] < 60, "overlap takes the earlier tile (%s)" % centre)
    check(outside[2] > 200 and outside[0] < 60, "outside the overlap stays untouched (%s)" % outside)
    near_edge = pixel(out, 500, 512)       # 12 px before the overlap edge
    check(near_edge[2] > 150, "feather keeps the own tile near the seam (%s)" % near_edge)

    print("[guide: switches]")
    same = stlu._apply_tile_guide(blue, 1, coords, generated,
                                  {"enabled": False, "feather": 0, "strength": 100, "tolerance": 0})
    check(torch.allclose(same, blue), "disabled guide returns the tile unchanged")
    same = stlu._apply_tile_guide(blue, 1, coords, generated,
                                  {"enabled": True, "feather": 0, "strength": 0, "tolerance": 0})
    check(torch.allclose(same, blue), "strength 0 returns the tile unchanged")

    faded = stlu._apply_tile_guide(blue, 1, coords, generated,
                                   {"enabled": True, "feather": 0, "strength": 50, "tolerance": 0})
    half = pixel(faded, 300, 512)
    check(120 <= half[0] <= 140, "strength 50 blends half the earlier tile (%s)" % half)

    print("[guide: source-agreement tolerance]")
    filtered = stlu._apply_tile_guide(blue, 1, coords, generated,
                                      {"enabled": True, "feather": 0, "strength": 100,
                                       "tolerance": 48})
    check(torch.allclose(filtered, blue),
          "a red reconstruction over blue source is NOT pasted (invented pixels stay out)")
    near = comp.copy()
    near[..., :3] = (0, 0, 250)
    close = stlu._apply_tile_guide(blue, 1, coords, {0: near},
                                   {"enabled": True, "feather": 0, "strength": 100,
                                    "tolerance": 48})
    close_px = pixel(close, 300, 512)
    check(close_px[2] > 240 and close_px[0] < 10,
          "a reconstruction close to the source is still pasted (%s)" % close_px)

    print("[guide: no earlier neighbour]")
    same = stlu._apply_tile_guide(blue, 0, coords, {1: comp},
                                  {"enabled": True, "feather": 0, "strength": 100, "tolerance": 0})
    check(torch.allclose(same, blue), "later tiles are never pulled backwards")

    print("[guide: a cached later tile may guide backwards]")
    backwards = stlu._apply_tile_guide(
        blue, 0, coords, {1: comp},
        {"enabled": True, "feather": 0, "strength": 100, "tolerance": 0},
        fixed={1})
    cached_px = pixel(backwards, 700, 512)
    check(cached_px[0] > 200 and cached_px[2] < 60,
          "a tile finished in an earlier run guides tile 0 (%s)" % cached_px)

    print("[guide: sequential node loop (fake SeeThrough nodes)]")
    seen_inputs = []

    class FakeLayer:
        def generate(self, image, model, seed, resolution, steps):
            seen_inputs.append(image)
            return ({"image": image}, None)

    class FakeDepth:
        def generate(self, layers, depth_model, seed, depth_resolution):
            # The real node returns (layers_depth, preview); the loop must
            # unwrap it before handing it to PostProcess.
            return ({"image": layers["image"]}, None)

    class FakePost:
        def process(self, depth, tblr_split, use_lama):
            arr = (depth["image"][0].numpy() * 255.0).astype(np.uint8)
            small = np.array(Image.fromarray(arr).resize((512, 512)))
            rgba = np.dstack([small, np.full((512, 512), 255, np.uint8)])
            parts = {"tag2pinfo": {"face": {
                "img": rgba, "depth": np.full((512, 512), 128, np.uint8),
                "xyxy": [0, 0, 512, 512], "depth_median": 0.5, "tag": "face"}},
                "frame_size": (512, 512)}
            return (parts, None)

    from PIL import Image
    doc = stlu.tile_psd.grid_document(
        (1536, 1024), (1024, 1024), 512, source_image="", name="grid",
        created="test")
    doc["guide"] = {"enabled": True, "feather": 0, "strength": 100, "tolerance": 0}
    doc["out"] = {"alpha_threshold": 8}
    grid_json = json.dumps(doc, ensure_ascii=False)

    original = nodes_shared._seethrough_node_class
    nodes_shared._seethrough_node_class = lambda name: {
        "SeeThrough_GenerateLayers": FakeLayer,
        "SeeThrough_GenerateDepth": FakeDepth,
        "SeeThrough_PostProcess": FakePost,
    }[name]
    try:
        node = stlu.STLU_GuidedLayers()
        parts_out, indices_out = node.execute(
            tiles=[tile_of((255, 0, 0)), tile_of((0, 0, 255))],
            indices=["0", "1"],
            layerdiff_model=[object()], depth_model=[object()],
            grid_doc=[grid_json], random_seed=42, resolution=512,
            num_inference_steps=1, depth_resolution=-1,
            tblr_split=True, use_lama=True)
    finally:
        nodes_shared._seethrough_node_class = original

    check(len(parts_out) == 2 and indices_out == ["0", "1"],
          "the loop hands one parts document per tile onward")
    check(len(seen_inputs) == 2, "both tiles reached GenerateLayers")
    guided = pixel(seen_inputs[1], 200, 512)
    check(guided[0] > 200 and guided[2] < 60,
          "tile 1 reaches the model with tile 0 pasted into the overlap (%s)" % guided)

    print("[guide: custom run order + per-window resolution]")
    seen = []

    class CustomLayer:
        def generate(self, image, model, seed, resolution, steps):
            seen.append({"image": image, "resolution": resolution})
            return ({"image": image}, None)

    class CustomDepth:
        def generate(self, layers, depth_model, seed, depth_resolution):
            seen[-1]["depth_resolution"] = depth_resolution
            return ({"image": layers["image"]}, None)

    class CustomPost:
        # Always red, so the pasted region is unmistakable against the blue
        # input tiles regardless of what the fake model echoed back.
        def process(self, depth, tblr_split, use_lama):
            parts = {"tag2pinfo": {"face": solid_part((255, 0, 0))},
                     "frame_size": (512, 512)}
            return (parts, None)

    custom = region_plan.custom_document(
        [{"x": 0, "y": 0, "size": 1024, "res": 512},
         {"x": 512, "y": 0, "size": 1024, "res": 768}],
        1536, 1024, global_resolution=512)
    custom["guide"] = {"enabled": True, "feather": 0, "strength": 100,
                       "tolerance": 0}
    custom["out"] = {"alpha_threshold": 8}

    nodes_shared._seethrough_node_class = lambda name: {
        "SeeThrough_GenerateLayers": CustomLayer,
        "SeeThrough_GenerateDepth": CustomDepth,
        "SeeThrough_PostProcess": CustomPost,
    }[name]
    try:
        # The user arranged window 1 before window 0: the fill order is the
        # run order, not the index order.
        custom_node = stlu.STLU_GuidedLayers()
        parts_out, indices_out = custom_node.execute(
            tiles=[tile_of((0, 0, 255)), tile_of((0, 0, 255))],
            indices=["1", "0"],
            layerdiff_model=[object()], depth_model=[object()],
            grid_doc=[json.dumps(custom, ensure_ascii=False)],
            random_seed=42, resolution=512, num_inference_steps=1,
            depth_resolution=640, tblr_split=True, use_lama=True)
    finally:
        nodes_shared._seethrough_node_class = original

    check(indices_out == ["1", "0"] and len(parts_out) == 2,
          "the loop keeps the user's run order")
    check([item["resolution"] for item in seen] == [768, 512],
          "each window reaches GenerateLayers with its own resolution (%s)"
          % [item["resolution"] for item in seen])
    check([item["depth_resolution"] for item in seen] == [640, 512],
          "depth is capped by each window's layer resolution (%s)"
          % [item["depth_resolution"] for item in seen])
    first = pixel(seen[0]["image"], 300, 512)
    check(first[2] > 200 and first[0] < 60,
          "the first window is not pulled back by a later index (%s)" % first)
    second = pixel(seen[1]["image"], 700, 512)
    check(second[0] > 200 and second[2] < 60,
          "the later window sees the one finished before it in run order (%s)"
          % second)

    print("[guide: pipeline input carries the switches]")
    old_prompt = nodes_shared._prompt_id
    try:
        nodes_shared._prompt_id = lambda: "abcd1234-ffff-0000-1111-222233334444"
        node = stlu.STLU_PipelineInput()
        out = node.execute(
            image=torch.rand(1, 700, 1200, 3), tile_width=1024, tile_height=1024,
            overlap=256, select="all", guide_context=False, guide_feather=48,
            guide_strength=75, guide_tolerance=48, out_dir="SeeThrough/GuideTest")
        doc = stlu.tile_psd.load_grid_ref(out["result"][2])
        guide = doc.get("guide") or {}
        check(guide == {"enabled": False, "feather": 48, "strength": 75, "tolerance": 48},
              "grid_doc records the guide settings (%s)" % guide)
    finally:
        nodes_shared._prompt_id = old_prompt

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
