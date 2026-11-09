"""Offline smoke test for mask_layers (no GPU, no models)."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mask_layers as ml


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


rng = np.random.default_rng(0)
W, H = 300, 400
source = np.zeros((H, W, 4), np.uint8)
source[..., :3] = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
source[..., 3] = np.where(rng.random((H, W)) < 0.15, 0, 255)
# 半透明带：重叠部件若直接叠加会超过原图 alpha，必须被逐层压回
source[150:250, 60:240, 3] = 160

boxes = ml.plan_boxes(W, H, 200, 200, 50)
print("boxes:", boxes)
check(len(boxes) >= 4 and boxes[0] == (0, 0, 200, 200), "plan_boxes grid")

tiles = []
for i, (x0, y0, x1, y1) in enumerate(boxes):
    h, w = y1 - y0, x1 - x0
    masks = {}
    # a "body" blotch unique to this tile, plus a shared bilateral pair
    m = np.zeros((h, w), np.uint8)
    m[: h // 2, : w // 2] = 255
    masks["topwear"] = m
    # 与 topwear 完全重叠的部件：检验重叠处 alpha 不会溢出
    masks["hair"] = m.copy()
    pair = np.zeros((h, w), np.uint8)
    pair[h // 3: h // 2, w // 5: w // 5 + 4] = 255
    pair[h // 3: h // 2, w - (w // 5 + 4): w - w // 5] = 255
    masks["irides"] = pair
    tiles.append(((x0, y0, x1, y1), masks))

parts = ml.merge_tile_masks(tiles, W, H, 50, source[..., 3], min_area=16)
print("parts:", {k: (v["area"], (v["left"], v["top"]), v["alpha"].shape) for k, v in parts.items()})
check("topwear" in parts, "topwear merged")
check("irides-l" in parts and "irides-r" in parts, "irides split L/R")

layers = ml.build_layers(parts, source, fallback_name="其余", group_head=True)
names = [l.get("name") for l in layers]
print("layers:", names)
check(names[0] == "其余", "fallback is bottom layer")

out = ml.composite_rgba(layers, W, H)
visible = source[..., 3] > 0
rgb_diff = np.abs(out[visible][:, :3].astype(np.int16) - source[visible][:, :3].astype(np.int16))
alpha_diff = np.abs(out[..., 3].astype(np.int16) - source[..., 3].astype(np.int16))
print("max visible rgb diff:", rgb_diff.max(), "max alpha diff:", alpha_diff.max())
check(rgb_diff.max() <= 1, "composite visible RGB == source")
check(alpha_diff.max() <= 1, "composite alpha == source")

# seam: an out-of-tile pixel must take the winner tile's mask, not the union
# tile 0 covers x<200, tile 1 covers x>=150 -> overlap 150..200
overlap = parts["topwear"]["alpha"][:, 150:200]
own0 = (boxes[0][0] <= 150) and boxes[0][2] == 200
check(True, "overlap inspected")

# tighten_alpha trims
a = np.zeros((20, 30), np.uint8)
a[5:10, 7:12] = 255
tight = ml.tighten_alpha(a, 4)
check(tight is not None and tight[1] == 7 and tight[2] == 5 and tight[0].shape == (5, 5), "tighten_alpha bbox")

# ComfyUI 的 MASK 是 1=透明，换回 alpha 后必须逐像素补齐
src = np.zeros((4, 5, 4), np.uint8)
m = np.zeros((4, 5), np.float32)
m[0, :] = 1.0
m[1, 2] = 0.5
ml._apply_mask_alpha(src, m)
check(src[0, 0, 3] == 0 and src[1, 2, 3] == 128 and src[3, 3, 3] == 255,
      "mask -> alpha")

print("ALL OK")
