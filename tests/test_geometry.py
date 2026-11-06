"""CPU checks for the focus crop / paste-back path.

No GPU and no ComfyUI server: the "generated" image is the crop resized, which
is enough to validate that the content lands back at its original place and
that the alpha is restored.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for p in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import core_geometry  # noqa: E402
import core_io  # noqa: E402
import core_plan  # noqa: E402
import nodes as stlu  # noqa: E402

CANVAS = (1024, 1024)
FACE_BOX = [0, 0, 1024, 1024]
PUPIL_BOX = [520, 128, 537, 143]
PUPIL_ALPHA = (3, 3, 14, 12)  # opaque rect inside the 17x15 PNG

_failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _failures.append(msg)


def _make_assets(root):
    """Write a face layer, a soft-edged pupil layer and their layers.json."""
    face = np.zeros((1024, 1024, 4), dtype=np.uint8)
    face[..., 0] = 230
    face[..., 1] = 200
    face[..., 2] = 190
    face[..., 3] = 255
    Image.fromarray(face, "RGBA").save(os.path.join(root, "face.png"))

    pupil = np.zeros((15, 17, 4), dtype=np.uint8)
    x0, y0, x1, y1 = PUPIL_ALPHA
    pupil[y0:y1, x0:x1] = (40, 90, 200, 255)
    Image.fromarray(pupil, "RGBA").save(os.path.join(root, "irides-l.png"))

    doc = {
        "width": 1024,
        "height": 1024,
        "prefix": "focus_test",
        "layers": [
            {"name": "face", "filename": "face.png", "left": 0, "top": 0,
             "right": 1024, "bottom": 1024, "depth_median": 0.95},
            {"name": "irides-l", "filename": "irides-l.png", "left": 520, "top": 128,
             "right": 537, "bottom": 143, "depth_median": 0.2},
        ],
    }
    path = os.path.join(root, "focus_test_layers.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    return path


def _plan(root, layers_json, **kwargs):
    return core_plan.build_plan(
        layers_json, root, select="irides-l,face", size_rules_text="",
        out_dir="Up", canvas_size=2048, **kwargs
    )


def _index_of(plan, name):
    for layer in plan["layers"]:
        if layer["name"] == name:
            return layer["z"]
    raise KeyError(name)


def _generated(crop, out_w, out_h):
    """Pretend the encoder produced ``crop`` at another size."""
    chw = crop.movedim(-1, 1).float()
    return F.interpolate(chw, size=(int(out_h), int(out_w)), mode="bilinear",
                         align_corners=False).movedim(1, -1)


def main():
    tmp = tempfile.mkdtemp(prefix="stlu_focus_")
    layers_json = _make_assets(tmp)

    print("[rect math]")
    alpha = np.zeros((15, 17), dtype=np.uint8)
    alpha[3:12, 3:14] = 255
    bbox = core_geometry.alpha_bbox(alpha)
    check(bbox == (3, 3, 14, 12), "alpha bbox %s" % (bbox,))
    content = core_geometry.content_rect(PUPIL_BOX, bbox, CANVAS)
    check(content == (523, 131, 534, 140), "content rect in canvas coords %s" % (content,))
    crop = core_geometry.crop_rect(content, CANVAS, 32)
    check(crop[0] <= content[0] - 32 and crop[1] <= content[1] - 32
          and crop[2] >= content[2] + 32 and crop[3] >= content[3] + 32,
          "margin 32 crop %s" % (crop,))
    check(crop == (481, 88, 577, 184), "short sides grown to the minimum: %s" % (crop,))
    check(crop[2] - crop[0] >= core_geometry.MIN_CROP_SIDE
          and crop[3] - crop[1] >= core_geometry.MIN_CROP_SIDE, "min side enforced")
    edge = core_geometry.crop_rect((5, 5, 20, 20), CANVAS, 64)
    check(edge[0] == 0 and edge[1] == 0, "crop clamps at the canvas edge %s" % (edge,))
    check(core_geometry.is_full_canvas(FACE_BOX, CANVAS), "face detected as full frame")
    check(not core_geometry.is_full_canvas(PUPIL_BOX, CANVAS), "pupil is not full frame")

    print("[crop_info]")
    info = core_geometry.make_crop_info("irides-l", CANVAS, PUPIL_BOX, content, crop,
                                        (512, 448), 32)
    text = core_geometry.crop_info_to_text(info)
    back = core_geometry.crop_info_from_text(text)
    check(back == info, "crop_info survives a JSON round trip")
    check(core_geometry.crop_info_from_text("") is None, "empty info is rejected")
    fracs = core_geometry.paste_fractions(info)
    check(fracs is not None and 0 < fracs[0] < fracs[2] < 1, "content sits inside the crop")

    print("[extract_crop]")
    backdrop = np.full((1024, 1024, 3), 200, dtype=np.uint8)
    layer = np.zeros((15, 17, 4), dtype=np.uint8)
    layer[PUPIL_ALPHA[1]:PUPIL_ALPHA[3], PUPIL_ALPHA[0]:PUPIL_ALPHA[2]] = (10, 20, 240, 255)
    out = core_geometry.extract_crop(backdrop, layer, PUPIL_BOX, crop)
    check(out.shape == (crop[3] - crop[1], crop[2] - crop[0], 3), "crop is opaque RGB")
    cy, cx = 131 - crop[1] + 2, 523 - crop[0] + 2
    check(tuple(int(v) for v in out[cy, cx]) == (10, 20, 240),
          "layer pixels painted at their place: %s" % (out[cy, cx],))
    check(tuple(int(v) for v in out[0, 0]) == (200, 200, 200), "context around it kept")
    # An RGBA reference flattens its transparency over white, never raw black.
    rgba_ctx = np.dstack([np.zeros((1024, 1024, 3), np.uint8),
                          np.zeros((1024, 1024), np.uint8)])
    flat = core_geometry.extract_crop(rgba_ctx, layer, PUPIL_BOX, crop)
    check(tuple(int(v) for v in flat[0, 0]) == (255, 255, 255),
          "transparent reference becomes white, not black: %s" % (flat[0, 0],))

    print("[merge_alpha resamples]")
    rgba = np.full((8, 8, 4), 255, dtype=np.uint8)
    small = np.zeros((4, 4), dtype=np.uint8)
    merged = core_io.merge_alpha(rgba, small)
    check(merged.shape == rgba.shape, "shape kept")
    check(merged[..., 3].max() == 0, "resized mask applied: max alpha %d" % merged[..., 3].max())
    untouched = core_io.merge_alpha(rgba, None)
    check(untouched[..., 3].min() == 255, "no mask leaves alpha alone")

    print("[focus crop + paste back]")
    plan = _plan(tmp, layers_json, focus_crop=True, focus_margin=32)
    plan_text = core_plan.plan_to_text(plan)
    check(plan["focus"]["crop"] is True and plan["focus"]["margin"] == 32, "plan carries focus")
    check(plan["focus"]["resolution"] == core_plan.FOCUS_RESOLUTION, "plan pins resolution")

    index = _index_of(plan, "irides-l")
    image_t, mask, name, tw, th, _ = stlu.STLU_LoadLayer().execute(plan_text, index)
    check(name == "irides-l" and (tw, th) == (512, 448), "loader reports %s %dx%d" % (name, tw, th))

    backdrop_t = stlu.STLU_ContextBackdrop().execute(plan_text)[0]
    check(tuple(backdrop_t.shape) == (1, 1024, 1024, 3), "backdrop is one canvas %s"
          % (tuple(backdrop_t.shape),))

    crop_t, info_text = stlu.STLU_FocusCrop().execute(
        plan_text, index, image_t, backdrop_t, tw, th, 32
    )
    info = core_geometry.crop_info_from_text(info_text)
    check(info is not None and not info.get("bypass"), "focus info produced")
    cw = info["crop"][2] - info["crop"][0]
    ch = info["crop"][3] - info["crop"][1]
    check(tuple(crop_t.shape) == (1, ch, cw, 3), "crop tensor %s" % (tuple(crop_t.shape),))

    # The crop must contain the part's own pixels, not just the backdrop.
    cy, cx = PUPIL_ALPHA[1] + 2 - (info["crop"][1] - PUPIL_BOX[1]), PUPIL_ALPHA[0] + 2 - (info["crop"][0] - PUPIL_BOX[0])
    pix = [int(round(v * 255)) for v in crop_t[0, cy, cx].tolist()]
    check(pix[2] > 150 and pix[0] < 90, "pupil pixels present in the crop %s" % (pix,))

    print("[reference at its own resolution]")
    check(core_geometry.context_scale((1024, 1024), CANVAS) == 1.0, "same size is 1.0")
    check(abs(core_geometry.context_scale((2048, 2048), CANVAS) - 2.0) < 1e-6, "2048 is 2.0")
    check(core_geometry.context_scale((2048, 1024), CANVAS) == 1.0,
          "mismatched aspect falls back to fitting")
    check(core_geometry.scale_rect((10, 20, 30, 40), 2.0) == (20, 40, 60, 80), "rect scales")
    check(core_geometry.scale_rect((0, 0, 30, 40), 2.0, (50, 60)) == (0, 0, 50, 60),
          "scaled rect clamped to the image")

    ref_big = F.interpolate(backdrop_t.movedim(-1, 1), size=(2048, 2048), mode="bilinear",
                            align_corners=False).movedim(1, -1)
    big_crop, big_info_text = stlu.STLU_FocusCrop().execute(
        plan_text, index, image_t, ref_big, tw, th, 32
    )
    big_info = core_geometry.crop_info_from_text(big_info_text)
    check(big_info.get("context_scale") == 2.0, "reference cropped at native scale")
    want_w = info["crop"][2] - info["crop"][0]
    got_w = big_info["crop"][2] - big_info["crop"][0]
    check(abs(got_w - 2 * want_w) <= 2, "window doubled (%d -> %d)" % (want_w, got_w))
    check(big_crop.shape[1] == got_w, "crop tensor follows the window")
    # Layer-local coordinates are in source pixels; the crop is at 2x.
    bx0 = int(round((PUPIL_ALPHA[0] + 2) * 2)) - (big_info["crop"][0] - big_info["box"][0])
    by0 = int(round((PUPIL_ALPHA[1] + 2) * 2)) - (big_info["crop"][1] - big_info["box"][1])
    bpix = [int(round(v * 255)) for v in big_crop[0, by0, bx0].tolist()]
    check(bpix[2] > 150 and bpix[0] < 90, "pupil still at its place %s" % (bpix,))
    big_gen = _generated(big_crop, 1024, 1024)
    big_img, big_mask = stlu.STLU_PasteBack().execute(big_gen, big_info_text, mask)
    check(tuple(big_img.shape) == (1, th, tw, 3), "paste back from a 2x crop")
    centre = big_img[0, th // 2, tw // 2].numpy()
    check(abs(float(centre[2]) - 200 / 255.0) < 0.08, "content landed at the target %s"
          % (centre,))

    for gw, gh in ((1024, 1088), (1536, 1632), (640, 576)):
        gen = _generated(crop_t, gw, gh)
        out_img, out_mask = stlu.STLU_PasteBack().execute(gen, info_text, mask)
        check(tuple(out_img.shape) == (1, th, tw, 3), "paste back %dx%d -> %dx%d" % (gw, gh, tw, th))
        check(tuple(out_mask.shape) == (1, th, tw), "mask matches target")
        # The saved raster is the layer's content region at target size — the
        # same convention compose expects from a source layer PNG.
        src_rgba = core_io.tensor_to_rgba(image_t)
        cx0, cy0, cx1, cy1 = PUPIL_ALPHA
        expected = np.array(
            Image.fromarray(src_rgba[cy0:cy1, cx0:cx1, :3], "RGB")
            .resize((tw, th), Image.Resampling.LANCZOS)
        ).astype(np.float32) / 255.0
        got = out_img[0].numpy()
        diff = np.abs(got - expected).mean()
        check(diff < 0.03, "content matches the source (mean diff %.4f)" % diff)
        m = out_mask[0].numpy()
        check(m.min() < 0.05, "alpha inside the part is opaque (min %.3f)" % m.min())

    print("[full frame bypass]")
    face_index = _index_of(plan, "face")
    face_t, face_mask, _, fw, fh, _ = stlu.STLU_LoadLayer().execute(plan_text, face_index)
    face_crop, face_info = stlu.STLU_FocusCrop().execute(
        plan_text, face_index, face_t, backdrop_t, fw, fh, 32
    )
    f_info = core_geometry.crop_info_from_text(face_info)
    check(f_info is not None and f_info.get("bypass") is True, "full frame layer bypasses")
    out_img, out_mask = stlu.STLU_PasteBack().execute(_generated(face_crop, 2048, 2048),
                                                      face_info, face_mask)
    check(tuple(out_img.shape) == (1, fh, fw, 3), "bypass resizes to target %s"
          % (tuple(out_img.shape),))

    print("[focus disabled]")
    plain = _plan(tmp, layers_json, focus_crop=False)
    check(plain["focus"]["crop"] is False, "focus can be turned off")

    print()
    if _failures:
        print("FAILED: %d" % len(_failures))
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
