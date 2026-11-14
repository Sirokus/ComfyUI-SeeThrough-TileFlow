"""Offline checks for the /stlu HTTP handlers (no ComfyUI server needed)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.parse

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for p in (PKG_DIR, COMFY_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import routes  # noqa: E402

_failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _failures.append(msg)


class Req:
    def __init__(self, **query):
        self.query = query


def body(resp):
    return json.loads(resp.text)


def main():
    print("[/stlu/layers @2048]")
    resp = asyncio.run(routes.list_layers(Req(dir="SeeThrough", canvas="2048")))
    data = body(resp)
    check(resp.status == 200, "status 200")
    check(data["count"] == 29, "29 layers")
    by_name = {l["name"]: l for l in data["layers"]}
    check(by_name["face"]["suggested"]["long_side"] == 2048, "face 2048")
    check(by_name["eyebrow-l"]["suggested"]["long_side"] == 512, "eyebrow 512 from default rules")
    check(by_name["nose"]["empty"] is False, "faint compact part is not empty")
    check(by_name["tail"]["empty"] is True, "dust-only layer is empty")
    check(by_name["face"]["thumb"].startswith("/stlu/thumb?"), "thumbnail url")
    check("upscaled_count" in data, "upscaled tally reported")
    upscaled_face = os.path.join(COMFY_ROOT, "output", data["out_dir"], "face.png")
    check(by_name["face"]["upscaled"] == os.path.isfile(upscaled_face),
          "upscaled flag matches the file on disk")

    print("[/stlu/layers @4096]")
    resp = asyncio.run(routes.list_layers(Req(dir="SeeThrough", canvas="4096")))
    data = body(resp)
    by_name = {l["name"]: l for l in data["layers"]}
    check(by_name["face"]["suggested"]["long_side"] == 4096, "canvas size drives suggestion")

    print("[/stlu/layers auto out_dir]")
    resp = asyncio.run(routes.list_layers(Req(dir="SeeThrough", canvas="2048")))
    data = body(resp)
    # The auto folder belongs to the folder that holds the layers.json (which
    # may be one level below the browsed folder).
    layers_json_rel = data["source"].get("layers_json") or ""
    holder = layers_json_rel.rsplit("/", 1)[0] if "/" in layers_json_rel else ""
    want_auto = (holder + "/UpScale") if holder else "UpScale"
    check(data["auto_out_dir"] == want_auto, "auto folder follows the split (%s)" % data["auto_out_dir"])
    check(data["out_dir"] == want_auto, "blank out_dir resolves to the auto folder")
    check(data["out_dir_auto"] is True, "flagged as auto")
    resp = asyncio.run(routes.list_layers(Req(dir="SeeThrough", canvas="2048", out_dir="custom/dir")))
    data = body(resp)
    check(data["out_dir"] == "custom/dir" and data["out_dir_auto"] is False,
          "explicit out_dir wins")
    check(data["auto_out_dir"] == want_auto, "auto folder still reported")
    # The legacy widget default now means "follow the split folder" too.
    resp = asyncio.run(routes.list_layers(
        Req(dir="SeeThrough", canvas="2048", out_dir="SeeThrough/UpScale")))
    data = body(resp)
    check(data["out_dir"] == data["auto_out_dir"], "legacy default migrates to auto")

    print("[/stlu/layers missing]")
    resp = asyncio.run(routes.list_layers(Req(dir="DoesNotExist", canvas="2048")))
    check(resp.status == 404, "missing source reported (%d)" % resp.status)

    print("[/stlu/thumb]")
    layer = body(asyncio.run(routes.list_layers(Req(dir="SeeThrough", canvas="2048"))))["layers"][5]
    query = urllib.parse.parse_qs(layer["thumb"].split("?", 1)[1])
    resp = asyncio.run(routes.thumb(Req(
        scope=query["scope"][0], path=query["path"][0], size=query["size"][0]
    )))
    check(getattr(resp, "status", None) == 200, "thumbnail served")
    full = routes._resolve(query["scope"][0], query["path"][0])
    check(os.path.isfile(routes._render_thumb(full, 64)), "thumbnail file cached")
    resp = asyncio.run(routes.thumb(Req(scope="output", path="../../ComfyUI.py")))
    check(resp.status in (400, 403), "thumbnail traversal rejected (%d)" % resp.status)

    print("[/stlu/preview]")
    resp = asyncio.run(routes.preview_plan(Req(
        dir="SeeThrough", canvas="512", rules="", select="face, back hair",
        exclude="", out_dir="SeeThrough/UpScale", skip_empty="1", coverage="0.0005", size="512",
    )))
    data = body(resp)
    check(resp.status == 200, "preview status 200")
    check(data["canvas"] == [512, 512], "preview canvas size")
    check(data["layers"] >= 15, "preview composites every non-empty layer")
    check(data["selected"] == 2, "preview honours the selection")
    check(data["url"].startswith("/view?"), "preview url")
    check(len(data["notes"]) > 0, "empty layers reported")

    print("[/stlu/imgsize]")
    ref = os.path.join(COMFY_ROOT, "input", "Qwen_image_2.1_00029_Improve_1K_mouse.png")
    if os.path.isfile(ref):
        resp = asyncio.run(routes.image_size(
            Req(scope="input", path="Qwen_image_2.1_00029_Improve_1K_mouse.png")))
        data = body(resp)
        check(resp.status == 200 and data["width"] > 0 and data["height"] > 0,
              "image size %sx%s" % (data.get("width"), data.get("height")))
    resp = asyncio.run(routes.image_size(Req(scope="input", path="../escape.png")))
    check(resp.status in (400, 403), "traversal rejected (%d)" % resp.status)
    resp = asyncio.run(routes.image_size(Req(scope="input", path="nope.png")))
    check(resp.status == 404, "missing image reported (%d)" % resp.status)

    print("[/stlu/tiles/strips]")
    if os.path.isfile(ref):
        resp = asyncio.run(routes.tiles_strips(Req(
            scope="input", path=os.path.basename(ref), overlap="256",
            max_side="2048", max_resolution="768", margin="32")))
        data = body(resp)
        check(resp.status == 200, "strip plan status 200")
        check("ok" in data, "strip plan verdict present")
        if data.get("ok"):
            check(len(data["regions"]) >= 1, "strip regions returned")
            check(all(len(region) == 4 for region in data["regions"]),
                  "strip regions are boxes")
            check(data["side"] > 0 and data["resolution"] >= 512,
                  "strip side/resolution reported")
        else:
            check(bool(data.get("error")), "strip fallback reason reported")
    else:
        print("  skip (no sample input image)")
    resp = asyncio.run(routes.tiles_strips(Req(
        scope="input", path="../escape.png", overlap="256",
        max_side="2048", max_resolution="768", margin="32")))
    check(resp.status in (400, 403, 404), "strip plan traversal rejected (%d)" % resp.status)

    print()
    if _failures:
        print("FAILED: %d" % len(_failures))
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
