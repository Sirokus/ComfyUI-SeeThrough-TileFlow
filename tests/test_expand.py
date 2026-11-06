"""Static checks for the subgraph expansion (no GPU, nothing executed)."""

from __future__ import annotations

import os
import sys

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for p in (PKG_DIR, COMFY_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import core_plan  # noqa: E402
import subgraph_clone  # noqa: E402

OUTPUT_ROOT = os.path.join(COMFY_ROOT, "output")


def _find_workflow():
    """The single-layer workflow, wherever the user filed it."""
    root = os.path.join(COMFY_ROOT, "user", "default", "workflows")
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name == "Live2D层级二次放大.json":
                return os.path.join(dirpath, name)
    return os.path.join(root, "Live2D层级二次放大.json")


def _find_layers_json():
    """Newest source layers doc under output/ (the user re-saves these)."""
    preferred = os.path.join(
        OUTPUT_ROOT, "SeeThrough", "seethrough_20261031_144549_2c644a74_layers.json"
    )
    if os.path.isfile(preferred):
        return preferred
    found = core_plan.find_latest_layers_json(OUTPUT_ROOT)
    return found or preferred


LAYERS_JSON = _find_layers_json()
WORKFLOW = _find_workflow()

_failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _failures.append(msg)


def main():
    print("[template]")
    check(os.path.isfile(WORKFLOW), "workflow found")
    template = subgraph_clone.load_template(WORKFLOW)
    print("  subgraph: %s (%d nodes, %d links)" % (template.name, len(template.nodes), len(template.links)))
    check(template.output_link in {(457, 0)}, "output link is VAEDecode -> image")
    print("  loader kinds: %s" % template.loader_kind)

    print("[plan + expand]")
    plan = core_plan.build_plan(
        LAYERS_JSON, OUTPUT_ROOT, select="eyebrow-l,face", size_rules_text="",
        out_dir="SeeThrough/UpScale", canvas_size=2048,
    )
    check(plan["total"] == 2, "two layers selected")
    indices = [l["z"] for l in plan["layers"] if l["selected"]]

    external = {
        "reference": ["9001", 0],
        "model": ["9002", 0],
        "clip": ["9003", 0],
        "vae": ["9004", 0],
        "pe_clip": ["9005", 0],
    }
    expand, last_save, report = subgraph_clone.build_expand(
        WORKFLOW, plan, core_plan.plan_to_text(plan), external, seed=100, layer_indices=indices
    )
    check(len(report) == 2, "report has two layers")
    node_ids = set(expand.keys())
    print("  expanded nodes: %d" % len(expand))

    # No leftover subgraph interface ids.
    leftovers = []
    for nid, node in expand.items():
        for key, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                if value[0] in ("-10", "-20"):
                    leftovers.append((nid, key, value))
    check(not leftovers, "no -10/-20 links left")

    # Every internal link points at a node in the expansion or an external id.
    external_ids = {"9001", "9002", "9003", "9004", "9005"}
    dangling = []
    for nid, node in expand.items():
        for key, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                if value[0] not in node_ids and value[0] not in external_ids:
                    dangling.append((nid, key, value))
    check(not dangling, "no dangling links")
    if dangling:
        for d in dangling[:5]:
            print("     ", d)

    # Loader nodes are not cloned.
    classes = [n.get("class_type") for n in expand.values()]
    for skipped in ("UNETLoader", "CLIPLoader", "VAELoader", "QwenImage21Cache"):
        check(skipped not in classes, "%s not cloned" % skipped)

    # One save node per layer, and the expanded seed/index differ.
    saves = [nid for nid, n in expand.items() if n.get("class_type") == "STLU_SaveLayer"]
    loads = [nid for nid, n in expand.items() if n.get("class_type") == "STLU_LoadLayer"]
    check(len(saves) == 2, "two save nodes (%d)" % len(saves))
    check(len(loads) == 2, "two load nodes (%d)" % len(loads))
    indexes = sorted(expand[n].get("inputs", {}).get("index") for n in loads)
    check(indexes == sorted(indices), "load indexes are literals: %s" % indexes)
    seeds = sorted(n.get("inputs", {}).get("noise_seed") for n in expand.values()
                   if n.get("class_type") == "RandomNoise")
    check(len(set(seeds)) == len(seeds), "seeds differ per layer: %s" % seeds)

    # Image inputs are wired to the per-layer LoadLayer node, or to the focus
    # crop node for parts small enough to be cropped.
    encoders = [n for n in expand.values() if n.get("class_type") == "TextEncodeQwenImage21"]
    check(encoders, "TextEncode nodes cloned")
    crops = {nid for nid, n in expand.items() if n.get("class_type") == "STLU_FocusCrop"}
    pastes = {nid for nid, n in expand.items() if n.get("class_type") == "STLU_PasteBack"}
    backdrops = {nid for nid, n in expand.items() if n.get("class_type") == "STLU_ContextBackdrop"}
    image_sources = set(loads) | crops
    img_links = [v for n in encoders for k, v in n.get("inputs", {}).items()
                 if k == "images.image_1"]
    check(len(img_links) == 2 and all(v[0] in image_sources for v in img_links),
          "image_1 wired to loaders / focus crops")
    vision = [n for n in expand.values() if n.get("class_type") == "BatchImagesNode"]
    vision_links = [v for n in vision for k, v in n.get("inputs", {}).items()
                    if k == "images.image0"]
    check(len(vision_links) == 2 and all(v[0] in image_sources for v in vision_links),
          "vision encoder also sees the crop")
    for enc in encoders:
        if enc["inputs"].get("images.image_1", [None])[0] in crops:
            continue
        prompt = enc.get("inputs", {}).get("prompt")
        check(isinstance(prompt, list) and prompt[0] in node_ids,
              "prompt input survives PreviewAny removal")

    print("[focus wiring]")
    check(plan["focus"]["crop"] is True, "focus on by default")
    check(len(backdrops) == 1, "one shared backdrop (%d)" % len(backdrops))
    check(len(crops) == 1 and len(pastes) == 1,
          "only the small layer is cropped (%d crop, %d paste)" % (len(crops), len(pastes)))
    focus_enc = [n for n in encoders if n["inputs"].get("images.image_1", [None])[0] in crops]
    check(len(focus_enc) == 1, "one encoder runs on the crop")
    if focus_enc:
        inputs = focus_enc[0]["inputs"]
        # 512x128 target: its long side (512) caps the text-encoder resolution.
        check(inputs.get("resolution") == 512, "resolution follows target long side (%r)"
              % (inputs.get("resolution"),))
        check(inputs.get("prompt") == plan["focus"]["prompt"],
              "prompt replaced with the focus prompt")
    samplers = [n for n in expand.values() if n.get("class_type") == "SamplerCustomAdvanced"]
    latent_switches = [expand.get(n["inputs"]["latent_image"][0]) for n in samplers
                       if isinstance(n["inputs"].get("latent_image"), list)]
    forced = [sw for sw in latent_switches if sw and sw["inputs"].get("switch") is False]
    check(len(forced) == 1, "latent switch forced to the encoder latent for the crop layer")
    save_nodes = [n for n in expand.values() if n.get("class_type") == "STLU_SaveLayer"]
    paste_masks = [n["inputs"]["mask"][0] for n in save_nodes
                   if isinstance(n["inputs"].get("mask"), list)]
    check(sum(1 for mid in paste_masks if mid in pastes) == 1,
          "the cropped layer is saved through paste-back")

    print("[focus disabled]")
    plain = core_plan.build_plan(
        LAYERS_JSON, OUTPUT_ROOT, select="eyebrow-l,face", size_rules_text="",
        out_dir="SeeThrough/UpScale", canvas_size=2048, focus_crop=False,
    )
    expand_plain, _, _ = subgraph_clone.build_expand(
        WORKFLOW, plain, core_plan.plan_to_text(plain), external, seed=100,
        layer_indices=[l["z"] for l in plain["layers"] if l["selected"]],
    )
    plain_classes = {n.get("class_type") for n in expand_plain.values()}
    check(not plain_classes & {"STLU_FocusCrop", "STLU_ContextBackdrop", "STLU_PasteBack"},
          "no focus nodes when disabled")
    plain_encoders = [n for n in expand_plain.values()
                      if n.get("class_type") == "TextEncodeQwenImage21"]
    plain_loads = {nid for nid, n in expand_plain.items()
                   if n.get("class_type") == "STLU_LoadLayer"}
    plain_links = [v for n in plain_encoders for k, v in n.get("inputs", {}).items()
                   if k == "images.image_1"]
    check(len(plain_links) == 2 and all(v[0] in plain_loads for v in plain_links),
          "image_1 back on the raw layer")
    check(all(n["inputs"].get("resolution") == 0 for n in plain_encoders),
          "resolution keeps the instance value")

    # PreviewAny is replaced by its source (the If/Else switch).
    check(not any(n.get("class_type") == "PreviewAny" for n in expand.values()),
          "PreviewAny dropped")
    switches = [n for n in expand.values() if n.get("class_type") == "ComfySwitchNode"]
    prompt_sources = {enc["inputs"]["prompt"][0] for enc in encoders
                      if isinstance(enc["inputs"].get("prompt"), list)}
    switch_ids = {sid for sid, n in expand.items() if n.get("class_type") == "ComfySwitchNode"}
    check(bool(prompt_sources) and prompt_sources <= switch_ids,
          "prompt source traced back to the If/Else switch")

    # Last save node reference is a link into the expansion.
    check(last_save is not None and last_save.id in node_ids, "last save node present")

    # Loader outputs are wired to the external model/clip/vae links.
    guides = [n for n in expand.values() if n.get("class_type") == "CFGGuider"]
    check(guides and guides[0]["inputs"]["model"] == ["9002", 0], "model link wired")

    print("[size clamp]")
    check(core_plan.compute_target_size(32, 8, {"long_side": 512})[0] == 512, "long side honoured")
    check(core_plan.compute_target_size(1024, 1024, {"long_side": 4096}) == (4096, 4096), "square 4096")
    check(core_plan.compute_target_size(32, 8, {"long_side": 64})[1] == 128, "min side clamp")

    print()
    if _failures:
        print("FAILED: %d" % len(_failures))
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
