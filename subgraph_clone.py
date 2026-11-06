"""Clone the Qwen-edit subgraph once per source layer.

The batch node expands a fresh copy of the subgraph for every selected layer.
Every expanded node gets a distinct id prefix and every per-layer node carries
literal (not linked) ``index`` / ``seed`` values, so ComfyUI's input-signature
cache can never reuse an earlier layer's result.  Model loaders are not
cloned: their outputs are rewired to the MODEL / CLIP / VAE inputs of the
batch node, so the checkpoints are loaded once.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from comfy_execution.graph_utils import GraphBuilder

try:
    from . import core_geometry
except ImportError:  # pragma: no cover - standalone import
    import core_geometry

SKIP_CLASSES = {"UNETLoader", "CLIPLoader", "VAELoader", "QwenImage21Cache"}
# Preview-only nodes inside the subgraph would fire once per layer; drop them.
DROP_CLASSES = {"PreviewAny", "PreviewImage"}

_MISSING = object()


def _focus_resolution_for(layer: dict, cap: int) -> int:
    """Text-encoder resolution for one crop: just enough for the target size.

    Pinning every crop to ``cap`` (1024) turned a 384 px eye into ~5000 vision
    tokens — about 90 s of 8B-VLM prefill per layer.  The target only needs its
    own long side, so capping there halves the prefill without losing output
    detail (the sampler latent and paste-back scale to the target anyway).
    """
    target = layer.get("target") or {}
    long_side = max(int(target.get("w") or 0), int(target.get("h") or 0))
    if long_side <= 0:
        return cap
    return max(512, min(cap, long_side))


class SubgraphTemplate:
    def __init__(self, doc: Dict[str, Any], subgraph: Dict[str, Any], instance: Dict[str, Any]):
        self.doc = doc
        self.sg = subgraph
        self.instance = instance
        self.nodes = {int(n["id"]): n for n in subgraph.get("nodes", [])}
        self.links = {int(l["id"]): l for l in subgraph.get("links", [])}
        self.sg_inputs = subgraph.get("inputs", [])
        self.sg_outputs = subgraph.get("outputs", [])
        self.instance_values = self._instance_values()
        self.loader_kind = self._classify_loaders()
        self.passthrough = self._passthrough_map()
        self.output_link = self._find_output_link()
        self.subgraph_id = subgraph.get("id")
        self.name = subgraph.get("name", "subgraph")

    # -- setup -------------------------------------------------------------

    def _instance_values(self) -> Dict[int, Any]:
        named = self.instance.get("widgets_values_named") or {}
        positional = self.instance.get("widgets_values") or []
        values: Dict[int, Any] = {}
        for idx, sgin in enumerate(self.sg_inputs):
            name = sgin.get("name")
            if name in named:
                values[idx] = named[name]
            elif idx < len(positional):
                values[idx] = positional[idx]
        return values

    def _classify_loaders(self) -> Dict[int, str]:
        kinds: Dict[int, str] = {}
        for node_id, node in self.nodes.items():
            ct = node.get("type")
            if ct in ("UNETLoader", "QwenImage21Cache"):
                kinds[node_id] = "model"
            elif ct == "VAELoader":
                kinds[node_id] = "vae"
            elif ct == "CLIPLoader":
                consumers = [l["target_id"] for l in self.links.values() if l["origin_id"] == node_id]
                is_pe = any(self.nodes.get(c, {}).get("type") == "TextGenerate" for c in consumers)
                kinds[node_id] = "pe_clip" if is_pe else "clip"
        return kinds

    def _passthrough_map(self) -> Dict[int, Tuple[int, int]]:
        """Map preview-only nodes to the input they should pass through.

        ``PreviewAny`` is dropped from the clone, but the subgraph can read
        from it (the prompt display feeds TextEncode).  Its single input is
        used as a replacement source so the real consumer keeps working.
        """
        mapping: Dict[int, Tuple[int, int]] = {}
        for node_id, node in self.nodes.items():
            if node.get("type") not in DROP_CLASSES:
                continue
            for inp in node.get("inputs") or []:
                if inp.get("link") is not None:
                    link = self.links.get(int(inp["link"]))
                    if link is not None:
                        mapping[node_id] = (int(link["origin_id"]), int(link["origin_slot"]))
                    break
        return mapping

    def _resolve_origin(self, origin: int, slot: int) -> Tuple[int, int]:
        seen = set()
        while origin in self.passthrough and origin not in seen:
            seen.add(origin)
            origin, slot = self.passthrough[origin]
        return origin, slot

    def _find_output_link(self) -> Tuple[int, int]:
        for link in self.links.values():
            if link.get("target_id") == -20:
                return int(link["origin_id"]), int(link["origin_slot"])
        raise ValueError("subgraph %r has no output link" % self.name)

    # -- expansion ---------------------------------------------------------

    def expand(
        self,
        graph: GraphBuilder,
        plan: Dict[str, Any],
        plan_ref: str,
        external: Dict[str, List[Any]],
        seed: int,
        layer_indices: List[int],
        batch_tag: str = "u",
    ):
        """Append one cloned subgraph per layer; return the last SaveLayer node."""
        layers = plan["layers"]
        last_save = None
        report: List[Dict[str, Any]] = []

        focus = plan.get("focus") or {}
        focus_on = bool(focus.get("crop", True))
        focus_margin = int(focus.get("margin", 32))
        focus_context = str(focus.get("context") or "backdrop")
        focus_prompt = str(focus.get("prompt") or "")
        focus_resolution = int(focus.get("resolution") or 1024)
        source = plan.get("source") or {}
        canvas = (int(source.get("width") or 1024), int(source.get("height") or 1024))

        # One shared composite for the whole batch: cropping each layer from
        # real surrounding artwork is what stops a 17 px pupil from being
        # completed into a whole head.
        backdrop_node = None

        # Pipeline barrier: every layer's TextEncodeQwenImage21 must finish
        # before any layer starts sampling.  Otherwise the executor walks one
        # layer end-to-end at a time and the 8.9 GB text encoder is pushed out
        # of VRAM by the diffusion model on every single layer (~90 s of disk
        # reload each).  With the barrier the encoder loads once, all layers
        # encode, then the diffusion model loads once for all samplers.
        all_te_nodes = []
        all_gates = []

        def _backdrop():
            nonlocal backdrop_node
            if backdrop_node is None:
                backdrop_node = graph.node(
                    "STLU_ContextBackdrop", id=batch_tag + "_backdrop", plan=plan_ref
                )
            return backdrop_node.out(0)

        for li in layer_indices:
            layer = layers[int(li)]
            prefix = "%s%d" % (batch_tag, li)
            load = graph.node(
                "STLU_LoadLayer",
                id=prefix + "_load",
                plan=plan_ref,
                index=int(li),
                reference=external["reference"],
            )
            box = layer.get("box") or [0, 0, canvas[0], canvas[1]]
            use_focus = focus_on and not core_geometry.is_full_canvas(box, canvas)
            crop = None
            image1 = load.out(0)
            if use_focus:
                context = _backdrop() if focus_context != "reference" else external["reference"]
                crop = graph.node(
                    "STLU_FocusCrop",
                    id=prefix + "_crop",
                    plan=plan_ref,
                    index=int(li),
                    image=load.out(0),
                    context=context,
                    width=load.out(3),
                    height=load.out(4),
                    margin=focus_margin,
                )
                image1 = crop.out(0)
            created: Dict[int, Any] = {}
            layer_gates: Dict[int, Any] = {}

            for node_id, node in self.nodes.items():
                ct = node.get("type")
                mode = int(node.get("mode", 0))
                if ct in SKIP_CLASSES or ct in DROP_CLASSES or mode == 2:
                    continue
                if mode == 4:
                    raise ValueError(
                        "subgraph node %s (%s) is bypassed; bypassed nodes are not supported"
                        % (node_id, ct)
                    )
                created[node_id] = graph.node(ct, id="%s_%s" % (prefix, node_id))
                if ct == "TextEncodeQwenImage21":
                    # Downstream consumers read the gate, not the encoder, so
                    # the encoder can be ordered ahead of every sampler.
                    gate = graph.node(
                        "STLU_Gate",
                        id="%s_%s_gate" % (prefix, node_id),
                        positive=created[node_id].out(0),
                        negative=created[node_id].out(1),
                        latent=created[node_id].out(2),
                    )
                    layer_gates[node_id] = gate
                    all_te_nodes.append(created[node_id])
                    all_gates.append(gate)

            for node_id, node in self.nodes.items():
                if node_id not in created:
                    continue
                graph_node = created[node_id]
                base: Dict[str, Any] = {}
                allowed = {i.get("name") for i in (node.get("inputs") or []) if i.get("name")}
                for key, value in (node.get("widgets_values_named") or {}).items():
                    if key in allowed and value is not None:
                        base[key] = value

                for inp in node.get("inputs") or []:
                    name = inp.get("name")
                    link_id = inp.get("link")
                    if name is None or link_id is None:
                        continue
                    link = self.links.get(int(link_id))
                    if link is None:
                        continue
                    origin = int(link["origin_id"])
                    slot = int(link["origin_slot"])
                    origin, slot = self._resolve_origin(origin, slot)

                    if origin == -10:
                        sgin = self.sg_inputs[slot] if slot < len(self.sg_inputs) else {}
                        sname = sgin.get("name")
                        if sname == "images.image_1":
                            # Both the text encoder and the prompt-enhancer
                            # vision tower read image_1; both must see the crop.
                            base[name] = image1
                        elif use_focus and sname == "switch":
                            # Qwen's own tooltip: the sampling latent has to
                            # match the reference image's size.  The empty
                            # latent branch is the target size, which is what
                            # let the model paint a whole frame; use the
                            # encoder latent (= the crop) instead.
                            base[name] = False
                        elif sname == "images.image_2":
                            base[name] = external["reference"]
                        elif sname == "width":
                            base[name] = load.out(3)
                        elif sname == "height":
                            base[name] = load.out(4)
                        elif str(sgin.get("type")) == "IMAGE":
                            # image_3..image_10 are not wired on the instance.
                            base.pop(name, None)
                        else:
                            value = self.instance_values.get(slot, _MISSING)
                            if value is _MISSING:
                                base.pop(name, None)
                            else:
                                base[name] = value
                    elif origin == -20:
                        continue
                    elif origin in self.loader_kind:
                        base[name] = external[self.loader_kind[origin]]
                    elif origin in layer_gates:
                        base[name] = layer_gates[origin].out(slot)
                    elif origin in created:
                        base[name] = created[origin].out(slot)
                    else:
                        # Source node was dropped (preview etc.).
                        base.pop(name, None)

                if node.get("type") == "RandomNoise":
                    base["noise_seed"] = int(seed) + int(li)
                if node.get("type") == "TextGenerate" and "sampling_mode.seed" in base:
                    base["sampling_mode.seed"] = int(seed) + int(li)

                if use_focus:
                    # The instance pins resolution=0, which keeps every
                    # reference at its *own* size rounded to 32 — a 17x15
                    # pupil reached the encoder as 32x32.  Ask for the size the
                    # model was trained at so the crop is actually legible.
                    if node.get("type") == "TextEncodeQwenImage21":
                        base["resolution"] = _focus_resolution_for(layer, focus_resolution)
                    if focus_prompt and node.get("type") in ("TextEncodeQwenImage21", "TextGenerate"):
                        base["prompt"] = focus_prompt

                for key, value in base.items():
                    graph_node.set_input(key, value)

            out_id, out_slot = self.output_link
            if out_id not in created:
                raise ValueError("subgraph output comes from a skipped node: %s" % out_id)
            output_image = created[out_id].out(out_slot)
            output_mask = load.out(1)
            out_type = (self.sg_outputs[0].get("type") if self.sg_outputs else "IMAGE")
            if crop is not None and str(out_type).upper() == "IMAGE":
                paste = graph.node(
                    "STLU_PasteBack",
                    id=prefix + "_paste",
                    image=output_image,
                    crop_info=crop.out(1),
                    mask=load.out(1),
                )
                output_image = paste.out(0)
                output_mask = paste.out(1)
            save = graph.node(
                "STLU_SaveLayer",
                id=prefix + "_save",
                images=output_image,
                mask=output_mask,
                # Linked (not literal) so a failed LoadLayer yields an empty
                # name and the save node can skip instead of writing a blank.
                name=load.out(2),
                out_dir=plan["out_dir"],
            )
            last_save = save
            report.append({
                "layer": layer["name"],
                "index": int(li),
                "target": layer["target"],
                "focus": bool(use_focus),
                "save_node": save.id,
            })

        # Chain the encoders through pass-through barriers.  Each gate waits
        # for the tail of the chain (= every encoder), never the other way, so
        # there is no cycle; the chain is what forces "all encode, then all
        # sample" instead of relying on executor walk order.
        if all_te_nodes:
            prev = None
            for i, te_node in enumerate(all_te_nodes):
                kwargs = {"value": te_node.out(0)}
                if prev is not None:
                    kwargs["prev"] = prev
                barrier = graph.node(
                    "STLU_Barrier", id="%s_encode_bar%d" % (batch_tag, i), **kwargs
                )
                prev = barrier.out(0)
            for gate in all_gates:
                gate.set_input("dummy", prev)

        return last_save, report


def load_template(workflow_path: str) -> SubgraphTemplate:
    """Load the first subgraph and its instance from a UI workflow JSON."""
    with open(workflow_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    subgraphs = (doc.get("definitions") or {}).get("subgraphs") or []
    if not subgraphs:
        raise ValueError("workflow has no subgraph definitions: %s" % workflow_path)
    subgraph = subgraphs[0]
    instance = None
    for node in doc.get("nodes", []):
        if node.get("type") == subgraph.get("id"):
            instance = node
            break
    if instance is None:
        raise ValueError(
            "workflow has no instance of subgraph %r on the canvas: %s"
            % (subgraph.get("name"), workflow_path)
        )
    return SubgraphTemplate(doc, subgraph, instance)


def build_expand(
    workflow_path: str,
    plan: Dict[str, Any],
    plan_ref: str,
    external: Dict[str, List[Any]],
    seed: int = 42,
    layer_indices: Optional[List[int]] = None,
) -> Tuple[Dict[str, Any], Optional[Any], List[Dict[str, Any]]]:
    """Build the expanded graph for the plan.

    Returns ``(expand_json, last_save_node, per_layer_report)``.
    """
    template = load_template(workflow_path)
    if layer_indices is None:
        layer_indices = [layer["z"] for layer in plan["layers"] if layer.get("selected")]

    graph = GraphBuilder()
    last_save, report = template.expand(graph, plan, plan_ref, external, seed, layer_indices)
    return graph.finalize(), last_save, report


def default_workflow_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    comfy_root = os.path.dirname(os.path.dirname(here))
    workflow_root = os.path.join(comfy_root, "user", "default", "workflows")
    name = "Live2D层级二次放大.json"
    preferred = os.path.join(workflow_root, "Comfy-My-Workflow", "图片", name)
    if os.path.isfile(preferred):
        return preferred
    # The user re-files workflows now and then; find it wherever it lives.
    for dirpath, _dirnames, filenames in os.walk(workflow_root):
        if name in filenames:
            return os.path.join(dirpath, name)
    return preferred
