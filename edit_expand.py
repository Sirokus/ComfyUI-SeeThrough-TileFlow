"""Expand the Qwen-Image-Edit subgraph once per Live2D part.

The template is the normal single-image edit workflow (or the batch workflow
itself, as long as it holds one subgraph definition plus its instance).  Every
part gets a fresh clone with a literal prompt and seed, so ComfyUI's signature
cache can never reuse another part's result.

Model loaders and the optional prompt-enhancer branch are dropped from the
clone: the loaders are rewired to the batch node's MODEL / CLIP / VAE inputs
(loaded once per run), and the enhancer would rewrite the extraction
instruction and load a second 9B model.

Encoding is chained through ``STLU_Barrier`` gates so all prompts are encoded
before the first sample starts -- the text encoder stays resident instead of
being pushed out of VRAM on every part.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from comfy_execution.graph_utils import GraphBuilder, Node

try:
    from . import edit_parts
except ImportError:  # pragma: no cover - standalone import from the tests
    import edit_parts


# Loaded once by the batch node; their outputs feed the clone from outside.
SKIP_CLASSES = {"UNETLoader", "CLIPLoader", "VAELoader", "QwenImage21Cache"}

# Preview-only nodes plus the whole prompt-enhancer branch.  Dropping the
# enhancer keeps the literal extraction prompt and skips a 9B text model.
DROP_CLASSES = {
    "PreviewAny", "PreviewImage", "MarkdownNote",
    "TextGenerate", "BatchImagesNode", "PrimitiveStringMultiline",
}

# Inputs of TextEncodeQwenImage21 we set directly instead of following links.
_ENCODER_LITERALS = {"prompt", "negative_prompt", "resolution"}

_MISSING = object()


class EditSubgraph:
    def __init__(self, doc: Dict[str, Any], subgraph: Dict[str, Any],
                 instance: Dict[str, Any]):
        self.doc = doc
        self.sg = subgraph
        self.instance = instance
        self.nodes = {int(n["id"]): n for n in subgraph.get("nodes", [])}
        self.links = {int(l["id"]): l for l in subgraph.get("links", [])}
        self.sg_inputs = subgraph.get("inputs", [])
        self.sg_outputs = subgraph.get("outputs", [])
        self.instance_values = self._instance_values()
        self.loader_kind = self._classify_loaders()
        self.drop_ids = self._classify_dropped_switches()
        self.output_link = self._find_output_link()
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
                consumers = [l["target_id"] for l in self.links.values()
                             if l["origin_id"] == node_id]
                is_pe = any(self.nodes.get(c, {}).get("type") == "TextGenerate"
                            for c in consumers)
                kinds[node_id] = "pe_clip" if is_pe else "clip"
        return kinds

    def _sg_input_name(self, link_id: Optional[int]) -> str:
        if link_id is None:
            return ""
        link = self.links.get(int(link_id))
        if link is None or int(link["origin_id"]) != -10:
            return ""
        slot = int(link["origin_slot"])
        if slot >= len(self.sg_inputs):
            return ""
        return str(self.sg_inputs[slot].get("name") or "")

    def _classify_dropped_switches(self) -> set:
        """ComfySwitchNode instances feeding the prompt enhancer are dropped.

        The subgraph has two: one selects the sampling latent (keep it, force
        ``switch=False`` so the latent follows image_1) and one selects raw vs
        enhanced prompt (drop it -- the encoder is fed the literal prompt).
        """
        drop = set()
        for node_id, node in self.nodes.items():
            if node.get("type") != "ComfySwitchNode":
                continue
            switch_link = None
            for inp in node.get("inputs") or []:
                if inp.get("name") == "switch":
                    switch_link = inp.get("link")
            if self._sg_input_name(switch_link) == "switch_1":
                drop.add(node_id)
        return drop

    def _find_output_link(self) -> Tuple[int, int]:
        for link in self.links.values():
            if link.get("target_id") == -20:
                return int(link["origin_id"]), int(link["origin_slot"])
        raise ValueError("subgraph %r has no output link" % self.name)

    # -- expansion ---------------------------------------------------------

    def expand(self, graph: GraphBuilder, parts: List[Dict[str, Any]],
               external: Dict[str, Any], seed: int, out_dir_rel: str,
               key: str, tolerance: float, softness: float, protect: bool,
               despill: float, negative_prompt: str, cfg: float,
               resolution: int, scheduler: str, sigmas: str,
               width: int, height: int):
        """Append one clone per part; return the list of SaveLayer nodes."""
        saves: List[Any] = []
        all_encoders: List[Any] = []
        all_gates: List[Any] = []

        for index, part in enumerate(parts):
            prefix = "ep%d" % index
            created: Dict[int, Any] = {}
            gate_by_node: Dict[int, Any] = {}

            for node_id, node in self.nodes.items():
                ct = node.get("type")
                if (ct in SKIP_CLASSES or ct in DROP_CLASSES
                        or node_id in self.drop_ids):
                    continue
                if int(node.get("mode", 0)) == 4:
                    raise ValueError(
                        "subgraph node %s (%s) is bypassed; bypassed nodes are not "
                        "supported" % (node_id, ct))
                created[node_id] = graph.node(ct, id="%s_%s" % (prefix, node_id))
                if ct == "TextEncodeQwenImage21":
                    gate_by_node[node_id] = graph.node(
                        "STLU_Gate", id="%s_%s_gate" % (prefix, node_id),
                        positive=created[node_id].out(0),
                        negative=created[node_id].out(1),
                        latent=created[node_id].out(2),
                    )
                    all_encoders.append(created[node_id])
                    all_gates.append(gate_by_node[node_id])

            for node_id, node in self.nodes.items():
                if node_id not in created:
                    continue
                graph_node = created[node_id]
                ct = node.get("type")
                allowed = {i.get("name") for i in (node.get("inputs") or [])
                           if i.get("name")}
                base: Dict[str, Any] = {}
                for key_name, value in (node.get("widgets_values_named") or {}).items():
                    if key_name in allowed and value is not None:
                        base[key_name] = value

                for inp in node.get("inputs") or []:
                    name = inp.get("name")
                    link_id = inp.get("link")
                    if name is None or link_id is None:
                        continue
                    if ct == "TextEncodeQwenImage21" and name in _ENCODER_LITERALS:
                        if name == "prompt":
                            base[name] = part["prompt"]
                        elif name == "negative_prompt":
                            base[name] = negative_prompt
                        else:
                            base[name] = int(resolution)
                        continue
                    link = self.links.get(int(link_id))
                    if link is None:
                        continue
                    origin = int(link["origin_id"])
                    slot = int(link["origin_slot"])

                    if origin == -10:
                        sgin = self.sg_inputs[slot] if slot < len(self.sg_inputs) else {}
                        sname = str(sgin.get("name") or "")
                        if sname == "images.image_1":
                            base[name] = external["reference"]
                        elif sname == "negative_prompt":
                            base[name] = negative_prompt
                        elif sname == "cfg":
                            base[name] = float(cfg)
                        elif sname == "switch":
                            base[name] = False   # sampling latent = image_1 size
                        elif sname == "switch_1":
                            base[name] = False   # raw prompt, no enhancer text
                        elif sname == "width":
                            base[name] = int(width)
                        elif sname == "height":
                            base[name] = int(height)
                        elif sname == "scheduler":
                            base[name] = str(scheduler)
                        elif sname == "seed":
                            base[name] = int(seed) + int(index)
                        elif sname == "resolution":
                            base[name] = int(resolution)
                        elif str(sgin.get("type")) == "IMAGE":
                            base.pop(name, None)   # image_2..10 unused
                        else:
                            value = self.instance_values.get(slot, _MISSING)
                            if value is _MISSING:
                                base.pop(name, None)
                            else:
                                base[name] = value
                    elif origin == -20:
                        continue
                    elif origin in self.loader_kind:
                        kind = self.loader_kind[origin]
                        if kind == "pe_clip":
                            base.pop(name, None)
                        else:
                            base[name] = external[kind]
                    elif origin in gate_by_node:
                        base[name] = gate_by_node[origin].out(slot)
                    elif origin in created:
                        base[name] = created[origin].out(slot)
                    else:
                        base.pop(name, None)

                # Literal overrides that must never inherit from another part.
                if ct == "RandomNoise":
                    base["noise_seed"] = int(seed) + int(index)
                if ct == "TextEncodeQwenImage21":
                    base["prompt"] = part["prompt"]
                    base["negative_prompt"] = negative_prompt
                    base["resolution"] = int(resolution)
                if ct == "CFGGuider":
                    base["cfg"] = float(cfg)
                if ct == "KSamplerSelect":
                    base["sampler_name"] = str(scheduler)
                if ct == "ManualSigmas":
                    base["sigmas"] = str(sigmas)
                if ct == "ComfySwitchNode":
                    base["switch"] = False
                if ct == "EmptyLatentImage":
                    base["width"] = int(width)
                    base["height"] = int(height)
                    base["batch_size"] = 1

                for key_name, value in base.items():
                    graph_node.set_input(key_name, value)

            out_id, out_slot = self.output_link
            if out_id not in created:
                raise ValueError("subgraph output comes from a skipped node: %s" % out_id)
            image = created[out_id].out(out_slot)

            chroma = graph.node(
                "STLU_ChromaToAlpha", id="%s_chroma" % prefix,
                image=image,
                reference=external["reference"],
                key_color=key,
                tolerance=float(tolerance),
                softness=float(softness),
                protect=bool(protect),
                despill=float(despill),
            )
            save = graph.node(
                "STLU_SaveLayer", id="%s_save" % prefix,
                images=chroma.out(0),
                mask=chroma.out(1),
                name=part["name"],
                out_dir=out_dir_rel,
            )
            saves.append(save)

        if all_encoders:
            prev = None
            for i, encoder in enumerate(all_encoders):
                kwargs: Dict[str, Any] = {"value": encoder.out(0)}
                if prev is not None:
                    kwargs["prev"] = prev
                barrier = graph.node("STLU_Barrier", id="ep_encode_bar%d" % i, **kwargs)
                prev = barrier.out(0)
            for gate_node in all_gates:
                gate_node.set_input("dummy", prev)

        return saves


def load_template(workflow_path: str) -> EditSubgraph:
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
        raise ValueError("workflow has no instance of subgraph %r on the canvas: %s"
                         % (subgraph.get("name"), workflow_path))
    return EditSubgraph(doc, subgraph, instance)


def build_expand(workflow_path: str, parts: List[Dict[str, Any]],
                 external: Dict[str, Any], seed: int, out_dir_rel: str,
                 layers_json_path: str, psd_name: str, **options):
    template = load_template(workflow_path)
    graph = GraphBuilder()
    saves = template.expand(graph, parts, external, seed, out_dir_rel, **options)

    # Chain every save into one barrier so the PSD assembler runs last.
    # NOTE: ``prev`` must be the link ([id, slot]), not the Node object --
    # a Node stored as an input value is not a link, so the executor would
    # treat the PSD trigger as a constant and run it before any sampling.
    prev = None
    for i, save in enumerate(saves):
        kwargs: Dict[str, Any] = {"value": save.out(0)}
        if prev is not None:
            kwargs["prev"] = prev
        prev = graph.node("STLU_Barrier", id="ep_save_bar%d" % i, **kwargs).out(0)
    psd = graph.node(
        "STLU_PartsPSD", id="ep_psd",
        layers_json=layers_json_path,
        trigger=prev,
        psd_filename=psd_name,
    )
    expand = graph.finalize()
    for node_id, info in expand.items():
        for name, value in (info.get("inputs") or {}).items():
            if isinstance(value, Node):
                raise RuntimeError(
                    "expansion left a Node object on %s.%s -- it must be "
                    "stored as a link via .out(slot)" % (node_id, name))
    return expand, psd, saves
