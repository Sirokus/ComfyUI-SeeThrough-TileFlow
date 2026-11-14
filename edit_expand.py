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


# Preview-only nodes: dropped from every clone regardless of the model.
BASE_DROP_CLASSES = {"PreviewAny", "PreviewImage", "MarkdownNote"}

# The whole prompt-enhancer branch goes with it: dropping the enhancer keeps
# the literal extraction prompt and skips a second ~9B text model.
ENHANCER_DROP_CLASSES = {"TextGenerate", "BatchImagesNode", "PrimitiveStringMultiline"}

# Default (Qwen Image Edit 2.1) description of the subgraph.  A template may
# ship a sibling ``<name>.adapter.json`` that overrides any top-level section
# here; without one, the built-in Qwen layout is assumed, so older templates
# and the legacy nodes behave exactly as before.
DEFAULT_ADAPTER: Dict[str, Any] = {
    "label": "Qwen Image Edit 2.1",
    # class name -> what it loads from outside the subgraph
    "loaders": {
        "UNETLoader": "model",
        "QwenImage21Cache": "model",
        "CLIPLoader": "clip",
        "VAELoader": "vae",
    },
    # The encoder node that receives the part's literal prompt.  ``literals``
    # maps its input names to the semantic value to write there.
    "prompt_enhancer": {
        "class": "TextEncodeQwenImage21",
        "enhancer_nodes": ["TextGenerate"],
        "literals": {
            "prompt": "prompt",
            "negative_prompt": "negative_prompt",
            "resolution": "resolution",
        },
        "gate": True,
        "drop_switch_inputs": ["switch_1"],
    },
    "seed": {"class": "RandomNoise", "field": "noise_seed"},
    "cfg": {"class": "CFGGuider", "field": "cfg"},
    "sampler": {"class": "KSamplerSelect", "field": "sampler_name"},
    "sigmas": {"class": "ManualSigmas", "field": "sigmas"},
    "size": {"class": "EmptyLatentImage", "fields": ["width", "height"],
             "batch_field": "batch_size"},
    # Latent-switch nodes kept (not dropped) but pinned to image_1's size.
    "switch_classes": ["ComfySwitchNode"],
}

_MISSING = object()

# The transparent-background instruction is part of the task prompt itself
# (``edit_plan.task_prompt``): every part asks the model for a real alpha
# channel, so no tail here has to key anything.


def merge_adapter(adapter: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Overlay a template's adapter onto the Qwen defaults.

    Dict sections (loaders / prompt_enhancer / seed / ...) merge one level
    deep, so a new template only spells out the parts it changes and inherits
    the rest (e.g. ``enhancer_nodes``) from the default.
    """
    merged = dict(DEFAULT_ADAPTER)
    for key, value in (adapter or {}).items():
        if value is None:
            continue
        base = merged.get(key)
        if isinstance(value, dict) and isinstance(base, dict):
            merged[key] = dict(base, **value)
        else:
            merged[key] = value
    return merged


class EditSubgraph:
    def __init__(self, doc: Dict[str, Any], subgraph: Dict[str, Any],
                 instance: Dict[str, Any], adapter: Optional[Dict[str, Any]] = None):
        self.doc = doc
        self.sg = subgraph
        self.instance = instance
        self.adapter = merge_adapter(adapter)
        self.nodes = {int(n["id"]): n for n in subgraph.get("nodes", [])}
        self.links = {int(l["id"]): l for l in subgraph.get("links", [])}
        self.sg_inputs = subgraph.get("inputs", [])
        self.sg_outputs = subgraph.get("outputs", [])
        self.instance_values = self._instance_values()

        enh = self.adapter.get("prompt_enhancer") or {}
        self.enhancer_class = enh.get("class")
        self.enhancer_classes = set(enh.get("enhancer_nodes") or [])
        self.encoder_literals = dict(enh.get("literals") or {})
        self.encoder_gate = bool(enh.get("gate", True))
        self.enhancer_switch_inputs = set(enh.get("drop_switch_inputs") or [])
        self.loader_map = dict(self.adapter.get("loaders") or {})
        self.switch_classes = set(self.adapter.get("switch_classes") or [])

        skip = set(self.loader_map)
        self.skip_classes = skip
        self.drop_classes = set(BASE_DROP_CLASSES)
        if enh:
            self.drop_classes |= ENHANCER_DROP_CLASSES

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
            kind = self.loader_map.get(ct)
            if not kind:
                continue
            if kind == "clip" and self.enhancer_classes:
                # A CLIP that only feeds the prompt enhancer must be dropped
                # with it, not rewired to the batch node's CLIP.
                consumers = [l["target_id"] for l in self.links.values()
                             if l["origin_id"] == node_id]
                if any(self.nodes.get(c, {}).get("type") in self.enhancer_classes
                       for c in consumers):
                    kind = "pe_clip"
            kinds[node_id] = kind
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
        Which subgraph input marks the droppable one comes from the adapter.
        """
        drop = set()
        if not self.enhancer_switch_inputs:
            return drop
        for node_id, node in self.nodes.items():
            if node.get("type") != "ComfySwitchNode":
                continue
            switch_link = None
            for inp in node.get("inputs") or []:
                if inp.get("name") == "switch":
                    switch_link = inp.get("link")
            if self._sg_input_name(switch_link) in self.enhancer_switch_inputs:
                drop.add(node_id)
        return drop

    def _find_output_link(self) -> Tuple[int, int]:
        for link in self.links.values():
            if link.get("target_id") == -20:
                return int(link["origin_id"]), int(link["origin_slot"])
        raise ValueError("subgraph %r has no output link" % self.name)

    # -- expansion ---------------------------------------------------------

    def _overrides(self, **values) -> Dict[str, Dict[str, Any]]:
        """Values that must never be inherited from another part, per class.

        Built entirely from the adapter: which class carries the seed, the
        cfg, the sampler name, the sigmas, the sampling size, and which inputs
        of the encoder take the literal prompt.
        """
        out: Dict[str, Dict[str, Any]] = {}

        def put(spec, **fields):
            if not isinstance(spec, dict):
                return
            cls = spec.get("class")
            if cls:
                out.setdefault(cls, {}).update(fields)

        put(self.adapter.get("seed"),
            **{self.adapter["seed"].get("field", "noise_seed"):
               int(values["seed"]) + int(values["seed_index"])})
        put(self.adapter.get("cfg"),
            **{self.adapter["cfg"].get("field", "cfg"): float(values["cfg"])})
        put(self.adapter.get("sampler"),
            **{self.adapter["sampler"].get("field", "sampler_name"):
               str(values["scheduler"])})
        put(self.adapter.get("sigmas"),
            **{self.adapter["sigmas"].get("field", "sigmas"):
               str(values["sigmas"])})

        size = self.adapter.get("size") or {}
        fields = list(size.get("fields") or ["width", "height"])
        size_values = [int(values["width"]), int(values["height"])]
        size_fields = {f: v for f, v in zip(fields, size_values)}
        batch = size.get("batch_field")
        if batch:
            size_fields[batch] = 1
        put(size, **size_fields)

        if self.enhancer_class:
            enc_values = {
                "prompt": values["prompt"],
                "negative_prompt": values["negative_prompt"],
                "resolution": int(values["resolution"]),
                "width": int(values["width"]),
                "height": int(values["height"]),
            }
            enc_fields = {inp: enc_values[sem]
                          for inp, sem in self.encoder_literals.items()
                          if sem in enc_values}
            out.setdefault(self.enhancer_class, {}).update(enc_fields)
        return out


    def expand(self, graph: GraphBuilder, parts: List[Dict[str, Any]],
               external: Dict[str, Any], seed: int, out_dir_rel: str,
               negative_prompt: str, cfg: float,
               resolution: int, scheduler: str, sigmas: str,
               width: int, height: int, crop_mode: bool = False):
        """Append one clone per part; return the list of save nodes.

        In ``crop_mode`` every part is a PSD crop: the clone is fed a
        ``STLU_LoadCrop`` instead of a shared reference image, its sampling
        size follows the part, and the tail cuts the model's own RGBA back
        to the layer with ``STLU_CutToLayer``.  Outside crop mode the model's
        decoded RGBA is saved as it came -- no key, no mask is invented.
        """
        saves: List[Any] = []
        all_encoders: List[Any] = []
        all_gates: List[Any] = []

        for index, part in enumerate(parts):
            prefix = "ep%d" % index
            # Skipped layers keep their slot so every part's seed stays put.
            seed_index = int(part.get("seed_index", index))
            created: Dict[int, Any] = {}
            gate_by_node: Dict[int, Any] = {}

            # Crop mode: the clone is edited at the crop's work size, not at
            # the batch-wide sampling size.
            load_crop = None
            load_refs: List[Any] = []
            part_w, part_h, part_res = int(width), int(height), int(resolution)
            if crop_mode:
                part_w = max(1, int(part["width"]))
                part_h = max(1, int(part["height"]))
                part_res = max(1, int(part["resolution"]))
                load_crop = graph.node(
                    "STLU_LoadCrop", id="%s_crop" % prefix,
                    file=part["crop_file"], width=part_w, height=part_h,
                )
                # Optional high-resolution reference: the same frame cut out
                # at the crop's box, read at its native size so the encoder
                # downscales it once (it does the ~resolution^2 fit itself).
                # It owns <image2>; the task's own pattern/style refs follow.
                if str(part.get("ref_file") or "").strip():
                    load_refs.append(graph.node(
                        "STLU_LoadCrop", id="%s_ref" % prefix,
                        file=part["ref_file"], width=0, height=0,
                    ))
                for ref_i, ref_path in enumerate(part.get("ref_files") or []):
                    ref_path = str(ref_path or "").strip()
                    if not ref_path:
                        continue
                    load_refs.append(graph.node(
                        "STLU_LoadCrop", id="%s_ref%d" % (prefix, ref_i + 1),
                        file=ref_path, width=0, height=0,
                    ))

            for node_id, node in self.nodes.items():
                ct = node.get("type")
                if (ct in self.skip_classes or ct in self.drop_classes
                        or node_id in self.drop_ids):
                    continue
                if int(node.get("mode", 0)) == 4:
                    raise ValueError(
                        "subgraph node %s (%s) is bypassed; bypassed nodes are not "
                        "supported" % (node_id, ct))
                created[node_id] = graph.node(ct, id="%s_%s" % (prefix, node_id))
                if ct == self.enhancer_class and self.encoder_gate:
                    gate_by_node[node_id] = graph.node(
                        "STLU_Gate", id="%s_%s_gate" % (prefix, node_id),
                        positive=created[node_id].out(0),
                        negative=created[node_id].out(1),
                        latent=created[node_id].out(2),
                    )
                    all_encoders.append(created[node_id])
                    all_gates.append(gate_by_node[node_id])

            edit_prompt = part["prompt"]

            overrides = self._overrides(
                prompt=edit_prompt, negative_prompt=negative_prompt,
                resolution=part_res if crop_mode else int(resolution),
                seed=seed, seed_index=seed_index, cfg=cfg, scheduler=scheduler,
                sigmas=sigmas, width=part_w, height=part_h)

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
                    if ct == self.enhancer_class and name in self.encoder_literals:
                        # Written as a literal further down; never inherit the
                        # template's wiring (its prompt comes from a text node).
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
                            if crop_mode:
                                base[name] = load_crop.out(0)
                            else:
                                base[name] = external["reference"]
                        elif (crop_mode and sname.startswith("images.image_")
                              and sname != "images.image_1"):
                            # <imageN> is numbered over the CONNECTED slots, so
                            # wire the references consecutively from image_2.
                            try:
                                img_idx = int(sname.rsplit("_", 1)[-1])
                            except ValueError:
                                img_idx = 0
                            if 2 <= img_idx <= len(load_refs) + 1:
                                base[name] = load_refs[img_idx - 2].out(0)
                            else:
                                base.pop(name, None)
                        elif sname == "negative_prompt":
                            base[name] = negative_prompt
                        elif sname == "cfg":
                            base[name] = float(cfg)
                        elif sname == "switch":
                            base[name] = False   # sampling latent = image_1 size
                        elif sname == "switch_1":
                            base[name] = False   # raw prompt, no enhancer text
                        elif sname == "width":
                            base[name] = part_w
                        elif sname == "height":
                            base[name] = part_h
                        elif sname == "scheduler":
                            base[name] = str(scheduler)
                        elif sname == "seed":
                            base[name] = int(seed) + seed_index
                        elif sname == "resolution":
                            base[name] = part_res
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
                # Kept byte-identical per part except the seed, so ComfyUI's
                # signature cache can never reuse a neighbour's result.
                for key_name, value in overrides.get(ct, {}).items():
                    base[key_name] = value
                if ct in self.switch_classes:
                    base["switch"] = False

                for key_name, value in base.items():
                    graph_node.set_input(key_name, value)

            out_id, out_slot = self.output_link
            if out_id not in created:
                raise ValueError("subgraph output comes from a skipped node: %s" % out_id)
            image = created[out_id].out(out_slot)

            if crop_mode:
                # The model decodes straight RGBA: its own alpha channel is
                # the silhouette, so the crop is cut back with exactly what
                # the model drew -- no key, no source lock.
                save = graph.node(
                    "STLU_CutToLayer", id="%s_cut" % prefix,
                    image=image,
                    geometry=json.dumps(part["geometry"]),
                )
                saves.append(save)
                continue

            # No crop: the model decodes straight RGBA, so the layer is that
            # image with its own alpha -- nothing is keyed here.
            alpha_mask = graph.node(
                "ImageToMask", id="%s_alpha" % prefix,
                image=image, channel="alpha",
            )
            transparent = graph.node(
                "InvertMask", id="%s_alpha_inv" % prefix,
                mask=alpha_mask.out(0),
            )
            save = graph.node(
                "STLU_SaveLayer", id="%s_save" % prefix,
                images=image,
                mask=transparent.out(0),
                keep_input_alpha=True,
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


def load_template(workflow_path: str,
                  adapter: Optional[Dict[str, Any]] = None) -> EditSubgraph:
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
    return EditSubgraph(doc, subgraph, instance, adapter=adapter)


def build_expand(workflow_path: str, parts: List[Dict[str, Any]],
                 external: Dict[str, Any], seed: int, out_dir_rel: str,
                 layers_json_path: str, psd_name: str, psd_plan: str = "",
                 adapter: Optional[Dict[str, Any]] = None, **options):
    """Clone the edit subgraph per part and chain the output nodes.

    ``psd_plan`` switches the tail from the layer merge to the PSD-upscale
    assembler, which rebuilds the full layered document.
    ``adapter`` describes where the template keeps its loaders / prompt / seed
    nodes; omit it for the built-in Qwen Image Edit 2.1 layout.
    """
    template = load_template(workflow_path, adapter=adapter)
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
    if psd_plan:
        psd = graph.node(
            "STLU_PsdAssemble", id="ep_psd",
            plan_json=psd_plan,
            trigger=prev,
            psd_filename=psd_name,
        )
    else:
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
