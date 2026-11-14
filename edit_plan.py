"""CPU-only planning for the PSD per-layer edit workflow.

The upscale workflow already knows how to cut a finished PSD into one crop per
layer and how to put the layers back together.  Editing reuses all of it and
only adds three ideas:

* a *task list* -- which layer, what instruction, which reference; the crop
  travels as straight RGBA, so the model's own alpha channel is the silhouette,
* *new layers* -- an accessory that does not exist in the PSD yet: a fully
  transparent canvas the model paints on, placed at a box the user picked,
  inserted at a chosen depth,
* *crop restore* -- a crop that came back from the model is resized and pasted
  back into its slot by alpha; it is never white-un-composited, because the
  model never painted on white.

Everything here is deterministic and file based; the nodes only move tensors.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import core_io, core_plan, psd_upscale
except ImportError:  # standalone import used by the tests
    import core_io
    import core_plan
    import psd_upscale

EDIT_PLAN_NAME = "edit_plan.json"
MODE = "psd_edit"

SILHOUETTE_ALPHA = "alpha"
SILHOUETTE_LOCK = "lock"

# The Qwen Image 2.1 VAE is RGBA, so the edit model already returns a real
# alpha channel.  The default is to take that channel straight to the layer.
DEFAULT_SILHOUETTE = SILHOUETTE_ALPHA

_ALPHA_ALIASES = {"alpha", "native", "model", "rgba", "transparent", "透明",
                  "原生", "原生透明", "透明背景"}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "margin": psd_upscale.DEFAULT_MARGIN_RATIO,
    "psd_name": "",
}
_LOCK_ALIASES = {"lock", "locked", "keep", "source", "锁定", "保持轮廓", "原轮廓"}

# The edit template exposes image_1..image_10.  image_1 is the layer crop and
# the global high-res original (if selected) takes image_2, so a task's own
# pattern/style references share the rest.
MAX_TASK_REFS = 9
MAX_TASK_REFS_WITH_GLOBAL = 8


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

def _normalize_silhouette(value: Any, default: str = DEFAULT_SILHOUETTE) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in _ALPHA_ALIASES:
        return SILHOUETTE_ALPHA
    if text in _LOCK_ALIASES:
        return SILHOUETTE_LOCK
    return default


def _parse_box(value: Any) -> Optional[List[int]]:
    """Accept ``x,y,w,h`` / ``l,t,r,b`` lists / ``{x,y,w,h}`` dicts."""
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        try:
            x, y = int(value["x"]), int(value["y"])
            w, h = int(value["w"]), int(value["h"])
        except (KeyError, TypeError, ValueError):
            return None
        return [x, y, x + w, y + h]
    if isinstance(value, str):
        text = value.strip().strip("()[]")
        if not text:
            return None
        parts = [p for p in text.replace("，", ",").replace(" ", ",").split(",") if p]
        if len(parts) != 4:
            return None
        try:
            nums = [int(round(float(p))) for p in parts]
        except ValueError:
            return None
        return [nums[0], nums[1], nums[0] + nums[2], nums[1] + nums[3]]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            nums = [int(round(float(v))) for v in value]
        except (TypeError, ValueError):
            return None
        return nums
    return None


def _split_refs(value: Any) -> List[str]:
    """Accept a list, or one string with ``;`` / ``；`` / newline separators."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        items = [str(v or "") for v in value]
    else:
        text = str(value)
        for sep in ("；", ";", "\n"):
            text = text.replace(sep, "|")
        items = text.split("|")
    return [i.strip().strip('"') for i in items if i.strip().strip('"')]


def _normalize_task(raw: Dict[str, Any]) -> Dict[str, Any]:
    name = str(raw.get("name") or "").strip()
    box = _parse_box(raw.get("box"))
    refs = _split_refs(raw.get("ref_files"))
    if not refs:
        refs = _split_refs(raw.get("ref_file") or raw.get("reference"))
    # A raw 4-list from the panel is already l,t,r,b; a 4-list from a JSON
    # task could be either, so the panel always sends {"x","y","w","h"}.
    return {
        "name": name,
        "prompt": str(raw.get("prompt") or raw.get("instruction") or "").strip(),
        # Pattern / style images for this task only.  The global high-res
        # original stays a separate ``reference_path`` and keeps <image2>.
        "ref_files": refs,
        "ref_file": refs[0] if refs else "",
        "silhouette": _normalize_silhouette(raw.get("silhouette")
                                            or raw.get("mode")),
        "box": box,
        "insert": str(raw.get("insert") or raw.get("insert_after") or "").strip(),
        "blend": str(raw.get("blend") or "norm").strip() or "norm",
        "opacity": max(0, min(255, int(raw.get("opacity", 255) or 255))),
        # "强制重跑": by default an already generated layer is skipped.
        "force": bool(raw.get("force") or raw.get("force_rerun")),
    }


def _parse_line(line: str) -> Dict[str, Any]:
    """One text row: ``name | prompt | reference | mode | box | insert | force``.

    The reference cell may list several files separated by ``;``.  The last
    cell is the "强制重跑" flag (``1``/``yes``/``force``).
    """
    cells = [c.strip() for c in line.replace("\t", "|").split("|")]
    cells += [""] * (7 - len(cells))
    name, prompt, ref, mode, box, insert, force = cells[:7]
    return _normalize_task({
        "name": name, "prompt": prompt, "ref_files": _split_refs(ref),
        "silhouette": mode, "box": box, "insert": insert,
        "force": force.strip().lower() in ("1", "y", "yes", "true", "force"),
    })


def parse_tasks(text: Any) -> List[Dict[str, Any]]:
    """Parse the task widget: a JSON list, ``{"tasks": [...]}`` or text rows."""
    if isinstance(text, (list, tuple)):
        data: Any = list(text)
    else:
        raw = str(text or "").strip()
        if not raw:
            return []
        if raw[0] in "[{":
            data = json.loads(raw)
            if isinstance(data, dict):
                data = data.get("tasks") or data.get("rows") or []
        else:
            tasks = []
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                task = _parse_line(line)
                if task["name"]:
                    tasks.append(task)
            return tasks
    if not isinstance(data, (list, tuple)):
        return []
    tasks = []
    for row in data:
        if not isinstance(row, dict):
            continue
        task = _normalize_task(row)
        if task["name"]:
            tasks.append(task)
    return tasks


_TRANSPARENT_TAIL = (
    "Keep everything that is not the edited part fully transparent: output "
    "the part with a real alpha channel, do not paint any background, "
    "backdrop, frame or solid colour behind it, and keep its outline crisp."
)


def task_prompt(task: Dict[str, Any]) -> str:
    """The instruction sent to the edit model for one task."""
    text = str(task.get("prompt") or "").strip()
    return ("%s\n\n%s" % (text, _TRANSPARENT_TAIL)) if text else _TRANSPARENT_TAIL


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def _clamp_box(box: Sequence[int], width: int, height: int) -> List[int]:
    left, top, right, bottom = (int(round(float(v))) for v in box)
    left = max(0, min(int(width) - 1, left))
    top = max(0, min(int(height) - 1, top))
    right = max(left + 1, min(int(width), right))
    bottom = max(top + 1, min(int(height), bottom))
    return [left, top, right, bottom]


def _unique_stem(name: str, used: set) -> str:
    base = core_plan.safe_name(name) or "layer"
    stem = base
    index = 2
    while stem in used:
        stem = "%s_%d" % (base, index)
        index += 1
    used.add(stem)
    return stem


def _insert_index(layers: List[Dict[str, Any]], insert: str) -> int:
    """Where a new layer goes: the PSD layer list runs bottom -> top."""
    target = str(insert or "").strip()
    if not target or target.lower() in ("top", "最上层", "最上"):
        return len(layers)
    if target.lower() in ("bottom", "最下层", "最下"):
        return 0
    for index in range(len(layers) - 1, -1, -1):
        if str(layers[index].get("name") or "") == target:
            return index + 1
    return len(layers)


def prepare_edit_plan(
    psd_path: str,
    out_dir_abs: str,
    tasks: List[Dict[str, Any]],
    margin_ratio: float = psd_upscale.DEFAULT_MARGIN_RATIO,
    reference_path: str = "",
    factor: float = 1.0,
    max_side: int = psd_upscale.DEFAULT_MAX_SIDE,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Scan the PSD, attach one edit task per layer and cut every crop.

    Existing layers keep their record (blend, opacity, position); the task only
    flips ``selected`` and stores its ``edit`` block.  A task whose name is not
    in the PSD is a *new layer*: it gets a transparent source PNG (so the
    assembler's fallback stays invisible), a transparent crop to paint on, and
    a slot in the layer list chosen by ``insert``.
    """
    tasks = [t for t in (tasks or []) if t.get("name")]
    if not tasks:
        raise ValueError("没有编辑任务：请至少填写一个图层名和编辑指令")

    plan = psd_upscale.prepare_plan(psd_path, out_dir_abs, factor=factor,
                                    max_side=max_side, select="none")
    width = int(plan["source"]["width"])
    height = int(plan["source"]["height"])
    used = {layer.get("stem") for layer in plan["layers"] if layer.get("stem")}

    by_name: Dict[str, Dict[str, Any]] = {}
    for layer in plan["layers"]:
        if layer.get("kind") == psd_upscale.LAYER:
            by_name.setdefault(str(layer["name"]), layer)

    ref_path = str(reference_path or "").strip().strip('"')
    # The template exposes image_1..image_10.  image_1 is the crop; the global
    # reference (if any) takes image_2, so the task's own pattern refs get at
    # most the remaining slots.
    max_task_refs = MAX_TASK_REFS_WITH_GLOBAL if ref_path else MAX_TASK_REFS

    prepared: List[Dict[str, Any]] = []
    for task in tasks:
        task = dict(task)
        name = task["name"]
        ref_files = [str(p) for p in (task.get("ref_files") or []) if str(p)]
        for ref in ref_files:
            if not os.path.isfile(ref):
                raise ValueError("参考图不存在: %s" % ref)
        if len(ref_files) > max_task_refs:
            raise ValueError(
                "图层 %r 参考图太多（%d 张）：%s最多 %d 张（提示词编号 <image2>..<image10>）"
                % (name, len(ref_files), "已设全局参考原图时" if ref_path else "",
                   max_task_refs))

        layer = by_name.pop(name, None)
        if layer is not None:
            layer["selected"] = True
            # The crop is written as straight RGBA (real alpha, RGB untouched),
            # so the model's own alpha channel is the silhouette.
            layer["edit"] = {
                "prompt": task_prompt(task),
                "silhouette": task["silhouette"],
                "new": False,
                "box": None,
                "ref_files": ref_files,
                "ref_file": ref_files[0] if ref_files else "",
                "force": bool(task.get("force")),
            }
            prepared.append({"task": task, "layer": layer, "stem": layer["stem"],
                             "new": False})
            continue

        box = task.get("box")
        if not box:
            raise ValueError("图层 %r 不在 PSD 里，作为新增层必须给出位置框" % name)
        box = _clamp_box(box, width, height)
        w, h = box[2] - box[0], box[3] - box[1]
        target, scale, capped = psd_upscale._targets(w, h, factor, max_side)
        stem = _unique_stem(name, used)
        paths = psd_upscale.plan_layer_paths(out_dir_abs, stem)
        # A brand-new part has no source pixels; keep the PNG transparent so an
        # unedited fallback simply shows nothing instead of a black box.
        core_io.write_rgba(paths["source"], np.zeros((h, w, 4), np.uint8))
        record = {
            "kind": psd_upscale.LAYER,
            "name": name,
            "blend": task.get("blend") or "norm",
            "opacity": int(task.get("opacity", 255)),
            "hidden": False,
            "stem": stem,
            "box": box,
            "size": [w, h],
            "target": [int(target[0]), int(target[1])],
            "scale": round(float(scale), 6),
            "capped": bool(capped),
            "file": paths["source"],
            "crop_file": paths["crop"],
            "out_file": paths["upscaled"],
            "selected": True,
            "new": True,
            "edit": {
                "prompt": task_prompt(task),
                "silhouette": task["silhouette"],
                "new": True,
                "box": box,
                "ref_files": ref_files,
                "ref_file": ref_files[0] if ref_files else "",
                "force": bool(task.get("force")),
            },
        }
        index = _insert_index(plan["layers"], task.get("insert") or "")
        plan["layers"].insert(index, record)
        prepared.append({"task": task, "layer": record, "stem": stem, "new": True})

    crops = write_task_crops(plan, margin_ratio=margin_ratio)
    # write_references reads plan["crops"], so it must already be attached --
    # otherwise the global reference silently never reaches the edit model.
    plan["crops"] = crops

    if ref_path:
        psd_upscale.write_references(plan, ref_path)
    # The global high-res crop keeps ``ref_file`` (<image2>); task references
    # are a separate ordered list and start at <image3>.
    ref_by_stem = {item["stem"]: list(item["task"].get("ref_files") or [])
                   for item in prepared}
    for job in crops:
        job["ref_files"] = ref_by_stem.get(job["stem"], [])

    plan["crops"] = crops
    plan["mode"] = MODE
    plan["select"] = "tasks"
    plan["max_task_refs"] = max_task_refs
    resolved = dict(DEFAULT_SETTINGS)
    resolved["margin"] = float(margin_ratio)
    for source in (settings or {}):
        if settings[source] is not None:
            resolved[source] = settings[source]
    plan["settings"] = resolved
    plan["tasks"] = [{
        "name": item["task"]["name"],
        "stem": item["stem"],
        "new": bool(item["new"]),
        "silhouette": item["task"]["silhouette"],
        "insert": item["task"].get("insert") or "",
        "ref_files": list(item["task"].get("ref_files") or []),
        "ref_file": str(item["task"].get("ref_file") or ""),
        "force": bool(item["task"].get("force")),
        "exists": _layer_generated(item["layer"]),
    } for item in prepared]
    plan["count"] = len(psd_upscale.selected_layers(plan))
    write_edit_plan(out_dir_abs, plan)
    return plan


# ---------------------------------------------------------------------------
# crops
# ---------------------------------------------------------------------------

def _new_crop(plan: Dict[str, Any], layer: Dict[str, Any]):
    """A fully transparent canvas for a part that does not exist yet."""
    width = int(plan["source"]["width"])
    height = int(plan["source"]["height"])
    x0, y0, x1, y1 = _clamp_box(layer["box"], width, height)
    crop_w, crop_h = x1 - x0, y1 - y0
    canvas = np.zeros((crop_h, crop_w, 4), np.uint8)
    core_io.write_rgba(layer["crop_file"], canvas)
    return {
        "name": layer["name"],
        "stem": layer["stem"],
        "crop_file": layer["crop_file"],
        "crop_box": [x0, y0, x1, y1],
        "crop_size": [crop_w, crop_h],
        "target": [int(layer["target"][0]), int(layer["target"][1])],
        "out_file": layer["out_file"],
    }


def write_task_crops(plan: Dict[str, Any],
                     margin_ratio: float = psd_upscale.DEFAULT_MARGIN_RATIO
                     ) -> List[Dict[str, Any]]:
    """One crop per edited layer, in layer order; new layers get a clear canvas."""
    existing = [layer for layer in plan["layers"]
                if layer.get("kind") == psd_upscale.LAYER
                and layer.get("selected") and not layer.get("new")]
    proxy = dict(plan)
    proxy["layers"] = existing
    by_stem = {job["stem"]: job
               for job in psd_upscale.write_crops(proxy, margin_ratio)}
    jobs: List[Dict[str, Any]] = []
    for layer in psd_upscale.selected_layers(plan):
        if layer.get("new"):
            jobs.append(_new_crop(plan, layer))
        else:
            jobs.append(by_stem[layer["stem"]])
    return jobs


# ---------------------------------------------------------------------------
# restore a crop
# ---------------------------------------------------------------------------

def restore_crop(
    rgba: np.ndarray,
    crop_size: Sequence[int],
    crop_box: Sequence[int],
    box: Sequence[int],
    size: Sequence[int],
    target: Sequence[int],
    alpha: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Paste a crop back at the layer's target size.

    Unlike :func:`psd_upscale.restore_layer` this never un-composites white:
    the model returned straight RGBA, so the alpha it produced is the
    silhouette and the colours are already the part's own.  The crop is
    resampled premultiplied, otherwise the old background bleeds into the
    edges.

    ``alpha`` (an HxW map on the *native crop* grid) overrides the output
    alpha channel; only the MASK fallback uses it (a model that returned RGB
    instead of RGBA).  It is resampled straight from the source crop to the
    work size exactly once, with Lanczos, so a hard edge is not smeared over
    the whole scale factor.
    """
    work_w, work_h = psd_upscale.work_size(crop_size, size, target)
    target_w = max(1, int(target[0]))
    target_h = max(1, int(target[1]))
    sx = target_w / float(max(1, int(size[0])))
    sy = target_h / float(max(1, int(size[1])))

    rgba = np.asarray(rgba, dtype=np.uint8)
    if rgba.ndim == 3 and rgba.shape[-1] >= 4:
        pixels = np.ascontiguousarray(rgba[..., :4])
    elif rgba.ndim == 3:
        pixels = np.dstack([rgba[..., :3],
                            np.full(rgba.shape[:2], 255, np.uint8)])
    else:
        pixels = np.dstack([np.repeat(rgba[..., None], 3, axis=2),
                            np.full(rgba.shape[:2], 255, np.uint8)])
    up = core_io.resize_rgba_premultiplied(pixels, work_w, work_h)
    if alpha is not None:
        a_up = core_io.resize_alpha(alpha, work_w, work_h, sharp=True)
        # dstack returns a fresh array: resize_rgba_premultiplied may hand the
        # caller's own buffer back when no resample was needed, and writing the
        # override alpha into it would mutate the input.
        up = np.dstack([up[..., :3], a_up])

    left, top = int(box[0]), int(box[1])
    ox = int(round((left - int(crop_box[0])) * sx))
    oy = int(round((top - int(crop_box[1])) * sy))

    out = np.zeros((target_h, target_w, 4), np.uint8)
    cx0, cy0 = max(0, ox), max(0, oy)
    cx1, cy1 = min(work_w, ox + target_w), min(work_h, oy + target_h)
    if cx1 <= cx0 or cy1 <= cy0:
        return out
    dx, dy = cx0 - ox, cy0 - oy
    out[dy:dy + (cy1 - cy0), dx:dx + (cx1 - cx0)] = up[cy0:cy1, cx0:cx1]
    return out


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def edit_plan_path(out_dir_abs: str) -> str:
    return os.path.join(out_dir_abs, EDIT_PLAN_NAME)


def write_edit_plan(out_dir_abs: str, plan: Dict[str, Any]) -> str:
    os.makedirs(out_dir_abs, exist_ok=True)
    # psd_plan.json stays a valid document so the upscale nodes can be chained
    # on the same folder without re-running the prepare step.
    psd_upscale.write_plan(out_dir_abs, plan)
    path = edit_plan_path(out_dir_abs)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=1)
    return path


def load_edit_plan(ref: Any) -> Dict[str, Any]:
    return psd_upscale.load_plan(ref)


def edit_layers(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Selected layers that carry an edit task, bottom -> top."""
    return [layer for layer in psd_upscale.selected_layers(plan)
            if layer.get("edit")]


def _layer_generated(layer: Dict[str, Any]) -> bool:
    """The layer's edited PNG already exists on disk.

    No fingerprints: an existing file means "done", the panel's "强制重跑"
    checkbox is the only way to override it.
    """
    out_file = str(layer.get("out_file") or "")
    return bool(out_file) and os.path.isfile(out_file)


def split_pending(layers: List[Dict[str, Any]]):
    """Split edit layers into ``(to_edit, skipped)``.

    The skipped ones keep their previous PNG; ③ reads it from ``out_file``.
    The force flag lives on ``layer["edit"]`` (written by ①).
    """
    pending, skipped = [], []
    for layer in layers:
        force = bool((layer.get("edit") or {}).get("force"))
        if not force and _layer_generated(layer):
            skipped.append(layer)
        else:
            pending.append(layer)
    return pending, skipped


def task_by_stem(plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(item.get("stem")): item for item in (plan.get("tasks") or [])}
