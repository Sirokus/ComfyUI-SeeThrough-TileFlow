"""HTTP routes backing the graphical nodes.

Everything is sandboxed to one of three roots:

* ``output``    -- ComfyUI output dir (layer sources, upscaled results, previews)
* ``input``     -- ComfyUI input dir (reference images)
* ``workflows`` -- user/default/workflows (the subgraph template)

The frontend uses ``/stlu/ls`` for the pickers, ``/stlu/layers`` for the Plan
list, ``/stlu/thumb`` for cached thumbnails and ``/stlu/preview`` for the
pre-flight composite preview.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from aiohttp import web
from PIL import Image

import folder_paths

if __package__:  # ComfyUI loads the folder as a package
    from . import (core_compose, core_plan, edit_parts, psd_reader, psd_writer,
                   tile_psd, tile_strips)
else:  # standalone import used by the tests
    import core_compose
    import core_plan
    import edit_parts
    import psd_reader
    import psd_writer
    import tile_psd
    import tile_strips

try:
    from server import PromptServer
    _ROUTES = PromptServer.instance.routes
except Exception:  # pragma: no cover - ComfyUI always has it at load time
    _ROUTES = None

SCOPES = ("output", "input", "workflows")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
THUMB_DIR = "_stlu_thumbs"
PREVIEW_DIR = "_stlu_preview"

# The preview panel asks for a plan again on every parameter change.  Planning
# a 4K source is CPU-only but not free, and doing it on the event loop would
# stall every other request.  Cache recent answers and run the work in a
# worker thread; keyed by the file's real path + mtime/size + plan settings.
_STRIP_PLAN_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_STRIP_PLAN_CACHE_MAX = 12
# Subject detection is independent of overlap/max_side/resolution/margin and
# costs seconds on a 4K image.  Cache it by file identity so tweaking a plan
# parameter only re-runs the cheap geometry step.
_SUBJECT_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_SUBJECT_CACHE_MAX = 8
_SUBJECT_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# path sandbox
# ---------------------------------------------------------------------------

def _scope_root(scope: str) -> str:
    if scope == "input":
        root = folder_paths.get_input_directory()
    elif scope == "workflows":
        root = os.path.join(folder_paths.get_user_directory(), "default", "workflows")
    else:
        root = folder_paths.get_output_directory()
    return os.path.realpath(root)


def _inside(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


def _resolve(scope: str, rel: str) -> str:
    """Resolve a scope-relative path, rejecting traversal and absolutes."""
    root = _scope_root(scope)
    rel = (rel or "").replace("\\", "/").strip()
    if rel.startswith("/") or ".." in rel.split("/"):
        raise web.HTTPBadRequest(text="invalid path")
    path = os.path.realpath(os.path.join(root, rel))
    if not _inside(root, path):
        raise web.HTTPForbidden(text="outside the %s directory" % scope)
    return path


def _relative(scope: str, path: str) -> Optional[str]:
    root = _scope_root(scope)
    path = os.path.realpath(path)
    if _inside(root, path):
        return os.path.relpath(path, root).replace(os.sep, "/")
    return None


def _resolve_any(path: str) -> str:
    """Resolve an absolute path (or output-relative path) inside a known root."""
    raw = (path or "").strip()
    if not raw:
        raise web.HTTPBadRequest(text="empty path")
    candidate = os.path.realpath(raw if os.path.isabs(raw) else os.path.join(_scope_root("output"), raw))
    for scope in SCOPES:
        if _inside(_scope_root(scope), candidate):
            return candidate
    raise web.HTTPForbidden(text="path is outside the workspace")


def _resolve_local(path: str) -> str:
    """Resolve a path the user picked in the local file browser.

    Unlike ``_resolve_any`` this accepts any absolute location: the PSD
    upscale flow starts from a finished artwork that usually lives outside
    ComfyUI's output tree.
    """
    raw = (path or "").strip().strip('"')
    if not raw:
        raise web.HTTPBadRequest(text="empty path")
    if not os.path.isabs(raw):
        raise web.HTTPBadRequest(text="absolute path required")
    return os.path.realpath(os.path.expanduser(raw))


def _json_error(message: str, status: int = 400) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _split_rel(rel: str) -> Tuple[str, str]:
    if "/" in rel:
        folder, name = rel.rsplit("/", 1)
        return folder, name
    return "", rel


# ---------------------------------------------------------------------------
# thumbnails
# ---------------------------------------------------------------------------

def _thumb_url(full: str, size: int = 96) -> str:
    if not full:
        return ""
    for scope in ("output", "input"):
        rel = _relative(scope, full)
        if rel is None:
            continue
        query = urllib.parse.urlencode({"scope": scope, "path": rel, "size": int(size)})
        try:
            query += "&v=%d" % int(os.path.getmtime(full))
        except OSError:
            pass
        return "/stlu/thumb?" + query
    return ""


def _resize_rgba(im: "Image.Image", size: int) -> "Image.Image":
    """Size an RGBA image down without letting hidden colour leak in.

    A straight-alpha LANCZOS resize averages the colour stored under alpha 0
    (SeeThrough drops dark hair colour there) into the visible pixels, which
    is exactly how a clean layer turns into a pink fringe at thumbnail size.
    Premultiply first, resize, then un-premultiply; fully transparent pixels
    become white so every backdrop flattens to pure white.
    """
    scale = min(size / float(im.width), size / float(im.height), 1.0)
    w = max(1, int(round(im.width * scale)))
    h = max(1, int(round(im.height * scale)))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    alpha = arr[..., 3:4]
    prem = np.concatenate([arr[..., :3] * alpha, alpha], axis=2)
    small = Image.fromarray(
        np.rint(prem * 255.0).astype(np.uint8), "RGBA"
    ).resize((w, h), Image.Resampling.LANCZOS)
    out = np.asarray(small, dtype=np.float32) / 255.0
    alpha = out[..., 3:4]
    rgb = np.where(alpha > 1e-6, out[..., :3] / np.maximum(alpha, 1e-6), 1.0)
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray(
        np.rint(np.concatenate([rgb, alpha], axis=2) * 255.0).astype(np.uint8), "RGBA")


def _render_thumb(full: str, size: int) -> str:
    root = _scope_root("output")
    cache_dir = os.path.join(root, THUMB_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        stamp = int(os.path.getmtime(full))
    except OSError:
        stamp = 0
    key = hashlib.sha1(("%s|%d|%d" % (full, stamp, size)).encode("utf-8", "replace")).hexdigest()
    cache = os.path.join(cache_dir, key + ".png")
    if not os.path.isfile(cache):
        with Image.open(full) as im:
            if im.mode in ("RGBA", "LA") or (
                    im.mode == "P" and "transparency" in im.info):
                im = _resize_rgba(im.convert("RGBA"), size)
            else:
                im = im.convert("RGB")
                im.thumbnail((size, size), Image.Resampling.LANCZOS)
            im.save(cache, format="PNG", compress_level=6)
        _prune_thumbs(cache_dir)
    return cache


def _prune_thumbs(cache_dir: str, limit: int = 600) -> None:
    """Keep the thumbnail cache bounded (keys change whenever a file does)."""
    try:
        names = os.listdir(cache_dir)
        if len(names) <= limit:
            return
        entries = []
        for name in names:
            path = os.path.join(cache_dir, name)
            try:
                entries.append((os.path.getmtime(path), path))
            except OSError:
                continue
        entries.sort()
        for _stamp, path in entries[: len(entries) - limit // 2]:
            try:
                os.remove(path)
            except OSError:
                pass
    except OSError:
        pass


async def thumb(request: web.Request) -> web.Response:
    scope = request.query.get("scope") or "output"
    if scope not in SCOPES:
        scope = "output"
    try:
        size = int(request.query.get("size", "96"))
    except ValueError:
        size = 96
    size = max(16, min(1024, size))
    try:
        full = _resolve(scope, request.query.get("path", ""))
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    if not os.path.isfile(full):
        return _json_error("not found", 404)
    if os.path.splitext(full)[1].lower() not in IMAGE_EXTS:
        return _json_error("not an image", 415)
    try:
        cache = await asyncio.get_running_loop().run_in_executor(None, _render_thumb, full, size)
    except Exception as exc:  # noqa: BLE001
        return _json_error("cannot read image: %s" % exc, 500)
    return web.FileResponse(cache, headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------------------
# /stlu/ls  -- folder / file browser
# ---------------------------------------------------------------------------

async def list_directory(request: web.Request) -> web.Response:
    scope = request.query.get("scope") or "output"
    if scope not in SCOPES:
        scope = "output"
    try:
        path = _resolve(scope, request.query.get("path", ""))
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    if not os.path.isdir(path):
        return _json_error("not a directory: %s" % request.query.get("path", ""), 404)

    dirs: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []
    try:
        names = sorted(os.listdir(path), key=lambda n: (not os.path.isdir(os.path.join(path, n)), n.lower()))
    except OSError as exc:
        return _json_error(str(exc), 500)

    for name in names:
        if name.startswith("."):
            continue
        full = os.path.join(path, name)
        if os.path.isdir(full):
            dirs.append({"name": name, "rel": _relative(scope, full) or name})
            continue
        lower = name.lower()
        if scope == "input":
            if not lower.endswith(IMAGE_EXTS):
                continue
        elif scope == "workflows":
            if not lower.endswith(".json"):
                continue
        elif not lower.endswith((".json", ".png", ".psd", ".txt")):
            continue
        try:
            size = os.path.getsize(full)
        except OSError:
            size = 0
        entry = {
            "name": name,
            "rel": _relative(scope, full) or name,
            "abs": full,
            "size": size,
            "is_image": lower.endswith(IMAGE_EXTS),
            "is_json": lower.endswith(".json"),
            "is_layers": lower.endswith("_layers.json") and not lower.endswith("_psdinfo.json"),
            "is_workflow": scope == "workflows" and lower.endswith(".json"),
        }
        if entry["is_image"]:
            entry["thumb"] = _thumb_url(full, 64)
        files.append(entry)

    rel = _relative(scope, path) or ""
    if rel == ".":
        rel = ""
    parent = ""
    if rel:
        parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
    return web.json_response({
        "scope": scope,
        "root": rel,
        "path": rel,
        "parent": parent,
        "at_root": not rel,
        "dirs": dirs,
        "files": files,
    })


# ---------------------------------------------------------------------------
# /stlu/fs_ls -- browser for the whole machine (PSD source picking)
# ---------------------------------------------------------------------------

def _drive_roots() -> List[Dict[str, str]]:
    if os.name != "nt":
        return [{"name": "/", "rel": "/"}]
    return [{"name": letter + ":\\", "rel": letter + ":\\"}
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if os.path.isdir(letter + ":\\")]


async def fs_list(request: web.Request) -> web.Response:
    """List drives / folders / files at an absolute path.

    The scope-based ``/stlu/ls`` cannot reach the user's artwork folder, so
    this endpoint takes a raw absolute path.  It only ever reads names and
    sizes; nothing is written here.  ``ext=image`` lists editable image types
    (the PSD crop picker) instead of the default ``.psd``.
    """
    raw = (request.query.get("path") or "").strip().strip('"')
    want_image = (request.query.get("ext") or "").strip().lower() == "image"
    if not raw:
        return web.json_response({
            "path": "", "parent": "", "at_root": True,
            "dirs": _drive_roots(), "files": [],
        })
    try:
        path = _resolve_local(raw)
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    if not os.path.isdir(path):
        return _json_error("not a directory: %s" % raw, 404)

    allowed = IMAGE_EXTS if want_image else (".psd",)
    dirs: List[Dict[str, str]] = []
    files: List[Dict[str, Any]] = []
    try:
        names = sorted(os.listdir(path),
                       key=lambda n: (not os.path.isdir(os.path.join(path, n)), n.lower()))
    except OSError as exc:
        return _json_error(str(exc), 500)

    for name in names:
        if name.startswith(".") or name.startswith("$") or name.startswith("~"):
            continue
        full = os.path.join(path, name)
        if os.path.isdir(full):
            dirs.append({"name": name, "rel": full})
            continue
        lower = name.lower()
        if not lower.endswith(allowed):
            continue
        try:
            size = os.path.getsize(full)
        except OSError:
            size = 0
        files.append({"name": name, "rel": full, "abs": full, "size": size,
                      "is_psd": lower.endswith(".psd"),
                      "is_image": lower.endswith(IMAGE_EXTS)})

    parent = os.path.dirname(path.rstrip("\\/"))
    # ``dirname('C:')`` is ``'C:'`` on Windows: a drive root has no parent
    # above it, so the picker falls back to the drive list.
    if not parent or len(parent.rstrip("\\/")) <= 2 or not os.path.isdir(parent):
        parent = ""
    return web.json_response({
        "path": path,
        "parent": parent,
        "at_root": False,
        "dirs": dirs,
        "files": files,
    })


# ---------------------------------------------------------------------------
# /stlu/layers -- layer list for the graphical Plan editor
# ---------------------------------------------------------------------------

async def list_layers(request: web.Request) -> web.Response:
    source = request.query.get("dir") or request.query.get("source") or ""
    try:
        canvas = int(request.query.get("canvas", "2048"))
    except ValueError:
        canvas = 2048
    rules_text = request.query.get("rules", "")

    try:
        doc = core_plan.resolve_layers_source(source, _scope_root("output"))
    except Exception as exc:  # noqa: BLE001
        return _json_error(str(exc), 404)

    # "" means 自动: <拆层文件夹>/UpScale, created on demand.
    basis = core_plan.auto_basis(doc)
    auto_out = core_plan.auto_out_dir(basis, _scope_root("output"))
    try:
        out_rel = core_plan.resolve_out_dir(request.query.get("out_dir", ""), basis,
                                            _scope_root("output"))
    except ValueError:
        out_rel = auto_out
    out_abs = os.path.join(_scope_root("output"), out_rel)
    if out_rel == auto_out:
        # Only the derived sidecar folder is created here (so the panel can
        # show/open it before the first run).  An explicitly picked folder is
        # left to the plan, which creates it when it actually writes.
        try:
            os.makedirs(out_abs, exist_ok=True)
        except OSError:
            pass

    src_w = max(1, int(doc.get("width") or 1024))
    src_h = max(1, int(doc.get("height") or src_w))
    scale = float(canvas) / float(src_w)

    try:
        rules = core_plan.parse_size_rules(rules_text)
    except ValueError:
        rules = core_plan.parse_size_rules("")

    ordered = sorted(
        doc["layers"],
        key=lambda layer: (-float(layer.get("depth_median", 1.0)), int(layer.get("index", 0))),
    )

    layers: List[Dict[str, Any]] = []
    upscaled_count = 0
    for z, layer in enumerate(ordered):
        left, top, right, bottom = layer["box"]
        lw, lh = max(1, right - left), max(1, bottom - top)
        settings = core_plan.merge_size_settings(layer["name"], rules)
        try:
            tw, th = core_plan.compute_target_size(lw, lh, settings, base_factor=scale)
        except Exception:  # noqa: BLE001
            tw, th = core_plan.compute_target_size(lw, lh, {}, base_factor=scale)

        up_path = os.path.join(out_abs, core_plan.safe_name(layer["name"]) + ".png")
        upscaled = os.path.isfile(up_path)
        if upscaled:
            upscaled_count += 1
        layers.append({
            "name": layer["name"],
            "z": z,
            "box": [left, top, right, bottom],
            "w": lw,
            "h": lh,
            "depth_median": float(layer.get("depth_median", 1.0)),
            "coverage": layer.get("alpha_coverage"),
            "empty": bool(layer.get("empty")),
            "filename": os.path.basename(layer["file"]),
            "thumb": _thumb_url(layer["file"], 96),
            "upscaled": upscaled,
            "upscaled_thumb": _thumb_url(up_path, 96) if upscaled else "",
            "suggested": {"w": tw, "h": th, "long_side": max(tw, th)},
        })

    return web.json_response({
        "source": {
            "layers_json": _relative("output", doc["path"]) or doc["path"],
            "source_dir": _relative("output", doc.get("source_dir") or os.path.dirname(doc["path"]))
            or doc.get("source_dir"),
            "synthetic": bool(doc.get("synthetic")),
            "canvas_w": src_w,
            "canvas_h": src_h,
            "prefix": doc.get("prefix", ""),
            "timestamp": doc.get("timestamp", ""),
        },
        "canvas": canvas,
        "scale": scale,
        "out_dir": out_rel,
        "auto_out_dir": auto_out,
        "out_dir_auto": out_rel == auto_out,
        "upscaled_count": upscaled_count,
        "count": len(layers),
        "layers": layers,
        "warnings": doc.get("warnings", []),
    })


# ---------------------------------------------------------------------------
# /stlu/preview -- pre-flight composite of the current selection
# ---------------------------------------------------------------------------

def _build_preview(args: Dict[str, Any]) -> Dict[str, Any]:
    plan = core_plan.build_plan(
        layers_json=args["source"],
        output_root=_scope_root("output"),
        select=args["select"],
        exclude=args["exclude"],
        size_rules_text=args["rules"],
        out_dir=args["out_dir"],
        canvas_size=args["canvas"],
        skip_empty=args["skip_empty"],
        empty_coverage=args["coverage"],
    )
    canvas, psd_layers, notes = core_compose.build_canvas(plan, _scope_root("output"))

    height, width = canvas.shape[:2]
    image = _resize_rgba(Image.fromarray(canvas, "RGBA"), int(args["max_side"]))

    preview_dir = os.path.join(_scope_root("output"), PREVIEW_DIR)
    os.makedirs(preview_dir, exist_ok=True)
    preview_path = os.path.join(preview_dir, "preview.png")
    image.save(preview_path, format="PNG", compress_level=4)

    folder, name = _split_rel(_relative("output", preview_path) or (PREVIEW_DIR + "/preview.png"))
    query = urllib.parse.urlencode({
        "filename": name,
        "subfolder": folder,
        "type": "output",
        "v": int(os.path.getmtime(preview_path) * 1000),
    })
    return {
        "url": "/view?" + query,
        "width": int(image.width),
        "height": int(image.height),
        "canvas": [int(width), int(height)],
        "scale": float(plan.get("scale") or 1.0),
        "layers": len(psd_layers),
        "selected": int(plan.get("total") or 0),
        "notes": notes[:12],
        "warnings": (plan.get("warnings") or [])[:12],
    }


async def preview_plan(request: web.Request) -> web.Response:
    query = request.query

    def _int(name: str, default: int) -> int:
        try:
            return int(query.get(name, default))
        except (TypeError, ValueError):
            return default

    args = {
        "source": query.get("dir", ""),
        "canvas": _int("canvas", 2048),
        "rules": query.get("rules", ""),
        "select": query.get("select", "all"),
        "exclude": query.get("exclude", ""),
        "out_dir": query.get("out_dir", ""),
        "skip_empty": str(query.get("skip_empty", "1")).lower() not in ("0", "false", "no"),
        "coverage": 0.0005,
        "max_side": max(128, min(2048, _int("size", 1024))),
    }
    try:
        args["coverage"] = float(query.get("coverage", "0.0005"))
    except ValueError:
        args["coverage"] = 0.0005
    try:
        args["canvas"] = core_plan.clamp_side(args["canvas"])
    except (TypeError, ValueError):
        args["canvas"] = 2048

    try:
        data = await asyncio.get_running_loop().run_in_executor(None, _build_preview, args)
    except Exception as exc:  # noqa: BLE001
        return _json_error(str(exc), 400)
    return web.json_response(data)


# ---------------------------------------------------------------------------
# open / reveal
# ---------------------------------------------------------------------------

def _open_target(target: str) -> None:
    if os.name == "nt":
        os.startfile(target)  # noqa: S606 - local desktop convenience
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


async def _open_route(request: web.Request, allow_local: bool,
                      as_folder: bool) -> web.Response:
    raw = request.query.get("path", "")
    try:
        path = _resolve_local(raw) if allow_local else _resolve_any(raw)
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    if not os.path.exists(path):
        return _json_error("not found: %s" % raw, 404)
    target = path
    if as_folder and not os.path.isdir(path):
        target = os.path.dirname(path)
    try:
        _open_target(target)
    except Exception as exc:  # noqa: BLE001
        return _json_error(str(exc), 500)
    return web.json_response({"ok": True, "path": target})


async def reveal_path(request: web.Request) -> web.Response:
    """Open the containing folder (or the folder itself) in the OS file manager."""
    return await _open_route(request, allow_local=False, as_folder=True)


async def open_path(request: web.Request) -> web.Response:
    """Open a file (or folder) with its default application."""
    return await _open_route(request, allow_local=False, as_folder=False)


async def reveal_local(request: web.Request) -> web.Response:
    """Like ``reveal_path`` but for any absolute path (PSD source folders)."""
    return await _open_route(request, allow_local=True, as_folder=True)


async def open_local(request: web.Request) -> web.Response:
    """Like ``open_path`` but for any absolute path (a PSD outside output/)."""
    return await _open_route(request, allow_local=True, as_folder=False)


async def edit_parts_catalogue(request: web.Request) -> web.Response:
    """Part table for the edit-parts selector panel."""
    return web.json_response({
        "groups": [{"id": gid, "label": label} for gid, label in edit_parts.GROUPS],
        "parts": edit_parts.catalogue(),
        "default": "all",
    })


# ---------------------------------------------------------------------------
# /stlu/tiles -- grid-split tile PSDs (list / inspect / merge)
# ---------------------------------------------------------------------------

TILE_MANIFEST = "manifest.json"
TILE_SCAN_DEPTH = 4


def _read_tile_manifest(directory: str) -> Dict[str, Any]:
    path = os.path.join(directory, TILE_MANIFEST)
    if not os.path.isfile(path):
        raise FileNotFoundError("manifest.json not found in %s" % directory)
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc.get("tiles"), list):
        raise ValueError("manifest has no tiles")
    return doc


def _resolve_tile_dir(raw: str) -> str:
    """Tile endpoints accept output-relative *and* absolute run folders."""
    return _resolve_any(raw)


def _tile_run_info(manifest: Dict[str, Any], run_dir: str, output_root: str) -> Dict[str, Any]:
    """The payload both the node and the panels render."""
    first = manifest["tiles"][0]
    layers = first.get("layers")
    layer_count = len(layers) if isinstance(layers, list) else int(layers or 0)
    total = 0
    for entry in manifest["tiles"]:
        path = os.path.join(run_dir, entry.get("file") or "")
        if os.path.isfile(path):
            total += os.path.getsize(path)
    rel = os.path.relpath(run_dir, output_root).replace(os.sep, "/")
    info: Dict[str, Any] = {
        "dir": rel,
        "subfolder": rel,
        "filename": first.get("file"),
        "count": manifest.get("count") or len(manifest["tiles"]),
        "layers": layer_count,
        "canvas": first.get("canvas"),
        "size": total,
        "merged": bool(manifest.get("merge_parts")),
        "merged_psd": None,
        "merge_error": manifest.get("merge_error"),
        "preview": None,
        "error": None,
        "source_kind": manifest.get("source_kind"),
        "grid": manifest.get("grid"),
    }
    merged = manifest.get("merged") or {}
    if merged.get("file"):
        merged_path = os.path.join(run_dir, merged["file"])
        if os.path.isfile(merged_path):
            merged_layers = merged.get("layers")
            info["merged_psd"] = {
                "filename": merged["file"],
                "subfolder": rel,
                "canvas": merged.get("canvas"),
                # Older manifests store the merged layer count, newer ones the
                # list of layer names; both have to read back.
                "layers": (len(merged_layers) if isinstance(merged_layers, list)
                           else int(merged_layers or 0)),
                "groups": merged.get("groups") or [],
                "size": os.path.getsize(merged_path),
            }
        if merged.get("preview"):
            preview_path = os.path.join(run_dir, merged["preview"])
            if os.path.isfile(preview_path):
                info["preview"] = {
                    "filename": merged["preview"],
                    "subfolder": rel,
                    "type": "output",
                    "v": int(os.path.getmtime(preview_path) * 1000),
                }
    if merged.get("depth"):
        depth_path = os.path.join(run_dir, merged["depth"])
        if os.path.isfile(depth_path):
            info["depth"] = {"filename": merged["depth"], "subfolder": rel,
                             "v": int(os.path.getmtime(depth_path) * 1000)}
    if merged.get("depth_psd"):
        depth_psd_path = os.path.join(run_dir, merged["depth_psd"])
        if os.path.isfile(depth_psd_path):
            info["depth_psd"] = {"filename": merged["depth_psd"], "subfolder": rel,
                                 "v": int(os.path.getmtime(depth_psd_path) * 1000)}
    if info["preview"] is None:
        # Newer waves store no aggregate preview: ``preview`` names the richest
        # tile's own preview (a separate file was byte-identical to it).
        preview_name = str(manifest.get("preview") or "")
        preview = os.path.join(run_dir, preview_name) if preview_name else ""
        if preview and os.path.isfile(preview):
            info["preview"] = {"filename": preview_name, "subfolder": rel,
                               "type": "output",
                               "v": int(os.path.getmtime(preview) * 1000)}
    return info


async def tiles_grids(request: web.Request) -> web.Response:
    """Every persisted grid document under output/ (stage one's product)."""
    root = _scope_root("output")
    skip = {THUMB_DIR, PREVIEW_DIR, "__pycache__"}
    found: List[Dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in skip]
        if depth >= TILE_SCAN_DEPTH:
            dirnames[:] = []
        for name in filenames:
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(doc, dict) or doc.get("kind") != "stlu_tile_grid":
                continue
            try:
                mtime = int(os.path.getmtime(path))
            except OSError:
                mtime = 0
            src = doc.get("source") or {}
            found.append({
                "path": _relative("output", path) or path,
                "name": name,
                "dir": _relative("output", dirpath) or "",
                "count": int(doc.get("count") or 0),
                "rows": int(doc.get("rows") or 0),
                "columns": int(doc.get("columns") or 0),
                "tile_size": doc.get("tile_size"),
                "overlap": doc.get("overlap"),
                "source": [src.get("width"), src.get("height")],
                "selection": doc.get("selection") or "all",
                "mtime": mtime,
            })
    found.sort(key=lambda item: item["mtime"], reverse=True)
    return web.json_response({"grids": found})


async def tiles_grid(request: web.Request) -> web.Response:
    try:
        path = _resolve_any(request.query.get("path", ""))
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    try:
        doc = tile_psd.load_grid_doc(path)
    except FileNotFoundError:
        return _json_error("grid document not found", 404)
    except Exception as exc:  # noqa: BLE001
        return _json_error(str(exc), 400)
    doc["path"] = _relative("output", path) or path
    return web.json_response(doc)


def _card_thumb(dirpath: str, filenames: List[str]) -> str:
    """A representative picture for a wave card, as an output-relative path.

    A rendered preview beats the plain reference: the wave folder keeps the
    merge preview (``merged_preview.png``) or its per-tile previews next to the
    ``images/`` directory, so two waves of the same character stay tellable
    apart.  Newer waves write no aggregate preview at all.
    """
    candidates = [os.path.join(dirpath, "merged_preview.png"),
                  os.path.join(dirpath, "_preview.png")]
    parent = os.path.dirname(dirpath)
    if parent and parent != dirpath:
        candidates += [os.path.join(parent, "merged_preview.png"),
                       os.path.join(parent, "_preview.png")]
        try:
            candidates += [os.path.join(parent, name) for name in sorted(os.listdir(parent))
                           if name.lower().endswith("_preview.png")]
        except OSError:
            pass
    candidates.append(os.path.join(dirpath, "original.png"))
    if parent and parent != dirpath:
        candidates.append(os.path.join(parent, "original.png"))
    for path in candidates:
        if os.path.isfile(path):
            return _relative("output", path) or ""
    # First PNG of the first tile folder (wave layout: <wave>/images/tile_xx/)
    try:
        for sub in sorted(os.listdir(dirpath)):
            subdir = os.path.join(dirpath, sub)
            if not os.path.isdir(subdir):
                continue
            for name in sorted(os.listdir(subdir)):
                if name.lower().endswith(IMAGE_EXTS):
                    return _relative("output", os.path.join(subdir, name)) or ""
    except OSError:
        pass
    return ""


def _tile_cards(dirpath: str, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-tile entries (with thumbnails) of an images.json run."""
    cards: List[Dict[str, Any]] = []
    for entry in doc.get("tiles") or []:
        folder = str(entry.get("dir") or "")
        tile_dir = os.path.join(dirpath, folder)
        thumb = ""
        try:
            for name in sorted(os.listdir(tile_dir)):
                if name.lower().endswith(IMAGE_EXTS) and not name.lower().endswith("_depth.png"):
                    thumb = _relative("output", os.path.join(tile_dir, name)) or ""
                    break
        except OSError:
            pass
        cards.append({
            "index": int(entry.get("index", len(cards))),
            "dir": folder,
            "layers": int(entry.get("layers") or 0),
            "thumb": thumb,
        })
    return cards


def _images_entry(dirpath: str, filenames: List[str], doc: Dict[str, Any],
                  mtime: int) -> Dict[str, Any]:
    # Wave layout puts the images in <wave>/images: the card is named after
    # the wave folder, not after "images".
    name = os.path.basename(dirpath)
    group = _relative("output", os.path.dirname(dirpath)) or ""
    wave = _relative("output", dirpath) or ""
    if name.lower() == "images" and os.path.dirname(dirpath) != dirpath:
        name = os.path.basename(os.path.dirname(dirpath))
        wave = _relative("output", os.path.dirname(dirpath)) or ""
        group = _relative("output", os.path.dirname(os.path.dirname(dirpath))) or ""
    return {
        "dir": _relative("output", dirpath) or "",
        "wave": wave,
        "name": name,
        "group": group,
        "kind": "images",
        "count": int(doc.get("count") or len(doc.get("tiles") or [])),
        "source": doc.get("source"),
        "tile_size": doc.get("tile_size"),
        "thumb": _card_thumb(dirpath, filenames),
        "tiles": _tile_cards(dirpath, doc),
        "mtime": mtime,
    }


async def tiles_images(request: web.Request) -> web.Response:
    """Wave folders that STLU_TilesPSD can rebuild tile PSDs from (stage three).

    ``?dir=`` returns just that folder's entry, so a panel can restore the
    summary (name, tile count, per-tile thumbnails) from a saved source.
    """
    root = _scope_root("output")
    wanted = (request.query.get("dir") or "").strip()
    if wanted:
        try:
            path = _resolve_any(wanted)
        except web.HTTPException as exc:
            return _json_error(exc.text or "invalid path", exc.status)
        if os.path.isdir(path) and not os.path.isfile(os.path.join(path, "images.json")):
            nested = os.path.join(path, "images")
            if os.path.isfile(os.path.join(nested, "images.json")):
                path = nested
        if not os.path.isdir(path):
            return _json_error("not a folder: %s" % wanted, 404)
        filenames = os.listdir(path)
        lower = {name.lower(): name for name in filenames}
        if "images.json" in lower:
            index_path = os.path.join(path, lower["images.json"])
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                mtime = int(os.path.getmtime(index_path))
            except Exception as exc:  # noqa: BLE001
                return _json_error("cannot read images.json: %s" % exc, 400)
            return web.json_response({"source": _images_entry(path, filenames, doc, mtime)})
        layers_files = [name for name in filenames if name.lower().endswith("_layers.json")]
        if layers_files:
            try:
                mtime = max(int(os.path.getmtime(os.path.join(path, n))) for n in layers_files)
            except OSError:
                mtime = 0
            rel_dir = _relative("output", path) or ""
            return web.json_response({"source": {
                "dir": rel_dir, "wave": rel_dir, "name": os.path.basename(path) or rel_dir,
                "group": _relative("output", os.path.dirname(path)) or "",
                "kind": "layers", "count": len(layers_files), "source": None,
                "tile_size": None, "thumb": _card_thumb(path, filenames),
                "tiles": [], "mtime": mtime,
            }})
        return _json_error("no images.json in %s" % wanted, 404)

    skip = {THUMB_DIR, PREVIEW_DIR, "__pycache__"}
    found: List[Dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in skip]
        if depth >= TILE_SCAN_DEPTH:
            dirnames[:] = []
        lower = {name.lower(): name for name in filenames}
        if "images.json" in lower:
            path = os.path.join(dirpath, lower["images.json"])
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                mtime = int(os.path.getmtime(path))
            except Exception:  # noqa: BLE001
                continue
            found.append(_images_entry(dirpath, filenames, doc, mtime))
            continue
        layers_files = [name for name in filenames if name.lower().endswith("_layers.json")]
        if layers_files:
            try:
                mtime = max(int(os.path.getmtime(os.path.join(dirpath, n))) for n in layers_files)
            except OSError:
                mtime = 0
            rel_dir = _relative("output", dirpath) or ""
            found.append({
                "dir": rel_dir,
                "wave": rel_dir,
                "name": os.path.basename(dirpath) or rel_dir,
                "group": _relative("output", os.path.dirname(dirpath)) or "",
                "kind": "layers",
                "count": len(layers_files),
                "source": None,
                "tile_size": None,
                "thumb": _card_thumb(dirpath, filenames),
                "tiles": [],
                "mtime": mtime,
            })
    found.sort(key=lambda item: item["mtime"], reverse=True)
    return web.json_response({"sources": found[:80]})


async def tiles_from_images(request: web.Request) -> web.Response:
    """Stage three from the browser: existing images -> tile PSDs (+ merge)."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _json_error("invalid JSON body", 400)
    if not isinstance(body, dict):
        return _json_error("invalid JSON body", 400)

    raw = str(body.get("source") or "").strip()
    if not raw:
        return _json_error("source is required", 400)
    try:
        source_path = _resolve_any(raw)
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    if not os.path.exists(source_path):
        return _json_error("source not found: %s" % raw, 404)

    grid_doc = None
    grid_ref = str(body.get("grid") or "").strip()
    if grid_ref:
        try:
            # Inline JSON from stage one, or a path to an older grids/*.json.
            grid_doc = (tile_psd.load_grid_ref(grid_ref) if grid_ref.startswith("{")
                        else tile_psd.load_grid_doc(_resolve_any(grid_ref)))
        except Exception as exc:  # noqa: BLE001
            return _json_error("cannot read grid: %s" % exc, 400)

    out_dir = str(body.get("out_dir") or "SeeThrough").strip() or "SeeThrough"
    group = str(body.get("group") or "").strip()
    name = str(body.get("name") or "tile").strip() or "tile"
    try:
        alpha = int(body.get("alpha_threshold", 8))
    except (TypeError, ValueError):
        alpha = 8

    raw_tiles = body.get("tiles")
    tiles: Optional[List[int]] = None
    if raw_tiles not in (None, "", []):
        try:
            tiles = [int(t) for t in raw_tiles]
        except (TypeError, ValueError):
            return _json_error("tiles must be integers", 400)

    import datetime as _dt
    stamp = "%s_%s" % (_dt.datetime.now().strftime("%Y%m%d_%H%M%S"), os.urandom(3).hex())
    # The user picked the folder; write there directly instead of a wave
    # subfolder so other tiles in it stay reusable.
    wave_dir = tile_psd.flat_directory(_scope_root("output"), out_dir, group)
    wave = os.path.basename(wave_dir.rstrip("/\\")) or "run"

    def _run():
        manifest = tile_psd.write_tiles_from_images(
            source_path,
            _scope_root("output"),
            out_dir=out_dir,
            filename_prefix=name,
            merge_parts=bool(body.get("merge_parts", True)),
            alpha_threshold=alpha,
            stamp=stamp,
            # Same fixed policy as the node: depth and the split sub layers
            # are always written, and the merge belongs to step four.
            merge_after=False,
            merged_name=str(body.get("merged_name") or "merged"),
            grid_doc=grid_doc,
            tile_index_map=None,
            write_depth=True,
            merge_group_layers=True,
            include_original=True,
            run_dir_abs=wave_dir,
            tiles=tiles,
        )
        return manifest, wave_dir

    try:
        manifest, run_dir = await asyncio.get_running_loop().run_in_executor(None, _run)
    except Exception as exc:  # noqa: BLE001
        return _json_error("tile PSD build failed: %s" % exc, 500)

    info = _tile_run_info(manifest, run_dir, _scope_root("output"))
    info["group"] = group
    info["wave"] = wave
    return web.json_response({"ok": True, **info})


async def tiles_dirs(request: web.Request) -> web.Response:
    """Every run folder under output/ that carries a tile manifest."""
    root = _scope_root("output")
    found: List[Dict[str, Any]] = []
    skip = {THUMB_DIR, PREVIEW_DIR, "_stlu_preview", "__pycache__"}

    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in skip
        ]
        if depth >= TILE_SCAN_DEPTH:
            dirnames[:] = []
        if TILE_MANIFEST not in filenames:
            continue
        try:
            doc = _read_tile_manifest(dirpath)
            stat = os.stat(os.path.join(dirpath, TILE_MANIFEST))
        except Exception:  # noqa: BLE001
            continue
        merged = doc.get("merged") or {}
        found.append({
            "dir": _relative("output", dirpath),
            "name": os.path.basename(dirpath),
            "count": int(doc.get("count") or len(doc.get("tiles") or [])),
            "source": doc.get("source"),
            "merged": merged.get("file"),
            "merge_parts": bool(doc.get("merge_parts")),
            "mtime": int(stat.st_mtime),
        })
        dirnames[:] = []  # a run folder never nests another one

    found.sort(key=lambda item: item["mtime"], reverse=True)
    return web.json_response({"dirs": found})


async def tiles_manifest(request: web.Request) -> web.Response:
    rel = (request.query.get("dir") or "").strip()
    try:
        path = _resolve_tile_dir(rel)
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    try:
        doc = _read_tile_manifest(path)
    except FileNotFoundError as exc:
        return _json_error(str(exc), 404)
    except Exception as exc:  # noqa: BLE001
        return _json_error(str(exc), 400)
    doc["dir"] = _relative("output", path)
    return web.json_response(doc)


async def tiles_clear(request: web.Request) -> web.Response:
    """Delete what this pipeline wrote into one run folder (panel button).

    Only managed outputs are removed -- tile PSDs, previews, depth, the wave
    documents and ``images/``.  Files the user put there are left alone.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _json_error("invalid JSON body", 400)
    if not isinstance(body, dict):
        return _json_error("invalid JSON body", 400)

    rel = str(body.get("dir") or "").strip()
    try:
        path = _resolve_tile_dir(rel)
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    if not os.path.isdir(path):
        return _json_error("directory not found: %s" % rel, 404)
    try:
        result = tile_psd.clear_run_outputs(path)
    except tile_psd.ClearFailed as exc:
        return _json_error(str(exc), 409)
    except Exception as exc:  # noqa: BLE001
        return _json_error("clear failed: %s" % exc, 500)
    return web.json_response({
        "ok": True,
        "dir": _relative("output", path) or rel,
        "removed": int(result.get("removed") or 0),
    })


def _tile_view_url(rel_dir: str, filename: str) -> Dict[str, Any]:
    folder, name = (rel_dir, filename) if "/" not in filename else filename.rsplit("/", 1)
    return {
        "filename": name,
        "subfolder": folder,
        "type": "output",
    }


async def tiles_merge(request: web.Request) -> web.Response:
    """Merge a subset of an existing run's tile PSDs -- no queue, no GPU."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _json_error("invalid JSON body", 400)
    if not isinstance(body, dict):
        return _json_error("invalid JSON body", 400)

    rel = str(body.get("dir") or "").strip()
    try:
        path = _resolve_tile_dir(rel)
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    try:
        doc = _read_tile_manifest(path)
    except FileNotFoundError as exc:
        return _json_error(str(exc), 404)
    except Exception as exc:  # noqa: BLE001
        return _json_error(str(exc), 400)

    raw = body.get("indices")
    indices: Optional[List[int]] = None
    if raw not in (None, "", []):
        try:
            indices = [int(i) for i in raw]
        except (TypeError, ValueError):
            return _json_error("indices must be integers", 400)
        known = {int(t.get("index")) for t in doc["tiles"]}
        missing = [i for i in indices if i not in known]
        if missing:
            return _json_error("unknown tile indices: %s" % missing, 400)

    name = str(body.get("name") or "merged").strip() or "merged"
    name = os.path.basename(name.replace("\\", "/"))

    loop = asyncio.get_running_loop()
    try:
        info = await loop.run_in_executor(None, lambda: tile_psd.merge_run(
            path,
            manifest=doc,
            indices=indices,
            name=name,
            merge_parts=body.get("merge_parts"),
            alpha_threshold=body.get("alpha_threshold"),
            group_layers=body.get("group_layers"),
            # Optional export, but always as its own PSD.
            include_depth=bool(body.get("include_depth", False)),
            include_original=bool(body.get("include_original", False)),
            seam_feather=body.get("seam_feather"),
        ))
    except Exception as exc:  # noqa: BLE001
        return _json_error("merge failed: %s" % exc, 500)

    info["dir"] = _relative("output", path)
    if info.get("preview"):
        info["preview"] = _tile_view_url(info["dir"], info["preview"])
    if info.get("depth"):
        info["depth"] = _tile_view_url(info["dir"], info["depth"])
    if info.get("depth_psd"):
        info["depth_psd"] = _tile_view_url(info["dir"], info["depth_psd"])
    if info.get("original"):
        info["original"] = _tile_view_url(info["dir"], info["original"])
    info["psd"] = _tile_view_url(info["dir"], info["file"])

    return web.json_response({"ok": True, **info})


async def tiles_merge_mixed(request: web.Request) -> web.Response:
    """Merge tiles picked from different runs (one source per tile).

    This is the "body from wave 1, head from wave 2" case: every entry names a
    run folder and a tile index; when the same part exists in several runs the
    higher priority wins, so a redrawn face is never blended with the old one.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _json_error("invalid JSON body", 400)
    if not isinstance(body, dict):
        return _json_error("invalid JSON body", 400)
    raw_entries = body.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        return _json_error("entries is required", 400)

    entries: List[Dict[str, Any]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            return _json_error("entries must be objects", 400)
        try:
            path = _resolve_tile_dir(str(item.get("dir") or ""))
        except web.HTTPException as exc:
            return _json_error(exc.text or "invalid path", exc.status)
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            return _json_error("index must be an integer", 400)
        entries.append({
            "dir": path,
            "index": index,
            "priority": int(item.get("priority", 0) or 0),
        })

    name = str(body.get("name") or "merged_mixed").strip() or "merged_mixed"
    name = os.path.basename(name.replace("\\", "/"))
    if not name.lower().endswith(".psd"):
        name += ".psd"

    out_dir = entries[0]["dir"]
    loop = asyncio.get_running_loop()
    try:
        info = await loop.run_in_executor(None, lambda: tile_psd.merge_mixed(
            entries,
            _scope_root("output"),
            out_name=name,
            merge_parts=bool(body.get("merge_parts", True)),
            alpha_threshold=int(body.get(
                "alpha_threshold", tile_psd.DEFAULT_ALPHA_THRESHOLD)),
            group_layers=bool(body.get("group_layers", True)),
            include_depth=bool(body.get("include_depth", False)),
            include_original=bool(body.get("include_original", False)),
            seam_feather=int(body.get("seam_feather", tile_psd.DEFAULT_SEAM_FEATHER)),
        ))
    except Exception as exc:  # noqa: BLE001
        return _json_error("merge failed: %s" % exc, 500)

    info["dir"] = _relative("output", out_dir)
    for key in ("preview", "depth", "depth_psd", "original"):
        if info.get(key):
            info[key] = _tile_view_url(info["dir"], info[key])
    info["psd"] = _tile_view_url(info["dir"], info["file"])
    return web.json_response({"ok": True, **info})


def _source_for_plan(path: str, scope: str = "") -> Tuple[str, str]:
    """Resolve a source image for the CPU strip planner.

    The panel usually knows the LoadImage path but not its scope.  Try the
    declared scope first, then input/output; never leave a sandbox root.
    """
    raw = str(path or "").strip()
    if not raw:
        raise web.HTTPBadRequest(text="empty path")
    scopes = [scope] if scope in ("input", "output") else ["input", "output"]
    for candidate in scopes:
        try:
            full = _resolve(candidate, raw)
        except web.HTTPException:
            continue
        if os.path.isfile(full):
            return candidate, full
    raise web.HTTPNotFound(text="not found: %s" % raw)


def _query_int(request: web.Request, name: str, default: int) -> int:
    try:
        return int(request.query.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _file_identity(full: str) -> Tuple[Any, ...]:
    stat = os.stat(full)
    return (os.path.realpath(full), int(stat.st_mtime_ns), int(stat.st_size))


def _subject_for_plan(full: str) -> Dict[str, Any]:
    """Detect (and cache) the subject independently of plan parameters."""
    key = _file_identity(full)
    with _SUBJECT_CACHE_LOCK:
        entry = _SUBJECT_CACHE.get(key)
    if entry is not None:
        return entry
    rgb = tile_strips.resolve_source_rgb(full)
    height, width = rgb.shape[:2]
    subject = tile_strips.detect_subject(rgb)
    # The foreground ndarray is full-size and of no use to the planner; keep
    # only the JSON-safe metadata so the cache stays small.
    entry = {
        "width": int(width),
        "height": int(height),
        "subject": {k: v for k, v in subject.items()
                    if k not in ("foreground", "components")},
    }
    with _SUBJECT_CACHE_LOCK:
        if len(_SUBJECT_CACHE) >= _SUBJECT_CACHE_MAX:
            _SUBJECT_CACHE.pop(next(iter(_SUBJECT_CACHE)))
        _SUBJECT_CACHE[key] = entry
    return entry


def _strip_plan_payload(full: str, overlap: int, max_side: int,
                        resolution: int, margin: int) -> Dict[str, Any]:
    """CPU-only plan work (runs in a worker thread)."""
    detected = _subject_for_plan(full)
    plan = tile_strips.plan_from_subject(
        detected["width"], detected["height"], detected["subject"],
        overlap=int(overlap), max_side=int(max_side),
        margin=int(margin), resolution=int(resolution))
    return tile_strips.public_plan(
        plan, detected["width"], detected["height"])


async def tiles_strips(request: web.Request) -> web.Response:
    """CPU-only preview of the subject-strip plan for the ① panel.

    This is the same planner the input node runs at execution time.  It reads
    the source file, detects the subject and returns square region boxes; it
    never starts a generation and writes no files.
    """
    path = (request.query.get("path") or "").strip()
    scope = (request.query.get("scope") or "").strip()
    try:
        scope, full = _source_for_plan(path, scope)
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    overlap = _query_int(request, "overlap", 256)
    max_side = _query_int(request, "max_side", 2048)
    resolution = _query_int(request, "resolution", 512)
    margin = _query_int(request, "margin", 32)
    try:
        key = _file_identity(full) + (
            overlap, max_side, resolution, margin)
        payload = _STRIP_PLAN_CACHE.get(key)
        if payload is None:
            loop = asyncio.get_running_loop()
            payload = await loop.run_in_executor(
                None, _strip_plan_payload, full, overlap, max_side,
                resolution, margin)
            if len(_STRIP_PLAN_CACHE) >= _STRIP_PLAN_CACHE_MAX:
                _STRIP_PLAN_CACHE.pop(next(iter(_STRIP_PLAN_CACHE)))
            _STRIP_PLAN_CACHE[key] = payload
        # The cached value is shared; never let one response's path/scope
        # leak into another caller's payload.
        payload = dict(payload)
        payload["path"] = path
        payload["scope"] = scope
        return web.json_response(payload)
    except Exception as exc:  # noqa: BLE001
        return _json_error("strip plan failed: %s" % exc, 500)


async def image_size(request: web.Request) -> web.Response:
    """Report an image's pixel size without transferring it.

    The Plan panel uses this to say whether a reference image is cropped at its
    own resolution or flattened back to the source canvas.
    """
    scope = (request.query.get("scope") or "input").strip()
    path = (request.query.get("path") or "").strip()
    try:
        full = _resolve(scope, path)
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    if not os.path.isfile(full):
        return _json_error("not found: %s" % path, 404)
    try:
        with Image.open(full) as im:
            width, height = im.size
    except Exception as exc:  # noqa: BLE001
        return _json_error("cannot read image: %s" % exc, 400)
    return web.json_response({
        "path": path, "scope": scope, "width": int(width), "height": int(height),
    })


# ---------------------------------------------------------------------------
# /stlu/psd_preview -- a merged PSD grouped by tile or by body part
# ---------------------------------------------------------------------------
#
# The ⑤ preview node can show the PSD two ways.  "tiles" is the tile wall the
# run was built from (cheap: each tile already wrote a white-backed
# ``_preview.png``); "parts" stacks every same-named copy back together so the
# finished body parts can be checked one by one.  Both answer from disk, so
# reopening the workflow shows the last merge without re-running anything.

_PSD_PREVIEW_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_PSD_PREVIEW_CACHE_MAX = 8
_TILE_SUFFIX_RE = re.compile(r"^(.*)_(\d+)$")
# Split thumbnails are contact-sheet sized; more than this per group is not
# previewable by eye and only slows the first panel load down.
_PSD_SPLIT_LIMIT = 320
_PSD_SPLIT_MAX = 80


def _file_stamp(path: str) -> str:
    try:
        stat = os.stat(path)
        return "%d|%d" % (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return ""


def _image_ref(path: str) -> Dict[str, Any]:
    """A /view reference for a PNG in output/ or in ComfyUI's temp dir."""
    try:
        version = int(os.path.getmtime(path) * 1000)
    except OSError:
        version = 0
    rel = _relative("output", path)
    if rel is not None:
        folder, name = _split_rel(rel)
        return {"filename": name, "subfolder": folder, "type": "output", "v": version}
    temp_root = os.path.realpath(folder_paths.get_temp_directory())
    real = os.path.realpath(path)
    if _inside(temp_root, real):
        folder, name = _split_rel(os.path.relpath(real, temp_root).replace(os.sep, "/"))
        return {"filename": name, "subfolder": folder, "type": "temp", "v": version}
    return {"filename": os.path.basename(path), "subfolder": "", "type": "output",
            "v": version}


def _write_group_png(rgba: np.ndarray, stem: str, version: str, limit: int
                     ) -> Tuple[Dict[str, Any], List[int]]:
    """Flatten straight-alpha RGBA onto white and cache it in ComfyUI's temp."""
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = np.clip(np.rint(rgba[..., :3].astype(np.float32) * alpha
                          + 255.0 * (1.0 - alpha)), 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb, "RGB")
    if max(image.width, image.height) > limit:
        scale = limit / float(max(image.width, image.height))
        image = image.resize(
            (max(1, int(round(image.width * scale))),
             max(1, int(round(image.height * scale)))),
            Image.Resampling.LANCZOS)
    temp = folder_paths.get_temp_directory()
    os.makedirs(temp, exist_ok=True)
    digest = hashlib.sha1(("%s|%s" % (stem, version)).encode("utf-8")).hexdigest()[:8]
    filename = "stlu_psdgroup_%s_%s.png" % (tile_psd._safe_stem(stem) or "group",
                                            digest)
    image.save(os.path.join(temp, filename), compress_level=3)
    ref = {"filename": filename, "subfolder": "", "type": "temp",
           "v": int(os.path.getmtime(os.path.join(temp, filename)) * 1000)}
    return ref, [int(image.width), int(image.height)]


def _split_thumb(layer: Dict[str, Any], stem: str, version: str, limit: int
                 ) -> Optional[Dict[str, Any]]:
    """One white-backed thumbnail for a single PSD layer, or None if empty."""
    px = np.asarray(layer.get("pixels"))
    if px.ndim != 3 or px.shape[-1] != 4 or not px[..., 3].any():
        return None
    # Crop to real content: a layer saved before the dust fix is a full tile
    # with a handful of rim speckles, and a plain >0 bbox would keep it huge.
    mask = tile_psd.content_mask(px[..., 3])
    if not mask.any():
        return None
    if not mask.all():
        rows = np.nonzero(mask.any(axis=1))[0]
        cols = np.nonzero(mask.any(axis=0))[0]
        px = px[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    image, size = _write_group_png(px, stem, version, limit)
    return {"image": image, "size": size}


def _tile_split(run_dir: str, entry: Dict[str, Any], limit: int
                ) -> Tuple[List[Dict[str, Any]], int]:
    """The individual parts a tile's own PSD holds, as thumbnails."""
    index = int(entry.get("index") or 0)
    name = str(entry.get("file") or "")
    path = os.path.join(run_dir, name)
    if not (name and os.path.isfile(path)):
        return [], 0
    try:
        data = psd_reader.read_psd(path)
    except Exception as exc:  # noqa: BLE001 - the tile preview still shows
        print("[STLU] PSD preview: cannot read %s: %s" % (path, exc), flush=True)
        return [], 0
    layers = data.get("layers") or []
    suppressed = psd_writer.hidden_record_mask(layers)
    version = _file_stamp(path)
    out: List[Dict[str, Any]] = []
    dropped = 0
    for pos, layer in enumerate(layers):
        if str(layer.get("kind") or "layer") != "layer":
            continue
        if layer.get("hidden") or suppressed[pos]:
            continue
        label = str(layer.get("name") or "")
        if not label or label.lower() == "original":
            continue
        if (int(layer.get("right", 0)) <= int(layer.get("left", 0))
                or int(layer.get("bottom", 0)) <= int(layer.get("top", 0))):
            continue
        if len(out) >= _PSD_SPLIT_MAX:
            dropped += 1
            continue
        thumb = _split_thumb(layer, "tile_%d_%d_%s" % (index, pos, label),
                             version, limit)
        if thumb is None:
            continue
        out.append(dict(thumb, name=label, tile=index))
    return out, dropped


def _layer_tile(layer: Dict[str, Any], raw_name: str, tile_indices: set,
                tile_boxes: Dict[int, Tuple[int, int, int, int]]) -> Optional[int]:
    """Which tile a merged layer's copy came from, when it can be known.

    Folder mode suffixes copies ``part_3``; a copy only one tile drew keeps
    the plain name, so fall back to the tile box that contains it.  A flat
    merge blends copies together and may span several tiles -> None.
    """
    match = _TILE_SUFFIX_RE.match(raw_name)
    if match:
        try:
            suffix = int(match.group(2))
        except ValueError:
            suffix = None
        if suffix in tile_indices:
            return suffix
    left = int(layer.get("left", 0))
    top = int(layer.get("top", 0))
    right = int(layer.get("right", 0))
    bottom = int(layer.get("bottom", 0))
    hits = [idx for idx, (x0, y0, x1, y1) in tile_boxes.items()
            if right > x0 and left < x1 and bottom > y0 and top < y1]
    return hits[0] if len(hits) == 1 else None


def _psd_target(path: str) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """(merged PSD, manifest, run folder) for a PSD file or a run folder."""
    full = _resolve_any(path)
    if os.path.isdir(full):
        run_dir = full
        manifest = tile_psd.read_manifest(run_dir)
        merged = str(((manifest or {}).get("merged") or {}).get("file") or "")
        candidate = os.path.join(run_dir, merged) if merged else ""
        if not (candidate and os.path.isfile(candidate)):
            names = sorted(
                name for name in os.listdir(run_dir)
                if name.lower().endswith(".psd")
                and not name.lower().endswith("_depth.psd"))
            if not names:
                raise FileNotFoundError("no PSD in %s" % path)
            candidate = os.path.join(run_dir, names[0])
        return candidate, manifest, run_dir
    if not os.path.isfile(full):
        raise FileNotFoundError("not found: %s" % path)
    run_dir = os.path.dirname(full)
    return full, tile_psd.read_manifest(run_dir), run_dir


def _psd_tile_groups(run_dir: str, manifest: Optional[Dict[str, Any]],
                     limit: int) -> List[Dict[str, Any]]:
    entries = (manifest or {}).get("tiles") or []
    if not entries:
        return []
    coords = tile_psd._coords_from_manifest(manifest)
    groups: List[Dict[str, Any]] = []
    for entry in sorted(entries, key=lambda e: int(e.get("index") or 0)):
        index = int(entry.get("index") or 0)
        image = None
        thumb = ""
        preview = str(entry.get("preview") or "")
        if preview:
            candidate = os.path.join(run_dir, preview)
            if os.path.isfile(candidate):
                image = _image_ref(candidate)  # written by the tile pass
                thumb = _thumb_url(candidate, 192)
        if image is None:
            tile_file = str(entry.get("file") or "")
            candidate = os.path.join(run_dir, tile_file)
            rgba = (tile_psd.composite_psd(candidate)
                    if tile_file and os.path.isfile(candidate) else None)
            if rgba is None:
                continue
            image, _size = _write_group_png(
                rgba, "tile_%d" % index, _file_stamp(candidate), limit)
        label = "块 %02d" % index
        try:
            row, col = tile_psd.tile_rc(coords, index)
            label += " · 行%d 列%d" % (int(row) + 1, int(col) + 1)
        except Exception:  # noqa: BLE001 - coordinates are a nicety
            pass
        split, dropped = _tile_split(
            run_dir, entry, min(limit, _PSD_SPLIT_LIMIT))
        if dropped:
            print("[STLU] PSD preview: tile %d has %d extra layers not shown"
                  % (index, dropped), flush=True)
        groups.append({
            "index": index,
            "label": label,
            "layers": len(entry.get("layers") or []),
            "box": entry.get("tile"),
            "image": image,
            "thumb": thumb,
            "split": split,
            # Two seeds this tile was made with, so a re-test can be aimed at
            # one tile without going back to the settings node.
            "seeds": entry.get("seeds") or None,
        })
    return groups


def _psd_part_groups(psd_path: str, manifest: Optional[Dict[str, Any]],
                     limit: int) -> List[Dict[str, Any]]:
    data = psd_reader.read_psd(psd_path)
    layers = data.get("layers") or []
    suppressed = psd_writer.hidden_record_mask(layers)
    tile_indices = set()
    tile_boxes: Dict[int, Tuple[int, int, int, int]] = {}
    for entry in (manifest or {}).get("tiles") or []:
        try:
            tile_indices.add(int(entry.get("index")))
        except (TypeError, ValueError):
            continue
        box = entry.get("tile")
        if box and len(box) >= 4:
            try:
                tile_boxes[int(entry["index"])] = tuple(int(v) for v in box[:4])
            except (TypeError, ValueError):
                pass
    version = _file_stamp(psd_path)
    split_limit = min(limit, _PSD_SPLIT_LIMIT)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for index, layer in enumerate(layers):
        if str(layer.get("kind") or "") != "layer":
            continue
        if layer.get("hidden") or suppressed[index]:
            continue
        if (int(layer.get("right", 0)) <= int(layer.get("left", 0))
                or int(layer.get("bottom", 0)) <= int(layer.get("top", 0))):
            continue
        name = str(layer.get("name") or "")
        if not name or name.lower() == "original":
            continue
        # Folder mode names copies ``hairf_3`` (part + tile).  Strip that
        # suffix only when it really is one of this run's tile numbers, so a
        # part literally called ``lp_2`` is not chopped.
        match = _TILE_SUFFIX_RE.match(name)
        base = (match.group(1)
                if match and int(match.group(2)) in tile_indices else name)
        key = tile_psd.merge_group_name(base)
        buckets.setdefault(key, []).append({
            "layer": layer,
            "model": name,
            "base": base,
            "tile": _layer_tile(layer, name, tile_indices, tile_boxes),
        })

    groups: List[Dict[str, Any]] = []
    threshold = int(tile_psd.DEFAULT_ALPHA_THRESHOLD)
    for key, members in buckets.items():
        if len(groups) >= 240:
            break
        left = min(int(m["layer"]["left"]) for m in members)
        top = min(int(m["layer"]["top"]) for m in members)
        right = max(int(m["layer"]["right"]) for m in members)
        bottom = max(int(m["layer"]["bottom"]) for m in members)
        if right <= left or bottom <= top:
            continue
        canvas = np.zeros((bottom - top, right - left, 4), np.uint8)
        for member in members:
            layer = member["layer"]
            core_compose._alpha_over(
                canvas, np.asarray(layer["pixels"]),
                int(layer["left"]) - left, int(layer["top"]) - top)
        alpha = canvas[..., 3]
        mask = tile_psd.content_mask(alpha, threshold)
        if not mask.any():
            continue
        rows = np.nonzero(mask.any(axis=1))[0]
        cols = np.nonzero(mask.any(axis=0))[0]
        crop = canvas[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        image, size = _write_group_png(crop, "part_" + key, version, limit)
        split: List[Dict[str, Any]] = []
        dropped = 0
        for pos, member in enumerate(members):
            if len(split) >= _PSD_SPLIT_MAX:
                dropped += 1
                continue
            thumb = _split_thumb(
                member["layer"],
                "part_%s_%d_%s" % (key, pos, member["model"]),
                version, split_limit)
            if thumb is None:
                continue
            split.append(dict(thumb, name=member["base"],
                              layer=member["model"], tile=member["tile"]))
        if dropped:
            print("[STLU] PSD preview: part %s has %d copies not shown"
                  % (key, dropped), flush=True)
        groups.append({
            "name": key,
            "label": key if len(members) == 1 else "%s ×%d" % (key, len(members)),
            "count": len(members),
            "size": size,
            "image": image,
            "split": split,
        })
    return groups


def _psd_preview_build(path: str, max_side: int, mode: str) -> Dict[str, Any]:
    psd_path, manifest, run_dir = _psd_target(path)
    limit = max(128, min(int(max_side or 768), 1024))
    version = _file_stamp(psd_path)

    canvas = None
    source = (manifest or {}).get("source") or []
    if len(source) >= 2:
        try:
            canvas = [int(source[0]), int(source[1])]
        except (TypeError, ValueError):
            canvas = None

    payload: Dict[str, Any] = {
        "file": os.path.basename(psd_path),
        "dir": _relative("output", os.path.dirname(psd_path)) or "",
        "mode": mode,
        "canvas": canvas,
        "group_layers": bool((manifest or {}).get("group_layers", True)),
        "main": None,
        "main_thumb": "",
        "tiles": [],
        "parts": [],
        "tiles_count": 0,
        "parts_count": 0,
    }

    # The merge writes a white-backed preview next to its PSD; reuse that file
    # instead of reading the PSD again just to show the whole picture.
    stem = os.path.splitext(os.path.basename(psd_path))[0]
    own_preview = os.path.join(run_dir, stem + "_preview.png")
    if os.path.isfile(own_preview):
        payload["main"] = _image_ref(own_preview)
        payload["main_thumb"] = _thumb_url(own_preview, 640)
    else:
        rgba = tile_psd.composite_psd(psd_path)
        if rgba is not None:
            ref, _size = _write_group_png(rgba, "main_" + stem, version, limit)
            payload["main"] = ref
            payload["canvas"] = payload["canvas"] or [
                int(rgba.shape[1]), int(rgba.shape[0])]

    if mode == "parts":
        payload["parts"] = _psd_part_groups(psd_path, manifest, limit)
        payload["parts_count"] = len(payload["parts"])
        # A part knows which tile it came from, not which seeds made it, so
        # hand the UI every tile's seeds once and let it look them up.
        payload["seeds"] = {
            str(int(entry.get("index"))): entry.get("seeds") or None
            for entry in (manifest or {}).get("tiles") or []
            if entry.get("seeds")
        }
    else:
        payload["tiles"] = _psd_tile_groups(run_dir, manifest, limit)
        payload["tiles_count"] = len(payload["tiles"])
    return payload


async def psd_preview(request: web.Request) -> web.Response:
    raw = (request.query.get("path") or "").strip()
    mode = (request.query.get("mode") or "tiles").strip().lower()
    if mode not in ("tiles", "parts"):
        mode = "tiles"
    try:
        max_side = int(request.query.get("max_side", 768))
    except (TypeError, ValueError):
        max_side = 768
    max_side = max(128, min(max_side, 1024))
    try:
        psd_path, _manifest, _run_dir = _psd_target(raw)
        stat = os.stat(psd_path)
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    except (OSError, FileNotFoundError) as exc:
        return _json_error(str(exc), 404)

    key = (psd_path, stat.st_mtime_ns, stat.st_size, max_side, mode)
    payload = _PSD_PREVIEW_CACHE.get(key)
    if payload is None:
        loop = asyncio.get_running_loop()
        try:
            payload = await loop.run_in_executor(
                None, _psd_preview_build, raw, max_side, mode)
        except web.HTTPException as exc:
            return _json_error(exc.text or "invalid path", exc.status)
        except (OSError, FileNotFoundError) as exc:
            return _json_error(str(exc), 404)
        except Exception as exc:  # noqa: BLE001
            return _json_error("PSD preview failed: %s" % exc, 500)
        if len(_PSD_PREVIEW_CACHE) >= _PSD_PREVIEW_CACHE_MAX:
            _PSD_PREVIEW_CACHE.pop(next(iter(_PSD_PREVIEW_CACHE)))
        _PSD_PREVIEW_CACHE[key] = payload
    payload = dict(payload)
    payload["path"] = raw
    return web.json_response(payload)


# ---------------------------------------------------------------------------
# /stlu/psd_layers -- the raw layer names of a finished PSD, for the edit table
# ---------------------------------------------------------------------------

def _psd_layers_build(path: str) -> Dict[str, Any]:
    """Layer names the edit planner can target.

    Same filter as ``_tile_split``: only real, visible, non-suppressed, named,
    non-empty layers -- exactly the set ``prepare_edit_plan`` can match, so
    the panel's dropdown never offers a name the backend would reject.
    """
    psd_path, _manifest, _run_dir = _psd_target(path)
    data = psd_reader.read_psd(psd_path)
    layers = data.get("layers") or []
    suppressed = psd_writer.hidden_record_mask(layers)
    rows: List[Dict[str, Any]] = []
    for pos, layer in enumerate(layers):
        if str(layer.get("kind") or "layer") != "layer":
            continue
        if layer.get("hidden") or suppressed[pos]:
            continue
        label = str(layer.get("name") or "")
        if not label or label.lower() == "original":
            continue
        left, top = int(layer.get("left", 0)), int(layer.get("top", 0))
        right, bottom = int(layer.get("right", 0)), int(layer.get("bottom", 0))
        if right <= left or bottom <= top:
            continue
        rows.append({
            "name": label,
            "bounds": [left, top, right, bottom],
            "opacity": int(layer.get("opacity", 255) or 255),
            "blend": str(layer.get("blend") or "norm"),
        })
    return {
        "file": os.path.basename(psd_path),
        "canvas": [int(data.get("width") or 0), int(data.get("height") or 0)],
        "layers": rows,
    }


async def psd_layers(request: web.Request) -> web.Response:
    raw = (request.query.get("path") or "").strip()
    if not raw:
        return _json_error("缺少 path", 400)
    loop = asyncio.get_running_loop()
    try:
        payload = await loop.run_in_executor(None, _psd_layers_build, raw)
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    except (OSError, FileNotFoundError) as exc:
        return _json_error(str(exc), 404)
    except Exception as exc:  # noqa: BLE001
        return _json_error("读取 PSD 图层失败: %s" % exc, 500)
    return web.json_response(payload)


# ---------------------------------------------------------------------------
# /stlu/preset -- persist every panel input as JSON inside the output folder
# ---------------------------------------------------------------------------

_PRESET_NAME = "stlu_inputs.json"


def _preset_read(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


async def preset_load(request: web.Request) -> web.Response:
    """Read ``<dir>/stlu_inputs.json``.  A missing file is ``found: false``,
    not an error: a fresh output folder simply has no saved inputs yet."""
    try:
        folder = _resolve("output", request.query.get("dir", ""))
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    path = os.path.join(folder, _PRESET_NAME)
    rel_file = _relative("output", path) or ""
    if not os.path.isfile(path):
        return web.json_response({"found": False, "path": rel_file})
    data = _preset_read(path)
    if data is None:
        return _json_error("存档文件不是有效的 JSON：%s" % rel_file, 500)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    return web.json_response({
        "found": True, "path": rel_file, "data": data, "mtime": mtime,
    })


async def preset_save(request: web.Request) -> web.Response:
    """Write ``body.data`` to ``<dir>/<name>`` (default ``stlu_inputs.json``).
    Written to a temp file first so a crash never leaves a truncated preset."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - aiohttp raises several types here
        return _json_error("invalid JSON body", 400)
    if not isinstance(body, dict) or not isinstance(body.get("data"), dict):
        return _json_error("data must be a JSON object", 400)
    try:
        folder = _resolve("output", str(body.get("dir") or ""))
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    name = str(body.get("name") or _PRESET_NAME).strip()
    if ("/" in name or "\\" in name or name.startswith(".")
            or not name.lower().endswith(".json")):
        name = _PRESET_NAME
    try:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(body["data"], fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError as exc:
        return _json_error("cannot write preset: %s" % exc, 500)
    return web.json_response({"ok": True, "path": _relative("output", path) or ""})


async def preset_read_file(request: web.Request) -> web.Response:
    """Read any output-relative JSON file (the "import from file" button)."""
    try:
        path = _resolve("output", request.query.get("path", ""))
    except web.HTTPException as exc:
        return _json_error(exc.text or "invalid path", exc.status)
    if not os.path.isfile(path) or not path.lower().endswith(".json"):
        return _json_error("not a JSON file", 404)
    data = _preset_read(path)
    if data is None:
        return _json_error("文件不是有效的 JSON 对象", 500)
    return web.json_response({
        "ok": True, "path": _relative("output", path) or "", "data": data,
    })


if _ROUTES is not None:
    _ROUTES.get("/stlu/ls")(list_directory)
    _ROUTES.get("/stlu/fs_ls")(fs_list)
    _ROUTES.get("/stlu/reveal_local")(reveal_local)
    _ROUTES.get("/stlu/open_local")(open_local)
    _ROUTES.get("/stlu/layers")(list_layers)
    _ROUTES.get("/stlu/thumb")(thumb)
    _ROUTES.get("/stlu/preview")(preview_plan)
    _ROUTES.get("/stlu/reveal")(reveal_path)
    _ROUTES.get("/stlu/open")(open_path)
    _ROUTES.get("/stlu/imgsize")(image_size)
    _ROUTES.get("/stlu/psd_preview")(psd_preview)
    _ROUTES.get("/stlu/psd_layers")(psd_layers)
    _ROUTES.get("/stlu/tiles/strips")(tiles_strips)
    _ROUTES.get("/stlu/edit_parts")(edit_parts_catalogue)
    _ROUTES.get("/stlu/tiles/dirs")(tiles_dirs)
    _ROUTES.get("/stlu/tiles/manifest")(tiles_manifest)
    _ROUTES.post("/stlu/tiles/clear")(tiles_clear)
    _ROUTES.post("/stlu/tiles/merge")(tiles_merge)
    _ROUTES.post("/stlu/tiles/merge-mixed")(tiles_merge_mixed)
    _ROUTES.get("/stlu/tiles/grids")(tiles_grids)
    _ROUTES.get("/stlu/tiles/grid")(tiles_grid)
    _ROUTES.get("/stlu/tiles/images")(tiles_images)
    _ROUTES.post("/stlu/tiles/from_images")(tiles_from_images)
    _ROUTES.get("/stlu/preset")(preset_load)
    _ROUTES.post("/stlu/preset")(preset_save)
    _ROUTES.get("/stlu/preset_file")(preset_read_file)
