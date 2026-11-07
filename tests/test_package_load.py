"""Load the plugin the way ComfyUI does, not the way the other tests do.

Most tests put the package folder on ``sys.path`` and ``import nodes``, which
exercises the standalone-import fallback.  ComfyUI does something different:
it executes ``__init__.py`` as a package while the folder is NOT on
``sys.path``.  A wrong relative import (``from . import folder_paths``) passes
every standalone test and still makes the whole pack vanish from the node
menu, so this script reproduces the real loader.

It also checks that ``__init__.py`` swallows nothing: if importing ``nodes``
raises, the pack silently registers zero nodes instead of showing the error.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))

REQUIRED = (
    "STLU_PipelineInput",
    "STLU_SimpleInput",
    "STLU_GuidedLayers",
    "STLU_TilePSD",
    "STLU_WaveIndex",
    "STLU_TileMerge",
    "STLU_PSDPreview",
    # Transparent (fine-grained) variant: one node per visible step.
    "STLU_PipelineSettings",
    "STLU_PlanGrid",
    "STLU_PickTiles",
    "STLU_PlanContext",
    "STLU_PrepareRun",
    "STLU_CropTiles",
    "STLU_GuideTile",
    "STLU_SeqTileLoop",
    "STLU_CropBack",
    "STLU_RestoreLocks",
    "STLU_WriteTilePSD",
    "STLU_WriteTileImages",
    "STLU_SaveOriginal",
    "STLU_WriteManifest",
    "STLU_MergeTiles",
)

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def load_like_comfyui():
    """Mirror ComfyUI's ``load_custom_node`` for a folder with ``__init__.py``."""
    module_name = os.path.basename(PKG_DIR)
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(PKG_DIR, "__init__.py"),
        submodule_search_locations=[PKG_DIR],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        # Leave no half-loaded package behind for the other checks.
        sys.modules.pop(module_name, None)
    return module


def main():
    # The fallback branch only makes sense when the folder is importable as a
    # top-level module; ComfyUI never does that, so neither do we.
    for path in (PKG_DIR, TESTS_DIR):
        while path in sys.path:
            sys.path.remove(path)
    check(PKG_DIR not in [os.path.abspath(p or ".") for p in sys.path],
          "the package folder is not on sys.path (same as ComfyUI)")
    check(os.getcwd().lower() == COMFY_ROOT.lower(),
          "running from the ComfyUI root (folder_paths importable)")

    print("[import through the real package loader]")
    try:
        module = load_like_comfyui()
    except Exception:
        traceback.print_exc()
        check(False, "importing the package does not raise")
        print("\nFAILED: import\n")
        return 1

    mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
    names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    check(bool(mappings), "NODE_CLASS_MAPPINGS is not empty (%d)" % len(mappings))
    check(len(mappings) == 45, "all 45 nodes register (%d)" % len(mappings))
    check(set(names) == set(mappings), "every node has a display name")

    missing = [name for name in REQUIRED if name not in mappings]
    check(not missing, "the guided pipeline nodes are registered (%s)"
          % (missing or "none missing"))

    check(getattr(module, "WEB_DIRECTORY", None) == "./web",
          "WEB_DIRECTORY points at web/")

    # The category prefix is how the menu groups them; a split module losing
    # it would still register but look "wrong" in the UI.
    categories = {str(getattr(cls, "CATEGORY", "")) for cls in mappings.values()}
    stray = sorted(cat for cat in categories if "SeeThrough" not in cat)
    check(not stray, "every node keeps a SeeThrough category (stray: %s)"
          % (stray or "none"))
    legacy = [name for name, cls in mappings.items()
              if "Legacy" in str(getattr(cls, "CATEGORY", ""))]
    check(len(legacy) == 17, "the 17 legacy nodes stay under /Legacy (%d)" % len(legacy))

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
