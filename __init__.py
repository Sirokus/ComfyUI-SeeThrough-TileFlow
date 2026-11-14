"""SeeThrough layer upscale batch nodes."""

import os
import traceback

WEB_DIRECTORY = "./web"

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:  # noqa: BLE001
    # Keep ComfyUI booting, but say it loudly: an empty mapping means every
    # node shows up as "missing" in the editor, with no other clue.
    print("[STLU] 节点注册失败，所有 SeeThrough 节点都不会出现。"
          "下面是真正的异常："
          " / node registration failed; no SeeThrough node will appear."
          " The real exception follows:")
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    # Registers the /stlu/* routes with the running prompt server.
    from . import routes as _routes  # noqa: F401
except Exception:  # noqa: BLE001
    traceback.print_exc()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
