"""ComfyUI entry point: aggregate the SeeThrough LayerUpscale node modules.

The implementation is split by responsibility:

* ``nodes_shared``   -- constants and helpers used by every module
* ``nodes_pipeline`` -- the guided tile pipeline, stages ①-⑤
* ``nodes_legacy``   -- older nodes, still registered under ``/Legacy``
* ``nodes_upscale``  -- plan / edit-parts / 4K-mask workflows
* ``nodes_transparent`` -- fine-grained steps of the guided pipeline

Importing ``nodes`` (as tests and tools do) still exposes every class.
"""

from __future__ import annotations

import folder_paths  # noqa: F401  (tests patch ``nodes.folder_paths``)

if __package__:
    from . import tile_psd  # noqa: F401  (tests reach ``nodes.tile_psd``)
else:
    import tile_psd  # noqa: F401

# ComfyUI executes this file as a package (``__package__`` set, folder NOT on
# sys.path); the offline tests import it as a top-level module instead.  Branch
# on ``__package__`` rather than on ``except ImportError``: a fallback would
# load the same submodules twice under two names, and it hid a broken relative
# import (``from . import folder_paths``) that emptied every mapping.
if __package__:  # ComfyUI
    from .nodes_shared import *  # noqa: F401,F403
    from .nodes_pipeline import *  # noqa: F401,F403
    from .nodes_legacy import *  # noqa: F401,F403
    from .nodes_upscale import *  # noqa: F401,F403
    from .nodes_transparent import *  # noqa: F401,F403
    from .nodes_pipeline import (
        NODE_CLASS_MAPPINGS as _PIPELINE_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _PIPELINE_NAMES,
    )
    from .nodes_legacy import (
        NODE_CLASS_MAPPINGS as _LEGACY_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _LEGACY_NAMES,
    )
    from .nodes_upscale import (
        NODE_CLASS_MAPPINGS as _UPSCALE_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _UPSCALE_NAMES,
    )
    from .nodes_transparent import (
        NODE_CLASS_MAPPINGS as _TRANSPARENT_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _TRANSPARENT_NAMES,
    )
else:  # standalone import used by the tests and tools
    import folder_paths  # noqa: F401
    from nodes_shared import *  # noqa: F401,F403
    from nodes_pipeline import *  # noqa: F401,F403
    from nodes_legacy import *  # noqa: F401,F403
    from nodes_upscale import *  # noqa: F401,F403
    from nodes_transparent import *  # noqa: F401,F403
    from nodes_pipeline import (
        NODE_CLASS_MAPPINGS as _PIPELINE_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _PIPELINE_NAMES,
    )
    from nodes_legacy import (
        NODE_CLASS_MAPPINGS as _LEGACY_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _LEGACY_NAMES,
    )
    from nodes_upscale import (
        NODE_CLASS_MAPPINGS as _UPSCALE_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _UPSCALE_NAMES,
    )
    from nodes_transparent import (
        NODE_CLASS_MAPPINGS as _TRANSPARENT_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _TRANSPARENT_NAMES,
    )

CATEGORY = "SeeThrough/LayerUpscale"

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(_UPSCALE_CLASSES)
NODE_CLASS_MAPPINGS.update(_LEGACY_CLASSES)
NODE_CLASS_MAPPINGS.update(_PIPELINE_CLASSES)
NODE_CLASS_MAPPINGS.update(_TRANSPARENT_CLASSES)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(_UPSCALE_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_LEGACY_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_PIPELINE_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_TRANSPARENT_NAMES)
