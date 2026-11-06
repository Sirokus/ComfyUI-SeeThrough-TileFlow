"""Atomic JSON reads/writes for the tile pipeline's state files.

A crash while ``json.dump`` was writing ``manifest.json`` used to leave a
truncated file.  ``read_manifest`` then treated the folder as an interrupted
run and could reuse tiles from a previous generation.  Writing to a sibling
temporary file and replacing the target in one step makes the state file
either the old complete document or the new complete document, never a mix.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Optional


class JsonReadError(ValueError):
    """The file exists but cannot be parsed as JSON."""


def write_json_atomic(path: str, payload: Any, *, indent: Optional[int] = 2,
                      ensure_ascii: bool = False,
                      sort_keys: bool = False) -> str:
    """Write ``payload`` to ``path`` without ever exposing a partial file."""
    target = os.path.abspath(str(path))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix="." + os.path.basename(target) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=ensure_ascii,
                      indent=indent, sort_keys=sort_keys)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return target


def read_json(path: str, default: Any = None) -> Any:
    """Read JSON, returning ``default`` for a missing file.

    A corrupt file raises :class:`JsonReadError` instead of looking like an
    absent one; callers that only want an optimisation may catch it, but the
    pipeline's cache gates must not mistake corruption for "no manifest".
    """
    target = str(path or "")
    if not target or not os.path.isfile(target):
        return default
    try:
        with open(target, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise JsonReadError("cannot read %s: %s" % (target, exc)) from exc
