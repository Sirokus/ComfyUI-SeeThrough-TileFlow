"""CPU-only checks for the atomic JSON writer (no GPU, no server)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest import mock

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from stlu import json_io  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")

        print("[atomic write]")
        json_io.write_json_atomic(path, {"a": 1})
        with open(path, "r", encoding="utf-8") as handle:
            check(json.load(handle) == {"a": 1}, "creates a readable file")

        json_io.write_json_atomic(path, {"a": 2})
        with open(path, "r", encoding="utf-8") as handle:
            check(json.load(handle) == {"a": 2}, "replaces an existing file")

        print("[a failed write keeps the old document]")
        with mock.patch.object(json_io.json, "dump",
                               side_effect=RuntimeError("disk full")):
            try:
                json_io.write_json_atomic(path, {"a": 3})
                check(False, "a failing dump should raise")
            except RuntimeError:
                check(True, "a failing dump raises")
        with open(path, "r", encoding="utf-8") as handle:
            check(json.load(handle) == {"a": 2}, "the old document is untouched")
        leftovers = [n for n in os.listdir(tmp) if n.endswith(".tmp")]
        check(not leftovers, "no temporary file is left behind (%s)" % leftovers)

        print("[read_json]")
        check(json_io.read_json(os.path.join(tmp, "missing.json"), {"d": 0}) == {"d": 0},
              "a missing file returns the default")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        try:
            json_io.read_json(path)
            check(False, "corrupt JSON should raise")
        except json_io.JsonReadError:
            check(True, "corrupt JSON raises JsonReadError")

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
