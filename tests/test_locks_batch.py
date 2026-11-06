"""The preview panel's batch lock endpoint.

``POST /stlu/locks`` accepts ``items`` as well as one ``tile``/``name`` pair,
so the row and page buttons lock a whole set in one request.  This test drives
the handler against real tile PSDs written to a temporary output root.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import folder_paths  # noqa: E402
import psd_writer  # noqa: E402
import routes  # noqa: E402
import tile_psd  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def body(resp):
    return json.loads(resp.text)


def solid(width, height, value):
    arr = np.zeros((height, width, 4), np.uint8)
    arr[..., :3] = value
    arr[..., 3] = 255
    return arr


def write_tile(run_dir, name, layers):
    path = os.path.join(run_dir, name)
    psd_writer.write_psd(
        path,
        [{"name": label, "pixels": pixels, "left": left, "top": top}
         for label, pixels, left, top in layers],
        64, 64)
    return path


class Req:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


def main():
    with tempfile.TemporaryDirectory() as tmp:
        old_output = folder_paths.get_output_directory
        folder_paths.get_output_directory = lambda: tmp
        try:
            run_dir = os.path.join(tmp, "SeeThrough", "Batch")
            os.makedirs(run_dir, exist_ok=True)
            write_tile(run_dir, "tile_00_r0_c0.psd", [
                ("face", solid(9, 7, (200, 10, 10)), 5, 6),
                ("hair", solid(11, 8, (10, 10, 200)), 4, 9),
                ("mouth", solid(6, 5, (0, 0, 220)), 30, 30),
            ])
            write_tile(run_dir, "tile_01_r0_c1.psd", [
                ("face", solid(8, 8, (20, 220, 20)), 12, 14),
            ])
            with open(os.path.join(run_dir, "manifest.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({
                    "prefix": "tile",
                    "source": [128, 128],
                    "tiles": [
                        {"index": 0, "file": "tile_00_r0_c0.psd"},
                        {"index": 1, "file": "tile_01_r0_c1.psd"},
                    ],
                }, handle)

            payload = {"path": run_dir, "items": [
                {"tile": 0, "name": "face", "locked": True},
                {"tile": 0, "name": "hair", "locked": True},
                {"tile": 1, "name": "face", "locked": True},
            ]}
            print("[batch lock]")
            data = body(asyncio.run(routes.locks_update(Req(payload))))
            check(data.get("ok") is True, "request succeeds")
            check(data.get("done") == 3, "three locks written (%s)"
                  % data.get("done"))
            check(data.get("failed") == [], "no failures")
            summary = data.get("tiles") or {}
            check(sorted(summary.get("0") or []) == ["face", "hair"],
                  "tile 0 holds face + hair (%s)" % summary.get("0"))
            check(summary.get("1") == ["face"], "tile 1 holds face")

            print("[batch lock is idempotent]")
            data = body(asyncio.run(routes.locks_update(Req(payload))))
            check(data.get("done") == 3, "re-locking succeeds")
            check(sorted(data["tiles"].get("0") or []) == ["face", "hair"],
                  "no duplicate records")

            print("[batch unlock]")
            data = body(asyncio.run(routes.locks_update(Req({
                "path": run_dir, "items": [
                    {"tile": 0, "name": "hair", "locked": False},
                    {"tile": 1, "name": "face", "locked": False},
                ],
            }))))
            check(data.get("done") == 2, "two unlocks applied")
            check(sorted(data["tiles"].get("0") or []) == ["face"],
                  "tile 0 keeps only face")
            check("1" not in data["tiles"], "tile 1 has no locks left")

            print("[batch with a missing layer]")
            data = body(asyncio.run(routes.locks_update(Req({
                "path": run_dir, "items": [
                    {"tile": 0, "name": "nose", "locked": True},
                    {"tile": 0, "name": "mouth", "locked": True},
                ],
            }))))
            check(data.get("done") == 1, "the valid item still lands")
            check(len(data.get("failed") or []) == 1,
                  "the missing layer is reported")
            check("nose" in str((data.get("failed") or [{}])[0].get("error")),
                  "the failure names the layer")

            print("[single lock still works]")
            data = body(asyncio.run(routes.locks_update(Req({
                "path": run_dir, "tile": 0, "name": "face", "locked": False,
            }))))
            check(data.get("ok") is True, "single unlock succeeds")
            check("0" not in data["tiles"] or "face" not in data["tiles"]["0"],
                  "face is gone")

            print("[invalid batch]")
            resp = asyncio.run(routes.locks_update(Req({
                "path": run_dir, "items": "not-a-list",
            })))
            check(resp.status == 400, "non-list items rejected (%d)" % resp.status)
            resp = asyncio.run(routes.locks_update(Req({
                "path": run_dir, "items": [{"tile": "x", "name": "face"}],
            })))
            data = body(resp)
            check(data.get("done") == 0 and len(data.get("failed") or []) == 1,
                  "a bad tile lands in failed, not a crash")
        finally:
            folder_paths.get_output_directory = old_output

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
