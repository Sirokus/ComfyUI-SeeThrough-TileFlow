"""CPU-only checks that a failed folder reset stops the run.

Windows keeps a file locked while Photoshop (or Explorer's preview pane) has
it open.  The pipeline must not write a new generation fingerprint after a
partial clear: surviving PSDs would otherwise look like current output.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest import mock

import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
for path in (TESTS_DIR, PKG_DIR, COMFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import nodes as stlu  # noqa: E402
import nodes_shared  # noqa: E402
import tile_psd  # noqa: E402

_failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        _failures.append(message)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = tile_psd.flat_directory(tmp, "SeeThrough/Locked")
        os.makedirs(run_dir, exist_ok=True)
        tile = os.path.join(run_dir, "tile_00.psd")
        with open(tile, "wb") as handle:
            handle.write(b"old psd")
        manifest = {
            "dir": "SeeThrough/Locked",
            "source": [128, 128],
            "tile_size": [64, 64],
            "overlap": 0,
            "offset": [0, 0],
            "tiles": [{"index": 0, "file": "tile_00.psd"}],
            "count": 1,
        }
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        old_generation = {"generation": {"version": 1, "random_seed": 1},
                          "generation_id": "old"}
        with open(os.path.join(run_dir, "generation.json"), "w", encoding="utf-8") as handle:
            json.dump(old_generation, handle)

        print("[clear failure is fatal]")
        with mock.patch("tile_psd.os.remove",
                        side_effect=PermissionError(13, "file is in use")):
            try:
                tile_psd.clear_run_outputs(run_dir, manifest)
                check(False, "a locked PSD should raise ClearFailed")
            except tile_psd.ClearFailed as exc:
                check(True, "a locked PSD raises ClearFailed")
                check("Photoshop" in str(exc), "the message tells the user what to close")
        check(os.path.isfile(tile), "the locked file is still there")
        with open(os.path.join(run_dir, "generation.json"), "r", encoding="utf-8") as handle:
            check(json.load(handle)["generation_id"] == "old",
                  "the old fingerprint is still on disk")

        print("[PipelineInput never writes a new fingerprint after a failed clear]")
        original_root = nodes_shared._output_root
        nodes_shared._output_root = lambda: tmp
        try:
            with mock.patch("tile_psd.os.remove",
                            side_effect=PermissionError(13, "file is in use")):
                try:
                    stlu.STLU_PipelineInput().execute(
                        image=torch.zeros(1, 128, 128, 3),
                        tile_width=64, tile_height=64, overlap=0,
                        select="all", resolution=512, num_inference_steps=1,
                        depth_resolution=-1, random_seed=99,
                        out_dir="SeeThrough/Locked")
                    check(False, "execute should stop when the folder cannot be cleared")
                except tile_psd.ClearFailed:
                    check(True, "execute stops when the folder cannot be cleared")
        finally:
            nodes_shared._output_root = original_root
        with open(os.path.join(run_dir, "generation.json"), "r", encoding="utf-8") as handle:
            check(json.load(handle)["generation_id"] == "old",
                  "execute did not overwrite generation.json")

    if _failures:
        print("\nFAILED:", len(_failures))
        for message in _failures:
            print("  -", message)
        return 1
    print("\nALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
