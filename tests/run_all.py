"""Run every offline test script with the interpreter that started this file.

The tests are plain scripts (not pytest), so this runner executes each one in
its own process, from the ComfyUI root (the cwd the tests expect), and reports
a single pass/fail summary.

    python_embeded\\python.exe custom_nodes/ComfyUI-SeeThrough-LayerUpscale/tests/run_all.py
    python_embeded\\python.exe ...\\tests\\run_all.py tile_guide tile_stages

Passing substrings only runs matching test files.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))


def _tail(text: str, limit: int = 40) -> str:
    lines = (text or "").strip().splitlines()
    return "\n".join(lines[-limit:])


def main(argv: list[str]) -> int:
    names = sorted(
        name for name in os.listdir(TESTS_DIR)
        if name.startswith("test_") and name.endswith(".py")
    )
    if argv:
        names = [name for name in names
                 if any(part in name for part in argv)]
        if not names:
            print("no test file matches:", " ".join(argv))
            return 2

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    failures = []
    started_all = time.time()
    for name in names:
        path = os.path.join(TESTS_DIR, name)
        started = time.time()
        proc = subprocess.run(
            [sys.executable, path],
            cwd=COMFY_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        elapsed = time.time() - started
        ok = proc.returncode == 0
        print("%s %-34s %6.1fs" % ("PASS" if ok else "FAIL", name, elapsed))
        if not ok:
            failures.append((name, proc, elapsed))

    total = time.time() - started_all
    print("\n%d/%d passed in %.1fs" % (len(names) - len(failures), len(names), total))
    if failures:
        for name, proc, _elapsed in failures:
            print("\n===== %s =====" % name)
            output = (proc.stdout or "") + (proc.stderr or "")
            print(_tail(output))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
