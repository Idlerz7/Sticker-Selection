#!/usr/bin/env python3
"""Patch torch.utils.tensorboard for setuptools>=60 (distutils.version).

PyTorch 1.10.x reads ``distutils.version.LooseVersion`` without importing
``distutils.version`` first, which breaks with newer setuptools. Run once per
environment (re-run after ``pip install -U torch``).

Usage:
  python scripts/patch_torch_tensorboard_distutils.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        print("Could not import torch:", e, file=sys.stderr)
        return 1

    path = Path(torch.__file__).resolve().parent / "utils" / "tensorboard" / "__init__.py"
    if not path.is_file():
        print("tensorboard __init__ not found:", path, file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    needle = "from setuptools import distutils\n"
    insert = needle + "import distutils.version  # setuptools>=60: load submodule before LooseVersion\n"

    if "import distutils.version" in text:
        print("Already patched:", path)
        return 0

    if needle not in text:
        print("Unexpected file layout, needle not found:", path, file=sys.stderr)
        return 1

    new_text = text.replace(needle, insert, 1)
    path.write_text(new_text, encoding="utf-8")
    print("Patched:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
