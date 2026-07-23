#!/usr/bin/env python3
"""Merge per-rank negative traces and enforce complete epoch/source coverage."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from style_shapes.io import command_record
from style_shapes.validation import merge_and_validate_traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-glob", required=True)
    parser.add_argument("--num-rows", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--membership-hash", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        paths = sorted(glob.glob(args.trace_glob))
        if not paths:
            raise FileNotFoundError("no trace files matched")
        value = merge_and_validate_traces(
            paths,
            args.num_rows,
            args.epochs,
            args.membership_hash,
            args.output,
        )
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()

