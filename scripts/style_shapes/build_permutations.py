#!/usr/bin/env python3
"""Freeze complete global training permutations for all formal epochs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from style_shapes.io import command_record, sha256_file
from style_shapes.permutations import create_permutation_manifest, load_permutation_manifest, save_permutation_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        with open(args.train_data, "r", encoding="utf-8") as handle:
            num_rows = len(json.load(handle))
        value = create_permutation_manifest(
            num_rows, args.epochs, args.world_size, args.seed
        )
        value["train_data"] = {
            "path": args.train_data,
            "sha256": sha256_file(args.train_data),
        }
        # train_data is provenance rather than hashed permutation content.
        core = {key: item for key, item in value.items() if key != "manifest_hash"}
        from style_shapes.io import hash_value

        value["manifest_hash"] = hash_value(core)
        if Path(args.output).exists():
            existing = load_permutation_manifest(args.output, num_rows)
            if existing["manifest_hash"] != value["manifest_hash"]:
                raise RuntimeError("refusing to overwrite incompatible permutation manifest: %s" % args.output)
        else:
            save_permutation_manifest(args.output, value)
        print(json.dumps({key: value[key] for key in value if key != "permutations"}, indent=2))


if __name__ == "__main__":
    main()

