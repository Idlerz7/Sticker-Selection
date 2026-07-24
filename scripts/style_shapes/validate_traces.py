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
from style_shapes.group_bank import GroupBank
from style_shapes.negative_sampling import (
    StickerChatDualLocalNegativeSampler,
    load_eligibility_manifest,
    load_id_to_pack,
    validate_dual_local_trace_record,
)
from style_shapes.validation import merge_and_validate_traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-glob", required=True)
    parser.add_argument("--num-rows", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--membership-hash", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eligibility-manifest")
    parser.add_argument("--pack-metadata")
    parser.add_argument("--group-bank")
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        paths = sorted(glob.glob(args.trace_glob))
        if not paths:
            raise FileNotFoundError("no trace files matched")
        expected_source_rows = None
        record_validator = None
        if args.eligibility_manifest:
            if not args.pack_metadata or not args.group_bank:
                raise ValueError(
                    "--eligibility-manifest requires --pack-metadata and --group-bank"
                )
            eligibility = load_eligibility_manifest(args.eligibility_manifest)
            expected_source_rows = eligibility["eligible_rows"]
            bank = GroupBank.load(args.group_bank)
            sampler = StickerChatDualLocalNegativeSampler(
                bank, load_id_to_pack(args.pack_metadata), seed=eligibility["seed"]
            )
            record_validator = lambda record: validate_dual_local_trace_record(
                record, sampler
            )
        elif args.num_rows is None:
            raise ValueError("--num-rows is required without --eligibility-manifest")
        value = merge_and_validate_traces(
            paths,
            args.num_rows,
            args.epochs,
            args.membership_hash,
            args.output,
            expected_source_rows=expected_source_rows,
            record_validator=record_validator,
        )
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
