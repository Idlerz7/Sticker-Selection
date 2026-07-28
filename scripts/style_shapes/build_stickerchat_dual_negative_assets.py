#!/usr/bin/env python3
"""Freeze eligible rows and permutations for the dual-local StickerChat variant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from style_shapes.group_bank import GroupBank
from style_shapes.io import atomic_write_json, command_record, hash_value, sha256_file
from style_shapes.negative_sampling import (
    StickerChatDualLocalNegativeSampler,
    build_eligibility_manifest,
    load_id_to_pack,
    validate_eligibility_manifest,
    validate_group_top32_bank,
)
from style_shapes.permutations import (
    create_permutation_manifest,
    load_permutation_manifest,
    save_permutation_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-data",
        default="stickerchat/processed/release_train_u_sticker_format_int.json",
    )
    parser.add_argument(
        "--pack-metadata",
        default="stickerchat/processed/sticker_metadata.json",
    )
    parser.add_argument(
        "--group-bank",
        default="artifacts/style_shapes/groups/stickerchat/vpd_pack/group_bank.json",
    )
    parser.add_argument(
        "--eligibility-output",
        default=(
            "artifacts/style_shapes/negative_sampling/"
            "stickerchat_same_pack_plus_vpd_top32/eligible_rows.json"
        ),
    )
    parser.add_argument(
        "--permutation-output",
        default=(
            "artifacts/style_shapes/permutations/"
            "stickerchat_dual_local_seed2021_ws8.json"
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with command_record(args, args.artifact_root):
        with open(args.train_data, "r", encoding="utf-8") as handle:
            train_rows = json.load(handle)
        bank = GroupBank.load(args.group_bank)
        id_to_pack = load_id_to_pack(args.pack_metadata)
        bank_audit = validate_group_top32_bank(bank, id_to_pack)
        sampler = StickerChatDualLocalNegativeSampler(
            bank, id_to_pack, seed=args.seed
        )
        inputs = {
            "train_data": {
                "path": args.train_data,
                "sha256": sha256_file(args.train_data),
            },
            "pack_metadata": {
                "path": args.pack_metadata,
                "sha256": sha256_file(args.pack_metadata),
            },
            "group_bank": {
                "path": args.group_bank,
                "sha256": sha256_file(args.group_bank),
            },
            "membership_hash": bank.membership_hash,
            "neighbor_content_hash": bank_audit["neighbor_content_hash"],
        }
        eligibility = build_eligibility_manifest(
            train_rows, sampler, inputs=inputs
        )
        validate_eligibility_manifest(eligibility)
        eligibility_path = Path(args.eligibility_output)
        if eligibility_path.exists():
            with eligibility_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            validate_eligibility_manifest(existing)
            if existing != eligibility:
                raise RuntimeError(
                    "refusing to overwrite incompatible eligibility manifest: %s"
                    % eligibility_path
                )
        else:
            atomic_write_json(eligibility_path, eligibility)

        permutation = create_permutation_manifest(
            eligibility["eligible_count"],
            args.epochs,
            args.world_size,
            args.seed,
        )
        permutation["eligibility_manifest"] = {
            "path": args.eligibility_output,
            "sha256": sha256_file(eligibility_path),
            "manifest_hash": eligibility["manifest_hash"],
        }
        core = {
            key: value
            for key, value in permutation.items()
            if key != "manifest_hash"
        }
        permutation["manifest_hash"] = hash_value(core)
        permutation_path = Path(args.permutation_output)
        if permutation_path.exists():
            existing = load_permutation_manifest(
                str(permutation_path), eligibility["eligible_count"]
            )
            if existing != permutation:
                raise RuntimeError(
                    "refusing to overwrite incompatible permutation manifest: %s"
                    % permutation_path
                )
        else:
            save_permutation_manifest(str(permutation_path), permutation)

        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "negative_policy": eligibility["negative_policy"],
                    "total_rows": eligibility["total_rows"],
                    "eligible_count": eligibility["eligible_count"],
                    "excluded_count": eligibility["excluded_count"],
                    "excluded_reason_counts": eligibility[
                        "excluded_reason_counts"
                    ],
                    "membership_hash": bank.membership_hash,
                    "neighbor_content_hash": bank_audit["neighbor_content_hash"],
                    "eligibility_manifest": {
                        "path": str(eligibility_path),
                        "sha256": sha256_file(eligibility_path),
                        "manifest_hash": eligibility["manifest_hash"],
                    },
                    "permutation_manifest": {
                        "path": str(permutation_path),
                        "sha256": sha256_file(permutation_path),
                        "manifest_hash": permutation["manifest_hash"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
