#!/usr/bin/env python3
"""Build audit-only StickerChat R10 candidates matching the released PBR bug."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from style_shapes.io import command_record
from style_shapes.pbr_bug_eval import write_pbr_bug_eval_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-zip", default="1.zip")
    parser.add_argument("--img2id", default="stickerchat/processed/img2id.json")
    parser.add_argument(
        "--processed-validation",
        default="stickerchat/processed/release_val_u_sticker_format_int_with_cand_r10.json",
    )
    parser.add_argument(
        "--processed-test",
        default="stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json",
    )
    parser.add_argument(
        "--gray-embedding",
        default=(
            "artifacts/style_shapes/candidates/stickerchat_fixed_same_pack_r10/"
            "gray_clip_embedding.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/style_shapes/candidates/"
            "stickerchat_pbr_bug_compatible_r10"
        ),
    )
    parser.add_argument(
        "--validation-output",
        default=(
            "stickerchat/processed/"
            "release_val_u_sticker_format_int_with_cand_pbr_bug_r10.json"
        ),
    )
    parser.add_argument(
        "--test-output",
        default=(
            "stickerchat/processed/"
            "release_test_u_sticker_format_int_with_cand_pbr_bug_r10.json"
        ),
    )
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with command_record(args, args.artifact_root):
        manifest = write_pbr_bug_eval_assets(
            zip_path=args.raw_zip,
            img2id_path=args.img2id,
            processed_validation_path=args.processed_validation,
            processed_test_path=args.processed_test,
            gray_embedding_path=args.gray_embedding,
            output_dir=args.output_dir,
            validation_output_path=args.validation_output,
            test_output_path=args.test_output,
        )
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "audit_only": True,
                    "manifest": str(Path(args.output_dir) / "manifest.json"),
                    "manifest_hash": manifest["manifest_hash"],
                    "splits": {
                        split: metadata["stats"]
                        for split, metadata in manifest["splits"].items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
