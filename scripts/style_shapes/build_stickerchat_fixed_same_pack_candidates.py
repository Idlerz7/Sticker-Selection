#!/usr/bin/env python3
"""Build clean, mapping-ordered StickerChat fixed same-pack R10 assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from main import HFClipSentenceEncoder
from style_shapes.fixed_same_pack import write_fixed_same_pack_assets
from style_shapes.io import command_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-zip", default="1.zip")
    parser.add_argument("--img2id", default="stickerchat/processed/img2id.json")
    parser.add_argument(
        "--processed-train",
        default="stickerchat/processed/release_train_u_sticker_format_int.json",
    )
    parser.add_argument(
        "--processed-validation",
        default="stickerchat/processed/release_val_u_sticker_format_int_with_cand_r10.json",
    )
    parser.add_argument(
        "--processed-test",
        default="stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/style_shapes/candidates/stickerchat_fixed_same_pack_r10",
    )
    parser.add_argument(
        "--validation-output",
        default=(
            "stickerchat/processed/"
            "release_val_u_sticker_format_int_with_cand_fixed_same_pack_r10.json"
        ),
    )
    parser.add_argument(
        "--test-output",
        default=(
            "stickerchat/processed/"
            "release_test_u_sticker_format_int_with_cand_fixed_same_pack_r10.json"
        ),
    )
    parser.add_argument(
        "--clip-model",
        default="ckpt/clip-ViT-B-32/0_CLIPModel",
    )
    parser.add_argument(
        "--gray-device",
        default="auto",
        help="'auto', 'cpu', or an explicit CUDA device such as cuda:0.",
    )
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    parser.add_argument(
        "--allow-stat-drift",
        action="store_true",
        help="Debug only: do not enforce the pre-registered row/gray counts.",
    )
    return parser.parse_args()


def encode_gray(args: argparse.Namespace) -> torch.Tensor:
    if args.gray_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.gray_device)
    encoder = HFClipSentenceEncoder(
        args.clip_model, local_files_only=True, device=device
    )
    encoder.eval()
    gray = Image.new("RGB", (224, 224), color=(127, 127, 127))
    with torch.no_grad():
        embedding = encoder(encoder.tokenize([gray]))["sentence_embedding"][0]
    return embedding.detach().cpu().float()


def main() -> None:
    args = parse_args()
    with command_record(args, args.artifact_root):
        manifest = write_fixed_same_pack_assets(
            zip_path=args.raw_zip,
            img2id_path=args.img2id,
            processed_train_path=args.processed_train,
            processed_val_path=args.processed_validation,
            processed_test_path=args.processed_test,
            output_dir=args.output_dir,
            val_output_path=args.validation_output,
            test_output_path=args.test_output,
            gray_embedding=encode_gray(args),
            enforce_expected_stats=not args.allow_stat_drift,
        )
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "schema_version": manifest["schema_version"],
                    "negative_policy": manifest["negative_policy"],
                    "manifest": str(Path(args.output_dir) / "manifest.json"),
                    "manifest_hash": manifest["manifest_hash"],
                    "splits": {
                        split: metadata["stats"]
                        for split, metadata in manifest["splits"].items()
                    },
                    "gray_embedding": manifest["gray_embedding"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
