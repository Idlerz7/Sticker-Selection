#!/usr/bin/env python3
import argparse
import json

from factorized_style_bank import (
    build_factorized_style_bank_dict,
    build_factorized_style_bank_from_style_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a factorized style bank for two-stage style/expression retrieval."
    )
    parser.add_argument(
        "--bank-source",
        choices=("pseudo_labels", "style_metadata"),
        default="pseudo_labels",
    )
    parser.add_argument(
        "--pseudo-label-path",
        default=(
            "pseudo_labels/"
            "sticker_identity_style_labels_from_data_meme_set_model_gemini-3-pro-preview_date_20260321.jsonl"
        ),
    )
    parser.add_argument("--style-metadata-path", default="")
    parser.add_argument("--style-neighbors-path", default="style_neighbors.json")
    parser.add_argument("--max-image-id", type=int, default=307)
    parser.add_argument("--min-fine-proto-size", type=int, default=2)
    parser.add_argument("--min-coarse-proto-size", type=int, default=2)
    parser.add_argument("--output-path", default="factorized_style_bank.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bank_source == "style_metadata":
        bank_dict = build_factorized_style_bank_from_style_metadata(
            style_metadata_path=args.style_metadata_path,
            style_neighbors_path=args.style_neighbors_path,
            max_image_id=args.max_image_id,
        )
    else:
        bank_dict = build_factorized_style_bank_dict(
            pseudo_label_path=args.pseudo_label_path,
            style_neighbors_path=args.style_neighbors_path,
            max_image_id=args.max_image_id,
            min_fine_proto_size=args.min_fine_proto_size,
            min_coarse_proto_size=args.min_coarse_proto_size,
        )
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(bank_dict, f, ensure_ascii=False, indent=2)
    print(
        "saved factorized style bank:",
        args.output_path,
        "num_prototypes=",
        bank_dict["meta"]["num_prototypes"],
    )


if __name__ == "__main__":
    main()
