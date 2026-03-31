#!/usr/bin/env python3
"""
Build stickerchat/processed/factorized_style_bank.json (or a custom output path).

Examples:
  # img_set prototypes (default StickerChat recipe)
  python scripts/build_stickerchat_factorized_bank.py --bank-source style_metadata

  # DSTC-style pseudo labels (fine/coarse/singleton)
  python scripts/build_stickerchat_factorized_bank.py \\
    --bank-source pseudo_labels \\
    --pseudo-label-path ./stickerchat/processed/stickerchat_pseudo_labels.jsonl \\
    --output-path ./stickerchat/processed/factorized_style_bank_pseudo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from factorized_style_bank import (  # noqa: E402
    build_factorized_style_bank_dict,
    build_factorized_style_bank_from_style_metadata,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bank-source",
        choices=("pseudo_labels", "style_metadata"),
        default="style_metadata",
    )
    p.add_argument(
        "--pseudo-label-path",
        type=Path,
        default=Path("stickerchat/processed/stickerchat_pseudo_labels.jsonl"),
    )
    p.add_argument(
        "--style-metadata-path",
        type=Path,
        default=Path("stickerchat/processed/sticker_metadata.json"),
    )
    p.add_argument(
        "--style-neighbors-path",
        type=Path,
        default=Path("stickerchat/processed/style_neighbors.json"),
    )
    p.add_argument("--max-image-id", type=int, default=174695)
    p.add_argument("--min-fine-proto-size", type=int, default=2)
    p.add_argument("--min-coarse-proto-size", type=int, default=2)
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("stickerchat/processed/factorized_style_bank.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.bank_source == "style_metadata":
        bank_dict = build_factorized_style_bank_from_style_metadata(
            style_metadata_path=str(args.style_metadata_path),
            style_neighbors_path=str(args.style_neighbors_path),
            max_image_id=int(args.max_image_id),
        )
    else:
        if not args.pseudo_label_path.exists():
            print(
                f"Missing pseudo label file: {args.pseudo_label_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        bank_dict = build_factorized_style_bank_dict(
            pseudo_label_path=str(args.pseudo_label_path),
            style_neighbors_path=str(args.style_neighbors_path),
            max_image_id=int(args.max_image_id),
            min_fine_proto_size=int(args.min_fine_proto_size),
            min_coarse_proto_size=int(args.min_coarse_proto_size),
        )
    out = args.output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(bank_dict, f, ensure_ascii=False, indent=2)
    meta = bank_dict.get("meta", {})
    print(
        "saved:",
        out,
        "num_prototypes=",
        meta.get("num_prototypes"),
        "meta=",
        {k: meta[k] for k in sorted(meta) if k.startswith("num_")},
    )


if __name__ == "__main__":
    main()
