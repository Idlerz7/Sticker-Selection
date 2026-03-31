#!/usr/bin/env python3
"""Print suggested commands for StickerChat data-side workflow (audit, short ablations, bank build)."""

from __future__ import annotations

LINES = r"""
# 1) Data audit (no GPU)
python scripts/audit_stickerchat_factorized_data.py \
  --max-image-id 174695 \
  --metadata-path ./stickerchat/processed/sticker_metadata.json \
  --bank-path ./stickerchat/processed/factorized_style_bank.json \
  --train-path ./stickerchat/processed/release_train_u_sticker_format_int.json \
  --test-path ./stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json \
  --include-neighbors \
  --neighbor-proto-consistency-sample 8000

# 2) Short training ablations (set epochs: 2 in YAML or CLI if supported)
python main_structured_factorized.py --config configs/structured_factorized/stickerchat_v6_base_only.yaml --gpus 1
python main_structured_factorized.py --config configs/structured_factorized/stickerchat_v6_aux_zero.yaml --gpus 1
python main_structured_factorized.py --config configs/structured_factorized/stickerchat_v6_proto_only.yaml --gpus 1
python main_structured_factorized.py --config configs/structured_factorized/stickerchat_v6_expr_only.yaml --gpus 1

# 3) Build pseudo-label bank (after generating stickerchat/processed/stickerchat_pseudo_labels.jsonl)
python scripts/build_stickerchat_factorized_bank.py --bank-source pseudo_labels \
  --pseudo-label-path ./stickerchat/processed/stickerchat_pseudo_labels.jsonl \
  --output-path ./stickerchat/processed/factorized_style_bank_pseudo.json

python main_structured_factorized.py --config configs/structured_factorized/stickerchat_v6_pseudo_labels_bank.yaml --gpus 1

# 4) Frequency-filtered train subset (optional curriculum)
python scripts/filter_stickerchat_train_by_frequency.py \
  --input ./stickerchat/processed/release_train_u_sticker_format_int.json \
  --output ./stickerchat/processed/release_train_u_sticker_format_int_freq_ge_8.json \
  --min-frequency 8
# Then point train_data_path in a small YAML override to the filtered JSON.
""".strip()


def main() -> None:
    print(LINES)


if __name__ == "__main__":
    main()
