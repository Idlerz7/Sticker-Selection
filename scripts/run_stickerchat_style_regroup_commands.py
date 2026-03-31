#!/usr/bin/env python3
"""Print suggested commands for StickerChat CLIP-centroid style regroup experiments."""

from __future__ import annotations

LINES = r"""
# 1) If missing, precompute StickerChat CLIP embeddings
python precompute_sticker_embeddings.py \
  --id2img_path ./stickerchat/processed/id2img.json \
  --img_dir ./stickerchat/png_stickers_flat \
  --max_image_id 174695 \
  --img_pretrain_path ./ckpt/clip-ViT-B-32 \
  --output_path ./stickerchat/processed/stickerchat_clip_embs.pt \
  --batch_size 512 \
  --workers 8

# 2) Analysis only: see whether raw img_set packs have visually similar neighbors
python scripts/build_stickerchat_style_regroup_assets.py \
  --img-emb-cache-path ./stickerchat/processed/stickerchat_clip_embs.pt \
  --output-dir ./stickerchat/processed_style_regroup_centroid \
  --merge-threshold 0.92 \
  --reciprocal-topk 1 \
  --analysis-only

# 3) Build regrouped assets
python scripts/build_stickerchat_style_regroup_assets.py \
  --img-emb-cache-path ./stickerchat/processed/stickerchat_clip_embs.pt \
  --output-dir ./stickerchat/processed_style_regroup_centroid \
  --merge-threshold 0.92 \
  --reciprocal-topk 1 \
  --split-large-packs-over 120 \
  --split-target-size 48

# 4) Short training comparison
python main_structured_factorized.py --config configs/structured_factorized/stickerchat_v6_base_only_regroup_centroid.yaml --gpus 1
python main_structured_factorized.py --config configs/structured_factorized/stickerchat_v6_minimal_regroup_centroid.yaml --gpus 1
""".strip()


def main() -> None:
    print(LINES)


if __name__ == "__main__":
    main()
