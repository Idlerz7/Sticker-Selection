#!/usr/bin/env bash
# Regroup v2 aggressive: train / test / diagnostics (StickerChat factorized).
# Requires: conda env with project deps; GPU for reasonable train speed.
# Set CKPT to a saved checkpoint path before running test/base_only lines.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

CFG_V2="configs/structured_factorized/stickerchat_v6_minimal_regroup_v2_aggressive.yaml"
CFG_V2_LEGACY="configs/structured_factorized/stickerchat_v6_minimal_regroup_v2_aggressive_eval_legacy_test.yaml"
CFG_V2_BASEONLY="configs/structured_factorized/stickerchat_v6_base_only_regroup_v2_aggressive.yaml"
CFG_CENTROID="configs/structured_factorized/stickerchat_v6_minimal_regroup_centroid.yaml"
CFG_MMBERT="configs/mmbbert/stickerchat_v1.yaml"

echo "=== 0) Data-side evaluation (run before long training) ==="
echo "$PY scripts/evaluate_stickerchat_regroup_assets.py --regroup-dir ./stickerchat/processed_style_regroup_v2_aggressive --original-metadata ./stickerchat/processed/sticker_metadata.json"
echo "# Optional: add --include-neighbors --neighbor-baseline ./stickerchat/processed/style_neighbors.json"

echo ""
echo "=== 1) Train v2 aggressive (full epochs; override --epochs as needed) ==="
echo "$PY main_structured_factorized.py --config $CFG_V2 --mode train --gpus 1"

echo ""
echo "=== 2) Test v2 (same test JSON as training config: regroup test) ==="
echo "# CKPT=logs/stickerchat_factorized_v6_minimal_regroup_v2_aggressive/lightning_logs/version_X/checkpoints/last.ckpt"
echo "$PY main_structured_factorized.py --config $CFG_V2 --mode test --gpus 1 --ckpt_path \"\$CKPT\""

echo ""
echo "=== 3) Base-only diagnostic (same ckpt; MM-BERT score only) ==="
echo "$PY main_structured_factorized.py --config $CFG_V2 --mode test --gpus 1 --ckpt_path \"\$CKPT\" --base_only true"

echo ""
echo "=== 4) Legacy-processed test JSON (per-epoch / test paths from stickerchat/processed/) ==="
echo "$PY main_structured_factorized.py --config $CFG_V2_LEGACY --mode test --gpus 1 --ckpt_path \"\$CKPT\""

echo ""
echo "=== 5) Baseline: old regroup centroid (same minimal recipe, different assets) ==="
echo "$PY main_structured_factorized.py --config $CFG_CENTROID --mode train --gpus 1"

echo ""
echo "=== 6) Baseline: MM-BERT StickerChat (main_mmbbert_yaml.py) ==="
echo "$PY main_mmbbert_yaml.py --config $CFG_MMBERT --mode train --gpus 1"

echo ""
echo "Done printing commands. Asset summary: stickerchat/processed_style_regroup_v2_aggressive/summary.json"
