#!/usr/bin/env bash
# Train regroup-centroid minimal core while per-epoch validation uses legacy
# stickerchat/processed/ R10+R20 (same benchmark as historical MMBERT runs).
# P2 data-align / model selection from the Factorized vs MMBERT gap plan.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_regroup_with_legacy_eval.sh 8

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPUS="${1:-8}"

exec python main_structured_factorized.py \
  --config configs/structured_factorized/stickerchat_v6_minimal_regroup_centroid_eval_legacy_test.yaml \
  --gpus "$GPUS"
