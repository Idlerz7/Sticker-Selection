#!/usr/bin/env bash
# P1: Example training commands for lambda_expr / lambda_style_proto variants on regroup data.
# Edit CUDA_VISIBLE_DEVICES and --gpus to match your cluster.
#
# Configs:
#   - stickerchat_v6_minimal_regroup_centroid.yaml           -> λ_expr=0.3, λ_proto=0.4 (default)
#   - stickerchat_v6_low_proto_low_expr_regroup_centroid.yaml -> λ_expr=0.2, λ_proto=0.1 (conservative)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPUS="${1:-8}"

echo "# Full lambda (current default)"
echo "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_structured_factorized.py \\"
echo "  --config configs/structured_factorized/stickerchat_v6_minimal_regroup_centroid.yaml \\"
echo "  --gpus $GPUS"
echo
echo "# Low proto / expr (smaller additive terms at eval)"
echo "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_structured_factorized.py \\"
echo "  --config configs/structured_factorized/stickerchat_v6_low_proto_low_expr_regroup_centroid.yaml \\"
echo "  --gpus $GPUS"
