#!/usr/bin/env python3
"""
Run `main_structured_factorized.py --mode test` twice with the same config/checkpoint:
  - baseline (MMBERT-only): --base_only true  -> structured_result_dir A
  - full model:             --base_only false -> structured_result_dir B

Writes two `*_structured_pred.json` files used by mine_case_study_pool.py.

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/case_study/dual_test_for_case_study.py \\
    --config configs/structured_factorized/stickerchat_v6_minimal_core.yaml \\
    --ckpt_path logs/.../last.ckpt \\
    --test_data_path data/validation_pair_with_cand.json \\
    --output_root result/case_study_runs/run01 \\
    --gpus 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML config (same as training).")
    ap.add_argument("--ckpt_path", required=True, help="Checkpoint .ckpt path.")
    ap.add_argument("--test_data_path", required=True, help="Validation JSON with cand[].")
    ap.add_argument(
        "--output_root",
        default="result/case_study_dual_test",
        help="Parent dir; creates baseline_mmbert/ and full_model/ under it.",
    )
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--extra", nargs="*", default=[], help="Extra argv passed through to main_structured_factorized.py.")
    args = ap.parse_args()

    root = _project_root()
    out = Path(args.output_root)
    base_dir = out / "baseline_mmbert"
    full_dir = out / "full_model"
    base_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)

    def run_one(*, structured_result_dir: Path, base_only: bool) -> None:
        cmd = [
            sys.executable,
            str(root / "main_structured_factorized.py"),
            "--config",
            args.config,
            "--mode",
            "test",
            "--gpus",
            str(args.gpus),
            "--ckpt_path",
            args.ckpt_path,
            "--test_data_path",
            args.test_data_path,
            "--structured_result_dir",
            str(structured_result_dir),
            "--base_only",
            "true" if base_only else "false",
        ]
        cmd.extend(args.extra)
        env = os.environ.copy()
        print("[dual_test_for_case_study] Running:", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(root), env=env)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    run_one(structured_result_dir=base_dir, base_only=True)
    run_one(structured_result_dir=full_dir, base_only=False)
    print(
        "[dual_test_for_case_study] Done.\n"
        f"  baseline preds dir: {base_dir}\n"
        f"  full model preds dir: {full_dir}\n"
        "  Next: mine_case_study_pool.py --baseline_pred ... --full_pred ...",
        flush=True,
    )


if __name__ == "__main__":
    main()
