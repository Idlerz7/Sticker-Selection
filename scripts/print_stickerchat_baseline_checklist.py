#!/usr/bin/env python3
"""
Print a comparability checklist for StickerChat MMBERT vs factorized minimal runs.

Use when reporting numbers (e.g. MMBERT R10 MRR 87 / R20 MRR 79): confirm the baseline
used the same test JSONs, batch size, and training data pipeline as the factorized run.

Does not import project training code; only reads YAML paths from configs.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _read_key_paths(config_path: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    keys = (
        "train_data_path",
        "test_data_path",
        "per_epoch_eval_test_r10_path",
        "per_epoch_eval_test_r20_path",
        "style_neighbors_path",
        "factorized_bank_path",
        "factorized_style_metadata_path",
        "img_emb_cache_path",
        "pl_root_dir",
        "base_only",
        "name",
    )
    print(f"\n=== {config_path} ===")
    for line in text.splitlines():
        stripped = line.strip()
        for k in keys:
            if stripped.startswith(f"{k}:"):
                print(line.rstrip())


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "configs" / "structured_factorized"
    paths = [
        cfg_dir / "stickerchat_base.yaml",
        cfg_dir / "stickerchat_v6_minimal_core.yaml",
        cfg_dir / "stickerchat_v6_base_only.yaml",
        cfg_dir / "stickerchat_v6_minimal_regroup_centroid.yaml",
        cfg_dir / "stickerchat_v6_base_only_regroup_centroid.yaml",
        cfg_dir / "stickerchat_v6_minimal_regroup_centroid_eval_legacy_test.yaml",
    ]
    print(
        "StickerChat baseline / factorized comparability checklist\n"
        "----------------------------------------------------------\n"
        "1) Same legacy test files for R10/R20:\n"
        "   ./stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json\n"
        "   ./stickerchat/processed/release_test_u_sticker_format_int_with_cand_r20.json\n"
        "2) MMBERT-only ablation in this repo: stickerchat_v6_base_only.yaml (base_only: true)\n"
        "   on processed train (see stickerchat_base extends chain).\n"
        "3) Regroup factorized trains on processed_style_regroup_centroid; legacy test is a\n"
        "   distribution shift unless you train with eval_legacy_test config.\n"
        "4) Record: epochs, seed, valtest_batch_size, gradient_accumulation_steps, gpus.\n"
    )
    for p in paths:
        if p.is_file():
            _read_key_paths(p)
        else:
            print(f"\n(missing) {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
