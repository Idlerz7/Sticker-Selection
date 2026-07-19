#!/usr/bin/env python3
"""
Join baseline vs full-model test outputs and build the primary case-study pool.

Expects two `*_structured_pred.json` files from dual_test_for_case_study.py (same
ordering as rows in --val_json).

Primary pool (default): baseline top-1 != GT and full model top-1 == GT.

Optional stratified sampling: at most one case per dialogue_id (see criteria.yaml).

Example:
  python scripts/case_study/mine_case_study_pool.py \\
    --val_json data/validation_pair_with_cand.json \\
    --baseline_pred result/case_study_runs/run01/baseline_mmbert/validation_pair_with_cand_True_structured_pred.json \\
    --full_pred result/case_study_runs/run01/full_model/validation_pair_with_cand_True_structured_pred.json \\
    --output_dir result/case_study_mined \\
    --sample_k 6
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dialogue_id_from_row(row: Dict[str, Any]) -> str:
    return str(row.get("dialogue_id") or row.get("user_id", ""))


def _tag_error_kind(
    *,
    gt: int,
    base_top1: int,
    full_top1: int,
    cand: List[int],
) -> str:
    """Lightweight tag for stratified reporting (no external model)."""
    if base_top1 == full_top1:
        return "no_disagreement"
    if gt not in cand:
        return "gt_not_in_cand"
    try:
        gi = cand.index(gt)
        bi = cand.index(base_top1)
    except ValueError:
        return "rank_unknown"
    if bi < gi:
        return "baseline_ranked_above_gt"
    return "baseline_ranked_below_gt"


def stratified_sample(
    pool: List[Dict[str, Any]],
    k: int,
    seed: int,
    max_per_dialogue: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    picked: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)
    for item in shuffled:
        if len(picked) >= k:
            break
        d = str(item.get("dialogue_id", ""))
        if counts[d] >= max_per_dialogue:
            continue
        picked.append(item)
        counts[d] += 1
    if len(picked) < k:
        seen = {x["index"] for x in picked}
        for item in shuffled:
            if len(picked) >= k:
                break
            if item["index"] in seen:
                continue
            picked.append(item)
            seen.add(item["index"])
    return picked[:k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val_json", required=True, help="Same file used for test_data_path.")
    ap.add_argument("--baseline_pred", required=True, help="structured_pred.json (base_only=true run).")
    ap.add_argument("--full_pred", required=True, help="structured_pred.json (base_only=false run).")
    ap.add_argument("--output_dir", default="result/case_study_mined", help="Writes pool + sampled JSON.")
    ap.add_argument("--sample_k", type=int, default=6, help="Stratified sample size for paper.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_dialogue", type=int, default=1)
    args = ap.parse_args()

    val_path = Path(args.val_json)
    rows = _load_json(val_path)
    if not isinstance(rows, list):
        raise SystemExit("val_json must be a JSON array.")

    base_preds = _load_json(Path(args.baseline_pred))
    full_preds = _load_json(Path(args.full_pred))
    if not isinstance(base_preds, list) or not isinstance(full_preds, list):
        raise SystemExit("pred files must be JSON arrays.")
    n = len(base_preds)
    if len(full_preds) != n or len(rows) != n:
        raise SystemExit(
            f"Length mismatch: val={len(rows)} baseline_pred={len(base_preds)} full_pred={len(full_preds)}"
        )

    primary: List[Dict[str, Any]] = []
    secondary: Dict[str, List[Dict[str, Any]]] = {
        "both_correct": [],
        "both_wrong": [],
        "ours_wrong_baseline_right": [],
    }

    for i in range(n):
        row = rows[i]
        bp = base_preds[i]
        fp = full_preds[i]
        gt = int(bp["answer"])
        if int(fp["answer"]) != gt:
            raise SystemExit(f"Row {i}: answer mismatch between pred files.")
        b_top = int(bp["pred"][0]) if bp.get("pred") else -1
        f_top = int(fp["pred"][0]) if fp.get("pred") else -1
        cand = list(row.get("cand") or [])
        meta = {
            "index": i,
            "dialogue_id": _dialogue_id_from_row(row),
            "user_id": row.get("user_id", ""),
            "gt": gt,
            "baseline_top1": b_top,
            "full_top1": f_top,
            "tag": _tag_error_kind(gt=gt, base_top1=b_top, full_top1=f_top, cand=cand),
        }
        if b_top != gt and f_top == gt:
            primary.append(meta)
        if b_top == gt and f_top == gt:
            secondary["both_correct"].append(meta)
        elif b_top != gt and f_top != gt:
            secondary["both_wrong"].append(meta)
        elif b_top == gt and f_top != gt:
            secondary["ours_wrong_baseline_right"].append(meta)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "primary_pool.jsonl"
    with open(pool_path, "w", encoding="utf-8") as f:
        for item in primary:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    sampled = stratified_sample(
        primary,
        k=int(args.sample_k),
        seed=int(args.seed),
        max_per_dialogue=int(args.max_per_dialogue),
    )
    sampled_path = out_dir / "sampled_cases.json"
    with open(sampled_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "criteria": "baseline_top1!=gt and full_top1==gt",
                "pool_size": len(primary),
                "sample_k": args.sample_k,
                "seed": args.seed,
                "sampled_indices": [x["index"] for x in sampled],
                "sampled": sampled,
                "secondary_counts": {k: len(v) for k, v in secondary.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    sec_path = out_dir / "secondary_pools.json"
    with open(sec_path, "w", encoding="utf-8") as f:
        json.dump(secondary, f, ensure_ascii=False, indent=2)

    print(
        f"[mine_case_study_pool] val_rows={len(rows)} primary_pool={len(primary)} "
        f"sampled={len(sampled)}\n"
        f"  wrote {pool_path}\n"
        f"  wrote {sampled_path}\n"
        f"  wrote {sec_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
