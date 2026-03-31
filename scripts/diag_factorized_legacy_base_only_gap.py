#!/usr/bin/env python3
"""
Run legacy stickerchat/processed R10/R20 test for each checkpoint twice:
  base_only=false (full minimal score) vs base_only=true (MMBERT branch only).

Parses [StructuredEval] lines from subprocess stdout/stderr and writes a TSV table
to quantify how much expr/proto additive terms change MRR vs the same checkpoint.

Example:
  cd /path/to/Sticker-Selection
  CUDA_VISIBLE_DEVICES=0 python scripts/diag_factorized_legacy_base_only_gap.py \\
    --config configs/structured_factorized/stickerchat_v6_minimal_regroup_centroid.yaml \\
    --ckpt-dir logs/.../version_1/checkpoints

See plan: Factorized vs MMBERT gap (P0 diagnostic).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STRUCTURED_EVAL_RE = re.compile(
    r"\[StructuredEval\].*?r@1=(?P<r1>[0-9.]+).*?mrr=(?P<mrr>[0-9.]+)",
    re.DOTALL,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_structured_eval_last_block(text: str) -> Optional[Dict[str, float]]:
    """Take last [StructuredEval] line (may span if logger broke lines)."""
    lines = []
    for line in text.splitlines():
        if "[StructuredEval]" in line:
            lines = [line]
        elif lines and ("mrr=" in line or "r@1=" in line):
            lines.append(line)
    if not lines:
        return None
    block = " ".join(lines)
    m = STRUCTURED_EVAL_RE.search(block.replace("\n", " "))
    if not m:
        # Fallback: find mrr= on the last line containing StructuredEval
        for seg in reversed(text.split("[StructuredEval]")):
            if "mrr=" in seg:
                m2 = re.search(r"r@1=([0-9.]+).*?mrr=([0-9.]+)", seg.replace("\n", " "))
                if m2:
                    return {"r1": float(m2.group(1)), "mrr": float(m2.group(2))}
                break
        return None
    return {"r1": float(m.group("r1")), "mrr": float(m.group("mrr"))}


def _run_one_test(
    *,
    project_root: Path,
    config: str,
    ckpt: str,
    test_path: str,
    base_only: bool,
    gpus: int,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    cmd = [
        sys.executable,
        str(project_root / "main_structured_factorized.py"),
        "--config",
        config,
        "--mode",
        "test",
        "--gpus",
        str(gpus),
        "--ckpt_path",
        ckpt,
        "--test_data_path",
        test_path,
        "--base_only",
        "true" if base_only else "false",
    ]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        env=env,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        return out, f"exit_code={proc.returncode}"
    return out, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="configs/structured_factorized/stickerchat_v6_minimal_regroup_centroid.yaml",
        help="Factorized config (must match training assets / lambdas).",
    )
    ap.add_argument("--ckpt-dir", type=str, default="", help="Directory of *.ckpt files.")
    ap.add_argument(
        "--ckpts",
        nargs="*",
        default=[],
        help="Explicit checkpoint paths (overrides --ckpt-dir glob).",
    )
    ap.add_argument(
        "--r10",
        default="./stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json",
    )
    ap.add_argument(
        "--r20",
        default="./stickerchat/processed/release_test_u_sticker_format_int_with_cand_r20.json",
    )
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument(
        "--out-tsv",
        type=str,
        default="",
        help="Write results here (default: diag_base_only_gap_<timestamp>.tsv under cwd).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not run tests.",
    )
    ap.add_argument(
        "--max-checkpoints",
        type=int,
        default=0,
        help="If >0, only first N checkpoints (sorted by name).",
    )
    args = ap.parse_args()

    root = _project_root()
    cfg = args.config
    if args.ckpts:
        ckpts = [Path(p).resolve() for p in args.ckpts]
    else:
        if not args.ckpt_dir:
            ap.error("Provide --ckpt-dir or --ckpts")
        d = Path(args.ckpt_dir)
        ckpts = sorted(d.glob("*.ckpt"))
        if args.max_checkpoints > 0:
            ckpts = ckpts[: args.max_checkpoints]
    if not ckpts:
        print("No checkpoints found.", file=sys.stderr)
        sys.exit(1)

    r10 = args.r10
    r20 = args.r20
    rows: List[Dict[str, object]] = []

    splits = [("R10", r10), ("R20", r20)]

    for ckpt_path in ckpts:
        ckpt_str = str(ckpt_path)
        short = ckpt_path.name
        for split_name, test_p in splits:
            for base_only in (False, True):
                mode = "base_only" if base_only else "full"
                if args.dry_run:
                    print(
                        f"# {short} {split_name} {mode}\n"
                        f"python main_structured_factorized.py --config {cfg} "
                        f"--mode test --gpus {args.gpus} --ckpt_path {ckpt_str} "
                        f"--test_data_path {test_p} --base_only {'true' if base_only else 'false'}"
                    )
                    continue
                out, err = _run_one_test(
                    project_root=root,
                    config=cfg,
                    ckpt=ckpt_str,
                    test_path=test_p,
                    base_only=base_only,
                    gpus=args.gpus,
                )
                metrics = _parse_structured_eval_last_block(out)
                row = {
                    "checkpoint": short,
                    "split": split_name,
                    "base_only": base_only,
                    "r1": metrics["r1"] if metrics else None,
                    "mrr": metrics["mrr"] if metrics else None,
                    "error": err or ("" if metrics else "parse_failed"),
                }
                rows.append(row)
                tag = "ok" if metrics and not err else "BAD"
                print(
                    f"[{tag}] {short} {split_name} base_only={base_only} "
                    f"mrr={row['mrr']} {err}",
                    flush=True,
                )

    if args.dry_run:
        return

    out_path = args.out_tsv
    if not out_path:
        from datetime import datetime

        out_path = f"diag_base_only_gap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"
    out_p = Path(out_path)
    with out_p.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["checkpoint", "split", "base_only", "r1", "mrr", "error"],
            delimiter="\t",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_p.resolve()}")

    # Per-checkpoint delta: full_mrr - base_only_mrr (positive => additive terms hurt)
    by_ck: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    for r in rows:
        key = (str(r["checkpoint"]), str(r["split"]))
        by_ck.setdefault(key, {})
        m = r["mrr"]
        if r["base_only"] is False:
            by_ck[key]["full_mrr"] = float(m) if m is not None else None
        else:
            by_ck[key]["base_mrr"] = float(m) if m is not None else None
    print("\n# Delta (full - base_only MRR); positive => structured terms reduced MRR vs MMBERT-only scoring")
    for (ck, sp), d in sorted(by_ck.items()):
        fm, bm = d.get("full_mrr"), d.get("base_mrr")
        if fm is not None and bm is not None:
            print(f"{ck}\t{sp}\tdelta_mrr={fm - bm:+.4f}\tfull={fm:.4f}\tbase_only={bm:.4f}")


if __name__ == "__main__":
    main()
