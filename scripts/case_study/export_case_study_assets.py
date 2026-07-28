#!/usr/bin/env python3
"""
For sampled case-study indices (from mine_case_study_pool.py):
  1) Dump per-candidate MMBERT vs fused scores (score_breakdown from forward_eval_batch).
  2) Write paper_table.csv (dialogue snippet, GT, baseline@1, full@1, ranks).
  3) Optional matplotlib figure grids of candidate stickers (requires id2img + img_dir).

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/case_study/export_case_study_assets.py \\
    --config configs/structured_factorized/stickerchat_v6_minimal_core.yaml \\
    --ckpt_path logs/.../last.ckpt \\
    --val_json data/validation_pair_with_cand.json \\
    --sampled_json result/case_study_mined/sampled_cases.json \\
    --output_dir result/case_study_export \\
    --figure
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from main import PLDataLoader
from structured_retrieval import load_checkpoint_to_model
from structured_retrieval_factorized import (
    StructuredFactorizedPLModel,
    parse_structured_factorized_args,
)


def _rank_desc(scores: Sequence[float], i: int) -> int:
    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-arr)
    pos = int(np.where(order == i)[0][0])
    return pos + 1


def _dialogue_snippet(row: Dict[str, Any], max_chars: int = 240) -> str:
    parts: List[str] = []
    for turn in row.get("dialog") or []:
        t = str(turn.get("text", "")).replace("\n", " ")
        parts.append(t)
    s = " | ".join(parts)
    if len(s) > max_chars:
        return s[: max_chars - 3] + "..."
    return s


def _resolve_img_path(img_dir: str, rel: str, project_root: Path) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    joined = Path(img_dir) / rel
    if joined.exists():
        return joined
    alt = project_root / joined
    if alt.exists():
        return alt
    return joined


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--val_json", required=True)
    ap.add_argument("--sampled_json", required=True, help="sampled_cases.json")
    ap.add_argument("--output_dir", default="result/case_study_export")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--figure", action="store_true", help="Write PNG grids per case.")
    ap.add_argument(
        "--extra_indices",
        type=int,
        nargs="*",
        default=[],
        help="Optional extra row indices (e.g. one failure case from secondary pool).",
    )
    args = ap.parse_args()

    argv = [
        "--config",
        args.config,
        "--mode",
        "test",
        "--ckpt_path",
        args.ckpt_path,
        "--test_data_path",
        args.val_json,
    ]
    parsed = parse_structured_factorized_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = StructuredFactorizedPLModel(parsed)
    load_checkpoint_to_model(
        model, args.ckpt_path, strict=bool(getattr(parsed, "strict_checkpoint_load", False))
    )
    model.eval()
    model.to(device)
    model.model.prepare_for_test()

    pld = PLDataLoader(parsed, model.model.bert_tokenizer)
    pld.setup(stage="test")
    ds = pld.test_dataset

    with open(args.val_json, encoding="utf-8") as f:
        val_rows = json.load(f)
    with open(args.sampled_json, encoding="utf-8") as f:
        sampled_doc = json.load(f)
    indices: List[int] = list(sampled_doc.get("sampled_indices") or [])
    if not indices and sampled_doc.get("sampled"):
        indices = [int(x["index"]) for x in sampled_doc["sampled"]]
    for x in args.extra_indices:
        if x not in indices:
            indices.append(int(x))

    id2name: Dict[int, str] = {}
    with open(parsed.id2name_path, encoding="utf-8") as f:
        raw = json.load(f)
        for k, v in raw.items():
            id2name[int(k)] = str(v)

    id2img_rel: Dict[int, str] = {}
    with open(parsed.id2img_path, encoding="utf-8") as f:
        raw = json.load(f)
        for k, v in raw.items():
            id2img_rel[int(k)] = str(v)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "case_score_breakdown.jsonl"
    csv_path = out_dir / "paper_table.csv"

    table_rows: List[Dict[str, Any]] = []

    if args.figure:
        import matplotlib.pyplot as plt
        from PIL import Image

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for idx in indices:
            row = val_rows[idx]
            raw = ds[idx]
            gt = int(raw["img_id"])
            batch = pld.collate_fn([raw])
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            with torch.no_grad():
                _, _labels, cands, dbg = model.model.forward_eval_batch(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["img_ids"],
                    batch.get("cands"),
                    return_debug=True,
                    score_breakdown=True,
                )
            if cands is None:
                raise SystemExit(f"Index {idx}: missing candidates.")
            cand_ids = [int(x) for x in cands]
            if gt not in cand_ids:
                raise SystemExit(
                    f"Index {idx}: gt sticker {gt} not in cand list (data/candidate mismatch)."
                )

            mmb = dbg["mmbert_score_per_cand"]
            fin = dbg["final_score_per_cand"]
            gi = cand_ids.index(gt)
            rank_m = _rank_desc(mmb, gi)
            rank_f = _rank_desc(fin, gi)
            m_top = cand_ids[int(np.argmax(mmb))]
            f_top = cand_ids[int(np.argmax(fin))]

            record = {
                "index": idx,
                "dialogue_id": row.get("dialogue_id", ""),
                "candidate_ids": cand_ids,
                "gt": gt,
                "mmbert_top1": m_top,
                "fused_top1": f_top,
                "gt_rank_mmbert": rank_m,
                "gt_rank_fused": rank_f,
                "mmbert_score_per_cand": mmb,
                "final_score_per_cand": fin,
            }
            if "expr_score_per_cand" in dbg:
                record["expr_score_per_cand"] = dbg["expr_score_per_cand"]
            if "graph_score_per_cand" in dbg:
                record["graph_score_per_cand"] = dbg["graph_score_per_cand"]
            if "style_score_per_cand" in dbg:
                record["style_score_per_cand"] = dbg["style_score_per_cand"]
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

            table_rows.append(
                {
                    "index": idx,
                    "dialogue_id": str(row.get("dialogue_id", "")),
                    "dialogue_snippet": _dialogue_snippet(row),
                    "gt_id": gt,
                    "gt_label": id2name.get(gt, ""),
                    "mmbert_top1_id": m_top,
                    "mmbert_top1_label": id2name.get(m_top, ""),
                    "fused_top1_id": f_top,
                    "fused_top1_label": id2name.get(f_top, ""),
                    "gt_rank_mmbert": rank_m,
                    "gt_rank_fused": rank_f,
                }
            )

            if args.figure:
                n = len(cand_ids)
                cols = min(5, n)
                rows_g = int(np.ceil(n / cols))
                fig, axes = plt.subplots(rows_g, cols, figsize=(2.2 * cols, 2.4 * rows_g))
                axes_arr = np.atleast_1d(axes).ravel()
                img_root = str(getattr(parsed, "img_dir", "") or "")
                for i, sid in enumerate(cand_ids):
                    ax = axes_arr[i]
                    rel = id2img_rel.get(sid, "")
                    pth = _resolve_img_path(img_root, rel, _PROJECT_ROOT)
                    try:
                        im = Image.open(pth).convert("RGBA")
                        ax.imshow(im)
                    except Exception:
                        ax.text(0.5, 0.5, f"missing\n{sid}", ha="center", va="center")
                    ax.axis("off")
                    tag = id2name.get(sid, str(sid))
                    title = tag[:18] + ("…" if len(tag) > 18 else "")
                    mark = []
                    if sid == gt:
                        mark.append("GT")
                    if sid == m_top:
                        mark.append("MMBERT@1")
                    if sid == f_top:
                        mark.append("Fused@1")
                    extra = " [" + ", ".join(mark) + "]" if mark else ""
                    ax.set_title(f"{sid}{extra}", fontsize=7)
                for j in range(i + 1, len(axes_arr)):
                    axes_arr[j].axis("off")
                fig.suptitle(f"case idx={idx} dialogue={row.get('dialogue_id', '')}", fontsize=9)
                fig.tight_layout()
                fig_path = out_dir / f"case_grid_idx{idx}.png"
                fig.savefig(fig_path, dpi=160)
                plt.close(fig)

    if table_rows:
        fieldnames = list(table_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(table_rows)

    print(
        f"[export_case_study_assets] wrote {jsonl_path}\n"
        f"  csv: {csv_path}\n"
        f"  figures: {'yes' if args.figure else 'no'} under {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
