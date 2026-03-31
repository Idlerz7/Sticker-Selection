#!/usr/bin/env python3
"""
Data-side evaluation for CLIP-centroid regroup outputs (before expensive training).

Answers: "Is this regroup too aggressive / inconsistent?" using only manageable JSON
(small analysis/summary, metadata, img_set_to_ids, optional style_neighbors).

Does NOT load multi-GB factorized_style_bank.json fully — only peeks meta.* from the file head.

Usage:
  python scripts/evaluate_stickerchat_regroup_assets.py \\
    --regroup-dir ./stickerchat/processed_style_regroup_v2_aggressive \\
    --original-metadata ./stickerchat/processed/sticker_metadata.json

  python scripts/evaluate_stickerchat_regroup_assets.py \\
    --regroup-dir ./stickerchat/processed_style_regroup_centroid \\
    --original-metadata ./stickerchat/processed/sticker_metadata.json \\
    --include-neighbors \\
    --neighbor-baseline ./stickerchat/processed/style_neighbors.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Reuse audit helpers (metadata / neighbors summaries).
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (REPO_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import audit_stickerchat_factorized_data as _audit  # noqa: E402

audit_metadata = _audit.audit_metadata
audit_neighbors = _audit.audit_neighbors
_load_json = _audit._load_json
_summarize_sizes = _audit._summarize_sizes


def _read_text_prefix(path: Path, max_bytes: int = 131072) -> str:
    with path.open("rb") as f:
        return f.read(max_bytes).decode("utf-8", errors="replace")


def peek_factorized_bank_meta(bank_path: Path) -> Dict[str, Any]:
    """Parse meta fields from the start of a large factorized_style_bank.json without full load."""
    head = _read_text_prefix(bank_path, max_bytes=262144)
    out: Dict[str, Any] = {"bank_path": str(bank_path), "peek_ok": False}
    m = re.search(r'"num_prototypes"\s*:\s*(\d+)', head)
    if m:
        out["num_prototypes"] = int(m.group(1))
    m = re.search(r'"num_missing_style_metadata_ids"\s*:\s*(\d+)', head)
    if m:
        out["num_missing_style_metadata_ids"] = int(m.group(1))
    m = re.search(r'"max_image_id"\s*:\s*(\d+)', head)
    if m:
        out["max_image_id"] = int(m.group(1))
    out["peek_ok"] = "num_prototypes" in out
    return out


def load_id_to_img_set(metadata_path: Path) -> Dict[int, str]:
    raw = _load_json(metadata_path)
    rows = raw.get("stickers", []) if isinstance(raw, dict) else raw
    out: Dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = row.get("internal_img_id", row.get("sticker_id", row.get("id")))
        if sid is None:
            continue
        try:
            iid = int(sid)
        except (TypeError, ValueError):
            continue
        out[iid] = str(row.get("img_set", "")).strip() or "missing"
    return out


def merge_purity_vs_original_packs(
    original_id_to_pack: Dict[int, str],
    img_set_to_ids: Dict[str, List[int]],
) -> Dict[str, Any]:
    """
    For each regrouped style key, count how many distinct *original* img_set packs it contains.
    High values => many original packs were merged into one visual group.
    """
    n_orig_per_group: List[int] = []
    merged_groups = 0
    sticker_ids_seen: Set[int] = set()
    missing_orig = 0
    for _gkey, ids in img_set_to_ids.items():
        if not isinstance(ids, list):
            continue
        packs: Set[str] = set()
        for sid in ids:
            try:
                iid = int(sid)
            except (TypeError, ValueError):
                continue
            sticker_ids_seen.add(iid)
            p = original_id_to_pack.get(iid)
            if p is None:
                missing_orig += 1
                continue
            packs.add(p)
        nu = len(packs)
        n_orig_per_group.append(nu)
        if nu > 1:
            merged_groups += 1

    s = sorted(n_orig_per_group) if n_orig_per_group else []
    return {
        "num_final_style_keys": len(img_set_to_ids),
        "sticker_ids_in_img_set_to_ids": len(sticker_ids_seen),
        "groups_spanning_more_than_one_original_pack": merged_groups,
        "distinct_original_packs_per_final_group": _summarize_sizes(s),
        "missing_original_pack_for_sticker_rows": missing_orig,
    }


def neighbor_id_lists(
    neighbors_path: Path,
) -> Tuple[Dict[int, List[int]], int]:
    raw = _load_json(neighbors_path)
    rows = raw.get("neighbors", []) if isinstance(raw, dict) else raw
    out: Dict[int, List[int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = row.get("id")
        if sid is None:
            continue
        try:
            iid = int(sid)
        except (TypeError, ValueError):
            continue
        nbs: List[int] = []
        for nb in row.get("neighbors") or []:
            if isinstance(nb, dict):
                nid = nb.get("id")
            else:
                nid = nb
            if nid is None:
                continue
            try:
                nbs.append(int(nid))
            except (TypeError, ValueError):
                continue
        out[iid] = nbs
    return out, len(rows)


def jaccard_topk_overlap_sample(
    a_map: Dict[int, List[int]],
    b_map: Dict[int, List[int]],
    sample_n: int,
    k: int,
    seed: int,
) -> Dict[str, Any]:
    common = [i for i in a_map if i in b_map and a_map[i] and b_map[i]]
    if not common:
        return {"error": "no overlapping ids with nonempty neighbors"}
    rng = random.Random(seed)
    sample_n = min(sample_n, len(common))
    picked = rng.sample(common, sample_n)
    overlaps: List[float] = []
    for iid in picked:
        sa = set(a_map[iid][:k])
        sb = set(b_map[iid][:k])
        inter = len(sa & sb)
        union = len(sa | sb) or 1
        overlaps.append(inter / union)
    return {
        "sample_size": sample_n,
        "k": k,
        "mean_jaccard_topk": float(statistics.mean(overlaps)) if overlaps else 0.0,
        "p50_jaccard": float(statistics.median(overlaps)) if overlaps else 0.0,
        "note": "Compares first-k neighbor ids only (CLIP kNN order may shift after regroup).",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--regroup-dir",
        type=Path,
        default=Path("./stickerchat/processed_style_regroup_v2_aggressive"),
        help="Directory containing style_regroup_analysis.json, summary.json, etc.",
    )
    p.add_argument(
        "--original-metadata",
        type=Path,
        default=Path("./stickerchat/processed/sticker_metadata.json"),
        help="Original (pre-regroup) sticker_metadata.json for pack purity stats.",
    )
    p.add_argument(
        "--max-image-id",
        type=int,
        default=174695,
    )
    p.add_argument(
        "--include-neighbors",
        action="store_true",
        help="Load regroup style_neighbors.json (~hundreds of MB) and summarize.",
    )
    p.add_argument(
        "--neighbor-baseline",
        type=Path,
        default=None,
        help="Optional baseline style_neighbors.json for sampled Jaccard overlap vs regroup.",
    )
    p.add_argument(
        "--neighbor-overlap-sample",
        type=int,
        default=2000,
        help="Sample size for Jaccard overlap (requires --neighbor-baseline).",
    )
    p.add_argument(
        "--neighbor-overlap-k",
        type=int,
        default=32,
        help="Top-k neighbor ids to compare for Jaccard.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d: Path = args.regroup_dir
    max_id = int(args.max_image_id)

    analysis_path = d / "style_regroup_analysis.json"
    summary_path = d / "summary.json"
    meta_path = d / "sticker_metadata.json"
    bank_path = d / "factorized_style_bank.json"
    group_path = d / "img_set_to_ids.json"
    nb_path = d / "style_neighbors.json"

    print("=== Regroup directory ===", json.dumps({"path": str(d)}, ensure_ascii=False))

    if analysis_path.exists():
        raw_a = _load_json(analysis_path)
        _OMIT = frozenset({"top_merge_edges", "top_neighbors_by_pack"})
        slim = {k: v for k, v in raw_a.items() if k not in _OMIT}
        print("\n=== style_regroup_analysis.json (summary; large lists omitted) ===")
        print(json.dumps(slim, indent=2, ensure_ascii=False))
    else:
        print(f"\n(missing {analysis_path})", file=sys.stderr)

    if summary_path.exists():
        print("\n=== summary.json ===")
        print(json.dumps(_load_json(summary_path), indent=2, ensure_ascii=False))
    else:
        print(f"\n(missing {summary_path})", file=sys.stderr)

    if meta_path.exists():
        print("\n=== Regroup sticker_metadata (audit) ===")
        summary_m, _pack_c, _miss = audit_metadata(meta_path, max_id)
        print(json.dumps(summary_m, indent=2, ensure_ascii=False))
    else:
        print(f"\n(missing {meta_path})", file=sys.stderr)

    if group_path.exists() and args.original_metadata.exists():
        print("\n=== Merge purity vs original packs ===")
        orig_map = load_id_to_img_set(args.original_metadata)
        groups = _load_json(group_path)
        if not isinstance(groups, dict):
            groups = {}
        purity = merge_purity_vs_original_packs(orig_map, groups)
        print(json.dumps(purity, indent=2, ensure_ascii=False))
        nu = purity["distinct_original_packs_per_final_group"]
        p99 = nu.get("p99", float("nan"))
        mx = nu.get("max", float("nan"))
        print("\n--- Interpretation (heuristic) ---")
        print(
            "  If 'groups_spanning_more_than_one_original_pack' is large, many visual merges "
            "combined different product packs — expected after aggressive regroup; check max/p99 "
            "of distinct_original_packs_per_final_group for 'too wild' merges."
        )
        if isinstance(mx, (int, float)) and mx > 500:
            print(
                f"  WARNING: at least one final group spans up to {mx:.0f} original packs — "
                "style/proto supervision may become very coarse."
            )
    else:
        print(
            "\n(skipping merge purity: need img_set_to_ids.json + --original-metadata)",
            file=sys.stderr,
        )

    if bank_path.exists():
        print("\n=== factorized_style_bank.json (meta peek only) ===")
        print(json.dumps(peek_factorized_bank_meta(bank_path), indent=2, ensure_ascii=False))
    else:
        print(f"\n(missing {bank_path})", file=sys.stderr)

    if args.include_neighbors and nb_path.exists():
        print("\n=== Regroup style_neighbors.json ===")
        print(json.dumps(audit_neighbors(nb_path, max_id), indent=2, ensure_ascii=False))

    if args.neighbor_baseline and args.neighbor_baseline.exists() and nb_path.exists():
        print("\n=== Neighbor list churn vs baseline (sampled Jaccard) ===")
        print("Loading two neighbor JSONs; may take tens of seconds and use ~0.5–1GB RAM ...")
        a_map, _ = neighbor_id_lists(args.neighbor_baseline)
        b_map, _ = neighbor_id_lists(nb_path)
        rep = jaccard_topk_overlap_sample(
            a_map,
            b_map,
            sample_n=int(args.neighbor_overlap_sample),
            k=int(args.neighbor_overlap_k),
            seed=42,
        )
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        print(
            "\n  Low mean_jaccard is expected when regroup changes within-style kNN pools; "
            "if it is near-zero, structured heads trained on new neighbors differ a lot from "
            "legacy-processed behavior."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
