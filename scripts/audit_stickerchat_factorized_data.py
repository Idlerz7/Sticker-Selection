#!/usr/bin/env python3
"""
Audit StickerChat metadata, factorized_style_bank.json, and optional train/test coverage.

Usage:
  python scripts/audit_stickerchat_factorized_data.py \\
    --max-image-id 174695 \\
    --metadata-path ./stickerchat/processed/sticker_metadata.json \\
    --bank-path ./stickerchat/processed/factorized_style_bank.json \\
    --train-path ./stickerchat/processed/release_train_u_sticker_format_int.json \\
    --test-path ./stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json

Omit paths that are not available; the script prints what it can.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _percentile(sorted_vals: List[int], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    k = (n - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def _summarize_sizes(sizes: List[int]) -> Dict[str, float]:
    if not sizes:
        return {"count": 0}
    s = sorted(sizes)
    return {
        "count": len(s),
        "min": float(s[0]),
        "max": float(s[-1]),
        "mean": float(statistics.mean(s)),
        "p50": _percentile(s, 50),
        "p90": _percentile(s, 90),
        "p99": _percentile(s, 99),
    }


def _collect_gold_img_ids(samples: List[Dict[str, Any]]) -> Set[int]:
    out: Set[int] = set()
    for sample in samples:
        dialog = sample.get("dialog") or []
        if not dialog:
            continue
        last = dialog[-1]
        if not isinstance(last, dict):
            continue
        raw = last.get("img_id")
        if raw is None:
            continue
        try:
            out.add(int(raw))
        except (TypeError, ValueError):
            continue
    return out


def audit_metadata(
    metadata_path: Path, max_image_id: int
) -> Tuple[Dict[str, Any], Counter, List[int]]:
    raw = _load_json(metadata_path)
    rows = raw.get("stickers", []) if isinstance(raw, dict) else raw
    present: Set[int] = set()
    img_set_by_id: Dict[int, str] = {}
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
        if iid < 0 or iid >= max_image_id:
            continue
        present.add(iid)
        img_set = str(row.get("img_set", "")).strip() or f"missing_{iid}"
        img_set_by_id[iid] = img_set

    missing_ids = [i for i in range(max_image_id) if i not in present]
    style_groups: Dict[str, List[int]] = {}
    for i in range(max_image_id):
        pack = img_set_by_id.get(i, f"missing_{i}")
        style_groups.setdefault(pack, []).append(i)

    singleton_packs = sum(1 for _k, v in style_groups.items() if len(v) == 1)
    sizes = [len(v) for v in style_groups.values()]

    summary = {
        "metadata_path": str(metadata_path),
        "max_image_id": max_image_id,
        "rows_in_file": len(rows),
        "ids_in_range_with_row": len(present),
        "coverage_in_range": len(present) / max_image_id if max_image_id else 0.0,
        "missing_id_count": len(missing_ids),
        "num_distinct_img_set": len(style_groups),
        "num_singleton_packs": singleton_packs,
        "img_set_size_distribution": _summarize_sizes(sizes),
    }
    pack_counter = Counter(img_set_by_id.get(i, f"missing_{i}") for i in range(max_image_id))
    return summary, pack_counter, missing_ids


def audit_bank(bank_path: Path) -> Dict[str, Any]:
    raw = _load_json(bank_path)
    meta = raw.get("meta", {})
    protos = raw.get("prototypes", [])
    records = raw.get("records", [])
    by_source = Counter()
    member_sizes: List[int] = []
    for p in protos:
        if not isinstance(p, dict):
            continue
        by_source[str(p.get("proto_source", "?"))] += 1
        mc = int(p.get("member_count", 0) or 0)
        if mc <= 0 and p.get("member_ids"):
            mc = len(p.get("member_ids") or [])
        member_sizes.append(mc)

    singleton_protos = sum(1 for m in member_sizes if m == 1)

    return {
        "bank_path": str(bank_path),
        "meta": meta,
        "num_prototypes": len(protos),
        "num_records": len(records),
        "proto_source_counts": dict(by_source),
        "singleton_prototype_count": singleton_protos,
        "member_count_distribution": _summarize_sizes(member_sizes),
    }


def audit_neighbors(neighbors_path: Path, max_image_id: int) -> Dict[str, Any]:
    raw = _load_json(neighbors_path)
    rows = raw.get("neighbors", []) if isinstance(raw, dict) else raw
    lens: List[int] = []
    zeros = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        nbs = row.get("neighbors") or []
        k = len(nbs)
        lens.append(k)
        if k == 0:
            zeros += 1
    covered = len(lens)
    return {
        "neighbors_path": str(neighbors_path),
        "rows": len(rows),
        "entries_with_neighbor_lists": covered,
        "empty_neighbor_lists": zeros,
        "neighbor_list_length_distribution": _summarize_sizes(lens),
        "max_image_id_hint": max_image_id,
    }


def coverage_vs_bank(
    gold_ids: Set[int],
    bank_path: Path,
    metadata_missing_ids: List[int],
) -> Dict[str, Any]:
    raw = _load_json(bank_path)
    records = raw.get("records", [])
    sticker_rows: Dict[int, Dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        sid = row.get("sticker_id")
        if sid is None:
            continue
        try:
            iid = int(sid)
        except (TypeError, ValueError):
            continue
        sticker_rows[iid] = row

    in_split_not_in_bank = [i for i in gold_ids if i not in sticker_rows]
    singleton_proto_gold = 0
    img_set_proto_gold = 0
    for iid in gold_ids:
        row = sticker_rows.get(iid)
        if row is None:
            continue
        mids = row.get("member_ids") or []
        if len(mids) <= 1:
            singleton_proto_gold += 1
        if str(row.get("proto_source", "")) == "img_set":
            img_set_proto_gold += 1

    return {
        "gold_sticker_ids_in_split": len(gold_ids),
        "gold_ids_missing_from_bank_records": len(in_split_not_in_bank),
        "gold_in_singleton_proto": singleton_proto_gold,
        "gold_with_proto_source_img_set": img_set_proto_gold,
        "metadata_missing_id_overlap_with_gold": len(set(metadata_missing_ids) & gold_ids),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-image-id", type=int, default=174695)
    p.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("stickerchat/processed/sticker_metadata.json"),
    )
    p.add_argument(
        "--bank-path",
        type=Path,
        default=Path("stickerchat/processed/factorized_style_bank.json"),
    )
    p.add_argument(
        "--neighbors-path",
        type=Path,
        default=Path("stickerchat/processed/style_neighbors.json"),
    )
    p.add_argument(
        "--train-path",
        type=Path,
        default=Path("stickerchat/processed/release_train_u_sticker_format_int.json"),
    )
    p.add_argument(
        "--test-path",
        type=Path,
        default=Path(
            "stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json"
        ),
    )
    p.add_argument(
        "--include-neighbors",
        action="store_true",
        help="Also summarize style_neighbors.json (can be large).",
    )
    p.add_argument(
        "--top-packs",
        type=int,
        default=15,
        help="Print top N img_set packs by member count.",
    )
    p.add_argument(
        "--neighbor-proto-consistency-sample",
        type=int,
        default=0,
        help="If >0, sample this many sticker ids and report share_same_proto with first neighbor.",
    )
    return p.parse_args()


def neighbor_proto_consistency_sample(
    bank_path: Path,
    neighbors_path: Path,
    sample_n: int,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    import random

    raw_bank = _load_json(bank_path)
    records = raw_bank.get("records", [])
    sticker_to_proto: Dict[int, int] = {}
    for row in records:
        if not isinstance(row, dict) or row.get("sticker_id") is None:
            continue
        try:
            sticker_to_proto[int(row["sticker_id"])] = int(row.get("proto_id", -1))
        except (TypeError, ValueError):
            continue

    raw_nb = _load_json(neighbors_path)
    rows = raw_nb.get("neighbors", []) if isinstance(raw_nb, dict) else raw_nb
    id_to_neighbors: Dict[int, List[int]] = {}
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
        id_to_neighbors[iid] = nbs

    rng = random.Random(rng_seed)
    pool = [i for i in id_to_neighbors if id_to_neighbors[i]]
    if not pool:
        return {"error": "no neighbor entries with nonempty lists"}
    sample_n = min(sample_n, len(pool))
    picked = rng.sample(pool, sample_n)
    same = 0
    checked = 0
    missing_proto = 0
    for iid in picked:
        nb0 = id_to_neighbors[iid][0]
        p0 = sticker_to_proto.get(iid)
        p1 = sticker_to_proto.get(nb0)
        if p0 is None or p1 is None:
            missing_proto += 1
            continue
        checked += 1
        if p0 == p1:
            same += 1
    rate = float(same) / float(checked) if checked else 0.0
    return {
        "sample_size": sample_n,
        "neighbors_checked_with_proto_pair": checked,
        "missing_proto_for_endpoint": missing_proto,
        "share_same_proto_as_first_neighbor": same,
        "rate_same_proto": rate,
        "note": "Low rate may be OK if neighbors are visual-kNN; high img_set fragmentation lowers this too.",
    }


def main() -> None:
    args = parse_args()
    max_id = int(args.max_image_id)
    missing_ids: List[int] = []

    if args.metadata_path.exists():
        meta_summary, pack_counter, missing_ids = audit_metadata(args.metadata_path, max_id)
        print("=== sticker_metadata ===")
        print(json.dumps(meta_summary, indent=2, ensure_ascii=False))
        if args.top_packs > 0:
            most_common = pack_counter.most_common(args.top_packs)
            print(f"\nTop {args.top_packs} img_set keys by sticker count (in 0..max_id-1):")
            for name, cnt in most_common:
                print(f"  {cnt:8d}  {name}")
    else:
        print(f"=== sticker_metadata (skipped: not found {args.metadata_path}) ===", file=sys.stderr)

    if args.bank_path.exists():
        print("\n=== factorized_style_bank.json ===")
        print(json.dumps(audit_bank(args.bank_path), indent=2, ensure_ascii=False))
    else:
        print(f"\n=== factorized_style_bank (skipped: not found {args.bank_path}) ===", file=sys.stderr)

    if args.include_neighbors:
        if args.neighbors_path.exists():
            print("\n=== style_neighbors.json ===")
            print(
                json.dumps(
                    audit_neighbors(args.neighbors_path, max_id),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(
                f"\n=== style_neighbors (skipped: not found {args.neighbors_path}) ===",
                file=sys.stderr,
            )

    train_gold: Set[int] = set()
    test_gold: Set[int] = set()
    if args.train_path.exists():
        train_gold = _collect_gold_img_ids(_load_json(args.train_path))
        print(f"\n=== train gold img_ids ===\nunique count: {len(train_gold)}")
    if args.test_path.exists():
        test_gold = _collect_gold_img_ids(_load_json(args.test_path))
        print(f"\n=== test gold img_ids ===\nunique count: {len(test_gold)}")

    if (
        int(args.neighbor_proto_consistency_sample) > 0
        and args.bank_path.exists()
        and args.neighbors_path.exists()
    ):
        print("\n=== neighbor vs bank proto (sampled) ===")
        print(
            json.dumps(
                neighbor_proto_consistency_sample(
                    args.bank_path,
                    args.neighbors_path,
                    int(args.neighbor_proto_consistency_sample),
                ),
                indent=2,
                ensure_ascii=False,
            )
        )

    if args.bank_path.exists() and (train_gold or test_gold):
        print("\n=== gold vs bank (train) ===")
        if train_gold:
            print(
                json.dumps(
                    coverage_vs_bank(train_gold, args.bank_path, missing_ids),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        print("\n=== gold vs bank (test) ===")
        if test_gold:
            print(
                json.dumps(
                    coverage_vs_bank(test_gold, args.bank_path, missing_ids),
                    indent=2,
                    ensure_ascii=False,
                )
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
