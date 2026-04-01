#!/usr/bin/env python3
"""
Build StickerChat style assets with **target K** groups via K-means on **pack CLIP centroids**.

This is separate from scripts/build_stickerchat_style_regroup_assets.py (reciprocal merge).
Use this when you need a guaranteed group count in a range (e.g. 200–500), not graph-based merging.

Pipeline:
1. Load original sticker_metadata + CLIP embedding cache (same as reciprocal script).
2. Compute one L2-normalized centroid per original img_set (pack).
3. Run K-means (cosine-friendly: normalized vectors, dot-product assignment) on P pack centroids
   into K clusters (K << P).
4. Assign every sticker to cluster group key `kc_{cluster_id:04d}`.
5. Reuse the same downstream steps as the reciprocal script: metadata, neighbors, factorized bank,
   train/val/test remapping.

Example (analysis only, fast):
  python scripts/build_stickerchat_style_kmeans_regroup_assets.py \\
    --num-clusters 384 \\
    --output-dir ./stickerchat/processed_style_kmeans_k384 \\
    --analysis-only

Full build (slow; delete incomplete output if interrupted — no resume):
  python scripts/build_stickerchat_style_kmeans_regroup_assets.py \\
    --num-clusters 384 \\
    --output-dir ./stickerchat/processed_style_kmeans_k384 \\
    --no-progress
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
STICKERCHAT_DIR = REPO_ROOT / "stickerchat"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STICKERCHAT_DIR) not in sys.path:
    sys.path.insert(0, str(STICKERCHAT_DIR))

# Reuse downstream from reciprocal regroup script (do not modify that file).
import build_stickerchat_style_regroup_assets as _regroup  # noqa: E402

from factorized_style_bank import build_factorized_style_bank_from_style_metadata  # noqa: E402
from build_stickerchat_assets import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("./stickerchat/processed/sticker_metadata.json"),
    )
    p.add_argument(
        "--train-path",
        type=Path,
        default=Path("./stickerchat/processed/release_train_u_sticker_format_int.json"),
    )
    p.add_argument(
        "--val-r10-path",
        type=Path,
        default=Path("./stickerchat/processed/release_val_u_sticker_format_int_with_cand_r10.json"),
    )
    p.add_argument(
        "--val-r20-path",
        type=Path,
        default=Path("./stickerchat/processed/release_val_u_sticker_format_int_with_cand_r20.json"),
    )
    p.add_argument(
        "--test-r10-path",
        type=Path,
        default=Path("./stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json"),
    )
    p.add_argument(
        "--test-r20-path",
        type=Path,
        default=Path("./stickerchat/processed/release_test_u_sticker_format_int_with_cand_r20.json"),
    )
    p.add_argument(
        "--img-emb-cache-path",
        type=Path,
        default=Path("./stickerchat/processed/stickerchat_clip_embs.pt"),
        help="Local CLIP embedding cache with shape [max_image_id, dim].",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./stickerchat/processed_style_kmeans_regroup"),
    )
    p.add_argument(
        "--num-clusters",
        type=int,
        required=True,
        help="Target K for K-means on pack centroids (e.g. 256–512 for 200–500 groups).",
    )
    p.add_argument(
        "--kmeans-iters",
        type=int,
        default=40,
        help="Lloyd-style refinement iterations on pack centroids.",
    )
    p.add_argument(
        "--neighbor-topk",
        type=int,
        default=32,
        help="How many same-style neighbors to export per sticker.",
    )
    p.add_argument(
        "--neg-mode",
        choices=("cross_style", "global"),
        default="cross_style",
        help="Negative/candidate sampling mode for rebuilt processed splits.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260330,
    )
    p.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only write style_regroup_analysis.json (no full assets).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars for split remapping.",
    )
    return p.parse_args()


def final_groups_from_pack_kmeans(
    pack_names: Sequence[str],
    pack_to_ids: Mapping[str, Sequence[int]],
    centroid_tensor: "torch.Tensor",
    num_clusters: int,
    kmeans_iters: int,
    seed: int,
) -> Tuple[Dict[str, List[int]], Any]:
    """Assign each pack to a cluster; merge stickers into group keys kc_0000 .. kc_{K-1}."""
    import torch

    _regroup.require_torch()
    assign = _regroup.kmeans_assignments(
        centroid_tensor,
        num_clusters=num_clusters,
        iterations=kmeans_iters,
        seed=seed,
    )
    cluster_to_ids: Dict[int, List[int]] = defaultdict(list)
    for pack_idx, pack in enumerate(pack_names):
        cid = int(assign[pack_idx].item())
        cluster_to_ids[cid].extend(int(x) for x in pack_to_ids[pack])

    final_group_to_ids: Dict[str, List[int]] = {}
    for cid, ids in cluster_to_ids.items():
        if not ids:
            continue
        key = f"kc_{cid:04d}"
        final_group_to_ids[key] = sorted(ids)
    return final_group_to_ids, assign


def main() -> None:
    args = parse_args()
    _regroup.require_torch()
    k = int(args.num_clusters)
    if k < 2:
        raise ValueError("--num-clusters must be >= 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stickers = _regroup.load_stickers(args.metadata_path)
    num_stickers = len(stickers)
    pack_to_ids = _regroup.build_pack_members(stickers)
    num_packs = len(pack_to_ids)
    if k >= num_packs:
        raise ValueError(
            f"--num-clusters ({k}) must be < num original packs ({num_packs}) "
            "for meaningful K-means pooling."
        )

    emb = _regroup.load_embeddings(args.img_emb_cache_path, expected_rows=num_stickers)
    pack_names, _pack_to_idx, centroid_tensor = _regroup.compute_pack_centroids(emb, pack_to_ids)

    final_group_to_ids, assign = final_groups_from_pack_kmeans(
        pack_names,
        pack_to_ids,
        centroid_tensor,
        num_clusters=k,
        kmeans_iters=int(args.kmeans_iters),
        seed=int(args.seed),
    )

    nonempty = len(final_group_to_ids)
    pack_counts = Counter(int(assign[i].item()) for i in range(len(pack_names)))
    empty_clusters = k - len([c for c in range(k) if pack_counts.get(c, 0) > 0])

    analysis: Dict[str, Any] = {
        "style_regroup_method": "clip_pack_centroid_kmeans",
        "input_metadata_path": str(args.metadata_path),
        "img_emb_cache_path": str(args.img_emb_cache_path),
        "num_clusters_requested": k,
        "num_clusters_nonempty": nonempty,
        "empty_clusters_after_kmeans": empty_clusters,
        "kmeans_iters": int(args.kmeans_iters),
        "seed": int(args.seed),
        "num_stickers": num_stickers,
        "num_original_packs": num_packs,
        "num_final_groups": nonempty,
        "original_group_size_distribution": _regroup.summarize_sizes(
            [len(v) for v in pack_to_ids.values()]
        ),
        "final_group_size_distribution": _regroup.summarize_sizes(
            [len(v) for v in final_group_to_ids.values()]
        ),
        "packs_per_cluster_distribution": _regroup.summarize_sizes(list(pack_counts.values())),
        "largest_final_groups": [
            {"group_name": gn, "size": len(ids)}
            for gn, ids in sorted(
                final_group_to_ids.items(),
                key=lambda kv: (-len(kv[1]), kv[0]),
            )[:50]
        ],
    }
    write_json(args.output_dir / "style_regroup_analysis.json", analysis)

    if args.analysis_only:
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return

    dp = bool(args.no_progress)
    regrouped_stickers, internal_id_to_group, final_group_to_ids = _regroup.build_regrouped_stickers(
        stickers,
        final_group_to_ids,
    )
    metadata = {
        "meta": {
            "source": "StickerChat",
            "style_regroup_method": "clip_pack_centroid_kmeans",
            "num_stickers": num_stickers,
            "num_original_packs": num_packs,
            "num_final_groups": len(final_group_to_ids),
            "num_clusters_requested": k,
            "kmeans_iters": int(args.kmeans_iters),
            "seed": int(args.seed),
        },
        "stickers": regrouped_stickers,
    }
    metadata_path = args.output_dir / "sticker_metadata.json"
    group_members_path = args.output_dir / "img_set_to_ids.json"
    neighbors_path = args.output_dir / "style_neighbors.json"
    bank_path = args.output_dir / "factorized_style_bank.json"

    print("[kmeans_regroup] Writing sticker_metadata + img_set_to_ids ...", flush=True)
    write_json(metadata_path, metadata)
    write_json(group_members_path, final_group_to_ids)

    print("[kmeans_regroup] Building CLIP within-style neighbors ...", flush=True)
    neighbors = _regroup.build_embedding_neighbor_json(
        emb,
        final_group_to_ids,
        int(args.neighbor_topk),
    )
    write_json(neighbors_path, neighbors)

    print("[kmeans_regroup] Building factorized_style_bank.json ...", flush=True)
    bank = build_factorized_style_bank_from_style_metadata(
        style_metadata_path=str(metadata_path),
        style_neighbors_path=str(neighbors_path),
        max_image_id=num_stickers,
    )
    write_json(bank_path, bank)

    train_out = args.output_dir / "release_train_u_sticker_format_int.json"
    val_r10_out = args.output_dir / "release_val_u_sticker_format_int_with_cand_r10.json"
    val_r20_out = args.output_dir / "release_val_u_sticker_format_int_with_cand_r20.json"
    test_r10_out = args.output_dir / "release_test_u_sticker_format_int_with_cand_r10.json"
    test_r20_out = args.output_dir / "release_test_u_sticker_format_int_with_cand_r20.json"

    print(
        "[kmeans_regroup] Remapping splits (slow on cross_style). "
        "If interrupted, delete partial JSON in output dir and rerun.",
        flush=True,
    )
    train_rows = _regroup.rebuild_split(
        args.train_path,
        train_out,
        candidate_size=None,
        universe_size=num_stickers,
        internal_id_to_group=internal_id_to_group,
        neg_mode=str(args.neg_mode),
        seed=int(args.seed) + 1000,
        progress_label="train",
        disable_progress=dp,
    )
    val_rows_r10 = _regroup.rebuild_split(
        args.val_r10_path,
        val_r10_out,
        candidate_size=10,
        universe_size=num_stickers,
        internal_id_to_group=internal_id_to_group,
        neg_mode=str(args.neg_mode),
        seed=int(args.seed) + 2000,
        progress_label="val_r10",
        disable_progress=dp,
    )
    val_rows_r20 = _regroup.rebuild_split(
        args.val_r20_path,
        val_r20_out,
        candidate_size=20,
        universe_size=num_stickers,
        internal_id_to_group=internal_id_to_group,
        neg_mode=str(args.neg_mode),
        seed=int(args.seed) + 3000,
        progress_label="val_r20",
        disable_progress=dp,
    )
    test_rows_r10 = _regroup.rebuild_split(
        args.test_r10_path,
        test_r10_out,
        candidate_size=10,
        universe_size=num_stickers,
        internal_id_to_group=internal_id_to_group,
        neg_mode=str(args.neg_mode),
        seed=int(args.seed) + 4000,
        progress_label="test_r10",
        disable_progress=dp,
    )
    test_rows_r20 = _regroup.rebuild_split(
        args.test_r20_path,
        test_r20_out,
        candidate_size=20,
        universe_size=num_stickers,
        internal_id_to_group=internal_id_to_group,
        neg_mode=str(args.neg_mode),
        seed=int(args.seed) + 5000,
        progress_label="test_r20",
        disable_progress=dp,
    )

    summary = {
        "num_stickers": num_stickers,
        "num_original_packs": num_packs,
        "num_clusters_requested": k,
        "num_final_groups": len(final_group_to_ids),
        "train_rows": train_rows,
        "val_rows_r10": val_rows_r10,
        "val_rows_r20": val_rows_r20,
        "test_rows_r10": test_rows_r10,
        "test_rows_r20": test_rows_r20,
        "paths": {
            "metadata": str(metadata_path),
            "img_set_to_ids": str(group_members_path),
            "style_neighbors": str(neighbors_path),
            "factorized_bank": str(bank_path),
            "train": str(train_out),
            "val_r10": str(val_r10_out),
            "val_r20": str(val_r20_out),
            "test_r10": str(test_r10_out),
            "test_r20": str(test_r20_out),
            "analysis": str(args.output_dir / "style_regroup_analysis.json"),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
