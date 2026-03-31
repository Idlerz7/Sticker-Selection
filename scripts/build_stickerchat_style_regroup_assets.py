#!/usr/bin/env python3
"""
Build regrouped StickerChat style assets without external APIs.

The script reuses the existing factorized-style bank pipeline, but replaces the
original `img_set -> style` assumption with a CLIP-centroid regroup:

1. compute one visual centroid per original img_set from a local embedding cache
2. merge visually similar packs with reciprocal nearest-neighbor thresholding
3. optionally split oversized regrouped styles with lightweight k-means
4. rebuild metadata / neighbors / factorized bank / remapped train-test splits

Example:
  python scripts/build_stickerchat_style_regroup_assets.py \
    --img-emb-cache-path ./stickerchat/processed/stickerchat_clip_embs.pt \
    --output-dir ./stickerchat/processed_style_regroup_centroid \
    --analysis-only

  python scripts/build_stickerchat_style_regroup_assets.py \
    --img-emb-cache-path ./stickerchat/processed/stickerchat_clip_embs.pt \
    --output-dir ./stickerchat/processed_style_regroup_centroid \
    --merge-threshold 0.92 \
    --reciprocal-topk 2 \
    --split-large-packs-over 120 \
    --split-target-size 48

If the process is killed mid-run, remove incomplete files under --output-dir (or the whole
directory) and rerun. There is no resume-from-partial mode.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - environment-dependent
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
STICKERCHAT_DIR = REPO_ROOT / "stickerchat"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STICKERCHAT_DIR) not in sys.path:
    sys.path.insert(0, str(STICKERCHAT_DIR))

from factorized_style_bank import build_factorized_style_bank_from_style_metadata  # noqa: E402
from build_stickerchat_assets import (  # noqa: E402
    iter_json_array_records,
    write_json,
    write_json_array,
)
from stickerchat_sampling import sample_cand_list, sample_train_neg_id  # noqa: E402


@dataclass(frozen=True)
class MergeEdge:
    left_pack: str
    right_pack: str
    cosine: float


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {str(x): str(x) for x in items}

    def find(self, x: str) -> str:
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


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
        default=Path("./stickerchat/processed_style_regroup_centroid"),
    )
    p.add_argument(
        "--merge-threshold",
        type=float,
        default=0.92,
        help="Minimum cosine similarity for reciprocal pack merge.",
    )
    p.add_argument(
        "--reciprocal-topk",
        type=int,
        default=1,
        help="Require pack A and B to appear in each other's top-k neighbors.",
    )
    p.add_argument(
        "--top-neighbors-report",
        type=int,
        default=10,
        help="How many nearest original packs to save for each pack in analysis.",
    )
    p.add_argument(
        "--top-merge-edges-report",
        type=int,
        default=200,
        help="How many strongest accepted merge edges to save in summary.",
    )
    p.add_argument(
        "--split-large-packs-over",
        type=int,
        default=0,
        help="If >0, split regrouped styles larger than this size with k-means.",
    )
    p.add_argument(
        "--split-target-size",
        type=int,
        default=48,
        help="Approximate subgroup size when splitting oversized regrouped styles.",
    )
    p.add_argument(
        "--split-kmeans-iters",
        type=int,
        default=20,
        help="K-means refinement iterations for oversized-style splitting.",
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
        help="Only write regroup analysis files, not rebuilt metadata/splits/bank assets.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars for split remapping.",
    )
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_json_array_record_lines(path: Path) -> int:
    """Match iter_json_array_records line-skipping so tqdm total matches row count."""
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text in {"[", "]"}:
                continue
            n += 1
    return n


def require_torch() -> None:
    if torch is None or F is None:
        raise ImportError(
            "This script requires torch. Run it from the training environment where torch is installed."
        )


def load_stickers(metadata_path: Path) -> List[Dict[str, Any]]:
    raw = load_json(metadata_path)
    stickers = raw.get("stickers") or []
    if not stickers:
        raise ValueError(f"No sticker rows found in {metadata_path}")
    return stickers


def build_pack_members(stickers: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    for row in stickers:
        iid = int(row["internal_img_id"])
        pack = str(row["img_set"])
        out[pack].append(iid)
    for key in out:
        out[key] = sorted(out[key])
    return dict(out)


def summarize_sizes(sizes: Sequence[int]) -> Dict[str, float]:
    if not sizes:
        return {"count": 0}
    s = sorted(int(x) for x in sizes)
    n = len(s)

    def pct(p: float) -> float:
        if n == 1:
            return float(s[0])
        k = (n - 1) * p / 100.0
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return float(s[lo])
        alpha = k - lo
        return float((1.0 - alpha) * s[lo] + alpha * s[hi])

    return {
        "count": float(n),
        "min": float(s[0]),
        "max": float(s[-1]),
        "mean": float(sum(s)) / float(n),
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
    }


def load_embeddings(path: Path, expected_rows: int) -> torch.Tensor:
    require_torch()
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {path}. Run precompute_sticker_embeddings.py first."
        )
    obj = torch.load(path, map_location="cpu")
    if not torch.is_tensor(obj):
        raise TypeError(f"Expected a tensor in {path}, got {type(obj)!r}")
    embs = obj.float()
    if embs.dim() != 2:
        raise ValueError(f"Expected [N, D] embeddings, got shape {tuple(embs.shape)}")
    if embs.size(0) < expected_rows:
        raise ValueError(
            f"Embedding rows {embs.size(0)} < num stickers {expected_rows}; id alignment would break."
        )
    return F.normalize(embs[:expected_rows], dim=1)


def compute_pack_centroids(
    emb: torch.Tensor,
    pack_to_ids: Mapping[str, Sequence[int]],
) -> Tuple[List[str], Dict[str, int], torch.Tensor]:
    require_torch()
    pack_names = sorted(pack_to_ids.keys())
    centroids: List[torch.Tensor] = []
    for pack in pack_names:
        ids = torch.tensor(list(pack_to_ids[pack]), dtype=torch.long)
        centroid = emb.index_select(0, ids).mean(dim=0, keepdim=False)
        centroids.append(F.normalize(centroid, dim=0))
    centroid_tensor = torch.stack(centroids, dim=0)
    pack_to_idx = {name: idx for idx, name in enumerate(pack_names)}
    return pack_names, pack_to_idx, centroid_tensor


def reciprocal_merge_plan(
    pack_names: Sequence[str],
    sim: torch.Tensor,
    threshold: float,
    reciprocal_topk: int,
) -> Tuple[Dict[str, str], List[MergeEdge], Dict[str, List[Tuple[str, float]]]]:
    n = len(pack_names)
    topk = max(1, min(int(reciprocal_topk), max(n - 1, 1)))
    sim_no_self = sim.clone()
    sim_no_self.fill_diagonal_(-1.0)
    vals, idxs = torch.topk(sim_no_self, k=topk, dim=1)
    neighbor_sets = [
        {int(j) for j in idxs[row].tolist() if int(j) != row} for row in range(n)
    ]

    uf = UnionFind(pack_names)
    edges: List[MergeEdge] = []
    nearest: Dict[str, List[Tuple[str, float]]] = {}
    for row, pack in enumerate(pack_names):
        pairs: List[Tuple[str, float]] = []
        for col, score in zip(idxs[row].tolist(), vals[row].tolist()):
            if col == row:
                continue
            pairs.append((pack_names[int(col)], float(score)))
        nearest[pack] = pairs

    for i in range(n):
        for j in neighbor_sets[i]:
            if j <= i:
                continue
            score = float(sim[i, j].item())
            if score < threshold:
                continue
            if i not in neighbor_sets[j]:
                continue
            uf.union(pack_names[i], pack_names[j])
            edges.append(MergeEdge(pack_names[i], pack_names[j], score))

    pack_to_root = {pack: uf.find(pack) for pack in pack_names}
    return pack_to_root, sorted(edges, key=lambda x: x.cosine, reverse=True), nearest


def regroup_members(
    pack_to_ids: Mapping[str, Sequence[int]],
    pack_to_root: Mapping[str, str],
) -> Dict[str, List[int]]:
    root_to_ids: Dict[str, List[int]] = defaultdict(list)
    for pack, ids in pack_to_ids.items():
        root = str(pack_to_root[pack])
        root_to_ids[root].extend(int(i) for i in ids)
    for root in root_to_ids:
        root_to_ids[root] = sorted(root_to_ids[root])
    return dict(root_to_ids)


def kmeans_assignments(
    vectors: torch.Tensor,
    num_clusters: int,
    iterations: int,
    seed: int,
) -> torch.Tensor:
    require_torch()
    num_items = int(vectors.size(0))
    if num_clusters <= 1 or num_items <= 1:
        return torch.zeros(num_items, dtype=torch.long)
    if num_clusters >= num_items:
        return torch.arange(num_items, dtype=torch.long)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    init_idx = torch.randperm(num_items, generator=rng)[:num_clusters]
    centers = vectors.index_select(0, init_idx).clone()

    for _ in range(max(1, int(iterations))):
        logits = torch.matmul(vectors, centers.t())
        assign = torch.argmax(logits, dim=1)

        new_centers: List[torch.Tensor] = []
        for cid in range(num_clusters):
            mask = assign == cid
            if mask.any():
                center = vectors[mask].mean(dim=0)
                center = F.normalize(center, dim=0)
            else:
                ridx = int(torch.randint(num_items, (1,), generator=rng).item())
                center = vectors[ridx]
            new_centers.append(center)
        centers = torch.stack(new_centers, dim=0)

    return assign


def maybe_split_large_groups(
    emb: torch.Tensor,
    regroup_to_ids: Mapping[str, Sequence[int]],
    split_large_packs_over: int,
    split_target_size: int,
    split_kmeans_iters: int,
    seed: int,
) -> Dict[str, List[int]]:
    if int(split_large_packs_over) <= 0:
        return {str(k): list(v) for k, v in regroup_to_ids.items()}

    final_groups: Dict[str, List[int]] = {}
    for group_name in sorted(regroup_to_ids.keys()):
        ids = sorted(int(i) for i in regroup_to_ids[group_name])
        if len(ids) <= int(split_large_packs_over):
            final_groups[str(group_name)] = ids
            continue

        target = max(1, int(split_target_size))
        num_clusters = max(2, int(math.ceil(len(ids) / float(target))))
        vecs = emb.index_select(0, torch.tensor(ids, dtype=torch.long))
        assign = kmeans_assignments(
            vecs,
            num_clusters=num_clusters,
            iterations=split_kmeans_iters,
            seed=seed + len(ids),
        )
        by_cluster: Dict[int, List[int]] = defaultdict(list)
        for iid, cid in zip(ids, assign.tolist()):
            by_cluster[int(cid)].append(int(iid))
        for sub_idx, cid in enumerate(sorted(by_cluster.keys())):
            final_groups[f"{group_name}__sub{sub_idx:02d}"] = sorted(by_cluster[cid])
    return final_groups


def build_regrouped_stickers(
    stickers: Sequence[Mapping[str, Any]],
    final_group_to_ids: Mapping[str, Sequence[int]],
) -> Tuple[List[Dict[str, Any]], Dict[int, str], Dict[str, List[int]]]:
    sticker_to_group: Dict[int, str] = {}
    for group_name, ids in final_group_to_ids.items():
        for iid in ids:
            sticker_to_group[int(iid)] = str(group_name)

    rows: List[Dict[str, Any]] = []
    group_to_ids: Dict[str, List[int]] = defaultdict(list)
    for row in stickers:
        iid = int(row["internal_img_id"])
        new_row = dict(row)
        new_row["original_img_set"] = str(row["img_set"])
        new_row["img_set"] = sticker_to_group[iid]
        rows.append(new_row)
        group_to_ids[new_row["img_set"]].append(iid)
    for key in group_to_ids:
        group_to_ids[key] = sorted(group_to_ids[key])
    return rows, sticker_to_group, dict(group_to_ids)


def build_embedding_neighbor_json(
    emb: torch.Tensor,
    group_to_ids: Mapping[str, Sequence[int]],
    neighbor_topk: int,
) -> Dict[str, Any]:
    require_torch()
    rows: List[Dict[str, Any]] = []
    for group_name in sorted(group_to_ids.keys()):
        members = sorted(int(i) for i in group_to_ids[group_name])
        ids_tensor = torch.tensor(members, dtype=torch.long)
        group_emb = emb.index_select(0, ids_tensor)
        k = min(max(int(neighbor_topk), 0), max(len(members) - 1, 0))
        if k > 0:
            sims = torch.matmul(group_emb, group_emb.t())
            sims.fill_diagonal_(-1.0)
            vals, idxs = torch.topk(sims, k=k, dim=1)
        else:
            vals, idxs = None, None
        for row_idx, internal_id in enumerate(members):
            neighbors: List[Dict[str, str]] = []
            if k > 0 and vals is not None and idxs is not None:
                for col_idx, score in zip(idxs[row_idx].tolist(), vals[row_idx].tolist()):
                    if score < 0.0:
                        continue
                    neighbors.append({"id": str(members[int(col_idx)])})
            rows.append(
                {
                    "id": str(internal_id),
                    "neighbors": neighbors,
                    "meta": {
                        "candidate_count": max(len(members) - 1, 0),
                        "used_fallback_to_all": False,
                        "img_set": group_name,
                    },
                }
            )
    return {
        "meta": {
            "builder": "stickerchat_clip_regroup_within_style_knn",
            "neighbor_topk": int(neighbor_topk),
            "num_stickers": int(emb.size(0)),
            "num_sets": len(group_to_ids),
        },
        "neighbors": rows,
    }


def rebuild_split(
    input_path: Path,
    output_path: Path,
    *,
    candidate_size: Optional[int],
    universe_size: int,
    internal_id_to_group: Mapping[int, str],
    neg_mode: str,
    seed: int,
    progress_label: Optional[str] = None,
    disable_progress: bool = False,
) -> int:
    total: Optional[int] = None
    if progress_label and not disable_progress:
        total = count_json_array_record_lines(input_path)

    def _rows() -> Iterator[Dict[str, Any]]:
        it = enumerate(iter_json_array_records(input_path))
        if progress_label and not disable_progress:
            it = tqdm(
                it,
                desc=progress_label,
                total=total,
                unit="row",
                mininterval=0.5,
            )
        for row_idx, sample in it:
            remapped = json.loads(json.dumps(sample, ensure_ascii=False))
            dialog = remapped.get("dialog") or []
            if not dialog:
                yield remapped
                continue
            target = dialog[-1]
            raw = target.get("img_id")
            if raw is None:
                if candidate_size is not None:
                    raise ValueError(f"{input_path}: row {row_idx} missing target img_id")
                yield remapped
                continue

            pos_id = int(raw)
            pos_group = str(internal_id_to_group[pos_id])
            target["img_set"] = pos_group

            if candidate_size is None:
                rng = random.Random(seed + row_idx)
                neg_id = sample_train_neg_id(
                    rng,
                    pos_id,
                    universe_size,
                    internal_id_to_group,  # type: ignore[arg-type]
                    neg_mode,
                )
                target["neg_img_id"] = int(neg_id)
                target["neg_img_set"] = str(internal_id_to_group[int(neg_id)])
                remapped.pop("cand", None)
            else:
                rng = random.Random(seed + row_idx)
                cand = sample_cand_list(
                    rng,
                    pos_id,
                    candidate_size,
                    universe_size,
                    internal_id_to_group,  # type: ignore[arg-type]
                    neg_mode,
                )
                remapped["cand"] = cand
                neg_id = int(cand[1])
                target["neg_img_id"] = neg_id
                target["neg_img_set"] = str(internal_id_to_group[neg_id])
            yield remapped

    return write_json_array(output_path, _rows())


def main() -> None:
    args = parse_args()
    require_torch()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stickers = load_stickers(args.metadata_path)
    num_stickers = len(stickers)
    pack_to_ids = build_pack_members(stickers)
    emb = load_embeddings(args.img_emb_cache_path, expected_rows=num_stickers)
    pack_names, pack_to_idx, centroids = compute_pack_centroids(emb, pack_to_ids)
    sim = torch.matmul(centroids, centroids.t()).cpu()

    pack_to_root, merge_edges, nearest = reciprocal_merge_plan(
        pack_names,
        sim,
        threshold=float(args.merge_threshold),
        reciprocal_topk=int(args.reciprocal_topk),
    )
    regroup_to_ids = regroup_members(pack_to_ids, pack_to_root)
    final_group_to_ids = maybe_split_large_groups(
        emb,
        regroup_to_ids,
        split_large_packs_over=int(args.split_large_packs_over),
        split_target_size=int(args.split_target_size),
        split_kmeans_iters=int(args.split_kmeans_iters),
        seed=int(args.seed),
    )

    analysis = {
        "input_metadata_path": str(args.metadata_path),
        "img_emb_cache_path": str(args.img_emb_cache_path),
        "num_stickers": num_stickers,
        "num_original_packs": len(pack_to_ids),
        "num_regroup_roots_before_split": len(regroup_to_ids),
        "num_final_groups": len(final_group_to_ids),
        "merge_threshold": float(args.merge_threshold),
        "reciprocal_topk": int(args.reciprocal_topk),
        "split_large_packs_over": int(args.split_large_packs_over),
        "split_target_size": int(args.split_target_size),
        "original_group_size_distribution": summarize_sizes([len(v) for v in pack_to_ids.values()]),
        "final_group_size_distribution": summarize_sizes([len(v) for v in final_group_to_ids.values()]),
        "accepted_merge_edge_count": len(merge_edges),
        "top_merge_edges": [
            {
                "left_pack": edge.left_pack,
                "right_pack": edge.right_pack,
                "cosine": edge.cosine,
            }
            for edge in merge_edges[: int(args.top_merge_edges_report)]
        ],
        "top_neighbors_by_pack": {
            pack: [
                {"neighbor_pack": neighbor_pack, "cosine": cosine}
                for neighbor_pack, cosine in pairs[: int(args.top_neighbors_report)]
            ]
            for pack, pairs in nearest.items()
        },
        "largest_final_groups": [
            {
                "group_name": group_name,
                "size": len(ids),
            }
            for group_name, ids in sorted(
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
    regrouped_stickers, internal_id_to_group, final_group_to_ids = build_regrouped_stickers(
        stickers,
        final_group_to_ids,
    )
    metadata = {
        "meta": {
            "source": "StickerChat",
            "style_regroup_method": "clip_centroid_regroup",
            "num_stickers": num_stickers,
            "num_original_packs": len(pack_to_ids),
            "num_final_groups": len(final_group_to_ids),
            "merge_threshold": float(args.merge_threshold),
            "reciprocal_topk": int(args.reciprocal_topk),
            "split_large_packs_over": int(args.split_large_packs_over),
            "split_target_size": int(args.split_target_size),
        },
        "stickers": regrouped_stickers,
    }
    metadata_path = args.output_dir / "sticker_metadata.json"
    group_members_path = args.output_dir / "img_set_to_ids.json"
    neighbors_path = args.output_dir / "style_neighbors.json"
    bank_path = args.output_dir / "factorized_style_bank.json"

    print("[style_regroup] Writing regrouped sticker_metadata + img_set_to_ids ...", flush=True)
    write_json(metadata_path, metadata)
    write_json(group_members_path, final_group_to_ids)
    print("[style_regroup] Building CLIP within-style neighbors ...", flush=True)
    neighbors = build_embedding_neighbor_json(
        emb,
        final_group_to_ids,
        int(args.neighbor_topk),
    )
    write_json(neighbors_path, neighbors)

    print("[style_regroup] Building factorized_style_bank.json ...", flush=True)
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
        "[style_regroup] Remapping splits (slow on cross_style: train >> val/test). "
        "If interrupted, delete partial JSON in output dir and rerun.",
        flush=True,
    )
    train_rows = rebuild_split(
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
    val_rows_r10 = rebuild_split(
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
    val_rows_r20 = rebuild_split(
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
    test_rows_r10 = rebuild_split(
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
    test_rows_r20 = rebuild_split(
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
        "num_original_packs": len(pack_to_ids),
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
