#!/usr/bin/env python3
"""
t-SNE visualization for factorized sticker factors u (shared), c (style), a (expression).

Loads StructuredFactorizedPLModel from checkpoint + YAML, samples stickers by prototype
from FactorizedStyleBank, runs decompose_sticker on CLIP embeddings, then L2 -> PCA -> t-SNE.

Example:
  python scripts/tsne_factorized_stickers.py \\
    --config configs/structured_factorized/stickerchat_v6_minimal_kmeans_k384_2.yaml \\
    --ckpt_path logs/.../last.ckpt \\
    --tsne_output_png figures/tsne_uac.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Project root on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from structured_retrieval import load_checkpoint_to_model, logger  # noqa: E402
from structured_retrieval_factorized import (  # noqa: E402
    StructuredFactorizedPLModel,
    parse_structured_factorized_args,
)


def _parse_tsne_flags(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--tsne_num_prototypes", type=int, default=10)
    p.add_argument(
        "--tsne_samples_per_proto",
        type=int,
        default=None,
        help="If set, sample exactly this many stickers per prototype (overrides min/max).",
    )
    p.add_argument(
        "--tsne_samples_per_proto_min",
        type=int,
        default=40,
        help="Per-prototype sample count lower bound (inclusive). Ignored if --tsne_samples_per_proto is set.",
    )
    p.add_argument(
        "--tsne_samples_per_proto_max",
        type=int,
        default=80,
        help="Per-prototype sample count upper bound (inclusive). Ignored if --tsne_samples_per_proto is set.",
    )
    p.add_argument("--tsne_output_png", type=str, default="tsne_factorized_uac.png")
    p.add_argument("--tsne_output_npz", type=str, default="")
    p.add_argument(
        "--tsne_proto_select",
        choices=("largest", "random", "diverse"),
        default="largest",
        help="largest=by member count; random=uniform random protos; "
        "diverse=greedy maximin on L2-normalized mean CLIP vectors (needs img emb cache).",
    )
    p.add_argument(
        "--tsne_diverse_min_proto_size",
        type=int,
        default=5,
        help="For diverse: minimum members to include a prototype in the candidate pool (CLIP centroid). "
        "Independent of per-proto sample counts; use << tsne_samples_per_proto_min when many clusters are small.",
    )
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_pca_dim", type=int, default=50)
    p.add_argument(
        "--tsne_seed",
        type=int,
        default=None,
        help="If set, overrides training config seed for numpy/sklearn/torch sampling.",
    )
    p.add_argument(
        "--tsne_metrics",
        action="store_true",
        help="Print silhouette score and mean kNN purity (same feature pipeline as t-SNE input).",
    )
    p.add_argument("--tsne_knn_k", type=int, default=15)
    return p.parse_known_args(list(argv))


def _proto_clip_centroid(
    members: Sequence[int],
    all_img_embs: torch.Tensor,
) -> np.ndarray:
    """Mean CLIP embedding with per-row L2 norm, then centroid L2-normalized."""
    from sklearn.preprocessing import normalize

    idx = torch.tensor(list(members), dtype=torch.long)
    h = all_img_embs.index_select(0, idx).numpy()
    hn = normalize(h, norm="l2", axis=1)
    c = hn.mean(axis=0)
    nrm = np.linalg.norm(c)
    if nrm < 1e-12:
        return hn[0].copy()
    return (c / nrm).astype(np.float64)


def _collect_diverse_candidates(
    bank: Any,
    max_image_id: int,
    all_img_embs: torch.Tensor,
    min_members: int,
) -> Tuple[List[int], List[np.ndarray]]:
    pids: List[int] = []
    centroids: List[np.ndarray] = []
    for pr in bank.prototypes:
        pid = int(pr.proto_id)
        members = [int(x) for x in pr.member_ids if 0 <= int(x) < max_image_id]
        if len(members) < min_members:
            continue
        pids.append(pid)
        centroids.append(_proto_clip_centroid(members, all_img_embs))
    return pids, centroids


def _greedy_diverse_prototypes_clip(
    bank: Any,
    max_image_id: int,
    all_img_embs: torch.Tensor,
    k: int,
    min_members: int,
) -> List[int]:
    """
    Pick k prototypes whose mean (L2-normalized) CLIP centroids are spread out:
    greedy maximin — repeatedly add the prototype farthest from the already-chosen set
    (distance = Euclidean between centroid unit vectors).
    First prototype is the eligible one with the largest member count.

    If fewer than k prototypes meet ``min_members``, the floor is relaxed down to 2.
    If still fewer than k prototypes exist in the bank, k is reduced to the available count.
    """
    requested_k = int(k)
    floor = max(2, int(min_members))
    initial_floor = floor
    pids, centroids = _collect_diverse_candidates(
        bank, max_image_id, all_img_embs, floor
    )
    while len(pids) < requested_k and floor > 2:
        logger.info(
            "[t-SNE] diverse: %d protos with >= %d members (need %d); lowering min_proto_size to %d",
            len(pids),
            floor,
            requested_k,
            floor - 1,
        )
        floor -= 1
        pids, centroids = _collect_diverse_candidates(
            bank, max_image_id, all_img_embs, floor
        )

    if initial_floor > floor:
        logger.warning(
            "[t-SNE] diverse: used effective min_proto_size=%d (requested >= %d) to get enough candidates.",
            floor,
            initial_floor,
        )

    if len(pids) < 2:
        raise RuntimeError(
            "diverse: fewer than 2 prototypes with at least 2 members in [0, max_image_id). "
            "Check factorized_style_bank.json and max_image_id."
        )

    k_eff = min(requested_k, len(pids))
    if k_eff < requested_k:
        logger.warning(
            "[t-SNE] diverse: only %d prototype(s) available; plotting %d (requested %d).",
            len(pids),
            k_eff,
            requested_k,
        )

    C = np.stack(centroids, axis=0)
    sizes = [
        len([x for x in bank.members_of_proto(pid) if 0 <= int(x) < max_image_id]) for pid in pids
    ]
    first = int(np.argmax(np.array(sizes)))
    selected = [first]

    while len(selected) < k_eff:
        best_j: Optional[int] = None
        best_score = -1.0
        for j in range(len(pids)):
            if j in selected:
                continue
            dmin = min(float(np.linalg.norm(C[j] - C[s])) for s in selected)
            if dmin > best_score:
                best_score = dmin
                best_j = j
        assert best_j is not None
        selected.append(best_j)

    return [pids[i] for i in selected]


def _sample_sticker_ids(
    bank: Any,
    max_image_id: int,
    num_prototypes: int,
    samples_per_proto_min: int,
    samples_per_proto_max: int,
    proto_select: str,
    rng: np.random.Generator,
    min_members_for_diverse: int,
    all_img_embs: Optional[torch.Tensor] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Returns sticker_ids, proto_ids (aligned), selected_proto_ids (the K prototypes used).
    """
    protos = list(bank.prototypes)
    if not protos:
        raise RuntimeError("Style bank has no prototypes.")

    sp_min = int(samples_per_proto_min)
    sp_max = int(samples_per_proto_max)
    if sp_min > sp_max:
        raise ValueError("tsne_samples_per_proto_min must be <= tsne_samples_per_proto_max.")

    if proto_select == "diverse":
        if all_img_embs is None:
            raise RuntimeError("diverse proto selection requires img embedding tensor.")
        selected_proto_ids = _greedy_diverse_prototypes_clip(
            bank=bank,
            max_image_id=max_image_id,
            all_img_embs=all_img_embs,
            k=int(num_prototypes),
            min_members=min_members_for_diverse,
        )
        logger.info(
            "[t-SNE] diverse_clip selected proto_ids=%s (greedy maximin on CLIP centroids)",
            selected_proto_ids,
        )
    else:
        scored: List[Tuple[int, int]] = []
        for pr in protos:
            pid = int(pr.proto_id)
            members = [int(x) for x in pr.member_ids if 0 <= int(x) < max_image_id]
            if len(members) < 1:
                continue
            scored.append((pid, len(members)))

        if not scored:
            raise RuntimeError("No prototypes with valid members in [0, max_image_id).")

        if proto_select == "largest":
            scored.sort(key=lambda x: -x[1])
        else:
            rng.shuffle(scored)

        k = min(num_prototypes, len(scored))
        picked = scored[:k]
        selected_proto_ids = [p for p, _ in picked]

    sticker_ids: List[int] = []
    proto_ids: List[int] = []
    for pid in selected_proto_ids:
        members = [int(x) for x in bank.members_of_proto(pid) if 0 <= int(x) < max_image_id]
        if not members:
            continue
        cap = len(members)
        lo = min(sp_min, cap)
        hi = min(sp_max, cap)
        if lo > hi:
            continue
        n = int(rng.integers(lo, hi + 1))
        chosen = rng.choice(members, size=n, replace=False).tolist()
        sticker_ids.extend(chosen)
        proto_ids.extend([pid] * len(chosen))

    if len(sticker_ids) < 2:
        raise RuntimeError("Too few sampled stickers for t-SNE.")

    return sticker_ids, proto_ids, selected_proto_ids


def _l2_pca_tsne(
    X: np.ndarray,
    seed: int,
    pca_dim: int,
    perplexity: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """L2 row-normalize -> PCA -> t-SNE. Returns (xy_2d, z_pca)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize

    X = np.asarray(X, dtype=np.float64)
    Xn = normalize(X, norm="l2", axis=1)
    n_samples, n_feat = Xn.shape
    n_comp = min(int(pca_dim), max(1, n_samples - 1), n_feat)
    pca = PCA(n_components=n_comp, random_state=seed)
    Z = pca.fit_transform(Xn)
    perp = float(perplexity)
    if perp >= n_samples:
        perp = max(2.0, float(n_samples - 1) * 0.99)
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=seed,
        perplexity=perp,
    )
    xy = tsne.fit_transform(Z)
    return xy, Z


def _silhouette_on_z(Z: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    uniq = np.unique(labels)
    if uniq.size < 2 or Z.shape[0] < 3:
        return float("nan")
    return float(silhouette_score(Z, labels, metric="euclidean"))


def _knn_purity(Z: np.ndarray, labels: np.ndarray, k: int) -> float:
    from sklearn.neighbors import NearestNeighbors

    n = Z.shape[0]
    kk = min(int(k), max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    neigh = idx[:, 1:]
    pur: List[float] = []
    for i in range(n):
        same = np.mean(labels[neigh[i]] == labels[i])
        pur.append(float(same))
    return float(np.mean(pur))


def _plot_triptych(
    coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: np.ndarray,
    title_suffix: str,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt

    u_xy, c_xy, a_xy = coords
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    uniq = np.unique(labels)
    if len(uniq) <= 10:
        cmap = plt.cm.get_cmap("tab10", 10)
    else:
        cmap = plt.cm.get_cmap("tab20", 20)

    def scatter(ax, xy: np.ndarray, title: str) -> None:
        for j, lab in enumerate(uniq):
            m = labels == lab
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=12,
                alpha=0.75,
                color=cmap(j % cmap.N),
                label=str(int(lab)),
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    scatter(axes[0], u_xy, "(a) u shared")
    scatter(axes[1], c_xy, "(b) c style")
    scatter(axes[2], a_xy, "(c) a expr")
    fig.suptitle(title_suffix, fontsize=10)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            leg_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            title="proto_id",
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def main() -> None:
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__ or "")
        print(
            "Required: same as training (--config ... --ckpt_path ...).\n"
            "Optional: --tsne_num_prototypes, --tsne_samples_per_proto[_min|_max], --tsne_diverse_min_proto_size, "
            "--tsne_output_png, --tsne_output_npz, --tsne_proto_select {largest,random,diverse}, "
            "--tsne_perplexity, --tsne_pca_dim, --tsne_seed, --tsne_metrics, --tsne_knn_k."
        )
        return
    tsne_args, rest = _parse_tsne_flags(sys.argv[1:])
    rest = [t for t in rest if t and str(t).strip()]
    if not rest:
        raise SystemExit(
            "Usage: python scripts/tsne_factorized_stickers.py [--tsne_* ...] "
            "--config <yaml> ... --ckpt_path <ckpt>"
        )
    args = parse_structured_factorized_args(rest)
    ckpt = (getattr(args, "ckpt_path", None) or "").strip()
    if not ckpt:
        raise ValueError("--ckpt_path is required.")

    seed = int(tsne_args.tsne_seed if tsne_args.tsne_seed is not None else args.seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    import pytorch_lightning as pl

    pl.seed_everything(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(args, "allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(getattr(args, "allow_tf32", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pl_model = StructuredFactorizedPLModel(args)
    load_checkpoint_to_model(pl_model, ckpt, strict=bool(getattr(args, "strict_checkpoint_load", False)))
    model = pl_model.model
    model.eval()
    model.to(device)

    bank = model.style_bank
    max_image_id = int(args.max_image_id)
    cache_path = (getattr(args, "img_emb_cache_path", None) or "").strip()
    if not cache_path or not os.path.isfile(cache_path):
        raise FileNotFoundError(f"img_emb_cache_path missing or not found: {cache_path!r}")

    all_img_embs = torch.load(cache_path, map_location="cpu").float()
    if all_img_embs.dim() != 2 or all_img_embs.size(0) != max_image_id:
        raise ValueError(
            f"Expected all_img_embs shape [max_image_id, D], got {tuple(all_img_embs.shape)} "
            f"vs max_image_id={max_image_id}"
        )

    if tsne_args.tsne_samples_per_proto is not None:
        sp_min = sp_max = int(tsne_args.tsne_samples_per_proto)
    else:
        sp_min = int(tsne_args.tsne_samples_per_proto_min)
        sp_max = int(tsne_args.tsne_samples_per_proto_max)

    proto_sel = str(tsne_args.tsne_proto_select)
    emb_for_sample: Optional[torch.Tensor] = all_img_embs if proto_sel == "diverse" else None
    # Diverse candidate pool must not require as many members as per-proto sampling (DSTC banks are often small).
    diverse_floor = max(2, int(tsne_args.tsne_diverse_min_proto_size)) if proto_sel == "diverse" else 0

    sticker_ids, proto_ids, selected_proto_ids = _sample_sticker_ids(
        bank=bank,
        max_image_id=max_image_id,
        num_prototypes=int(tsne_args.tsne_num_prototypes),
        samples_per_proto_min=sp_min,
        samples_per_proto_max=sp_max,
        proto_select=proto_sel,
        rng=rng,
        min_members_for_diverse=diverse_floor,
        all_img_embs=emb_for_sample,
    )
    logger.info(
        "[t-SNE] sampled N=%d stickers from %d prototypes (per-proto count in [%d, %d] unless capped by membership).",
        len(sticker_ids),
        len(selected_proto_ids),
        sp_min,
        sp_max,
    )

    idx = torch.tensor(sticker_ids, dtype=torch.long)
    h = all_img_embs.index_select(0, idx).to(device)
    with torch.no_grad():
        u_t, c_t, a_t = model.decompose_sticker(h)

    u = u_t.detach().cpu().numpy()
    c = c_t.detach().cpu().numpy()
    a = a_t.detach().cpu().numpy()
    labels = np.array(proto_ids, dtype=np.int64)

    if tsne_args.tsne_output_npz:
        npz_path = tsne_args.tsne_output_npz
        os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
        np.savez(
            npz_path,
            sticker_ids=np.array(sticker_ids, dtype=np.int64),
            proto_ids=labels,
            selected_proto_ids=np.array(selected_proto_ids, dtype=np.int64),
            u=u,
            c=c,
            a=a,
            seed=np.int64(seed),
        )
        logger.info("Wrote %s", npz_path)

    perp = float(tsne_args.tsne_perplexity)
    pca_dim = int(tsne_args.tsne_pca_dim)

    u_xy, u_z = _l2_pca_tsne(u, seed, pca_dim, perp)
    c_xy, c_z = _l2_pca_tsne(c, seed, pca_dim, perp)
    a_xy, a_z = _l2_pca_tsne(a, seed, pca_dim, perp)

    if tsne_args.tsne_metrics:
        for name, Z in (("u", u_z), ("c", c_z), ("a", a_z)):
            sil = _silhouette_on_z(Z, labels)
            pur = _knn_purity(Z, labels, k=int(tsne_args.tsne_knn_k))
            logger.info(
                "[t-SNE metrics] %s: silhouette=%.4f kNN_purity(k=%d)=%.4f (PCA-%dd pre-t-SNE)",
                name,
                sil,
                int(tsne_args.tsne_knn_k),
                pur,
                Z.shape[1],
            )

    caption = (
        f"ckpt={os.path.basename(ckpt)} seed={seed} "
        f"proto_select={proto_sel} prototypes={len(selected_proto_ids)} n={len(sticker_ids)} "
        f"perplexity={perp} pca_dim={pca_dim}"
    )
    _plot_triptych((u_xy, c_xy, a_xy), labels, caption, tsne_args.tsne_output_png)


if __name__ == "__main__":
    main()
