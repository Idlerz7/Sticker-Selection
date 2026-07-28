#!/usr/bin/env python3
"""
Aggregate four ``export_pair_features_for_tsne.py`` output directories, run identical
L2 → PCA(30) → t-SNE(2) on features, compute 2D inter/intra separation metrics and bar plots.

Outputs (under ``--out_dir``)
---------------------------
``figures/tsne_4panel.pdf``                 2×2 panels (a) Base (b) w/o PSP (c) w/o L_proto (d) Full.
``figures/avg_positive_matching_score.pdf`` four bars: mean matching score on positive pairs (from ``scores.npy``).
``figures/inter_intra_ratio_bar.pdf``       four bars: inter/intra ratio on 2D t-SNE (geometry only; unrelated scale to scores).
Both bar charts use the same four colors for Base / w/o PSP / w/o L_proto / Full.
``metrics.json``                            Per-variant metrics + t-SNE / PCA hyperparameters.
``tsne_xy_<variant>.npy``            Optional ``[1000, 2]`` arrays (--save_tsne_xy).

Example
-------

::

    python scripts/plot_pair_tsne_and_metrics.py \\
        --base_dir exports/base \\
        --wo_psp_dir exports/wo_psp \\
        --wo_lproto_dir exports/wo_lproto \\
        --full_dir exports/full \\
        --out_dir eigml_analysis \\
        --save_tsne_xy

Requires ``features.npy``, ``labels.npy``, ``scores.npy`` in each input directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


VARIANT_ORDER = ("base", "wo_psp", "wo_lproto", "full")
PANEL_LABELS = ("(a) Base", "(b) w/o PSP", "(c) w/o L_proto", "(d) Full")

# Wide / short aspect for the two single-row bar figures (slightly taller than ultra-flat).
_BAR_FIGSIZE_INCHES = (8.2, 2.85)

# Same color per variant on both bar figures: Base | w/o PSP | w/o L_proto | Full
_BAR_COLORS = ("#4C72B0", "#DD8452", "#55A868", "#C44E52")


def _expand_ylim_for_bar_labels(ax, *, top_frac: float = 0.14, bottom_frac: float = 0.06) -> None:
    """Extra vertical room so value labels are not clipped by axes spines."""
    lo, hi = ax.get_ylim()
    span = max(hi - lo, 1e-9)
    ax.set_ylim(lo - bottom_frac * span, hi + top_frac * span)


def _load_triplet(d: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fp = os.path.join(d, "features.npy")
    lp = os.path.join(d, "labels.npy")
    sp = os.path.join(d, "scores.npy")
    for p in (fp, lp, sp):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    X = np.load(fp)
    y = np.load(lp)
    s = np.load(sp)
    return X, y.astype(np.int64).ravel(), s.astype(np.float64).ravel()


def _tsne_pipeline(
    X: np.ndarray,
    seed: int,
    pca_dim: int,
    tsne_perp: float,
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize

    X = np.asarray(X, dtype=np.float64)
    Xn = normalize(X, norm="l2", axis=1)
    n_samples, n_feat = Xn.shape
    n_comp = min(int(pca_dim), max(1, n_samples - 1), n_feat)
    pca = PCA(n_components=n_comp, random_state=seed)
    Z = pca.fit_transform(Xn)
    perp = float(tsne_perp)
    if perp >= n_samples:
        perp = max(2.0, float(n_samples - 1) * 0.99)
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=seed,
        perplexity=perp,
    )
    xy = tsne.fit_transform(Z)
    return xy.astype(np.float64), Z.astype(np.float64)


def _metrics_2d(
    xy: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    only_correct: bool,
    case_correct: Optional[np.ndarray],
) -> Dict[str, Any]:
    pos = y == 1
    neg = y == 0
    mu_p = xy[pos].mean(axis=0)
    mu_n = xy[neg].mean(axis=0)
    d_inter = float(np.linalg.norm(mu_p - mu_n))

    intra_p = np.linalg.norm(xy[pos] - mu_p, axis=1).mean() if pos.any() else 0.0
    intra_n = np.linalg.norm(xy[neg] - mu_n, axis=1).mean() if neg.any() else 0.0
    d_intra = float(0.5 * (intra_p + intra_n))

    ratio = float(d_inter / d_intra) if d_intra > 1e-12 else float("inf")

    if only_correct and case_correct is not None:
        pos_scores = scores[pos]
        ok = case_correct.astype(bool)
        if ok.size != pos_scores.size:
            raise ValueError(
                f"case_correct length {ok.size} != number of positive rows {pos_scores.size}"
            )
        avg_pos_score = float(pos_scores[ok].mean()) if ok.any() else float("nan")
    else:
        avg_pos_score = float(scores[pos].mean()) if pos.any() else float("nan")

    return {
        "avg_positive_score": avg_pos_score,
        "inter_distance": d_inter,
        "intra_distance": d_intra,
        "ratio": ratio,
        "n_positive_points": int(pos.sum()),
        "n_negative_points": int(neg.sum()),
    }


def _plot_four_panels(
    xys: Dict[str, np.ndarray],
    ys: Dict[str, np.ndarray],
    out_pdf: str,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    _style = {
        "figure.facecolor": "#f6f7f9",
        "axes.facecolor": "#f6f7f9",
        "font.size": 10,
    }
    c_pos, c_neg = "#C44E52", "#4C72B0"
    with mpl.rc_context(_style):
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2), constrained_layout=True)
        for ax, key, title in zip(
            axes.ravel(),
            VARIANT_ORDER,
            PANEL_LABELS,
        ):
            xy = xys[key]
            y = ys[key]
            m = y > 0.5
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                c=c_pos,
                s=22,
                alpha=0.78,
                label="Positive",
                edgecolors="white",
                linewidths=0.35,
            )
            ax.scatter(
                xy[~m, 0],
                xy[~m, 1],
                c=c_neg,
                s=22,
                alpha=0.78,
                label="Negative",
                edgecolors="white",
                linewidths=0.35,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(title, fontsize=11)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)


def _plot_bar_avg_positive_score(
    metrics: Dict[str, Dict[str, Any]],
    out_pdf: str,
) -> None:
    """Mean of final matching scores on positive pairs only (same scale as training eval)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    labels = ["Base", "w/o PSP", "w/o L_proto", "Full"]
    keys = list(VARIANT_ORDER)
    scores = [metrics[k]["avg_positive_score"] for k in keys]

    with mpl.rc_context({"font.size": 10}):
        fig, ax = plt.subplots(figsize=_BAR_FIGSIZE_INCHES, constrained_layout=True)
        x = np.arange(len(labels))
        bars = ax.bar(
            x,
            scores,
            color=list(_BAR_COLORS),
            alpha=0.92,
            edgecolor="0.25",
            linewidth=0.65,
            zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Avg. positive score")
        ax.axhline(0.0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
        _expand_ylim_for_bar_labels(ax)
        ax.spines["top"].set_visible(False)
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        dy = 0.015 * yr
        for rect in bars:
            h = rect.get_height()
            if np.isfinite(h):
                xc = rect.get_x() + rect.get_width() / 2.0
                if h >= 0:
                    ax.text(
                        xc,
                        h + dy,
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        zorder=3,
                    )
                else:
                    ax.text(
                        xc,
                        h - dy,
                        f"{h:.3f}",
                        ha="center",
                        va="top",
                        fontsize=8,
                        zorder=3,
                    )
        os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)


def _plot_bar_inter_intra_ratio(
    metrics: Dict[str, Dict[str, Any]],
    out_pdf: str,
) -> None:
    """Inter/intra on 2D t-SNE coordinates only; not comparable to matching scores."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    labels = ["Base", "w/o PSP", "w/o L_proto", "Full"]
    keys = list(VARIANT_ORDER)
    ratios = [metrics[k]["ratio"] for k in keys]

    with mpl.rc_context({"font.size": 10}):
        fig, ax = plt.subplots(figsize=_BAR_FIGSIZE_INCHES, constrained_layout=True)
        x = np.arange(len(labels))
        bars = ax.bar(
            x,
            ratios,
            color=list(_BAR_COLORS),
            alpha=0.92,
            edgecolor="0.25",
            linewidth=0.65,
            zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Inter / intra")
        _expand_ylim_for_bar_labels(ax)
        ax.spines["top"].set_visible(False)
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        dy = 0.015 * yr
        for rect in bars:
            h = rect.get_height()
            if np.isfinite(h):
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    h + dy,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    zorder=3,
                )
        os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="PCA+t-SNE four-panel figure + metrics JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--wo_psp_dir", type=str, required=True)
    p.add_argument("--wo_lproto_dir", type=str, required=True)
    p.add_argument("--full_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="eigml_analysis")
    p.add_argument("--seed", type=int, default=2021, help="PCA + t-SNE random_state.")
    p.add_argument("--pca_dim", type=int, default=30)
    p.add_argument("--tsne_perplexity", type=float, default=20.0)
    p.add_argument(
        "--avg_score_only_correct",
        action="store_true",
        help="Average positive score only when argmax matches gold (needs --gold_argmax_npz).",
    )
    p.add_argument(
        "--gold_argmax_npz",
        type=str,
        default="",
        help="Optional npz with boolean 'correct' [500] per case (export script may add later).",
    )
    p.add_argument("--save_tsne_xy", action="store_true")
    args = p.parse_args()

    dirs = {
        "base": os.path.abspath(args.base_dir),
        "wo_psp": os.path.abspath(args.wo_psp_dir),
        "wo_lproto": os.path.abspath(args.wo_lproto_dir),
        "full": os.path.abspath(args.full_dir),
    }

    gold_argmax: Optional[np.ndarray] = None
    if args.avg_score_only_correct:
        if not args.gold_argmax_npz or not os.path.isfile(args.gold_argmax_npz):
            raise ValueError("--avg_score_only_correct requires --gold_argmax_npz with 'correct' [N_cases]")
        z = np.load(args.gold_argmax_npz, allow_pickle=True)
        gold_argmax = z["correct"].astype(bool).ravel()

    data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for k, d in dirs.items():
        data[k] = _load_triplet(d)

    seed = int(args.seed)
    pca_dim = int(args.pca_dim)
    perp = float(args.tsne_perplexity)

    xys: Dict[str, np.ndarray] = {}
    zs: Dict[str, np.ndarray] = {}
    ys: Dict[str, np.ndarray] = {}
    metrics: Dict[str, Any] = {
        "tsne": {
            "seed": seed,
            "pca_dim": pca_dim,
            "tsne_perplexity_requested": perp,
            "l2_normalize_rows": True,
            "tsne_init": "pca",
        },
        "variants": {},
    }

    for k in VARIANT_ORDER:
        X, y, s = data[k]
        xy, Z = _tsne_pipeline(X, seed, pca_dim, perp)
        xys[k] = xy
        zs[k] = Z
        ys[k] = y.astype(np.float64)
        m = _metrics_2d(
            xy,
            y,
            s,
            only_correct=bool(args.avg_score_only_correct),
            case_correct=gold_argmax,
        )
        metrics["variants"][k] = m
        if args.save_tsne_xy:
            out_npy = os.path.join(args.out_dir, f"tsne_xy_{k}.npy")
            os.makedirs(args.out_dir, exist_ok=True)
            np.save(out_npy, xy.astype(np.float32))

    out_dir = os.path.abspath(args.out_dir)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    _plot_four_panels(xys, ys, os.path.join(fig_dir, "tsne_4panel.pdf"))
    _plot_bar_avg_positive_score(
        metrics["variants"], os.path.join(fig_dir, "avg_positive_matching_score.pdf")
    )
    _plot_bar_inter_intra_ratio(
        metrics["variants"], os.path.join(fig_dir, "inter_intra_ratio_bar.pdf")
    )

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(
        f"Wrote {fig_dir}/tsne_4panel.pdf , "
        f"{fig_dir}/avg_positive_matching_score.pdf , "
        f"{fig_dir}/inter_intra_ratio_bar.pdf , {out_dir}/metrics.json"
    )


if __name__ == "__main__":
    main()
