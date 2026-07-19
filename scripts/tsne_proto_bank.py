#!/usr/bin/env python3
"""
t-SNE of factorized prototype bank vectors (style-space means), one point per prototype.

Uses model._get_eval_or_fresh_bank_factorization(device) -> proto_vectors aligned with the bank.

Example:
  python scripts/tsne_proto_bank.py \\
    --config configs/structured_factorized/v6_00_minimal_core.yaml \\
    --ckpt_path logs/.../epoch=9-step=33059.ckpt \\
    --tsne_output_png figures/tsne_proto_bank_dstc.png \\
    --tsne_output_npz figures/tsne_proto_bank_dstc.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from structured_retrieval import load_checkpoint_to_model, logger  # noqa: E402
from structured_retrieval_factorized import (  # noqa: E402
    StructuredFactorizedPLModel,
    parse_structured_factorized_args,
)

# Load _l2_pca_tsne from sibling script (scripts/ is not a Python package).
_tsne_fac_path = os.path.join(os.path.dirname(__file__), "tsne_factorized_stickers.py")
_tsne_spec = importlib.util.spec_from_file_location("tsne_factorized_stickers_fac", _tsne_fac_path)
_tsne_mod = importlib.util.module_from_spec(_tsne_spec)
assert _tsne_spec.loader is not None
_tsne_spec.loader.exec_module(_tsne_mod)
_l2_pca_tsne = _tsne_mod._l2_pca_tsne


def _parse_proto_tsne_flags(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--tsne_output_png", type=str, default="tsne_proto_bank.png")
    p.add_argument("--tsne_output_npz", type=str, default="")
    p.add_argument("--tsne_seed", type=int, default=None)
    p.add_argument("--tsne_perplexity", type=float, default=8.0)
    p.add_argument("--tsne_pca_dim", type=int, default=20)
    p.add_argument(
        "--proto_color_by",
        choices=("visual_style", "subject_category", "proto_source"),
        default="visual_style",
    )
    p.add_argument("--annotate_topk", type=int, default=8)
    return p.parse_known_args(list(argv))


def _safe_field(pr: Any, name: str, fallback: str = "") -> str:
    try:
        v = getattr(pr, name, fallback)
    except Exception:
        return fallback
    if v is None:
        return fallback
    s = str(v).strip()
    return s if s else fallback


def _is_singleton_source(src: str) -> bool:
    return str(src).strip().lower() == "singleton"


def _marker_for_source(src: str) -> str:
    s = str(src).strip().lower()
    if s == "fine":
        return "o"
    if s == "coarse":
        return "s"
    if s == "singleton":
        return "^"
    return "o"


def _effective_perplexity(user_perp: float, n_proto: int) -> float:
    cap = max(3.0, float((n_proto - 1) // 3))
    return float(min(float(user_perp), cap))


def _align_proto_metadata(
    bank: Any,
    proto_vectors: torch.Tensor,
    proto_density_t: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str], List[str]]:
    """
    Row i of proto_vectors corresponds to prototype proto_id == i.
    proto_density_t[j] corresponds to bank.prototypes[j] iteration order.
    """
    n = int(proto_vectors.shape[0])
    pv = proto_vectors.detach().cpu().numpy()
    pd_1d = proto_density_t.detach().cpu().numpy().reshape(-1)
    density_by_row = np.zeros(n, dtype=np.float64)
    id_map = {int(p.proto_id): p for p in bank.prototypes}
    for j, pr in enumerate(bank.prototypes):
        pid = int(pr.proto_id)
        if j < len(pd_1d) and 0 <= pid < n:
            density_by_row[pid] = float(pd_1d[j])

    sources: List[str] = []
    vstyles: List[str] = []
    subcats: List[str] = []
    idsumm: List[str] = []
    for i in range(n):
        pr = id_map.get(i)
        if pr is None:
            sources.append("unknown")
            vstyles.append("unknown")
            subcats.append("unknown")
            idsumm.append("")
            continue
        sources.append(_safe_field(pr, "proto_source", "unknown"))
        vstyles.append(_safe_field(pr, "visual_style", "unknown"))
        subcats.append(_safe_field(pr, "subject_category", "unknown"))
        idsumm.append(_safe_field(pr, "identity_summary", ""))

    return pv, density_by_row, sources, vstyles, subcats, idsumm


def _color_maps(
    labels: Sequence[str],
    max_named: int = 12,
) -> Tuple[np.ndarray, Dict[str, tuple]]:
    """Top frequent labels get tab10 colors; others gray."""
    cleaned = [str(x) if str(x).strip() else "(empty)" for x in labels]
    counts = Counter(cleaned)
    top_keys = [k for k, _ in counts.most_common(max_named)]
    import matplotlib

    cmap = matplotlib.cm.get_cmap("tab10", 10)
    key_to_rgba: Dict[str, tuple] = {}
    for i, k in enumerate(top_keys):
        key_to_rgba[k] = tuple(float(x) for x in cmap(i % 10))
    gray = (0.55, 0.55, 0.55, 1.0)
    colors = []
    for lab in cleaned:
        colors.append(key_to_rgba.get(lab, gray))
    return np.array(colors), key_to_rgba


def _annotate_indices(
    proto_sources: Sequence[str],
    density: np.ndarray,
    topk: int,
) -> List[int]:
    """Prefer non-singleton by density; fill with singletons if needed."""
    n = len(density)
    order = np.argsort(-density)
    non_sing = [i for i in order if not _is_singleton_source(proto_sources[i])]
    sing = [i for i in order if _is_singleton_source(proto_sources[i])]
    out: List[int] = []
    for i in non_sing:
        if len(out) >= topk:
            break
        out.append(int(i))
    for i in sing:
        if len(out) >= topk:
            break
        if int(i) not in out:
            out.append(int(i))
    return out[:topk]


def _plot_proto_bank(
    xy: np.ndarray,
    colors: np.ndarray,
    proto_sources: Sequence[str],
    density: np.ndarray,
    annotate_idx: Sequence[int],
    annotate_labels: Sequence[str],
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt

    d = np.asarray(density, dtype=np.float64)
    dmin, dmax = float(np.min(d)), float(np.max(d))
    span = max(dmax - dmin, 1e-12)
    sizes = 25.0 + 95.0 * (d - dmin) / span

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    by_marker: Dict[str, List[int]] = {"o": [], "s": [], "^": []}
    for i, src in enumerate(proto_sources):
        m = _marker_for_source(src)
        if m not in by_marker:
            m = "o"
        by_marker[m].append(i)

    for m, idxs in by_marker.items():
        if not idxs:
            continue
        idxs_arr = np.array(idxs, dtype=int)
        ax.scatter(
            xy[idxs_arr, 0],
            xy[idxs_arr, 1],
            s=sizes[idxs_arr],
            c=colors[idxs_arr],
            marker=m,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.3,
            label=f"{m} ({_marker_label(m)})",
        )

    for i, txt in zip(annotate_idx, annotate_labels):
        ax.annotate(
            txt,
            (xy[i, 0], xy[i, 1]),
            fontsize=7,
            alpha=0.95,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=8)
    fig.suptitle("t-SNE of prototype bank on DSTC10-MOD", fontsize=12)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _marker_label(m: str) -> str:
    return {"o": "fine", "s": "coarse", "^": "singleton"}.get(m, m)


def main() -> None:
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__ or "")
        print(
            "Required: --config ... --ckpt_path ...\n"
            "Optional: --tsne_output_png, --tsne_output_npz, --tsne_seed, --tsne_perplexity, "
            "--tsne_pca_dim, --proto_color_by, --annotate_topk."
        )
        return

    extra, rest = _parse_proto_tsne_flags(sys.argv[1:])
    rest = [t for t in rest if t and str(t).strip()]
    if not rest:
        raise SystemExit(
            "Usage: python scripts/tsne_proto_bank.py [--tsne_* ...] --config <yaml> ... --ckpt_path <ckpt>"
        )
    args = parse_structured_factorized_args(rest)
    ckpt = (getattr(args, "ckpt_path", None) or "").strip()
    if not ckpt:
        raise ValueError("--ckpt_path is required.")

    seed = int(extra.tsne_seed if extra.tsne_seed is not None else args.seed)
    np.random.seed(seed)
    import pytorch_lightning as pl

    pl.seed_everything(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(args, "allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(getattr(args, "allow_tf32", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pl_model = StructuredFactorizedPLModel(args)
    load_checkpoint_to_model(
        pl_model, ckpt, strict=bool(getattr(args, "strict_checkpoint_load", False))
    )
    model = pl_model.model
    model.eval()
    model.to(device)

    with torch.no_grad():
        _all_h, _sc, _sa, proto_vectors, proto_density = model._get_eval_or_fresh_bank_factorization(
            device
        )

    bank = model.style_bank
    if proto_vectors.shape[0] < 2:
        raise RuntimeError("Need at least 2 prototypes for t-SNE.")

    _pv, dens_align, sources, vstyles, subcats, idsumm = _align_proto_metadata(
        bank, proto_vectors, proto_density
    )

    color_key = str(extra.proto_color_by)
    if color_key == "visual_style":
        label_list = vstyles
    elif color_key == "subject_category":
        label_list = subcats
    else:
        label_list = sources

    colors_rgba, _ = _color_maps(label_list)
    n_proto = int(_pv.shape[0])
    perp_user = float(extra.tsne_perplexity)
    perp = _effective_perplexity(perp_user, n_proto)
    if perp < perp_user - 1e-6:
        logger.info(
            "[t-SNE proto bank] perplexity %.2f -> %.2f (n_proto=%d)",
            perp_user,
            perp,
            n_proto,
        )

    xy, z_pca = _l2_pca_tsne(_pv, seed, int(extra.tsne_pca_dim), perp)

    ann_idx = _annotate_indices(sources, dens_align, int(extra.annotate_topk))
    style_counts = Counter(vstyles)
    ann_texts: List[str] = []
    for i in ann_idx:
        st = vstyles[i] if i < len(vstyles) else ""
        if not st or st == "unknown":
            ann_texts.append(f"id={i}")
        elif style_counts[st] > 1:
            ann_texts.append(f"{st} ({i})"[:48])
        else:
            ann_texts.append(st[:48])

    if extra.tsne_output_png:
        _plot_proto_bank(
            xy=xy,
            colors=colors_rgba,
            proto_sources=sources,
            density=dens_align,
            annotate_idx=ann_idx,
            annotate_labels=ann_texts,
            out_path=extra.tsne_output_png,
        )

    if extra.tsne_output_npz:
        out = extra.tsne_output_npz
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        id_map = {int(p.proto_id): p for p in bank.prototypes}
        bank_proto_ids = np.array(
            [
                int(id_map[i].proto_id) if id_map.get(i) is not None else i
                for i in range(n_proto)
            ],
            dtype=np.int64,
        )
        np.savez(
            out,
            xy=xy.astype(np.float32),
            z_pca=z_pca.astype(np.float32),
            proto_ids=bank_proto_ids,
            proto_source=np.array(sources, dtype=object),
            visual_style=np.array(vstyles, dtype=object),
            subject_category=np.array(subcats, dtype=object),
            identity_summary=np.array(idsumm, dtype=object),
            proto_density=np.asarray(dens_align, dtype=np.float32),
            tsne_perplexity_used=np.float32(perp),
            tsne_pca_dim=np.int32(int(extra.tsne_pca_dim)),
            proto_color_by=np.array(color_key, dtype=object),
            seed=np.int64(seed),
        )
        logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
