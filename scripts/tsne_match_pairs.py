#!/usr/bin/env python3
"""
Unsupervised 2-D embedding (t-SNE or UMAP) of (dialogue, sticker) pair features:
True = gold sticker, False = negative sticker.

Feature modes:
  mmbert_pooler (default): CLS pooled representation from the multimodal BERT encoder (input to the
    2-way classification head — same path as outputs.logits / mmbert_score). Uses model.bert AFTER
    load_checkpoint_to_model(--ckpt_path): the BERT weights are exactly those stored in your
    checkpoint (same module as training), not a separate download.
    ca_only: concat(c, a) sticker factors only (no MM-BERT, ignores checkpoint BERT).
    concat_qeca: concat(q_style, q_expr, c, a); q is shared per dialogue — often overlaps in t-SNE.
    score_triplet: [mmbert_score, expr_score, graph_score] scalars.

Embeddings (unsupervised for paper figures): --tsne_embed_method tsne (default) or umap (needs
umap-learn). Tune --tsne_perplexity / --tsne_early_exaggeration (t-SNE) or --umap_n_neighbors /
--umap_min_dist (UMAP). True/False may overlap (shared dialogue); labels are for color only.

Optional --tsne_embed_method lda is supervised (not for claiming unsupervised separation).

Fair comparison vs baseline MMBERT (tsne_match_pairs_main.py):
  1) Same match_data_path, max_image_id (e.g. 307), pair_feature_mode=mmbert_pooler, --mmbert_repr pooler on baseline.
  2) Same --tsne_seed and all --tsne_* / --umap_* / --plot_dpi.
  3) Run one model with --match_save_pair_spec_npz figures/pair_spec.npz; run the other with
     --match_pair_spec_npz figures/pair_spec.npz so dialogues and random negatives match exactly.

Example (factorized):
  python scripts/tsne_match_pairs.py \\
    --config configs/structured_factorized/v6_00_minimal_core.yaml \\
    --ckpt_path logs/.../epoch=9.ckpt \\
    --match_data_path data/validation_pair_with_cand.json \\
    --tsne_output_png figures/tsne_match_dstc.png

Example (structured_residual — mmbert_pooler only):
  python scripts/tsne_match_pairs.py --structured_model residual \\
    --config configs/structured_residual/v3_01_expr_residual.yaml \\
    --ckpt_path logs/.../epoch=8.ckpt \\
    --match_data_path data/validation_pair_with_cand.json \\
    --tsne_output_png figures/tsne_match_residual.png
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from main import PLDataset  # noqa: E402
from structured_retrieval import load_checkpoint_to_model, logger  # noqa: E402
from structured_retrieval_factorized import (  # noqa: E402
    StructuredFactorizedPLModel,
    parse_structured_factorized_args,
)
from structured_retrieval_residual import (  # noqa: E402
    StructuredResidualPLModel,
    parse_structured_residual_args,
)

def _embed_pair_2d(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    pca_dim: int,
    perplexity: float,
    method: str,
    l2_norm: bool,
    early_exaggeration: float,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map features to 2D for plotting.
    tsne: L2 (optional) -> PCA -> t-SNE (same as sticker script).
    umap: L2 (optional) -> PCA -> UMAP on PCA features (unsupervised).
    lda: L2 (optional) -> LDA 1st axis (supervised) + PCA 2nd axis on X (unsupervised y).
    """
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize

    X = np.asarray(X, dtype=np.float64)
    if l2_norm:
        Xp = normalize(X, norm="l2", axis=1)
    else:
        Xp = X.copy()
    n_samples, n_feat = Xp.shape
    y_int = np.asarray(y, dtype=np.int64).ravel()

    n_comp = min(int(pca_dim), max(1, n_samples - 1), n_feat)
    pca = PCA(n_components=n_comp, random_state=seed)
    Z = pca.fit_transform(Xp)

    if method == "lda":
        lda = LinearDiscriminantAnalysis(n_components=1)
        x1 = lda.fit_transform(Xp, y_int).ravel()
        want = max(2, min(int(pca_dim), n_feat, max(1, n_samples - 1)))
        pca2 = PCA(n_components=want, random_state=seed)
        Z2 = pca2.fit_transform(Xp)
        y2 = Z2[:, 1] if Z2.shape[1] >= 2 else np.zeros(n_samples, dtype=np.float64)
        xy = np.column_stack([x1, y2])
        return xy, Z

    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as e:
            raise ImportError(
                "tsne_embed_method=umap requires the umap-learn package. "
                "Install with: pip install umap-learn"
            ) from e
        nn = int(umap_n_neighbors)
        nn = max(2, min(nn, n_samples - 1))
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=nn,
            min_dist=float(umap_min_dist),
            metric=str(umap_metric),
            verbose=False,
        )
        xy = reducer.fit_transform(Z)
        return xy, Z

    if method == "tsne":
        perp = float(perplexity)
        if perp >= n_samples:
            perp = max(2.0, float(n_samples - 1) * 0.99)
        tsne = TSNE(
            n_components=2,
            init="pca",
            random_state=seed,
            perplexity=perp,
            early_exaggeration=float(early_exaggeration),
        )
        xy = tsne.fit_transform(Z)
        return xy, Z

    raise ValueError(f"Unknown tsne_embed_method={method!r}")


def _features_mmbert_pooler(
    model: Any,
    bank_h: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    sticker_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Multimodal BERT encoder pooled output (feeds the 2-class classifier that defines mmbert_score).
    Same forward path as _compute_pair_logits -> compute_base_score.
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    sid = sticker_ids.to(device).long().clamp(0, bank_h.size(0) - 1)
    cand_h = bank_h.index_select(0, sid)
    cand_ids = [int(sticker_ids[i].item()) for i in range(int(sticker_ids.size(0)))]
    text_emb = model._get_text_word_embeddings(input_ids)
    input_emb, full_attention_mask, token_type_ids = model._build_multimodal_inputs(
        text_emb=text_emb,
        attention_mask=attention_mask,
        img_emb=cand_h,
        img_ids=cand_ids,
        dialogue_q=None,
    )
    inner = model.bert.bert(
        inputs_embeds=input_emb,
        attention_mask=full_attention_mask,
        token_type_ids=token_type_ids,
        return_dict=True,
    )
    pooled = getattr(inner, "pooler_output", None)
    if pooled is None:
        pooled = inner.last_hidden_state[:, 0, :]
    return pooled


def _parse_match_flags(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--match_data_path",
        type=str,
        default="./data/validation_pair_with_cand.json",
        help="DSTC-style pair JSON (same as PLDataset).",
    )
    p.add_argument(
        "--match_n_pairs",
        type=int,
        default=200,
        help="Number of dialogues to use; each yields one True + one False pair (2 points each).",
    )
    p.add_argument(
        "--pair_feature_mode",
        choices=("mmbert_pooler", "ca_only", "concat_qeca", "score_triplet"),
        default="mmbert_pooler",
        help="mmbert_pooler: BERT CLS pooler (pre-classifier, same as ranking logits). "
        "ca_only / concat_qeca / score_triplet: see script docstring.",
    )
    p.add_argument("--match_batch_size", type=int, default=16)
    p.add_argument(
        "--tsne_output_png",
        type=str,
        default="tsne_match_pairs.png",
        help="Output figure path: use .pdf for vector PDF or .png for raster.",
    )
    p.add_argument(
        "--plot_show_title",
        action="store_true",
        help="If set, add a figure title (default: no title, for paper panels).",
    )
    p.add_argument("--tsne_output_npz", type=str, default="")
    p.add_argument("--tsne_seed", type=int, default=None)
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_pca_dim", type=int, default=50)
    p.add_argument(
        "--tsne_embed_method",
        choices=("tsne", "umap", "lda"),
        default="tsne",
        help="tsne or umap: unsupervised (paper-friendly). lda: supervised linear axis (internal only).",
    )
    p.add_argument(
        "--tsne_no_l2_norm",
        action="store_true",
        help="Disable L2 row normalization before PCA/t-SNE/LDA (default: normalize).",
    )
    p.add_argument(
        "--tsne_early_exaggeration",
        type=float,
        default=12.0,
        help="t-SNE early_exaggeration (only for tsne). Try 24-48 for tighter clusters.",
    )
    p.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (lower = more local structure; only for tsne_embed_method=umap).",
    )
    p.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.08,
        help="UMAP min_dist (smaller = tighter clusters; only for umap).",
    )
    p.add_argument(
        "--umap_metric",
        type=str,
        default="cosine",
        help="UMAP distance metric (e.g. cosine, euclidean).",
    )
    p.add_argument(
        "--plot_dpi",
        type=int,
        default=200,
        help="Figure DPI for PNG output.",
    )
    p.add_argument(
        "--match_pair_spec_npz",
        type=str,
        default="",
        help="Load dataset_indices + neg_sticker_id from a prior run (fair comparison vs another model).",
    )
    p.add_argument(
        "--match_save_pair_spec_npz",
        type=str,
        default="",
        help="Save dataset_indices + neg_sticker_id after this run; pass to --match_pair_spec_npz for the other model.",
    )
    p.add_argument(
        "--structured_model",
        choices=("factorized", "residual"),
        default="factorized",
        help="factorized: YAML from structured_factorized + factorized style bank. "
        "residual: YAML from structured_residual + CLIP all_img_embs (mmbert_pooler only).",
    )
    return p.parse_known_args(list(argv))


def _collate_sents(
    tokenizer: Any,
    sents: Sequence[str],
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    res = tokenizer(
        list(sents),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
    )
    return res["input_ids"], res["attention_mask"]


def _load_pair_spec_npz(path: str) -> Tuple[List[int], Dict[int, int]]:
    """Returns (ordered dataset row indices, mapping row_idx -> neg sticker id used)."""
    spec = np.load(path, allow_pickle=True)
    ind = spec["dataset_indices"].astype(np.int64).tolist()
    neg = spec["neg_sticker_id"].astype(np.int64).tolist()
    if len(ind) != len(neg):
        raise ValueError(
            f"pair spec length mismatch: dataset_indices={len(ind)} neg_sticker_id={len(neg)}"
        )
    return ind, {int(i): int(n) for i, n in zip(ind, neg)}


def _gather_valid_indices(
    ds: PLDataset,
    max_image_id: int,
    rng: random.Random,
    n_pairs: int,
) -> List[int]:
    """Indices into ds that have img_id and a usable negative."""
    valid: List[int] = []
    for i in range(len(ds)):
        sample = ds[i]
        pos = sample.get("img_id")
        if pos is None:
            continue
        pos = int(pos)
        if pos < 0 or pos >= max_image_id:
            continue
        neg = sample.get("neg_img_id")
        if neg is not None:
            neg = int(neg)
            if neg == pos or neg < 0 or neg >= max_image_id:
                neg = None
        if neg is None:
            pool = [j for j in range(max_image_id) if j != pos]
            if not pool:
                continue
        valid.append(i)
    if not valid:
        raise RuntimeError("No valid samples with img_id in range [0, max_image_id).")
    rng.shuffle(valid)
    return valid[: min(n_pairs, len(valid))]


def _features_ca_only(
    bank_c: torch.Tensor,
    bank_a: torch.Tensor,
    sticker_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Only sticker style + expression (2*D). No query — avoids duplicate q for pos/neg pairs."""
    sid = sticker_ids.to(device).long().clamp(0, bank_c.size(0) - 1)
    c = bank_c.index_select(0, sid)
    a = bank_a.index_select(0, sid)
    return torch.cat([c, a], dim=-1)


def _features_concat(
    model: Any,
    bank_c: torch.Tensor,
    bank_a: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    sticker_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """sticker_ids [B] long on CPU or device."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    sid = sticker_ids.to(device).long().clamp(0, bank_c.size(0) - 1)
    q_style, q_expr, _ = model.encode_style_expr_queries(input_ids, attention_mask)
    c = bank_c.index_select(0, sid)
    a = bank_a.index_select(0, sid)
    return torch.cat([q_style, q_expr, c, a], dim=-1)


def _features_score_triplet(
    model: Any,
    bank_h: torch.Tensor,
    bank_c: torch.Tensor,
    bank_a: torch.Tensor,
    proto_vectors: torch.Tensor,
    proto_density: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    sticker_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """[B, 3] mmbert, expr, graph."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    b = int(input_ids.size(0))
    sid = sticker_ids.to(device).long().clamp(0, bank_h.size(0) - 1)
    cand_h = bank_h.index_select(0, sid)
    cand_c = bank_c.index_select(0, sid)
    cand_a = bank_a.index_select(0, sid)
    cand_ids = [int(sticker_ids[i].item()) for i in range(b)]

    mmbert = model._compute_mmbert_score_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        candidate_ids=cand_ids,
        candidate_h=cand_h,
    )
    q_style, q_expr, _ = model.encode_style_expr_queries(input_ids, attention_mask)
    _ss, es = model._compute_candidate_factor_scores(q_style, q_expr, cand_c, cand_a)
    proto_logits = model._compute_proto_logits(q_style, proto_vectors)
    gs = model._gather_proto_scores_for_batch(
        proto_logits, cand_ids, proto_density=proto_density
    )
    return torch.stack([mmbert.view(-1), es.view(-1), gs.view(-1)], dim=-1)


def _plot_match(
    xy: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    feature_mode: str,
    embed_method: str,
    dpi: int,
    show_title: bool = False,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    _style = {
        "figure.facecolor": "#f6f7f9",
        "axes.facecolor": "#f6f7f9",
        "axes.edgecolor": "#d0d4dc",
        "axes.linewidth": 0.8,
        "font.size": 10,
        "legend.frameon": True,
        "legend.fancybox": True,
    }
    with mpl.rc_context(_style):
        fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
        fig.patch.set_facecolor("#f6f7f9")
        ax.set_facecolor("#f6f7f9")

        m_true = labels > 0.5
        m_false = ~m_true
        c_pos, c_neg = "#C44E52", "#4C72B0"
        ax.scatter(
            xy[m_true, 0],
            xy[m_true, 1],
            c=c_pos,
            s=32,
            alpha=0.82,
            label="Positive pair",
            edgecolors="white",
            linewidths=0.45,
            zorder=2,
        )
        ax.scatter(
            xy[m_false, 0],
            xy[m_false, 1],
            c=c_neg,
            s=32,
            alpha=0.82,
            label="Negative pair",
            edgecolors="white",
            linewidths=0.45,
            zorder=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal", adjustable="datalim")

        pretty = {"tsne": "t-SNE", "umap": "UMAP", "lda": "LDA"}.get(embed_method, embed_method)
        if embed_method == "lda":
            title = f"Supervised linear embedding ({pretty}) — {feature_mode}"
        else:
            title = f"Unsupervised embedding ({pretty}) — {feature_mode}"
        ax.legend(loc="upper right", fontsize=9, framealpha=0.92, edgecolor="#e0e3e8")
        if show_title:
            fig.suptitle(
                title,
                fontsize=12,
                fontweight="medium",
                color="#2a2a2a",
                y=1.02,
            )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".pdf":
            fig.savefig(
                out_path,
                format="pdf",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
            )
        else:
            fig.savefig(out_path, dpi=int(dpi), facecolor=fig.get_facecolor())
        plt.close(fig)
    logger.info("Wrote %s", out_path)


def main() -> None:
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__ or "")
        return

    extra, rest = _parse_match_flags(sys.argv[1:])
    rest = [t for t in rest if t and str(t).strip()]
    if not rest:
        raise SystemExit(
            "Usage: python scripts/tsne_match_pairs.py [--match_* ...] --config <yaml> ... --ckpt_path <ckpt>"
        )
    backend = str(getattr(extra, "structured_model", "factorized"))
    if backend == "residual":
        args = parse_structured_residual_args(rest)
    else:
        args = parse_structured_factorized_args(rest)
    ckpt = (getattr(args, "ckpt_path", None) or "").strip()
    if not ckpt:
        raise ValueError("--ckpt_path is required.")

    seed = int(extra.tsne_seed if extra.tsne_seed is not None else args.seed)
    random.seed(seed)
    np.random.seed(seed)
    import pytorch_lightning as pl

    pl.seed_everything(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(args, "allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(getattr(args, "allow_tf32", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if backend == "residual":
        pl_model = StructuredResidualPLModel(args)
        load_checkpoint_to_model(
            pl_model, ckpt, strict=bool(getattr(args, "strict_checkpoint_load", False))
        )
        logger.info(
            "[tsne_match_pairs] Checkpoint weights applied to StructuredResidualPLModel (%s).",
            ckpt,
        )
    else:
        pl_model = StructuredFactorizedPLModel(args)
        load_checkpoint_to_model(
            pl_model, ckpt, strict=bool(getattr(args, "strict_checkpoint_load", False))
        )
        logger.info(
            "[tsne_match_pairs] Checkpoint weights applied to StructuredFactorizedPLModel (%s).",
            ckpt,
        )

    model = pl_model.model
    model.eval()
    model.to(device)
    tokenizer = model.bert_tokenizer
    logger.info(
        "[tsne_match_pairs] structured_model=%s pair_feature_mode=%s (in-memory BERT).",
        backend,
        str(extra.pair_feature_mode),
    )

    if backend == "residual" and str(extra.pair_feature_mode) != "mmbert_pooler":
        raise ValueError(
            "structured_model=residual only supports pair_feature_mode=mmbert_pooler "
            "(no factorized style bank for ca_only / concat_qeca / score_triplet)."
        )

    if getattr(model, "uses_full_variant", lambda: False)():
        logger.warning(
            "[tsne_match_pairs] factorized_variant=full: score_triplet uses minimal-style scalars "
            "(mmbert+expr+graph); for fusion-aligned scores use concat_qeca or inspect forward_eval."
        )

    mode = str(extra.pair_feature_mode)
    if mode == "concat_qeca":
        logger.info(
            "[tsne_match_pairs] concat_qeca: True/False from the same dialogue share identical "
            "q_style,q_expr — expect t-SNE overlap; prefer mmbert_pooler (default)."
        )

    data_path = (extra.match_data_path or "").strip()
    if not data_path or not os.path.isfile(data_path):
        raise FileNotFoundError(f"match_data_path not found: {data_path!r}")

    ds = PLDataset(data_path, "test", args, tokenizer)
    rng = random.Random(seed)
    max_image_id = int(args.max_image_id)
    spec_path = (getattr(extra, "match_pair_spec_npz", None) or "").strip()
    idx_to_neg: Optional[Dict[int, int]] = None
    if spec_path:
        if not os.path.isfile(spec_path):
            raise FileNotFoundError(f"match_pair_spec_npz not found: {spec_path!r}")
        indices, idx_to_neg = _load_pair_spec_npz(spec_path)
        _spec = np.load(spec_path, allow_pickle=True)
        if "max_image_id" in _spec.files and int(_spec["max_image_id"]) != max_image_id:
            logger.warning(
                "[tsne_match_pairs] pair spec max_image_id=%s != current %s (unfair if sticker id space differs)",
                int(_spec["max_image_id"]),
                max_image_id,
            )
        logger.info(
            "[tsne_match_pairs] Loaded %d dialogues from pair spec %s",
            len(indices),
            spec_path,
        )
    else:
        indices = _gather_valid_indices(ds, max_image_id, rng, int(extra.match_n_pairs))

    style_bank_c: Any
    style_bank_a: Any
    proto_vectors: Any
    proto_density: Any

    with torch.no_grad():
        if backend == "residual":
            logger.info("[tsne_match_pairs] prepare_for_test() (CLIP sticker bank for residual)...")
            model.prepare_for_test()
            bank_all_h = model.all_img_embs
            if bank_all_h is None:
                raise RuntimeError("Residual model: all_img_embs is None after prepare_for_test().")
            bank_all_h = bank_all_h.to(device)
            style_bank_c = None
            style_bank_a = None
            proto_vectors = None
            proto_density = None
        else:
            bank_all_h, style_bank_c, style_bank_a, proto_vectors, proto_density = (
                model._get_eval_or_fresh_bank_factorization(device)
            )

    feats: List[np.ndarray] = []
    labs: List[float] = []
    pos_ids: List[int] = []
    neg_ids: List[int] = []

    bs = max(1, int(extra.match_batch_size))

    for start in range(0, len(indices), bs):
        chunk_idx = indices[start : start + bs]
        sents: List[str] = []
        p_ids: List[int] = []
        n_ids: List[int] = []
        for ix in chunk_idx:
            sample = ds[ix]
            pos = int(sample["img_id"])
            if idx_to_neg is not None:
                neg = idx_to_neg[ix]
            else:
                neg = sample.get("neg_img_id")
                if neg is not None:
                    neg = int(neg)
                    if neg == pos or neg < 0 or neg >= max_image_id:
                        neg = None
                if neg is None:
                    pool = [j for j in range(max_image_id) if j != pos]
                    neg = rng.choice(pool)
            sents.append(sample["sent"])
            p_ids.append(pos)
            n_ids.append(neg)

        inp, mask = _collate_sents(
            tokenizer, sents, int(getattr(args, "max_dialogue_length", 490) or 490)
        )
        p_t = torch.tensor(p_ids, dtype=torch.long)
        n_t = torch.tensor(n_ids, dtype=torch.long)

        if mode == "mmbert_pooler":
            fp = _features_mmbert_pooler(model, bank_all_h, inp, mask, p_t, device)
            fn = _features_mmbert_pooler(model, bank_all_h, inp, mask, n_t, device)
        elif mode == "ca_only":
            fp = _features_ca_only(style_bank_c, style_bank_a, p_t, device)
            fn = _features_ca_only(style_bank_c, style_bank_a, n_t, device)
        elif mode == "concat_qeca":
            fp = _features_concat(
                model, style_bank_c, style_bank_a, inp, mask, p_t, device
            )
            fn = _features_concat(
                model, style_bank_c, style_bank_a, inp, mask, n_t, device
            )
        elif mode == "score_triplet":
            fp = _features_score_triplet(
                model,
                bank_all_h,
                style_bank_c,
                style_bank_a,
                proto_vectors,
                proto_density,
                inp,
                mask,
                p_t,
                device,
            )
            fn = _features_score_triplet(
                model,
                bank_all_h,
                style_bank_c,
                style_bank_a,
                proto_vectors,
                proto_density,
                inp,
                mask,
                n_t,
                device,
            )
        else:
            raise ValueError(f"Unknown pair_feature_mode={mode!r}")

        fp_np = fp.detach().cpu().numpy()
        fn_np = fn.detach().cpu().numpy()
        for i in range(fp_np.shape[0]):
            feats.append(fp_np[i])
            feats.append(fn_np[i])
            labs.append(1.0)
            labs.append(0.0)
            pos_ids.append(p_ids[i])
            neg_ids.append(n_ids[i])

    save_spec = (getattr(extra, "match_save_pair_spec_npz", None) or "").strip()
    if save_spec and idx_to_neg is None:
        os.makedirs(os.path.dirname(save_spec) or ".", exist_ok=True)
        np.savez_compressed(
            save_spec,
            dataset_indices=np.array(indices, dtype=np.int64),
            neg_sticker_id=np.array(neg_ids, dtype=np.int64),
            match_data_path=np.array(data_path, dtype=object),
            max_image_id=np.int32(max_image_id),
            tsne_seed=np.int64(seed),
        )
        logger.info("[tsne_match_pairs] Wrote pair spec for fair replay: %s", save_spec)

    if len(feats) < 4:
        raise RuntimeError("Too few pair features for t-SNE.")

    X = np.stack(feats, axis=0)
    y = np.array(labs, dtype=np.float32)
    n = X.shape[0]
    perp_user = float(extra.tsne_perplexity)
    perp = float(min(perp_user, max(2.0, float(n - 1) * 0.99)))
    if perp < perp_user - 1e-3:
        logger.info("[tsne_match_pairs] perplexity %.2f -> %.2f (n=%d)", perp_user, perp, n)

    embed_m = str(extra.tsne_embed_method)
    l2 = not bool(getattr(extra, "tsne_no_l2_norm", False))
    xy, z_pca = _embed_pair_2d(
        X,
        y,
        seed,
        int(extra.tsne_pca_dim),
        perp,
        embed_m,
        l2,
        float(getattr(extra, "tsne_early_exaggeration", 12.0)),
        int(getattr(extra, "umap_n_neighbors", 15)),
        float(getattr(extra, "umap_min_dist", 0.08)),
        str(getattr(extra, "umap_metric", "cosine")),
    )
    if embed_m == "lda":
        logger.info(
            "[tsne_match_pairs] lda: x=LDA (separates True/False linearly on this sample), "
            "y=2nd PCA component of features (for spread)."
        )
    elif embed_m == "umap":
        logger.info(
            "[tsne_match_pairs] umap: unsupervised 2-D on PCA features "
            "(n_neighbors=%s, min_dist=%s, metric=%s).",
            getattr(extra, "umap_n_neighbors", 15),
            getattr(extra, "umap_min_dist", 0.08),
            getattr(extra, "umap_metric", "cosine"),
        )

    plot_dpi = int(getattr(extra, "plot_dpi", 200))
    if extra.tsne_output_png:
        _plot_match(
            xy,
            y,
            extra.tsne_output_png,
            mode,
            embed_m,
            plot_dpi,
            show_title=bool(getattr(extra, "plot_show_title", False)),
        )

    if extra.tsne_output_npz:
        out = extra.tsne_output_npz
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        sticker_per_point: List[int] = []
        for pi, ni in zip(pos_ids, neg_ids):
            sticker_per_point.append(int(pi))
            sticker_per_point.append(int(ni))
        pair_label = np.tile(np.array([1.0, 0.0], dtype=np.float32), len(pos_ids))
        np.savez(
            out,
            xy=xy.astype(np.float32),
            z_pca=z_pca.astype(np.float32),
            labels=pair_label,
            sticker_id=np.array(sticker_per_point, dtype=np.int64),
            feature_mode=np.array(mode, dtype=object),
            structured_model=np.array(backend, dtype=object),
            pos_sticker_id=np.array(pos_ids, dtype=np.int64),
            neg_sticker_id=np.array(neg_ids, dtype=np.int64),
            tsne_perplexity_used=np.float32(perp),
            tsne_pca_dim=np.int32(int(extra.tsne_pca_dim)),
            tsne_embed_method=np.array(embed_m, dtype=object),
            tsne_l2_norm=np.bool_(l2),
            umap_n_neighbors=np.int32(int(getattr(extra, "umap_n_neighbors", 15))),
            umap_min_dist=np.float32(float(getattr(extra, "umap_min_dist", 0.08))),
            umap_metric=np.array(str(getattr(extra, "umap_metric", "cosine")), dtype=object),
            plot_dpi=np.int32(plot_dpi),
            seed=np.int64(seed),
        )
        logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
