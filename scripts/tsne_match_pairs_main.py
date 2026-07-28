#!/usr/bin/env python3
"""
Unsupervised t-SNE / UMAP for (dialogue, sticker) pairs using the baseline MM-BERT from main.py
(PLModel + Model), e.g. checkpoints produced by run.sh with --model_choice=use_img_clip.

What you are embedding (important):
  It is NOT "pure text BERT" and NOT raw CLIP vectors. The model builds one sequence:
  [ dialogue word embeddings | img_ff(CLIP sticker) as one token | SEP ], then runs the
  **BERT encoder** (same stack as training/inference). Default --mmbert_repr pooler uses the
  encoder pooler (or CLS fallback), i.e. the representation the classifier head is built on.
  t-SNE often shows arcs / horseshoes on semantic manifolds; that is a known projection artifact,
  not proof that the code skipped BERT.

This is NOT for structured_factorized checkpoints; use scripts/tsne_match_pairs.py for those.

Requires the same data args as training (fix_img, add_ocr_info, max_image_id, bert/clip paths).
Pass --mmbert_baseline_defaults to prepend run.sh-style training defaults; override any flag after.

Example:
  python scripts/tsne_match_pairs_main.py --mmbert_baseline_defaults \\
    --ckpt_path logs/clip/lightning_logs/version_6/checkpoints/epoch=7-step=211575.ckpt \\
    --match_data_path data/validation_pair_with_cand.json \\
    --tsne_output_png figures/tsne_match_baseline_clip_v6.png
"""

from __future__ import annotations

import argparse
import importlib.util
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

from main import Arguments, PLDataset, PLModel  # noqa: E402
from structured_retrieval import load_checkpoint_to_model, logger  # noqa: E402
from transformers import HfArgumentParser  # noqa: E402

_pair_mod_path = os.path.join(os.path.dirname(__file__), "tsne_match_pairs.py")
_pair_spec = importlib.util.spec_from_file_location("tsne_match_pairs_baseline", _pair_mod_path)
assert _pair_spec and _pair_spec.loader
_m = importlib.util.module_from_spec(_pair_spec)
_pair_spec.loader.exec_module(_m)
_collate_sents = _m._collate_sents
_embed_pair_2d = _m._embed_pair_2d
_gather_valid_indices = _m._gather_valid_indices
_parse_match_flags = _m._parse_match_flags
_plot_match = _m._plot_match
_load_pair_spec_npz = _m._load_pair_spec_npz

# Matches run.sh train block (mode 0) for model/data switches.
_RUN_SH_MODEL_DEFAULTS: List[str] = [
    "--model_choice",
    "use_img_clip",
    "--max_image_id",
    "307",
    "--fix_text",
    "false",
    "--fix_img",
    "true",
    "--add_ocr_info",
    "false",
    "--add_emotion_task",
    "false",
    "--add_predict_context_task",
    "false",
    "--add_predict_img_label_task",
    "false",
    "--bert_pretrain_path",
    "./ckpt/bert-base-chinese",
    "--img_pretrain_path",
    "./ckpt/clip-ViT-B-32",
    "--local_files_only",
    "true",
]


def _features_mmbert_pooler_main(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    sticker_ids: torch.Tensor,
    device: torch.device,
    repr_kind: str,
) -> torch.Tensor:
    """Encoder output; same multimodal layout as Model.forward (no OCR path when add_ocr_info=False)."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    cand_ids = [int(sticker_ids[i].item()) for i in range(int(sticker_ids.size(0)))]
    img_emb = model.get_emb_by_imgids(cand_ids).to(device)
    batch_size = img_emb.size(0)
    sep_id = torch.tensor(
        model.bert_tokenizer.sep_token_id, device=device, dtype=torch.long
    )
    sep_emb = model.bert.bert.embeddings.word_embeddings(sep_id).unsqueeze(0).unsqueeze(0).repeat(
        batch_size, 1, 1
    )
    text_emb = model.bert.bert.embeddings.word_embeddings(input_ids)

    if getattr(model.args, "use_visual_personalization_token", False):
        raise NotImplementedError(
            "tsne_match_pairs_main: use_visual_personalization_token=True is not supported."
        )

    img_tok = model.img_ff(img_emb).unsqueeze(1)
    if model.args.add_ocr_info:
        cls_inputs_ts: List[Any] = []
        for img_id in cand_ids:
            _a, _b, cls_inputs = model.get_input_output_imglabel_by_imgid(int(img_id))
            cls_inputs_ts.append(cls_inputs)
        cls_inputs_ts_t = torch.tensor(cls_inputs_ts, device=device, dtype=torch.long)
        cls_inputs_emb = model.bert.bert.embeddings.word_embeddings(cls_inputs_ts_t)
        input_emb = torch.cat([text_emb, cls_inputs_emb, img_tok, sep_emb], dim=1)
    else:
        input_emb = torch.cat([text_emb, img_tok, sep_emb], dim=1)

    extra_len = int(input_emb.size(1) - text_emb.size(1))
    ones_mask = torch.ones(batch_size, extra_len, device=device)
    full_attention_mask = torch.cat([attention_mask, ones_mask], dim=1)
    token_type_ids = torch.zeros_like(full_attention_mask, dtype=torch.long)
    token_type_ids[:, -extra_len:] = 1

    inner = model.bert.bert(
        inputs_embeds=input_emb,
        attention_mask=full_attention_mask,
        token_type_ids=token_type_ids,
        return_dict=True,
    )
    if repr_kind == "last_hidden_cls":
        return inner.last_hidden_state[:, 0, :].contiguous()
    # pooler: matches the vector fed to dropout+classifier in BertForSequenceClassification
    pooled = getattr(inner, "pooler_output", None)
    if pooled is None:
        logger.warning(
            "[tsne_match_pairs_main] pooler_output is None; using last_hidden_state[:,0] (raw CLS)."
        )
        pooled = inner.last_hidden_state[:, 0, :]
    return pooled.contiguous()


def _preprocess_argv(argv: Sequence[str]) -> Tuple[List[str], str]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--mmbert_baseline_defaults",
        action="store_true",
        help="Prepend run.sh-style model args (user flags later still override if duplicated).",
    )
    p.add_argument(
        "--mmbert_repr",
        choices=("pooler", "last_hidden_cls"),
        default="pooler",
        help="pooler: encoder pooler (input to classifier); last_hidden_cls: last-layer CLS before pooler MLP.",
    )
    known, unknown = p.parse_known_args(list(argv))
    out = list(unknown)
    if known.mmbert_baseline_defaults:
        out = _RUN_SH_MODEL_DEFAULTS + out
    return out, str(known.mmbert_repr)


def main() -> None:
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__ or "")
        return

    argv, mmbert_repr = _preprocess_argv(sys.argv[1:])
    extra, rest = _parse_match_flags(argv)
    rest = [t for t in rest if t and str(t).strip()]
    if not rest:
        raise SystemExit(
            "Usage: python scripts/tsne_match_pairs_main.py [--mmbert_baseline_defaults] "
            "[--match_* ...] --ckpt_path <ckpt> [--bert_pretrain_path ...] ..."
        )

    hf_parser = HfArgumentParser(Arguments)
    args = hf_parser.parse_args_into_dataclasses(rest)[0]

    ckpt = (getattr(args, "ckpt_path", None) or "").strip()
    if not ckpt:
        raise ValueError("--ckpt_path is required.")

    mode = str(extra.pair_feature_mode)
    if mode != "mmbert_pooler":
        raise ValueError(
            f"tsne_match_pairs_main only supports pair_feature_mode=mmbert_pooler (got {mode!r}). "
            "Use scripts/tsne_match_pairs.py for factorized modes."
        )

    seed = int(extra.tsne_seed if extra.tsne_seed is not None else args.seed)
    random.seed(seed)
    np.random.seed(seed)
    import pytorch_lightning as pl

    pl.seed_everything(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.local_files_only:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    pl_model = PLModel(args)
    load_checkpoint_to_model(
        pl_model,
        ckpt,
        strict=bool(getattr(args, "strict_checkpoint_load", False)),
    )
    model = pl_model.model
    model.eval()
    model.to(device)
    logger.info(
        "[tsne_match_pairs_main] MM-BERT features: repr=%s — encoder on sequence "
        "[dialogue | img_ff(CLIP) | SEP]; not CLIP-only, not text-only BERT.",
        mmbert_repr,
    )
    logger.info("[tsne_match_pairs_main] prepare_for_test() (full sticker bank embeddings)...")
    model.prepare_for_test()

    tokenizer = model.bert_tokenizer
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
                "[tsne_match_pairs_main] pair spec max_image_id=%s != current %s",
                int(_spec["max_image_id"]),
                max_image_id,
            )
        logger.info(
            "[tsne_match_pairs_main] Loaded %d dialogues from pair spec %s",
            len(indices),
            spec_path,
        )
    else:
        indices = _gather_valid_indices(ds, max_image_id, rng, int(extra.match_n_pairs))

    feats: List[np.ndarray] = []
    labs: List[float] = []
    pos_ids: List[int] = []
    neg_ids: List[int] = []
    bs = max(1, int(extra.match_batch_size))

    with torch.no_grad():
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

            fp = _features_mmbert_pooler_main(
                model, inp, mask, p_t, device, mmbert_repr
            )
            fn = _features_mmbert_pooler_main(
                model, inp, mask, n_t, device, mmbert_repr
            )

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
        logger.info("[tsne_match_pairs_main] Wrote pair spec for fair replay: %s", save_spec)

    if len(feats) < 4:
        raise RuntimeError("Too few pair features for t-SNE.")

    X = np.stack(feats, axis=0)
    y = np.array(labs, dtype=np.float32)
    n = X.shape[0]
    perp_user = float(extra.tsne_perplexity)
    perp = float(min(perp_user, max(2.0, float(n - 1) * 0.99)))
    if perp < perp_user - 1e-3:
        logger.info("[tsne_match_pairs_main] perplexity %.2f -> %.2f (n=%d)", perp_user, perp, n)

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
    if embed_m == "umap":
        logger.info("[tsne_match_pairs_main] UMAP on PCA features (unsupervised).")

    plot_dpi = int(getattr(extra, "plot_dpi", 200))
    mode_label = f"{mode}+{mmbert_repr}"
    if extra.tsne_output_png:
        _plot_match(
            xy,
            y,
            extra.tsne_output_png,
            mode_label,
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
            feature_mode=np.array(mode_label, dtype=object),
            mmbert_repr=np.array(mmbert_repr, dtype=object),
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
            checkpoint=np.array(ckpt, dtype=object),
        )
        logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
