#!/usr/bin/env python3
"""
EIGML-style pair analysis: sample fixed (pos, neg) cases from DSTC validation JSON, then export
MM-BERT pooler features and eval-aligned scalar scores for Base / w/o PSP / w/o L_proto / Full.

Subcommands
-----------
sample-pairs
    Draw N cases (default 500), write ``analysis_pairs_500.json`` shared by all models.

export
    Load one checkpoint (--variant + model args after ``--``), write ``features.npy``,
    ``labels.npy``, ``scores.npy``, ``meta.json`` under ``--out_dir``.

Example commands (four checkpoints)
-------------------------------------

1) Sample pairs (random negative from candidates; fallback to global pool is recorded in meta)::

    python scripts/export_pair_features_for_tsne.py sample-pairs \\
        --data_path data/validation_pair_with_cand.json \\
        --out_json analysis_pairs_500.json \\
        --n_pairs 500 --seed_sample 2021 \\
        --neg_mode random_from_cand

2) Sample pairs (hard negative: Base model top-1 wrong among candidates)::

    python scripts/export_pair_features_for_tsne.py sample-pairs \\
        --data_path data/validation_pair_with_cand.json \\
        --out_json analysis_pairs_500.json \\
        --neg_mode hard_base_top1_wrong \\
        --base_ckpt logs/.../epoch=X.ckpt \\
        -- --mmbert_baseline_defaults --max_image_id 307 --ckpt_path dummy

3) Export Base (MM-BERT) features + scores::

    python scripts/export_pair_features_for_tsne.py export \\
        --pair_json analysis_pairs_500.json \\
        --out_dir exports/base \\
        --variant base \\
        -- --mmbert_baseline_defaults --ckpt_path logs/.../baseline.ckpt

4) Export w/o PSP (structured residual)::

    python scripts/export_pair_features_for_tsne.py export \\
        --pair_json analysis_pairs_500.json \\
        --out_dir exports/wo_psp \\
        --variant wo_psp \\
        -- --config configs/structured_residual/v3_01_expr_residual.yaml \\
           --ckpt_path logs/.../residual.ckpt

5) Export w/o L_proto / Full (factorized; use ablate vs full YAML)::

    python scripts/export_pair_features_for_tsne.py export \\
        --pair_json analysis_pairs_500.json \\
        --out_dir exports/wo_lproto \\
        --variant wo_lproto \\
        -- --config configs/structured_factorized/..._ablate.yaml --ckpt_path ...

    python scripts/export_pair_features_for_tsne.py export \\
        --pair_json analysis_pairs_500.json \\
        --out_dir exports/full \\
        --variant full \\
        -- --config configs/structured_factorized/v6_00_minimal_core.yaml --ckpt_path ...

Output files (export)
---------------------
``out_dir/features.npy``  shape ``[1000, D]``, row order
``case0_pos, case0_neg, case1_pos, case1_neg, ...``.
``out_dir/labels.npy``    ``1`` = positive sticker, ``0`` = negative.
``out_dir/scores.npy``    Scalar matching score per row (same definition as each model's eval).
``out_dir/meta.json``     Feature definition, paths, variant name, D.

For ``variant=base``, use ``--base_score margin`` (default: ``logit_pos - logit_neg`` on the
2-way head) or ``--base_score logit1`` (legacy positive-class logit only).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
from structured_retrieval_factorized import (  # noqa: E402
    StructuredFactorizedPLModel,
    parse_structured_factorized_args,
)
from structured_retrieval_residual import (  # noqa: E402
    StructuredResidualPLModel,
    parse_structured_residual_args,
)
from transformers import HfArgumentParser  # noqa: E402

# Reuse pooler + collate from tsne_match_pairs
_pair_mod_path = os.path.join(os.path.dirname(__file__), "tsne_match_pairs.py")
_pair_spec = importlib.util.spec_from_file_location("tsne_match_pairs_export", _pair_mod_path)
assert _pair_spec and _pair_spec.loader
_tsne_pairs = importlib.util.module_from_spec(_pair_spec)
_pair_spec.loader.exec_module(_tsne_pairs)
_collate_sents = _tsne_pairs._collate_sents
_features_mmbert_pooler = _tsne_pairs._features_mmbert_pooler

_main_mod_path = os.path.join(os.path.dirname(__file__), "tsne_match_pairs_main.py")
_main_spec = importlib.util.spec_from_file_location("tsne_match_pairs_main_export", _main_mod_path)
assert _main_spec and _main_spec.loader
_tsne_main = importlib.util.module_from_spec(_main_spec)
_main_spec.loader.exec_module(_tsne_main)
_features_mmbert_pooler_main = _tsne_main._features_mmbert_pooler_main

FEATURE_DEF = (
    "MM-BERT encoder pooler: multimodal sequence [dialogue | sticker | …] → bert encoder → "
    "pooler_output (or last_hidden_state[:,0] if pooler missing). Aligned across variants via "
    "the same pooling path as scripts/tsne_match_pairs.py / tsne_match_pairs_main.py."
)

# Matches scripts/tsne_match_pairs_main.py — prepended when --mmbert_baseline_defaults is present.
_MMBERT_BASELINE_DEFAULTS: List[str] = [
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


def _apply_mmbert_baseline_defaults(argv: List[str]) -> List[str]:
    """
    Consume --mmbert_baseline_defaults (not an HfArgumentParser field) and prepend run.sh-style
    defaults, same as scripts/tsne_match_pairs_main._preprocess_argv.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mmbert_baseline_defaults", action="store_true")
    known, unknown = p.parse_known_args(argv)
    out = list(unknown)
    if known.mmbert_baseline_defaults:
        out = _MMBERT_BASELINE_DEFAULTS + out
    return out


def _user_ids_from_sample(sample: Dict[str, Any]) -> List[str]:
    u = sample.get("user_id")
    if u is None:
        return ["__UNK__"]
    if isinstance(u, str):
        return [u]
    return [str(u)]


def _gather_eigml_valid_indices(
    ds: PLDataset,
    max_image_id: int,
) -> List[int]:
    """Indices with valid gold sticker and at least one negative id in [0, max_image_id)."""
    valid: List[int] = []
    for i in range(len(ds)):
        sample = ds[i]
        pos = sample.get("img_id")
        if pos is None:
            continue
        pos = int(pos)
        if pos < 0 or pos >= max_image_id:
            continue
        cand = sample.get("cand")
        has_alt = False
        if isinstance(cand, list) and len(cand) > 0:
            for c in cand:
                if int(c) != pos and 0 <= int(c) < max_image_id:
                    has_alt = True
                    break
        pool = [j for j in range(max_image_id) if j != pos]
        if not has_alt and not pool:
            continue
        valid.append(i)
    if not valid:
        raise RuntimeError("No valid samples for pair sampling (check max_image_id and data).")
    return valid


def _pick_random_neg(
    sample: Dict[str, Any],
    pos: int,
    max_image_id: int,
    rng: random.Random,
) -> Tuple[int, str]:
    cand = sample.get("cand")
    if isinstance(cand, list) and len(cand) > 0:
        others = [int(c) for c in cand if int(c) != pos and 0 <= int(c) < max_image_id]
        if others:
            return rng.choice(others), "from_cand"
    pool = [j for j in range(max_image_id) if j != pos]
    if not pool:
        raise ValueError(f"No negative available for pos={pos}")
    return rng.choice(pool), "global_fallback"


def _hard_neg_base(
    pl_model: PLModel,
    sample: Dict[str, Any],
    pos: int,
    device: torch.device,
    tokenizer: Any,
    max_dialogue_length: int,
) -> int:
    """Top-scoring wrong candidate under baseline PLModel (same ordering as validation)."""
    inp, mask = _collate_sents(tokenizer, [sample["sent"]], max_dialogue_length)
    inp = inp.to(device)
    mask = mask.to(device)
    cand = sample.get("cand")
    if not isinstance(cand, list) or len(cand) == 0:
        raise ValueError("hard_base_top1_wrong requires non-empty cand in dataset row")
    cands = [int(x) for x in cand]
    batch = {
        "input_ids": inp,
        "attention_mask": mask,
        "img_ids": [pos],
        "neg_img_ids": [0],
        "user_ids": _user_ids_from_sample(sample),
        "cands": [cands],
    }
    with torch.no_grad():
        logits, _labels, _c = pl_model.run_model_from_batch(batch, 0, test=True)
    scores = logits[0]
    wrong_idx = [i for i, cid in enumerate(cands) if int(cid) != int(pos)]
    if not wrong_idx:
        raise ValueError(f"No wrong candidate among cands for pos={pos}")
    best = max(wrong_idx, key=lambda i: float(scores[i].item()))
    return int(cands[best])


def _score_baseline(
    pl_model: PLModel,
    inp: torch.Tensor,
    mask: torch.Tensor,
    sid: int,
    user_ids: List[str],
    device: torch.device,
) -> float:
    """Legacy: only the positive-class logit (same as ``run_model_from_batch`` return slice)."""
    batch = {
        "input_ids": inp,
        "attention_mask": mask,
        "img_ids": [sid],
        "neg_img_ids": [0],
        "user_ids": user_ids,
        "cands": [[sid]],
    }
    with torch.no_grad():
        logits, _l, _c = pl_model.run_model_from_batch(batch, 0, test=True)
    return float(logits[0, 0].item())


def _score_baseline_margin(
    pl_model: PLModel,
    inp: torch.Tensor,
    mask: torch.Tensor,
    sid: int,
    user_ids: List[str],
    device: torch.device,
) -> float:
    """
    ``logit[pos class] - logit[neg class]`` on the MM-BERT classification head, same idea as
    ``structured_retrieval.StructuredStickerModel.compute_base_score`` (ranking margin).

    Replicates ``main.py`` ``Model.forward`` test branch for ``add_ocr_info=False`` and
    ``use_visual_personalization_token=False`` (dialogue | sticker | SEP), then
    ``res.logits[0, 1] - res.logits[0, 0]``.
    """
    _ = user_ids  # unused on this path (matches forward when not using personalization)
    _ = mask  # main.py test path does not pass attention_mask into bert here
    m = pl_model.model
    if getattr(m.args, "add_ocr_info", False) or getattr(
        m.args, "use_visual_personalization_token", False
    ):
        raise NotImplementedError(
            "export --base_score margin is only implemented for add_ocr_info=False and "
            "use_visual_personalization_token=False. Use --base_score logit1 for this checkpoint, "
            "or extend export_pair_features_for_tsne._score_baseline_margin."
        )
    inp = inp.to(device)
    cand_ids = [int(sid)]
    img_emb = m.all_img_embs[cand_ids]
    img_num = int(img_emb.size(0))
    sep_id = torch.tensor(
        m.bert_tokenizer.sep_token_id, device=device, dtype=torch.long
    )
    sep_emb = m.bert.bert.embeddings.word_embeddings(sep_id).unsqueeze(0).unsqueeze(0).repeat(
        img_num, 1, 1
    )
    text_emb = m.bert.bert.embeddings.word_embeddings(inp)
    text_emb = text_emb.repeat(img_num, 1, 1)
    img_emb = m.img_ff(img_emb).unsqueeze(1)
    input_emb = torch.cat([text_emb, img_emb, sep_emb], dim=1)
    extra_len = int(input_emb.size(1) - text_emb.size(1))
    token_type_ids = torch.zeros(
        (input_emb.size(0), input_emb.size(1)), device=device, dtype=torch.long
    )
    token_type_ids[:, -extra_len:] = 1
    with torch.no_grad():
        res = m.bert(
            inputs_embeds=input_emb,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
    logits2 = res.logits
    return float(logits2[0, 1].item() - logits2[0, 0].item())


def _score_structured_common(
    pl_model: Any,
    inp: torch.Tensor,
    mask: torch.Tensor,
    sid: int,
    device: torch.device,
) -> float:
    batch = {
        "input_ids": inp,
        "attention_mask": mask,
        "img_ids": [sid],
        "cands": [[sid]],
    }
    with torch.no_grad():
        out = pl_model.run_eval_batch(batch, return_debug=False)
    scores = out[0]
    return float(scores[0, 0].item())


def _dataset_bootstrap_argv(data_path: str, max_image_id: int) -> List[str]:
    """Minimal `Arguments` argv so PLDataset matches training/eval sticker space."""
    return [
        "--max_image_id",
        str(max_image_id),
        "--test_with_cand",
        "true",
        "--train_data_path",
        data_path,
        "--val_data_path",
        data_path,
        "--test_data_path",
        data_path,
        "--bert_pretrain_path",
        "./ckpt/bert-base-chinese",
        "--img_pretrain_path",
        "./ckpt/clip-ViT-B-32",
        "--local_files_only",
        "true",
        "--model_choice",
        "use_img_clip",
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
    ]


def cmd_sample_pairs(args: argparse.Namespace) -> None:
    seed = int(args.seed_sample)
    random.seed(seed)
    np.random.seed(seed)

    data_path = os.path.abspath(args.data_path)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(data_path)

    hf_parser = HfArgumentParser(Arguments)

    if args.neg_mode == "hard_base_top1_wrong":
        rest = _strip_leading_dd(list(args.base_rest or []))
        if not rest:
            raise ValueError(
                "hard_base_top1_wrong requires base model argv (e.g. --mmbert_baseline_defaults "
                "--ckpt_path ... --max_image_id 307) — place flags after the subcommand."
            )
        rest = _apply_mmbert_baseline_defaults(rest)
        base_args = hf_parser.parse_args_into_dataclasses(rest)[0]
        ckpt = (getattr(base_args, "ckpt_path", None) or "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            raise ValueError("hard mode: provide a real existing --ckpt_path to the Base checkpoint.")
        import pytorch_lightning as pl

        pl.seed_everything(int(getattr(base_args, "seed", seed)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pl_model = PLModel(base_args)
        load_checkpoint_to_model(
            pl_model,
            ckpt,
            strict=bool(getattr(base_args, "strict_checkpoint_load", False)),
        )
        pl_model.model.eval()
        pl_model.model.to(device)
        pl_model.model.prepare_for_test()
        tokenizer = pl_model.model.bert_tokenizer
        max_image_id = int(base_args.max_image_id)
        max_len = int(getattr(base_args, "max_dialogue_length", 490) or 490)
        data_args = base_args
    else:
        pl_model = None
        device = None
        max_image_id = int(args.max_image_id)
        max_len = int(args.max_dialogue_length)
        bootstrap = _dataset_bootstrap_argv(data_path, max_image_id)
        data_args = hf_parser.parse_args_into_dataclasses(bootstrap)[0]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            data_args.bert_pretrain_path,
            local_files_only=bool(getattr(data_args, "local_files_only", True)),
        )

    ds = PLDataset(data_path, "test", data_args, tokenizer)

    valid = _gather_eigml_valid_indices(ds, max_image_id)
    rng = random.Random(seed)
    n_pairs = int(args.n_pairs)
    if len(valid) < n_pairs:
        raise RuntimeError(f"Only {len(valid)} valid indices; need {n_pairs}.")
    chosen = rng.sample(valid, n_pairs)

    pairs_out: List[Dict[str, Any]] = []
    for case_id, dataset_index in enumerate(chosen):
        sample = ds[dataset_index]
        pos = int(sample["img_id"])
        if args.neg_mode == "hard_base_top1_wrong":
            assert pl_model is not None and tokenizer is not None and device is not None
            neg = _hard_neg_base(
                pl_model, sample, pos, device, tokenizer, max_len
            )
            neg_src = "hard_base_top1_wrong"
        else:
            neg, neg_src = _pick_random_neg(sample, pos, max_image_id, rng)
        pairs_out.append(
            {
                "case_id": case_id,
                "dataset_index": int(dataset_index),
                "pos_img_id": pos,
                "neg_img_id": neg,
                "neg_source": neg_src,
            }
        )

    payload = {
        "version": 1,
        "data_path": data_path,
        "seed_sample": seed,
        "neg_mode": args.neg_mode,
        "n_pairs": n_pairs,
        "max_image_id": max_image_id,
        "pairs": pairs_out,
    }
    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s (%d pairs)", out_path, len(pairs_out))


def _load_pair_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _strip_leading_dd(argv: List[str]) -> List[str]:
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def cmd_export(args: argparse.Namespace) -> None:
    pair_path = os.path.abspath(args.pair_json)
    spec = _load_pair_json(pair_path)
    pairs: List[Dict[str, Any]] = spec["pairs"]
    n_pairs = len(pairs)
    seed = int(spec.get("seed_sample", 2021))

    rest = _strip_leading_dd(list(args.model_rest or []))
    if not rest:
        raise ValueError("Pass model arguments after -- (see script docstring).")

    import pytorch_lightning as pl

    pl.seed_everything(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(
            getattr(args, "allow_tf32", True) if hasattr(args, "allow_tf32") else True
        )
        torch.backends.cudnn.allow_tf32 = bool(
            getattr(args, "allow_tf32", True) if hasattr(args, "allow_tf32") else True
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = str(args.variant)

    if variant == "base":
        rest = _apply_mmbert_baseline_defaults(rest)
        hf_parser = HfArgumentParser(Arguments)
        model_args = hf_parser.parse_args_into_dataclasses(rest)[0]
        ckpt = (getattr(model_args, "ckpt_path", None) or "").strip()
        if not ckpt:
            raise ValueError("--ckpt_path required in model args for base.")
        pl_model = PLModel(model_args)
        load_checkpoint_to_model(
            pl_model,
            ckpt,
            strict=bool(getattr(model_args, "strict_checkpoint_load", False)),
        )
        pl_model.model.eval()
        pl_model.model.to(device)
        pl_model.model.prepare_for_test()
        model = pl_model.model
        tokenizer = model.bert_tokenizer
        max_image_id = int(model_args.max_image_id)
        max_len = int(getattr(model_args, "max_dialogue_length", 490) or 490)
        mmbert_repr = str(getattr(args, "mmbert_repr", "pooler"))

        def pooler_features(inp, mask, sticker_ids_1d):
            return _features_mmbert_pooler_main(
                model, inp, mask, sticker_ids_1d, device, mmbert_repr
            )

        base_score_mode = str(getattr(args, "base_score", "margin"))

        def score_one(inp, mask, sid, sample):
            u = _user_ids_from_sample(sample)
            if base_score_mode == "margin":
                return _score_baseline_margin(pl_model, inp, mask, sid, u, device)
            return _score_baseline(pl_model, inp, mask, sid, u, device)

        backend = "base"
    elif variant == "wo_psp":
        model_args = parse_structured_residual_args(rest)
        ckpt = (getattr(model_args, "ckpt_path", None) or "").strip()
        if not ckpt:
            raise ValueError("--ckpt_path required.")
        pl_model = StructuredResidualPLModel(model_args)
        load_checkpoint_to_model(
            pl_model,
            ckpt,
            strict=bool(getattr(model_args, "strict_checkpoint_load", False)),
        )
        pl_model.model.eval()
        pl_model.model.to(device)
        pl_model.model.prepare_for_test()
        model = pl_model.model
        tokenizer = model.bert_tokenizer
        bank_all_h = model.all_img_embs
        if bank_all_h is None:
            raise RuntimeError("Residual: all_img_embs is None after prepare_for_test().")
        bank_all_h = bank_all_h.to(device)
        max_image_id = int(model_args.max_image_id)
        max_len = int(getattr(model_args, "max_dialogue_length", 490) or 490)

        def pooler_features(inp, mask, sticker_ids_1d):
            return _features_mmbert_pooler(
                model, bank_all_h, inp, mask, sticker_ids_1d, device
            )

        def score_one(inp, mask, sid, sample):
            return _score_structured_common(pl_model, inp, mask, sid, device)

        backend = "structured_residual"
    elif variant in ("wo_lproto", "full"):
        model_args = parse_structured_factorized_args(rest)
        ckpt = (getattr(model_args, "ckpt_path", None) or "").strip()
        if not ckpt:
            raise ValueError("--ckpt_path required.")
        pl_model = StructuredFactorizedPLModel(model_args)
        load_checkpoint_to_model(
            pl_model,
            ckpt,
            strict=bool(getattr(model_args, "strict_checkpoint_load", False)),
        )
        pl_model.model.eval()
        pl_model.model.to(device)
        bank_all_h, _, _, _, _ = pl_model.model._get_eval_or_fresh_bank_factorization(device)
        tokenizer = pl_model.model.bert_tokenizer
        max_image_id = int(model_args.max_image_id)
        max_len = int(getattr(model_args, "max_dialogue_length", 490) or 490)

        def pooler_features(inp, mask, sticker_ids_1d):
            return _features_mmbert_pooler(
                pl_model.model, bank_all_h, inp, mask, sticker_ids_1d, device
            )

        def score_one(inp, mask, sid, sample):
            return _score_structured_common(pl_model, inp, mask, sid, device)

        backend = "structured_factorized"
    else:
        raise ValueError(f"Unknown variant {variant!r}")

    data_path = os.path.abspath(spec["data_path"])
    ds_args = model_args
    ds = PLDataset(data_path, "test", ds_args, tokenizer)

    feats: List[np.ndarray] = []
    labs: List[int] = []
    scos: List[float] = []

    bs = max(1, int(args.batch_size))
    with torch.no_grad():
        for start in range(0, n_pairs, bs):
            chunk = pairs[start : start + bs]
            sents: List[str] = []
            pos_list: List[int] = []
            neg_list: List[int] = []
            samples: List[Dict[str, Any]] = []
            for p in chunk:
                ix = int(p["dataset_index"])
                sample = ds[ix]
                pos = int(p["pos_img_id"])
                neg = int(p["neg_img_id"])
                if int(sample["img_id"]) != pos:
                    raise ValueError(
                        f"pair_json dataset_index={ix} pos_img_id {pos} != dataset img_id {sample['img_id']}"
                    )
                sents.append(sample["sent"])
                pos_list.append(pos)
                neg_list.append(neg)
                samples.append(sample)

            inp, mask = _collate_sents(tokenizer, sents, max_len)
            inp_d = inp.to(device)
            mask_d = mask.to(device)
            p_t = torch.tensor(pos_list, dtype=torch.long)
            n_t = torch.tensor(neg_list, dtype=torch.long)

            fp = pooler_features(inp_d, mask_d, p_t)
            fn = pooler_features(inp_d, mask_d, n_t)
            fp_np = fp.detach().cpu().numpy()
            fn_np = fn.detach().cpu().numpy()

            for i in range(len(chunk)):
                for row, lab, sid in (
                    (fp_np[i], 1, pos_list[i]),
                    (fn_np[i], 0, neg_list[i]),
                ):
                    feats.append(row.astype(np.float32))
                    labs.append(lab)
                    inp_i = inp[i : i + 1].to(device)
                    mask_i = mask[i : i + 1].to(device)
                    scos.append(score_one(inp_i, mask_i, sid, samples[i]))

    X = np.stack(feats, axis=0)
    y = np.asarray(labs, dtype=np.int8)
    s = np.asarray(scos, dtype=np.float32)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "features.npy"), X)
    np.save(os.path.join(out_dir, "labels.npy"), y)
    np.save(os.path.join(out_dir, "scores.npy"), s)

    meta = {
        "feature_definition": FEATURE_DEF,
        "D": int(X.shape[1]),
        "num_rows": int(X.shape[0]),
        "variant": variant,
        "backend": backend,
        "pair_json": pair_path,
        "data_path": data_path,
        "ckpt": ckpt,
        "model_argv": rest,
    }
    if variant == "base":
        meta["mmbert_repr"] = str(getattr(args, "mmbert_repr", "pooler"))
        meta["base_score"] = str(getattr(args, "base_score", "margin"))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(
        "Wrote features [N,D]=%s, labels, scores under %s",
        X.shape,
        out_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    epilog = """
Examples:
  sample-pairs (random negatives):
    python scripts/export_pair_features_for_tsne.py sample-pairs \\
      --data_path data/validation_pair_with_cand.json --out_json analysis_pairs_500.json

  sample-pairs (hard negatives, Base model):
    python scripts/export_pair_features_for_tsne.py sample-pairs \\
      --data_path data/validation_pair_with_cand.json --out_json analysis_pairs_500.json \\
      --neg_mode hard_base_top1_wrong \\
      -- --mmbert_baseline_defaults --ckpt_path logs/base/epoch=1.ckpt --max_image_id 307

  export (four runs, model args after --):
    python scripts/export_pair_features_for_tsne.py export --pair_json analysis_pairs_500.json \\
      --out_dir exports/base --variant base \\
      -- --mmbert_baseline_defaults --ckpt_path logs/base/epoch=1.ckpt

    python scripts/export_pair_features_for_tsne.py export --pair_json analysis_pairs_500.json \\
      --out_dir exports/wo_psp --variant wo_psp \\
      -- --config configs/structured_residual/v3_01_expr_residual.yaml --ckpt_path logs/r/epoch=1.ckpt

    python scripts/export_pair_features_for_tsne.py export --pair_json analysis_pairs_500.json \\
      --out_dir exports/wo_lproto --variant wo_lproto \\
      -- --config configs/structured_factorized/ablate.yaml --ckpt_path logs/f/epoch=1.ckpt

    python scripts/export_pair_features_for_tsne.py export --pair_json analysis_pairs_500.json \\
      --out_dir exports/full --variant full \\
      -- --config configs/structured_factorized/v6_00_minimal_core.yaml --ckpt_path logs/f/epoch=1.ckpt

Output files (export): out_dir/features.npy, labels.npy, scores.npy, meta.json
"""
    p = argparse.ArgumentParser(
        description="Sample DSTC pair lists and export MM-BERT pooler features + scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("sample-pairs", help="Write analysis_pairs JSON (500 cases × pos/neg).")
    ps.add_argument("--data_path", type=str, required=True)
    ps.add_argument("--out_json", type=str, default="analysis_pairs_500.json")
    ps.add_argument("--n_pairs", type=int, default=500)
    ps.add_argument("--seed_sample", type=int, default=2021)
    ps.add_argument(
        "--neg_mode",
        type=str,
        choices=("random_from_cand", "hard_base_top1_wrong"),
        default="random_from_cand",
    )
    ps.add_argument(
        "--max_image_id",
        type=int,
        default=307,
        help="Only used when neg_mode=random_from_cand (dataset bootstrap).",
    )
    ps.add_argument("--max_dialogue_length", type=int, default=490)
    ps.add_argument(
        "base_rest",
        nargs=argparse.REMAINDER,
        default=[],
        help="For hard_base_top1_wrong: use `--` then base flags, e.g. "
        "`-- --mmbert_baseline_defaults --ckpt_path /path/baseline.ckpt --max_image_id 307`.",
    )

    pe = sub.add_parser("export", help="Export npy + meta for one checkpoint.")
    pe.add_argument("--pair_json", type=str, required=True)
    pe.add_argument("--out_dir", type=str, required=True)
    pe.add_argument(
        "--variant",
        type=str,
        choices=("base", "wo_psp", "wo_lproto", "full"),
        required=True,
    )
    pe.add_argument("--batch_size", type=int, default=16)
    pe.add_argument(
        "--mmbert_repr",
        type=str,
        choices=("pooler", "last_hidden_cls"),
        default="pooler",
        help="Only for variant=base.",
    )
    pe.add_argument(
        "--base_score",
        type=str,
        choices=("margin", "logit1"),
        default="margin",
        help="Only for variant=base. margin=logit_pos-logit_neg (ranking margin, recommended); "
        "logit1=positive-class logit only (legacy, same scale as old exports).",
    )
    pe.add_argument(
        "model_rest",
        nargs=argparse.REMAINDER,
        default=[],
        help="Model argv after `--`, e.g. `-- --config ... --ckpt_path ...`",
    )
    return p


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
    p = _build_parser()
    # Allow `export -- ...` : model_rest may need `--` consumed by shell; user passes remaining after --
    args = p.parse_args()
    if args.command == "sample-pairs":
        cmd_sample_pairs(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        raise SystemExit(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    main()
