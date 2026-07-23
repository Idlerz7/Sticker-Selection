"""Exact-length-bucket extraction of frozen pair representations."""

from __future__ import annotations

import collections
import math
from typing import Dict, List, Sequence

import torch

from .legacy import dialogue_text


@torch.no_grad()
def extract_pair_cache(model, legacy_args, source_rows: Sequence[dict], manifest_rows: Sequence[dict], pair_batch_size: int = 256) -> dict:
    core = model.model
    tokenizer = core.bert_tokenizer
    device = next(core.parameters()).device
    candidate_count = len(manifest_rows[0]["candidate_ids"])
    if any(len(row["candidate_ids"]) != candidate_count for row in manifest_rows):
        raise ValueError("pair cache requires a fixed candidate count")
    token_rows = []
    groups = collections.defaultdict(list)
    for query_index, manifest in enumerate(manifest_rows):
        source = source_rows[int(manifest["source_row"])]
        ids = tokenizer(
            dialogue_text(source, legacy_args.sent_num), truncation=True,
            max_length=legacy_args.max_dialogue_length, add_special_tokens=True,
        )["input_ids"]
        token_rows.append(ids)
        groups[len(ids)].append(query_index)
    q_count = len(manifest_rows)
    h = torch.empty((q_count, candidate_count, 768), dtype=torch.float32)
    b = torch.empty((q_count, candidate_count), dtype=torch.float32)
    candidates = torch.tensor([row["candidate_ids"] for row in manifest_rows], dtype=torch.long)
    positive = torch.tensor([row["positive_index"] for row in manifest_rows], dtype=torch.long)
    batch_orders = []
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = core.bert.bert.pooler.register_forward_hook(hook)
    try:
        for length in sorted(groups):
            query_indices = groups[length]
            # CUDA GEMM kernels change with the outer batch dimension.  Combining multiple
            # queries produced small (but forbidden) differences from legacy batch-size-one
            # evaluation.  We retain exact-length bucket scheduling while executing exactly
            # one legacy candidate group per call, which is the only elementwise-identical path.
            queries_per_batch = 1
            for start in range(0, len(query_indices), queries_per_batch):
                q_indices = query_indices[start:start + queries_per_batch]
                input_ids = torch.tensor([token_rows[index] for index in q_indices], dtype=torch.long, device=device)
                input_ids = input_ids.repeat_interleave(candidate_count, dim=0)
                flat_candidates = candidates[q_indices].reshape(-1).to(device)
                text_emb = core.bert.bert.embeddings.word_embeddings(input_ids)
                image_emb = core.img_ff(core.all_img_embs[flat_candidates]).unsqueeze(1)
                sep_id = torch.tensor(tokenizer.sep_token_id, device=device, dtype=torch.long)
                sep = core.bert.bert.embeddings.word_embeddings(sep_id).view(1, 1, -1).expand(len(flat_candidates), 1, -1)
                inputs = torch.cat((text_emb, image_emb, sep), dim=1)
                token_types = torch.zeros(inputs.shape[:2], dtype=torch.long, device=device)
                token_types[:, -2:] = 1
                captured.clear()
                result = core.bert(inputs_embeds=inputs, token_type_ids=token_types, return_dict=True)
                if len(captured) != 1 or captured[0].shape != (len(flat_candidates), 768):
                    raise RuntimeError("pooler capture failed")
                pooled = captured[0].view(len(q_indices), candidate_count, 768).cpu().float()
                logits = result.logits[:, 1].view(len(q_indices), candidate_count).cpu().float()
                h[q_indices] = pooled
                b[q_indices] = logits
                batch_orders.append({"token_length": length, "query_indices": q_indices})
    finally:
        handle.remove()
    if not torch.isfinite(h).all() or not torch.isfinite(b).all():
        raise ValueError("non-finite pair cache")
    return {
        "h": h, "b": b, "candidate_ids": candidates, "positive_index": positive,
        "query_ids": torch.tensor([row["query_id"] for row in manifest_rows]),
        "source_rows": torch.tensor([row["source_row"] for row in manifest_rows]),
        "token_lengths": torch.tensor([len(value) for value in token_rows]),
        "extraction_batches": batch_orders,
    }


@torch.no_grad()
def validate_against_legacy(model, legacy_args, source_row: dict, manifest_row: dict, cache_h: torch.Tensor, cache_b: torch.Tensor) -> dict:
    """Compare one cache row with the unmodified `Model.forward(test=True)` path."""
    core = model.model
    tokenizer = core.bert_tokenizer
    device = next(core.parameters()).device
    encoded = tokenizer(
        [dialogue_text(source_row, legacy_args.sent_num)], return_tensors="pt", padding=True,
        truncation=True, max_length=legacy_args.max_dialogue_length,
    )
    captured = []
    handle = core.bert.bert.pooler.register_forward_hook(lambda _m, _i, out: captured.append(out.detach().cpu()))
    try:
        logits, _labels, returned = core(
            input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device),
            img_ids=[int(source_row["dialog"][-1]["img_id"])], neg_img_ids=[None],
            user_ids=[source_row.get("user_id")], test=True, cands=[manifest_row["candidate_ids"]],
        )
    finally:
        handle.remove()
    direct_h = captured[0].float()
    direct_b = logits.cpu().float().squeeze(0)
    return {
        "candidate_ids_equal": list(returned) == list(manifest_row["candidate_ids"]),
        "h_max_abs": float((direct_h - cache_h.float()).abs().max()),
        "b_max_abs": float((direct_b - cache_b.float()).abs().max()),
        "h_shape": list(direct_h.shape), "b_shape": list(direct_b.shape),
    }
