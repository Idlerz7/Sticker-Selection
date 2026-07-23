"""Exact, leakage-isolated reciprocal local presentation context."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _tie_key(similarity: torch.Tensor, ids: torch.Tensor, global_max_id: int) -> torch.Tensor:
    # Float64 makes the secondary ID key far smaller than float32 feature precision.
    return similarity.double() + (global_max_id - ids.double()).view(1, -1) * 1e-12


def exact_topk(
    queries: torch.Tensor,
    index: torch.Tensor,
    index_ids: torch.Tensor,
    k: int,
    query_ids: Optional[torch.Tensor] = None,
    exclude_equal_id: bool = False,
    query_batch_size: int = 2048,
    index_batch_size: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked exact cosine top-k, resolving exact similarity ties by lower integer ID."""
    if k <= 0 or k > index.shape[0] - int(exclude_equal_id):
        raise ValueError("invalid k")
    device = queries.device
    q = F.normalize(queries.float(), dim=-1)
    x = F.normalize(index.float(), dim=-1)
    index_ids = index_ids.to(device=device, dtype=torch.long)
    global_max_id = int(index_ids.max().item()) if index_ids.numel() else 0
    output_ids = []
    output_sims = []
    for q_start in range(0, len(q), query_batch_size):
        q_end = min(q_start + query_batch_size, len(q))
        q_chunk = q[q_start:q_end]
        best_keys = torch.full((len(q_chunk), k), -float("inf"), dtype=torch.float64, device=device)
        best_sims = torch.full((len(q_chunk), k), -float("inf"), dtype=torch.float32, device=device)
        best_ids = torch.full((len(q_chunk), k), -1, dtype=torch.long, device=device)
        q_ids = None if query_ids is None else query_ids[q_start:q_end].to(device)
        for x_start in range(0, len(x), index_batch_size):
            x_end = min(x_start + index_batch_size, len(x))
            ids = index_ids[x_start:x_end]
            sims = q_chunk.matmul(x[x_start:x_end].T)
            if exclude_equal_id:
                if q_ids is None:
                    raise ValueError("query_ids required when excluding equal IDs")
                sims = sims.masked_fill(q_ids.view(-1, 1) == ids.view(1, -1), -float("inf"))
            keys = _tie_key(sims, ids, global_max_id)
            merged_keys = torch.cat((best_keys, keys), dim=1)
            merged_sims = torch.cat((best_sims, sims), dim=1)
            merged_ids = torch.cat((best_ids, ids.view(1, -1).expand(len(q_chunk), -1)), dim=1)
            best_keys, positions = torch.topk(merged_keys, k=k, dim=1, largest=True, sorted=True)
            best_sims = torch.gather(merged_sims, 1, positions)
            best_ids = torch.gather(merged_ids, 1, positions)
        output_ids.append(best_ids.cpu())
        output_sims.append(best_sims.cpu())
    return torch.cat(output_ids), torch.cat(output_sims)


def exact_topk_with_group_filtered(
    queries: torch.Tensor,
    index: torch.Tensor,
    index_ids: torch.Tensor,
    query_group: torch.Tensor,
    index_group: torch.Tensor,
    k: int = 10,
    query_ids: Optional[torch.Tensor] = None,
    exclude_equal_id: bool = False,
    query_batch_size: int = 2048,
    index_batch_size: int = 8192,
):
    """Compute ordinary and exact group-excluded top-k from each shared similarity block."""
    device = queries.device
    q = F.normalize(queries.float(), dim=-1); x = F.normalize(index.float(), dim=-1)
    index_ids = index_ids.to(device=device, dtype=torch.long)
    query_group = query_group.to(device=device, dtype=torch.long); index_group = index_group.to(device=device, dtype=torch.long)
    global_max_id = int(index_ids.max().item())
    ordinary_parts = []; filtered_parts = []
    for q_start in range(0, len(q), query_batch_size):
        q_end = min(q_start + query_batch_size, len(q)); q_chunk = q[q_start:q_end]
        shape = (len(q_chunk), k)
        ordinary_keys = torch.full(shape, -float("inf"), dtype=torch.float64, device=device)
        filtered_keys = ordinary_keys.clone()
        ordinary_ids = torch.full(shape, -1, dtype=torch.long, device=device)
        filtered_ids = ordinary_ids.clone()
        q_ids = None if query_ids is None else query_ids[q_start:q_end].to(device)
        q_groups = query_group[q_start:q_end]
        for x_start in range(0, len(x), index_batch_size):
            x_end = min(x_start + index_batch_size, len(x)); ids = index_ids[x_start:x_end]
            sims = q_chunk.matmul(x[x_start:x_end].T)
            if exclude_equal_id:
                if q_ids is None: raise ValueError("query_ids required when excluding equal IDs")
                sims = sims.masked_fill(q_ids.view(-1, 1) == ids.view(1, -1), -float("inf"))
            keys = _tie_key(sims, ids, global_max_id)
            block_ids = ids.view(1, -1).expand(len(q_chunk), -1)
            merged = torch.cat((ordinary_keys, keys), dim=1)
            merged_ids = torch.cat((ordinary_ids, block_ids), dim=1)
            ordinary_keys, positions = torch.topk(merged, k=k, dim=1, largest=True, sorted=True)
            ordinary_ids = torch.gather(merged_ids, 1, positions)
            filtered_block = keys.masked_fill(q_groups.view(-1, 1) == index_group[x_start:x_end].view(1, -1), -float("inf"))
            merged_filtered = torch.cat((filtered_keys, filtered_block), dim=1)
            merged_filtered_ids = torch.cat((filtered_ids, block_ids), dim=1)
            filtered_keys, positions = torch.topk(merged_filtered, k=k, dim=1, largest=True, sorted=True)
            filtered_ids = torch.gather(merged_filtered_ids, 1, positions)
        ordinary_parts.append(ordinary_ids.cpu()); filtered_parts.append(filtered_ids.cpu())
    return torch.cat(ordinary_parts), torch.cat(filtered_parts)


def _aggregate(features_by_id: dict, query_ids: torch.Tensor, neighbors: torch.Tensor, width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = []
    degree = []
    fallback = []
    for query_id, row in zip(query_ids.tolist(), neighbors.tolist()):
        valid = [int(value) for value in row if int(value) >= 0]
        degree.append(len(valid))
        if not valid:
            values.append(features_by_id[int(query_id)])
            fallback.append(True)
        else:
            values.append(torch.stack([features_by_id[value] for value in valid]).mean(dim=0))
            fallback.append(False)
    return F.normalize(torch.stack(values).float(), dim=-1), torch.tensor(degree), torch.tensor(fallback)


def build_training_lpc(features: torch.Tensor, ids: torch.Tensor, k: int = 10, **topk_kwargs) -> dict:
    candidate_ids, candidate_sims = exact_topk(
        features, features, ids, k, query_ids=ids, exclude_equal_id=True, **topk_kwargs
    )
    row_by_id = {int(value): row for row, value in enumerate(ids.tolist())}
    candidate_sets = {int(query): set(int(value) for value in row) for query, row in zip(ids.tolist(), candidate_ids.tolist())}
    mutual = torch.full_like(candidate_ids, -1)
    for row_index, (query_id, row) in enumerate(zip(ids.tolist(), candidate_ids.tolist())):
        kept = [neighbor for neighbor in row if int(query_id) in candidate_sets[int(neighbor)]]
        if kept:
            mutual[row_index, :len(kept)] = torch.tensor(kept, dtype=torch.long)
    features_by_id = {int(value): features[row].cpu() for row, value in enumerate(ids.tolist())}
    context, degree, fallback = _aggregate(features_by_id, ids.cpu(), mutual, features.shape[1])
    kth_sims = candidate_sims[:, -1].clone()
    kth_ids = candidate_ids[:, -1].clone()
    return {
        "ids": ids.cpu().long(),
        "features": context.cpu().float(),
        "neighbors": mutual.cpu().long(),
        "degree": degree.long(),
        "fallback": fallback.bool(),
        "candidate_neighbors": candidate_ids.cpu().long(),
        "candidate_similarities": candidate_sims.cpu().float(),
        "kth_similarities": kth_sims.cpu().float(),
        "kth_ids": kth_ids.cpu().long(),
    }


def query_training_lpc(
    query_features: torch.Tensor,
    query_ids: torch.Tensor,
    train_features: torch.Tensor,
    train_bundle: dict,
    k: int = 10,
    **topk_kwargs
) -> dict:
    train_ids = train_bundle["ids"].long()
    candidate_ids, candidate_sims = exact_topk(query_features, train_features, train_ids, k, **topk_kwargs)
    threshold_by_id = {
        int(value): (float(train_bundle["kth_similarities"][row]), int(train_bundle["kth_ids"][row]))
        for row, value in enumerate(train_ids.tolist())
    }
    reciprocal = torch.full_like(candidate_ids, -1)
    for row_index, (neighbors, sims) in enumerate(zip(candidate_ids.tolist(), candidate_sims.tolist())):
        kept = []
        query_id = int(query_ids[row_index])
        for neighbor, similarity in zip(neighbors, sims):
            kth_similarity, kth_id = threshold_by_id[int(neighbor)]
            if similarity > kth_similarity or (similarity == kth_similarity and query_id < kth_id):
                kept.append(int(neighbor))
        if kept:
            reciprocal[row_index, :len(kept)] = torch.tensor(kept, dtype=torch.long)
    features_by_id = {int(value): train_features[row].cpu() for row, value in enumerate(train_ids.tolist())}
    # Query fallback vectors live outside the training mapping.
    for row, value in enumerate(query_ids.tolist()):
        features_by_id[int(value)] = query_features[row].cpu()
    context, degree, fallback = _aggregate(features_by_id, query_ids.cpu(), reciprocal, query_features.shape[1])
    return {
        "ids": query_ids.cpu().long(),
        "features": context.cpu().float(),
        "neighbors": reciprocal.cpu().long(),
        "degree": degree.long(),
        "fallback": fallback.bool(),
        "candidate_neighbors": candidate_ids.cpu().long(),
        "candidate_similarities": candidate_sims.cpu().float(),
    }
