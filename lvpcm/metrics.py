"""Ranking metrics, neighbor stability, and deterministic paired bootstrap."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import torch


def ranking_metrics(scores: torch.Tensor, positive_index: torch.Tensor) -> dict:
    if scores.ndim != 2 or positive_index.shape != (scores.shape[0],):
        raise ValueError("invalid score/positive shapes")
    order = torch.argsort(scores, dim=1, descending=True)
    ranks = (order == positive_index.view(-1, 1)).nonzero(as_tuple=False)[:, 1] + 1
    result = {"mrr": float((1.0 / ranks.float()).mean())}
    for k in (1, 2, 5, 10):
        result["r@%d" % k] = float((ranks <= min(k, scores.shape[1])).float().mean())
    return {**result, "ranks": ranks.cpu()}


def jaccard_rows(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("neighbor shapes differ")
    values = []
    for a, b in zip(left.tolist(), right.tolist()):
        aset = set(int(value) for value in a if int(value) >= 0)
        bset = set(int(value) for value in b if int(value) >= 0)
        union = aset | bset
        values.append(len(aset & bset) / len(union) if union else 1.0)
    return np.asarray(values, dtype=np.float64)


def paired_bootstrap_delta(left: Sequence[float], right: Sequence[float], replicates: int = 10000, seed: int = 2021) -> dict:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("paired bootstrap needs non-empty equal vectors")
    rng = np.random.RandomState(seed)
    samples = np.empty(replicates, dtype=np.float64)
    delta = a - b
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        indices = rng.randint(0, len(delta), size=(count, len(delta)))
        samples[start:start + count] = delta[indices].mean(axis=1)
    return {
        "estimate": float(delta.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "replicates": int(replicates),
        "seed": int(seed),
    }
