"""Per-query retrieval metrics, paired bootstrap, and preregistered gates."""

from __future__ import annotations

import random
from typing import Dict, Mapping, Sequence


def query_contributions(ranks: Sequence[int]) -> Dict[str, list]:
    values = [int(rank) for rank in ranks]
    if not values or any(rank <= 0 for rank in values):
        raise ValueError("ranks must be non-empty positive integers")
    return {
        "r@1": [float(rank <= 1) for rank in values],
        "r@2": [float(rank <= 2) for rank in values],
        "r@5": [float(rank <= 5) for rank in values],
        "r@10": [float(rank <= 10) for rank in values],
        "mrr": [1.0 / rank for rank in values],
    }


def metrics_from_ranks(ranks: Sequence[int]) -> Dict[str, float]:
    values = query_contributions(ranks)
    return {key: sum(row) / len(row) for key, row in values.items()}


def _percentile(values, q):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(q)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(
    left_ranks: Sequence[int],
    right_ranks: Sequence[int],
    iterations: int = 10000,
    seed: int = 2021,
) -> dict:
    if len(left_ranks) != len(right_ranks) or not left_ranks:
        raise ValueError("paired ranks must have equal non-zero length")
    left = query_contributions(left_ranks)
    right = query_contributions(right_ranks)
    import numpy as np

    n = len(left_ranks)
    result = {}
    chunk_size = 256
    for metric_index, metric in enumerate(left):
        per_query = np.asarray(left[metric], dtype=np.float64) - np.asarray(
            right[metric], dtype=np.float64
        )
        point = float(per_query.mean())
        rng = np.random.RandomState(int(seed) + metric_index)
        chunks = []
        remaining = int(iterations)
        while remaining > 0:
            count = min(chunk_size, remaining)
            indices = rng.randint(0, n, size=(count, n))
            chunks.append(per_query[indices].mean(axis=1))
            remaining -= count
        samples = np.concatenate(chunks)
        result[metric] = {
            "difference": point,
            "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            "iterations": int(iterations),
            "seed": int(seed),
        }
    return result


def scientific_gate(dataset: str, comparisons: Mapping[str, Mapping]) -> dict:
    failures = []

    def diff(name, metric):
        return comparisons[name][metric]["difference"]

    def lower(name, metric):
        return comparisons[name][metric]["ci95"][0]

    for metric in ("r@1", "mrr"):
        if lower("vpd_vs_random", metric) <= 0:
            failures.append("vpd_vs_random %s CI lower <= 0" % metric)
    if diff("vpd_vs_final_clip", "r@1") <= 0 or diff("vpd_vs_final_clip", "mrr") <= 0:
        failures.append("vpd_vs_final_clip point difference is not positive")
    if lower("vpd_vs_final_clip", "mrr") <= 0:
        failures.append("vpd_vs_final_clip MRR CI lower <= 0")
    if lower("vpd_vs_base_only", "mrr") <= 0 or diff("vpd_vs_base_only", "r@1") <= 0:
        failures.append("vpd group branch efficacy gate failed")
    if dataset == "dstc":
        for metric in ("r@1", "mrr"):
            if lower("vpd_vs_reference", metric) < -0.01:
                failures.append("DSTC VPD not within 0.01 of LLM on %s" % metric)
    elif dataset == "stickerchat":
        for metric in ("r@1", "mrr"):
            if diff("vpd_vs_reference", metric) < 0 or lower("vpd_vs_reference", metric) < -0.005:
                failures.append("StickerChat VPD reference non-inferiority failed on %s" % metric)
    else:
        raise ValueError("unknown dataset")
    return {"verdict": "GO" if not failures else "STOP", "failures": failures}

