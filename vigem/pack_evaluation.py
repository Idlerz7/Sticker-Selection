"""Paired R10 evaluation and preregistered gate for pack-relative VIGEM."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from style_shapes.evaluation import (
    metrics_from_ranks,
    paired_bootstrap,
)
from style_shapes.io import atomic_write_json, sha256_file


def _load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not value:
        raise ValueError("score file must contain a non-empty query list")
    return [dict(row) for row in value]


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if int(row.get("query_index", -1)) != index:
            raise ValueError("query indices are not dense and ordered")
        candidates = [int(value) for value in row["candidate_ids"]]
        if len(candidates) != 10:
            raise ValueError("pack-relative evaluation must use R10")
        if int(row["gold"]) not in candidates:
            raise ValueError("gold is absent from candidate row")
        for name in (
            "holistic_scores",
            "instance_scores",
            "group_scores",
            "final_scores",
            "final_without_instance_scores",
        ):
            values = [float(value) for value in row[name]]
            if len(values) != 10 or not all(math.isfinite(v) for v in values):
                raise ValueError("invalid %s at query %d" % (name, index))


def _aligned(left, right) -> None:
    if len(left) != len(right):
        raise ValueError("baseline score row count differs")
    for index, (new, old) in enumerate(zip(left, right)):
        if (
            int(new["query_index"]) != int(old["query_index"])
            or int(new["gold"]) != int(old["gold"])
            or [int(v) for v in new["candidate_ids"]]
            != [int(v) for v in old["candidate_ids"]]
        ):
            raise ValueError("baseline candidates are not aligned at row %d" % index)


def _score_stats(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float]:
    values = [
        float(value) for row in rows for value in row[key]
    ]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def evaluate_pack_relative_scores(
    score_path: str,
    output_path: str,
    baseline_score_path: str = "",
    iterations: int = 10000,
    seed: int = 2021,
) -> Dict[str, Any]:
    rows = _load_rows(score_path)
    _validate_rows(rows)
    full_ranks = [int(row["rank"]) for row in rows]
    without_ranks = [int(row["rank_without_instance"]) for row in rows]
    within = paired_bootstrap(full_ranks, without_ranks, iterations, seed)
    full_metrics = metrics_from_ranks(full_ranks)
    without_metrics = metrics_from_ranks(without_ranks)
    full_metrics["map"] = full_metrics["mrr"]
    without_metrics["map"] = without_metrics["mrr"]
    result: Dict[str, Any] = {
        "schema_version": "vigem.pack_relative_evaluation.v1",
        "score_file": {
            "path": score_path,
            "sha256": sha256_file(score_path),
        },
        "queries": len(rows),
        "full": full_metrics,
        "without_instance": without_metrics,
        "full_vs_without_instance": within,
        "rank_changes": {
            "improved": sum(a < b for a, b in zip(full_ranks, without_ranks)),
            "degraded": sum(a > b for a, b in zip(full_ranks, without_ranks)),
            "unchanged": sum(a == b for a, b in zip(full_ranks, without_ranks)),
            "top1_flip": sum(
                (a == 1) != (b == 1)
                for a, b in zip(full_ranks, without_ranks)
            )
            / float(len(rows)),
        },
        "score_stats": {
            "holistic": _score_stats(rows, "holistic_scores"),
            "instance": _score_stats(rows, "instance_scores"),
            "group": _score_stats(rows, "group_scores"),
            "final": _score_stats(rows, "final_scores"),
        },
        "bootstrap": {"iterations": int(iterations), "seed": int(seed)},
    }
    failures = []
    if full_metrics["r@1"] <= without_metrics["r@1"]:
        failures.append("full R@1 is not above Instance=0")
    if full_metrics["mrr"] <= without_metrics["mrr"]:
        failures.append("full MRR is not above Instance=0")

    baseline_path = str(baseline_score_path or "").strip()
    if baseline_path:
        baseline = _load_rows(baseline_path)
        _validate_rows(baseline)
        _aligned(rows, baseline)
        baseline_ranks = [int(row["rank"]) for row in baseline]
        comparison = paired_bootstrap(
            full_ranks, baseline_ranks, iterations, seed
        )
        baseline_metrics = metrics_from_ranks(baseline_ranks)
        baseline_metrics["map"] = baseline_metrics["mrr"]
        result["baseline"] = {
            "score_file": {
                "path": baseline_path,
                "sha256": sha256_file(baseline_path),
            },
            "metrics": baseline_metrics,
            "comparison": comparison,
        }
        if full_metrics["mrr"] - baseline_metrics["mrr"] < 0.003:
            failures.append("MRR improvement over current VIGEM is below 0.003")
        if full_metrics["r@1"] - baseline_metrics["r@1"] < 0.005:
            failures.append("R@1 improvement over current VIGEM is below 0.005")
        if comparison["mrr"]["ci95"][0] <= 0.0:
            failures.append("MRR bootstrap CI lower bound is not positive")
        result["verdict"] = "GO" if not failures else "STOP"
    else:
        result["verdict"] = "BASELINE_PENDING"
        failures.append("current VIGEM aligned score file was not supplied")
    result["failures"] = failures
    atomic_write_json(output_path, result)
    return result
