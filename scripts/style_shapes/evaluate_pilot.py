#!/usr/bin/env python3
"""Compute formal metrics, 10k paired bootstrap, structural gates, and verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from style_shapes.evaluation import metrics_from_ranks, paired_bootstrap, scientific_gate
from style_shapes.io import atomic_write_json, atomic_write_text, command_record, sha256_file


def load_scores(path):
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    required = {
        "query_index",
        "candidate_ids",
        "gold",
        "positive_index",
        "base_scores",
        "instance_scores",
        "group_scores",
        "final_scores",
        "rank",
        "membership_hash",
    }
    for index, row in enumerate(rows):
        if not required.issubset(row):
            raise ValueError("score row %d misses fields" % index)
        candidates = [int(value) for value in row["candidate_ids"]]
        gold = int(row["gold"])
        if candidates.count(gold) != 1 or candidates[int(row["positive_index"])] != gold:
            raise ValueError("score row %d candidate/gold mismatch" % index)
        if any(len(row[name]) != len(candidates) for name in (
            "base_scores", "instance_scores", "group_scores", "final_scores"
        )):
            raise ValueError("score row %d score length mismatch" % index)
    return rows


def aligned_ranks(named):
    reference = None
    output = {}
    for name, rows in named.items():
        identity = [
            (int(row["query_index"]), tuple(int(value) for value in row["candidate_ids"]), int(row["gold"]))
            for row in rows
        ]
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("%s query/candidate order differs from other variants" % name)
        output[name] = [int(row["rank"]) for row in rows]
    return output


def structural_gate(dataset, construction):
    vpd_name = "vpd_multi" if dataset == "dstc" else "vpd_pack"
    reference_name = "llm_original" if dataset == "dstc" else "final_clip_pack_original"
    vpd = construction["sources"][vpd_name]
    reference = construction["sources"][reference_name]
    failures = []
    if vpd["min_group_size"] <= 0:
        failures.append("empty VPD group")
    if vpd["max_group_fraction"] > max(0.20, 2.0 * reference["max_group_fraction"]):
        failures.append("maximum group share gate failed")
    if vpd["effective_group_count"] < 0.5 * reference["effective_group_count"]:
        failures.append("effective group count gate failed")
    if vpd["same_group_negative_coverage"] < 0.75:
        failures.append("same-group negative coverage below 75%")
    if (
        vpd["same_group_negative_coverage"]
        < reference["same_group_negative_coverage"] - 0.05
    ):
        failures.append("same-group negative coverage more than 5pp below reference")
    return {"verdict": "PASS" if not failures else "STOP", "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dstc", "stickerchat"), required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--vpd", required=True)
    parser.add_argument("--random", required=True)
    parser.add_argument("--final-clip", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base-only", required=True)
    parser.add_argument("--group-construction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        paths = {
            "vpd": args.vpd,
            "random": args.random,
            "final_clip": args.final_clip,
            "reference": args.reference,
            "base_only": args.base_only,
        }
        scores = {name: load_scores(path) for name, path in paths.items()}
        ranks = aligned_ranks(scores)
        metrics = {name: metrics_from_ranks(value) for name, value in ranks.items()}
        comparisons = {
            "vpd_vs_random": paired_bootstrap(
                ranks["vpd"], ranks["random"], args.iterations, args.seed
            ),
            "vpd_vs_final_clip": paired_bootstrap(
                ranks["vpd"], ranks["final_clip"], args.iterations, args.seed
            ),
            "vpd_vs_reference": paired_bootstrap(
                ranks["vpd"], ranks["reference"], args.iterations, args.seed
            ),
            "vpd_vs_base_only": paired_bootstrap(
                ranks["vpd"], ranks["base_only"], args.iterations, args.seed
            ),
        }
        science = scientific_gate(args.dataset, comparisons)
        with open(args.group_construction, "r", encoding="utf-8") as handle:
            construction = json.load(handle)
        structure = structural_gate(args.dataset, construction)
        verdict = (
            "GO"
            if science["verdict"] == "GO" and structure["verdict"] == "PASS"
            else "STOP"
        )
        result = {
            "status": "COMPLETE",
            "dataset": args.dataset,
            "protocol": args.protocol,
            "queries": len(ranks["vpd"]),
            "metrics": metrics,
            "comparisons": comparisons,
            "scientific_gate": science,
            "structural_gate": structure,
            "verdict": verdict,
            "score_files": {
                name: {"path": path, "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
            "bootstrap": {"iterations": args.iterations, "seed": args.seed},
        }
        atomic_write_json(args.output, result)
        lines = [
            "# %s %s Style Shapes Result" % (args.dataset, args.protocol),
            "",
            "Verdict: **%s**" % verdict,
            "",
            "| Variant | R@1 | R@2 | R@5 | R@10 | MRR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, row in metrics.items():
            lines.append(
                "| %s | %.4f | %.4f | %.4f | %.4f | %.4f |"
                % (name, row["r@1"], row["r@2"], row["r@5"], row["r@10"], row["mrr"])
            )
        atomic_write_text(str(Path(args.output).with_suffix(".md")), "\n".join(lines) + "\n")
        print(json.dumps({"verdict": verdict, "queries": len(ranks["vpd"])}, indent=2))


if __name__ == "__main__":
    main()

