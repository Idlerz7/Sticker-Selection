#!/usr/bin/env python
"""Run the frozen six-variant one-seed Stage-B comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lvpcm.io import atomic_torch_save, atomic_write_json, atomic_write_text, load_yaml, sha256_file
from lvpcm.provenance import command_record
from lvpcm.scorers import trainable_parameter_count
from lvpcm.stage_b import (
    VARIANTS, centered_pca_condition, condition_tensor, epoch_permutations, evaluate_variant,
    paired_comparisons, slice_metrics, split_safe_shuffle, train_variant,
)


def main(args):
    stage_a = __import__("json").load(open(args.stage_a_report, encoding="utf-8"))
    if stage_a["stage_a_verdict"] not in ("GO", "CONDITIONAL GO"):
        raise RuntimeError("Stage B is not authorized by Stage A: %s" % stage_a["stage_a_verdict"])
    config = load_yaml(args.config)
    train_cache = torch.load(args.train_pair_cache, map_location="cpu")
    eval_cache = torch.load(args.eval_pair_cache, map_location="cpu")
    descriptor_dir = Path(args.descriptor_dir)
    final_bundle = centered_pca_condition(torch.load(descriptor_dir / "final_clip_clean.pt", map_location="cpu"), config["condition_dim"], config["seed"])
    single_bundle = torch.load(descriptor_dir / "multi_clean.pt", map_location="cpu")
    local_bundle = torch.load(args.lpc_bundle, map_location="cpu")
    shuffled_bundle, shuffle_map = split_safe_shuffle(local_bundle, config["seed"])
    bundles = {"final_clip": final_bundle, "single_vpd": single_bundle, "local_lpc": local_bundle, "shuffled_lpc": shuffled_bundle}
    train_conditions = {name: condition_tensor(bundle, train_cache["candidate_ids"]) for name, bundle in bundles.items()}
    eval_conditions = {name: condition_tensor(bundle, eval_cache["candidate_ids"]) for name, bundle in bundles.items()}
    sigma_b = float(train_cache["b"].float().std(unbiased=False))
    permutations = epoch_permutations(len(train_cache["b"]), config["epochs"], config["seed"])
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(output_dir / "epoch_permutations.pt", {"seed": config["seed"], "permutations": permutations})
    atomic_write_json(output_dir / "shuffled_lpc_bijection.json", {"seed": config["seed"], "mapping": shuffle_map, "split_safe": True})
    results = {}; score_by_variant = {}
    for variant in VARIANTS:
        condition_name = variant if variant in bundles else None
        model, losses = train_variant(
            variant, train_cache, train_conditions.get(condition_name), sigma_b, config, permutations, args.device
        )
        scores, delta, metrics = evaluate_variant(
            model, variant, eval_cache, eval_conditions.get(condition_name), args.device, config["batch_size"]
        )
        score_by_variant[variant] = scores
        parameter_count = 0 if model is None else trainable_parameter_count(model)
        checkpoint_path = None
        if model is not None:
            checkpoint_path = output_dir / "checkpoints" / (variant + ".pt")
            atomic_torch_save(checkpoint_path, {"variant": variant, "state_dict": model.state_dict(), "sigma_b": sigma_b, "config": config})
        atomic_torch_save(output_dir / ("%s_query_scores.pt" % variant), {
            "scores": scores, "delta": delta, "positive_index": eval_cache["positive_index"], "candidate_ids": eval_cache["candidate_ids"],
        })
        results[variant] = {
            "metrics": {key: value for key, value in metrics.items() if key != "ranks"}, "losses": losses,
            "active_trainable_parameters": parameter_count, "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "slices": slice_metrics(scores, eval_cache["positive_index"], local_bundle, eval_cache["candidate_ids"]),
        }
    comparisons = paired_comparisons(score_by_variant, eval_cache["positive_index"], 10000, config["seed"])
    local_unconditional = comparisons["local_minus_unconditional"]
    local_gain = local_unconditional["mrr"]["estimate"]
    shuffled_gap = comparisons["local_minus_shuffled_lpc"]["mrr"]["estimate"]
    dstc_evidence_pass = bool(
        all(comparisons["local_minus_%s" % other]["mrr"]["ci95"][0] > 0 for other in ("unconditional", "final_clip", "single_vpd"))
        and local_unconditional["r@1"]["estimate"] > 0
        and local_gain > 0 and shuffled_gap >= 0.75 * local_gain
    )
    # Formal preregistration requires both StickerChat protocol families, which are audited missing.
    verdict = "STOP"
    report = {
        "status": "complete", "stage_b_verdict": verdict, "dataset": "dstc", "seed": config["seed"],
        "sigma_b_population_std": sigma_b, "results": results, "paired_bootstrap": comparisons,
        "dstc_evidence_pass": dstc_evidence_pass,
        "formal_gate": {
            "pass": False, "reason": "StickerChat formal comparisons require an unavailable neutral checkpoint; same-pack reconstruction also has 16 validation and 5 test rows without original negatives",
            "stickerchat_global": "BLOCKED", "stickerchat_same_pack": "BLOCKED",
        },
        "inputs": {"train_pair_cache": args.train_pair_cache, "eval_pair_cache": args.eval_pair_cache, "lpc_bundle": args.lpc_bundle},
    }
    atomic_write_json(output_dir / "results.json", report)
    lines = [
        "# LVPCM Stage-B Report", "", "Formal verdict: **STOP**", "",
        "DSTC one-seed evidence: **%s**." % ("PASS" if dstc_evidence_pass else "FAIL"),
        "The preregistered overall gate cannot pass because both StickerChat global and same-pack comparisons are BLOCKED by the missing neutral checkpoint.",
        "", "All six variants used the same pair caches, saved epoch permutations, optimizer schedule, and final-checkpoint evaluation.",
    ]
    atomic_write_text(output_dir / "report.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a-report", default="artifacts/lvpcm/stage_a_report.json")
    parser.add_argument("--config", default="configs/lvpcm/stage_b.yaml")
    parser.add_argument("--train-pair-cache", required=True)
    parser.add_argument("--eval-pair-cache", required=True)
    parser.add_argument("--descriptor-dir", default="artifacts/lvpcm/descriptors/dstc")
    parser.add_argument("--lpc-bundle", default="artifacts/lvpcm/lpc/dstc/lpc_clean.pt")
    parser.add_argument("--output-dir", default="artifacts/lvpcm/stage_b")
    parser.add_argument("--device", default="cuda:0")
    options = parser.parse_args()
    with command_record(options):
        main(options)
