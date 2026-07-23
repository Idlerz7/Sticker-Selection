#!/usr/bin/env python
"""Construct leakage-isolated reciprocal LPC bundles for all perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lvpcm.io import atomic_torch_save, atomic_write_json, hash_mapping, load_yaml, sha256_file
from lvpcm.lpc import build_training_lpc, query_training_lpc
from lvpcm.provenance import command_record


def main(args):
    common = load_yaml(args.common_config)
    descriptor_dir = Path(common["artifact_root"]) / "descriptors" / args.dataset
    output_dir = Path(common["artifact_root"]) / "lpc" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for perturbation in args.perturbations:
        source = descriptor_dir / ("multi_%s.pt" % perturbation)
        bundle = torch.load(source, map_location="cpu")
        ids = bundle["ids"].long()
        features = bundle["features"].float()
        train_ids = bundle["train_ids"].long()
        row_by_id = {int(value): row for row, value in enumerate(ids.tolist())}
        train_rows = torch.tensor([row_by_id[int(value)] for value in train_ids.tolist()])
        train_features = features[train_rows].to(args.device)
        training = build_training_lpc(
            train_features, train_ids.to(args.device), common["lpc_k"],
            query_batch_size=args.query_batch_size, index_batch_size=args.index_batch_size,
        )
        nontrain_mask = ~torch.isin(ids, train_ids)
        nontrain_ids = ids[nontrain_mask]
        if len(nontrain_ids):
            queried = query_training_lpc(
                features[nontrain_mask].to(args.device), nontrain_ids.to(args.device), train_features,
                training, common["lpc_k"], query_batch_size=args.query_batch_size,
                index_batch_size=args.index_batch_size,
            )
        else:
            queried = None
        combined_features = torch.empty_like(features)
        neighbors = torch.full((len(ids), common["lpc_k"]), -1, dtype=torch.long)
        degree = torch.zeros(len(ids), dtype=torch.long)
        fallback = torch.zeros(len(ids), dtype=torch.bool)
        for part in (training, queried):
            if part is None:
                continue
            for part_row, image_id in enumerate(part["ids"].tolist()):
                row = row_by_id[int(image_id)]
                combined_features[row] = part["features"][part_row]
                neighbors[row] = part["neighbors"][part_row]
                degree[row] = part["degree"][part_row]
                fallback[row] = part["fallback"][part_row]
        input_hash = hash_mapping({
            "descriptor_sha256": sha256_file(source), "k": common["lpc_k"],
            "algorithm": "exact_cosine_reciprocal_train_index", "seed": common["seed"],
        })
        output = output_dir / ("lpc_%s.pt" % perturbation)
        payload = {
            "ids": ids, "features": combined_features.float(), "neighbors": neighbors,
            "degree": degree, "fallback": fallback, "eligible": bundle.get("eligible", torch.ones(len(ids), dtype=torch.bool)),
            "train_ids": train_ids, "perturbation": perturbation, "catalog_hash": bundle["catalog_hash"],
            "input_config_hash": input_hash, "k": common["lpc_k"], "test_isolation": "queries_use_training_index_only",
        }
        atomic_torch_save(output, payload)
        manifests.append({
            "perturbation": perturbation, "path": str(output), "sha256": sha256_file(output),
            "input_config_hash": input_hash, "coverage": float((degree > 0).float().mean()),
            "fallback_rate": float(fallback.float().mean()), "mean_degree": float(degree.float().mean()),
        })
        del train_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_write_json(output_dir / "manifest.json", {
        "status": "complete", "dataset": args.dataset, "bundles": manifests,
        "catalog_hash": manifests and torch.load(manifests[0]["path"], map_location="cpu")["catalog_hash"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("dstc", "stickerchat"), required=True)
    parser.add_argument("--common-config", default="configs/lvpcm/common.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--index-batch-size", type=int, default=8192)
    parser.add_argument("--perturbations", nargs="+", default=["clean", "resize75", "jpeg75", "alpha_bbox"])
    options = parser.parse_args()
    with command_record(options):
        main(options)
