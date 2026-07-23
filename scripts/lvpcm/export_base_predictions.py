#!/usr/bin/env python
"""Export immutable per-query base predictions from an exact pair cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lvpcm.catalog import read_json
from lvpcm.io import atomic_write_json, sha256_file
from lvpcm.provenance import command_record


def main(args):
    cache = torch.load(args.pair_cache, map_location="cpu")
    manifest = read_json(args.candidate_manifest)["rows"]
    scores = cache["b"].float(); positive = cache["positive_index"].long()
    order = torch.argsort(scores, dim=1, descending=True)
    ranks = (order == positive.view(-1, 1)).nonzero(as_tuple=False)[:, 1] + 1
    sorted_scores = torch.sort(scores, dim=1, descending=True).values
    rows = []
    for index, source in enumerate(manifest):
        rows.append({
            "query_id": source["query_id"], "source_row": source["source_row"], "dialogue_id": source["dialogue_id"],
            "candidate_ids": source["candidate_ids"], "positive_index": int(positive[index]),
            "base_logits": scores[index].tolist(), "rank": int(ranks[index]),
            "margin": float(sorted_scores[index, 0] - sorted_scores[index, 1]),
        })
    atomic_write_json(args.output, {
        "status": "complete", "pair_cache": args.pair_cache, "pair_cache_sha256": sha256_file(args.pair_cache),
        "candidate_manifest": args.candidate_manifest, "rows": rows,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-cache", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output", default="artifacts/lvpcm/base_predictions/dstc_validation.json")
    options = parser.parse_args()
    with command_record(options):
        main(options)
