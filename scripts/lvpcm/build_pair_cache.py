#!/usr/bin/env python
"""Build a frozen DSTC `h`/`b` pair cache and exactness evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lvpcm.catalog import read_json
from lvpcm.io import atomic_torch_save, atomic_write_json, hash_mapping, load_yaml, sha256_file
from lvpcm.legacy import load_dstc_plmodel
from lvpcm.pair_cache import extract_pair_cache, validate_against_legacy
from lvpcm.provenance import command_record


def main(args):
    config = load_yaml(args.dataset_config)
    if config["dataset"] != "dstc":
        raise RuntimeError("StickerChat pair cache is blocked: no strict-loadable neutral checkpoint")
    manifest_payload = read_json(args.candidate_manifest)
    manifest_rows = manifest_payload["rows"]
    source_paths = sorted(set(row["source"] for row in manifest_rows))
    if len(source_paths) != 1:
        raise ValueError("candidate manifest must use exactly one source")
    source_rows = read_json(source_paths[0])
    model, legacy_args = load_dstc_plmodel(config, args.device)
    cache = extract_pair_cache(model, legacy_args, source_rows, manifest_rows, args.pair_batch_size)
    exact = validate_against_legacy(
        model, legacy_args, source_rows[manifest_rows[0]["source_row"]], manifest_rows[0], cache["h"][0], cache["b"][0]
    )
    if not exact["candidate_ids_equal"] or exact["h_max_abs"] != 0.0 or exact["b_max_abs"] != 0.0:
        raise RuntimeError("pair cache does not exactly reconstruct legacy forward: %s" % exact)
    cache_hash = hash_mapping({
        "checkpoint": sha256_file(config["checkpoint"]), "candidate_manifest": sha256_file(args.candidate_manifest),
        "mmbbert_config": load_yaml(config["mmbbert_config"]), "source": source_paths[0],
    })
    cache["input_config_hash"] = cache_hash
    cache["checkpoint"] = config["checkpoint"]
    cache["checkpoint_sha256"] = sha256_file(config["checkpoint"])
    cache["tokenizer_rows"] = len(model.model.bert_tokenizer)
    cache["candidate_manifest"] = args.candidate_manifest
    output = Path(args.output)
    atomic_torch_save(output, cache)
    atomic_write_json(output.with_suffix(".manifest.json"), {
        "status": "complete", "input_config_hash": cache_hash, "cache": str(output),
        "cache_sha256": sha256_file(output), "checkpoint": config["checkpoint"],
        "checkpoint_sha256": cache["checkpoint_sha256"], "candidate_manifest": args.candidate_manifest,
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest), "tokenizer_rows": cache["tokenizer_rows"],
        "tensor_schema": {"h": list(cache["h"].shape), "b": list(cache["b"].shape), "candidate_ids": list(cache["candidate_ids"].shape), "positive_index": list(cache["positive_index"].shape)},
        "dtypes": {"h": str(cache["h"].dtype), "b": str(cache["b"].dtype)}, "legacy_exactness": exact,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", default="configs/lvpcm/dstc.yaml")
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pair-batch-size", type=int, default=256)
    options = parser.parse_args()
    with command_record(options):
        main(options)
