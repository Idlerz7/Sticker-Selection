#!/usr/bin/env python
"""Validate and merge formal VPD shards, fitting reducers only on clean train IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lvpcm.catalog import dstc_catalog, stickerchat_catalog
from lvpcm.io import atomic_torch_save, atomic_write_json, hash_mapping, load_yaml, sha256_file
from lvpcm.provenance import command_record
from lvpcm.vpd import FrozenReducer, fit_reducer, l2_normalize


def get_catalog(config):
    if config["dataset"] == "dstc": return dstc_catalog(config["id2img"], config["train_pairs"])
    return stickerchat_catalog(config["id2img"], config["img2id"], config["raw_train_pairs"], config["metadata"])


def transform_chunks(values, reducer, chunk=8192):
    output = torch.empty((len(values), reducer.components.shape[0]), dtype=torch.float32)
    for start in range(0, len(values), chunk):
        output[start:start + chunk] = torch.from_numpy(reducer.transform_numpy(values[start:start + chunk].numpy()))
    return output


def main(args):
    common = load_yaml(args.common_config); config = load_yaml(args.dataset_config); catalog = get_catalog(config)
    shard_dir = Path(args.shard_dir) / config["dataset"]
    shard_paths = [shard_dir / ("%s_%03d-of-%03d.pt" % (args.perturbation, index, args.num_shards)) for index in range(args.num_shards)]
    if not all(path.is_file() for path in shard_paths): raise FileNotFoundError("missing VPD shards")
    shards = [torch.load(path, map_location="cpu") for path in shard_paths]
    ids = torch.cat([value["ids"] for value in shards]); expected = torch.tensor(catalog["ids"])
    if not torch.equal(ids, expected): raise ValueError("shards do not exactly reconstruct frozen ID order")
    if len({value["catalog_hash"] for value in shards}) != 1 or shards[0]["catalog_hash"] != catalog["hash"]: raise ValueError("shard catalog hash mismatch")
    raw = {"single": torch.cat([value["single_raw"] for value in shards]), "multi": torch.cat([value["multi_raw"] for value in shards])}
    eligible = torch.cat([value["eligible"] for value in shards]); output_dir = Path(common["artifact_root"]) / "descriptors" / config["dataset"]; output_dir.mkdir(parents=True, exist_ok=True)
    reducers = {}
    for family in ("single", "multi"):
        clean_path = output_dir / (family + "_clean.pt")
        if args.perturbation == "clean":
            reducers[family] = fit_reducer(raw[family], ids, catalog["train_ids"], common["pca_components"], common["seed"])
        else:
            clean = torch.load(clean_path, map_location="cpu")
            reducers[family] = FrozenReducer.from_state_dict(clean["reducer"])
        features = transform_chunks(raw[family], reducers[family])
        input_hash = hash_mapping({"shards": [sha256_file(path) for path in shard_paths], "catalog_hash": catalog["hash"], "family": family, "perturbation": args.perturbation})
        payload = {
            "ids": ids, "features": features, "eligible": eligible, "family": family, "perturbation": args.perturbation,
            "train_ids": torch.tensor(catalog["train_ids"]), "catalog_hash": catalog["hash"], "input_config_hash": input_hash,
            "shards": [str(path) for path in shard_paths],
        }
        if args.perturbation == "clean": payload["reducer"] = reducers[family].state_dict()
        atomic_torch_save(output_dir / ("%s_%s.pt" % (family, args.perturbation)), payload)
    if args.perturbation == "clean":
        final = torch.load(config["final_clip_cache"], map_location="cpu").float()
        if final.shape != (len(ids), 512) or not torch.isfinite(final).all(): raise ValueError("invalid final CLIP cache")
        atomic_torch_save(output_dir / "final_clip_clean.pt", {
            "ids": ids, "features": l2_normalize(final), "family": "final_clip", "perturbation": "clean",
            "train_ids": torch.tensor(catalog["train_ids"]), "catalog_hash": catalog["hash"],
            "source": config["final_clip_cache"], "source_sha256": sha256_file(config["final_clip_cache"]),
        })
        image_manifest = sum((value["image_manifest"] for value in shards), [])
        if [value["id"] for value in image_manifest] != catalog["ids"]: raise ValueError("clean image manifest order mismatch")
        atomic_write_json(output_dir / "clean_image_manifest.json", {
            "status": "complete", "catalog_hash": catalog["hash"], "image_manifest_hash": hash_mapping(image_manifest), "image_manifest": image_manifest,
        })
    atomic_write_json(output_dir / ("%s_merge_manifest.json" % args.perturbation), {
        "status": "complete", "perturbation": args.perturbation, "catalog_hash": catalog["hash"],
        "shards": [{"path": str(path), "sha256": sha256_file(path)} for path in shard_paths],
        "outputs": [str(output_dir / (family + "_" + args.perturbation + ".pt")) for family in ("single", "multi")],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--common-config", default="configs/lvpcm/common.yaml")
    parser.add_argument("--perturbation", choices=("clean", "resize75", "jpeg75", "alpha_bbox"), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-dir", default="artifacts/lvpcm/work/vpd_shards")
    options = parser.parse_args()
    with command_record(options): main(options)
