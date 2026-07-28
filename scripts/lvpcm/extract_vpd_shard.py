#!/usr/bin/env python
"""Content-addressed contiguous shard worker for formal VPD extraction."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from lvpcm.catalog import dstc_catalog, stickerchat_catalog
from lvpcm.clip_intermediate import CLIPIntermediateExtractor
from lvpcm.images import perturb_image
from lvpcm.io import atomic_torch_save, hash_mapping, load_yaml, sha256_file
from lvpcm.provenance import command_record


def get_catalog(config):
    if config["dataset"] == "dstc":
        return dstc_catalog(config["id2img"], config["train_pairs"])
    return stickerchat_catalog(config["id2img"], config["img2id"], config["raw_train_pairs"], config["metadata"])


def main(args):
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("invalid shard index")
    common = load_yaml(args.common_config); config = load_yaml(args.dataset_config); catalog = get_catalog(config)
    total = len(catalog["ids"]); start = total * args.shard_index // args.num_shards; end = total * (args.shard_index + 1) // args.num_shards
    ids = torch.tensor(catalog["ids"][start:end], dtype=torch.long)
    shard_hash = hash_mapping({
        "catalog_hash": catalog["hash"], "clip_sha256": sha256_file(Path(common["clip_model"]) / "pytorch_model.bin"),
        "perturbation": args.perturbation, "shard_index": args.shard_index, "num_shards": args.num_shards,
        "ids": ids.tolist(), "algorithm": "lvpcm_vpd_v1",
    })
    output = Path(args.output_dir) / config["dataset"] / ("%s_%03d-of-%03d.pt" % (args.perturbation, args.shard_index, args.num_shards))
    if output.is_file():
        existing = torch.load(output, map_location="cpu")
        if existing.get("input_config_hash") != shard_hash:
            raise RuntimeError("refusing to overwrite incompatible VPD shard: %s" % output)
        if existing.get("status") == "complete":
            print("compatible shard already complete: %s" % output)
            return
    extractor = CLIPIntermediateExtractor(common["clip_model"], args.device)
    single = torch.empty((len(ids), 1536), dtype=torch.float32)
    multi = torch.empty((len(ids), 4608), dtype=torch.float32)
    eligible = torch.zeros(len(ids), dtype=torch.bool); image_manifest = []
    for offset in range(0, len(ids), args.batch_size):
        images = []; flags = []
        for image_id in ids[offset:offset + args.batch_size].tolist():
            path = Path(config["image_root"]) / catalog["id2img"][int(image_id)]
            data = path.read_bytes(); image, flag = perturb_image(data, args.perturbation)
            images.append(image); flags.append(flag)
            if args.perturbation == "clean":
                canonical = hashlib.sha256(); canonical.update(("%dx%d:" % image.size).encode("ascii")); canonical.update(image.tobytes())
                image_manifest.append({
                    "id": int(image_id), "path": str(path), "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "canonical_rgb_sha256": canonical.hexdigest(),
                })
        a, b = extractor.extract(images)
        single[offset:offset + len(images)] = a; multi[offset:offset + len(images)] = b
        eligible[offset:offset + len(images)] = torch.tensor(flags)
        if offset == 0 or (offset // args.batch_size) % 50 == 0:
            print("%s shard %d/%d %d/%d" % (args.perturbation, args.shard_index + 1, args.num_shards, min(offset + len(images), len(ids)), len(ids)), flush=True)
    atomic_torch_save(output, {
        "status": "complete", "ids": ids, "single_raw": single, "multi_raw": multi, "eligible": eligible,
        "image_manifest": image_manifest, "catalog_hash": catalog["hash"], "train_ids": torch.tensor(catalog["train_ids"]),
        "perturbation": args.perturbation, "shard_index": args.shard_index, "num_shards": args.num_shards,
        "input_config_hash": shard_hash,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--common-config", default="configs/lvpcm/common.yaml")
    parser.add_argument("--perturbation", choices=("clean", "resize75", "jpeg75", "alpha_bbox"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", default="artifacts/lvpcm/work/vpd_shards")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    options = parser.parse_args()
    with command_record(options):
        main(options)
