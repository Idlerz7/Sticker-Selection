#!/usr/bin/env python
"""Extract clean/perturbed VPD bundles using frozen train-only reducers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from lvpcm.catalog import dstc_catalog, stickerchat_catalog
from lvpcm.clip_intermediate import CLIPIntermediateExtractor
from lvpcm.images import perturb_image
from lvpcm.io import atomic_torch_save, atomic_write_json, hash_mapping, load_yaml, refuse_incompatible_manifest, sha256_file
from lvpcm.provenance import command_record
from lvpcm.vpd import fit_reducer, l2_normalize


def catalog_for(config):
    if config["dataset"] == "dstc":
        return dstc_catalog(config["id2img"], config["train_pairs"])
    return stickerchat_catalog(config["id2img"], config["img2id"], config["raw_train_pairs"], config["metadata"])


def transform_chunks(memmap, reducer, chunk=8192):
    output = np.empty((len(memmap), reducer.components.shape[0]), dtype=np.float32)
    for start in range(0, len(memmap), chunk):
        output[start:start + chunk] = reducer.transform_numpy(np.asarray(memmap[start:start + chunk]))
    return torch.from_numpy(output)


def main(args):
    common = load_yaml(args.common_config)
    config = load_yaml(args.dataset_config)
    catalog = catalog_for(config)
    ids = torch.tensor(catalog["ids"], dtype=torch.long)
    paths = [Path(config["image_root"]) / catalog["id2img"][int(value)] for value in ids.tolist()]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing images: %s" % missing[:5])
    output_dir = Path(common["artifact_root"]) / "descriptors" / config["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)
    input_hash = hash_mapping({
        "catalog_hash": catalog["hash"], "clip": sha256_file(Path(common["clip_model"]) / "pytorch_model.bin"),
        "common": common, "dataset": config, "perturbations": args.perturbations,
    })
    run_manifest = output_dir / "extraction_manifest.json"
    if refuse_incompatible_manifest(run_manifest, input_hash) and not args.force:
        print("compatible extraction already complete")
        return
    extractor = CLIPIntermediateExtractor(common["clip_model"], args.device)
    reducers = {}
    image_manifest = []
    perturbations = args.perturbations
    if "clean" not in perturbations:
        perturbations = ["clean"] + perturbations
    for perturbation in perturbations:
        raw_single_path = output_dir / (".%s_single.npy" % perturbation)
        raw_multi_path = output_dir / (".%s_multi.npy" % perturbation)
        single = np.lib.format.open_memmap(raw_single_path, mode="w+", dtype=np.float32, shape=(len(ids), 1536))
        multi = np.lib.format.open_memmap(raw_multi_path, mode="w+", dtype=np.float32, shape=(len(ids), 4608))
        eligible = torch.zeros(len(ids), dtype=torch.bool)
        for start in range(0, len(ids), args.batch_size):
            images = []
            flags = []
            for path in paths[start:start + args.batch_size]:
                data = path.read_bytes()
                image, flag = perturb_image(data, perturbation)
                images.append(image)
                flags.append(flag)
                if perturbation == "clean":
                    canonical = __import__("hashlib").sha256()
                    canonical.update(("%dx%d:" % image.size).encode("ascii"))
                    canonical.update(image.tobytes())
                    image_manifest.append({
                        "path": str(path), "size": len(data),
                        "sha256": __import__("hashlib").sha256(data).hexdigest(),
                        "canonical_rgb_sha256": canonical.hexdigest(),
                    })
            raw_single, raw_multi = extractor.extract(images)
            single[start:start + len(images)] = raw_single.numpy()
            multi[start:start + len(images)] = raw_multi.numpy()
            eligible[start:start + len(images)] = torch.tensor(flags)
            if start == 0 or (start // args.batch_size) % 100 == 0:
                print("%s %d/%d" % (perturbation, min(start + len(images), len(ids)), len(ids)), flush=True)
        single.flush(); multi.flush()
        if perturbation == "clean":
            reducers["single"] = fit_reducer(torch.from_numpy(single), ids, catalog["train_ids"], common["pca_components"], common["seed"])
            reducers["multi"] = fit_reducer(torch.from_numpy(multi), ids, catalog["train_ids"], common["pca_components"], common["seed"])
        for family, raw in (("single", single), ("multi", multi)):
            features = transform_chunks(raw, reducers[family])
            payload = {
                "ids": ids, "features": features.float(), "eligible": eligible,
                "family": family, "perturbation": perturbation,
                "train_ids": torch.tensor(catalog["train_ids"], dtype=torch.long),
                "catalog_hash": catalog["hash"], "input_config_hash": input_hash,
            }
            if perturbation == "clean":
                payload["reducer"] = reducers[family].state_dict()
            atomic_torch_save(output_dir / ("%s_%s.pt" % (family, perturbation)), payload)
        del single, multi
        os.unlink(raw_single_path); os.unlink(raw_multi_path)
    final = torch.load(config["final_clip_cache"], map_location="cpu").float()
    if final.shape != (len(ids), 512) or not torch.isfinite(final).all():
        raise ValueError("invalid existing final CLIP cache")
    atomic_torch_save(output_dir / "final_clip_clean.pt", {
        "ids": ids, "features": l2_normalize(final), "family": "final_clip", "perturbation": "clean",
        "train_ids": torch.tensor(catalog["train_ids"]), "catalog_hash": catalog["hash"], "input_config_hash": input_hash,
        "source": config["final_clip_cache"], "source_sha256": sha256_file(config["final_clip_cache"]),
    })
    manifest = {
        "status": "complete", "input_config_hash": input_hash, "dataset": config["dataset"],
        "catalog_hash": catalog["hash"], "catalog_size": len(ids), "train_catalog_size": len(catalog["train_ids"]),
        "image_manifest_hash": hash_mapping(image_manifest), "image_manifest": image_manifest,
        "outputs": sorted(str(path) for path in output_dir.glob("*.pt")),
    }
    atomic_write_json(run_manifest, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--common-config", default="configs/lvpcm/common.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--perturbations", nargs="+", default=["clean", "resize75", "jpeg75", "alpha_bbox"])
    parser.add_argument("--force", action="store_true")
    options = parser.parse_args()
    with command_record(options):
        main(options)
