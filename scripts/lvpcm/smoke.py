#!/usr/bin/env python
"""Real-image, explicitly non-research smoke for extraction/PCA/LPC."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lvpcm.catalog import dstc_catalog
from lvpcm.clip_intermediate import CLIPIntermediateExtractor
from lvpcm.images import perturb_image
from lvpcm.io import atomic_write_json, load_yaml
from lvpcm.lpc import build_training_lpc
from lvpcm.provenance import command_record
from lvpcm.vpd import fit_reducer


def main(args):
    common = load_yaml(args.common_config); cfg = load_yaml(args.dataset_config)
    catalog = dstc_catalog(cfg["id2img"], cfg["train_pairs"])
    selected = catalog["train_ids"][:args.images]
    if len(selected) < 256:
        raise ValueError("real smoke needs at least 256 train images for frozen PCA width")
    extractor = CLIPIntermediateExtractor(common["clip_model"], args.device)
    results = {}
    reduced = {}
    for perturbation in ("clean", "resize75"):
        raw = []
        for start in range(0, len(selected), args.batch_size):
            images = []
            for image_id in selected[start:start + args.batch_size]:
                path = Path(cfg["image_root"]) / catalog["id2img"][image_id]
                images.append(perturb_image(path.read_bytes(), perturbation)[0])
            _single, multi = extractor.extract(images); raw.append(multi)
        raw = torch.cat(raw)
        if perturbation == "clean":
            reducer = fit_reducer(raw, torch.tensor(selected), selected, 256, common["seed"])
        reduced[perturbation] = reducer.transform(raw)
        results[perturbation] = {"raw_shape": list(raw.shape), "reduced_shape": list(reduced[perturbation].shape), "finite": bool(torch.isfinite(reduced[perturbation]).all())}
    lpc = build_training_lpc(reduced["clean"].to(args.device), torch.tensor(selected, device=args.device), 10, query_batch_size=256, index_batch_size=256)
    atomic_write_json(args.output, {
        "status": "complete", "smoke": True, "research_eligible": False, "dataset": "dstc",
        "images": len(selected), "results": results, "lpc_coverage": float((lpc["degree"] > 0).float().mean()),
        "lpc_fallback_rate": float(lpc["fallback"].float().mean()),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-config", default="configs/lvpcm/common.yaml")
    parser.add_argument("--dataset-config", default="configs/lvpcm/dstc.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--images", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="artifacts/lvpcm/smoke/dstc_real_smoke.json")
    options = parser.parse_args()
    with command_record(options):
        main(options)
