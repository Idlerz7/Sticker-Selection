#!/usr/bin/env python
"""Finalize a merged VPD family only after all required bundles validate."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lvpcm.catalog import dstc_catalog, stickerchat_catalog
from lvpcm.io import atomic_write_json, hash_mapping, load_yaml, sha256_file
from lvpcm.provenance import command_record


def main(args):
    common=load_yaml(args.common_config); cfg=load_yaml(args.dataset_config)
    catalog = dstc_catalog(cfg["id2img"],cfg["train_pairs"]) if cfg["dataset"]=="dstc" else stickerchat_catalog(cfg["id2img"],cfg["img2id"],cfg["raw_train_pairs"],cfg["metadata"])
    root=Path(common["artifact_root"])/"descriptors"/cfg["dataset"]
    required=[root/(family+"_"+perturb+".pt") for perturb in ("clean","resize75","jpeg75","alpha_bbox") for family in ("single","multi")]+[root/"final_clip_clean.pt"]
    for path in required:
        bundle=torch.load(path,map_location="cpu")
        if not torch.equal(bundle["ids"],torch.tensor(catalog["ids"])) or not torch.isfinite(bundle["features"]).all(): raise ValueError("invalid descriptor bundle: %s"%path)
    clean=__import__("json").load(open(root/"clean_image_manifest.json",encoding="utf-8"))
    atomic_write_json(root/"extraction_manifest.json",{
        "status":"complete","execution":"content_addressed_shards","dataset":cfg["dataset"],"catalog_hash":catalog["hash"],
        "catalog_size":len(catalog["ids"]),"train_catalog_size":len(catalog["train_ids"]),
        "image_manifest_hash":clean["image_manifest_hash"],"image_manifest":clean["image_manifest"],
        "input_config_hash":hash_mapping({"common":common,"dataset":cfg,"catalog_hash":catalog["hash"]}),
        "outputs":[{"path":str(path),"size":path.stat().st_size,"sha256":sha256_file(path)} for path in required],
    })


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--dataset-config",required=True);parser.add_argument("--common-config",default="configs/lvpcm/common.yaml")
    options=parser.parse_args()
    with command_record(options):main(options)
