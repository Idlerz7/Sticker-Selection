#!/usr/bin/env python3
"""Build deterministic CLIP+VPD group-centred residual bundles for VIGEM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lvpcm.io import atomic_torch_save
from style_shapes.io import atomic_write_json, command_record, sha256_file
from vigem.residuals import (
    InstanceResidualBundle,
    build_instance_residual_bundle,
)


DATASETS = {
    "dstc": {
        "group_bank": "artifacts/style_shapes/groups/dstc/vpd_multi/group_bank.json",
        "final_clip": "artifacts/lvpcm/descriptors/dstc/final_clip_clean.pt",
        "vpd": "artifacts/lvpcm/descriptors/dstc/multi_clean.pt",
        "output": "artifacts/vigem/residuals/dstc_vpd_multi.pt",
    },
    "stickerchat": {
        "group_bank": "artifacts/style_shapes/groups/stickerchat/vpd_pack/group_bank.json",
        "final_clip": "artifacts/lvpcm/descriptors/stickerchat/final_clip_clean.pt",
        "vpd": "artifacts/lvpcm/descriptors/stickerchat/multi_clean.pt",
        "output": "artifacts/vigem/residuals/stickerchat_vpd_pack.pt",
    },
}


def build_one(dataset: str):
    config = DATASETS[dataset]
    output = Path(config["output"])
    manifest_path = output.with_suffix(".manifest.json")
    expected_hashes = {
        name: sha256_file(config[name])
        for name in ("group_bank", "final_clip", "vpd")
    }
    if output.exists() or manifest_path.exists():
        if not (output.exists() and manifest_path.exists()):
            raise RuntimeError("refusing incomplete residual artifact pair")
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        actual_hashes = {
            name: existing.get("inputs", {}).get(name, {}).get("sha256")
            for name in expected_hashes
        }
        if actual_hashes != expected_hashes:
            raise RuntimeError(
                "refusing incompatible residual asset; archive it before rebuilding"
            )
        bundle = InstanceResidualBundle.load(
            str(output),
            expected_membership_hash=existing["membership_hash"],
        )
        if bundle.manifest_hash != existing.get("manifest_hash"):
            raise RuntimeError("residual bundle/manifest hash mismatch")
        return {
            "status": "ALREADY_COMPLETE",
            "dataset": dataset,
            "output": str(output),
            "sha256": sha256_file(output),
            "manifest": str(manifest_path),
            "manifest_hash": bundle.manifest_hash,
        }

    payload, manifest = build_instance_residual_bundle(
        dataset=dataset,
        group_bank_path=config["group_bank"],
        final_clip_path=config["final_clip"],
        vpd_path=config["vpd"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(str(output), payload)
    manifest["bundle"] = {
        "path": str(output),
        "sha256": sha256_file(output),
        "tensor_schema": {
            "ids": list(payload["ids"].shape),
            "residuals": list(payload["residuals"].shape),
            "group_ids": list(payload["group_ids"].shape),
            "group_sizes": list(payload["group_sizes"].shape),
        },
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "COMPLETE",
        "dataset": dataset,
        "output": str(output),
        "sha256": sha256_file(output),
        "manifest": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "dstc", "stickerchat"),
        default="all",
    )
    parser.add_argument("--artifact-root", default="artifacts/vigem")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        names = list(DATASETS) if args.dataset == "all" else [args.dataset]
        results = [build_one(name) for name in names]
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
