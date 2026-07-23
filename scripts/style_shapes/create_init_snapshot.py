#!/usr/bin/env python3
"""Create a deterministic weights-only initialization shared by all dataset variants."""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
from pathlib import Path

import pytorch_lightning as pl
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lvpcm.io import atomic_torch_save
from structured_retrieval_factorized import (
    StructuredFactorizedPLModel,
    parse_structured_factorized_args,
)
from style_shapes.io import atomic_write_json, command_record, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--group-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        if os.environ.get("CONDA_DEFAULT_ENV") != "stickr-select":
            raise RuntimeError("initialization snapshot must use stickr-select")
        manifest_path = str(args.output) + ".manifest.json"
        expected_base_hash = sha256_file(args.base_config)
        expected_bank_hash = sha256_file(args.group_bank)
        if Path(args.output).exists() or Path(manifest_path).exists():
            if not (Path(args.output).exists() and Path(manifest_path).exists()):
                raise RuntimeError("refusing incomplete initialization artifact pair")
            with open(manifest_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            compatible = (
                existing.get("status") == "COMPLETE"
                and int(existing.get("seed", -1)) == int(args.seed)
                and existing.get("base_config", {}).get("sha256") == expected_base_hash
                and existing.get("group_bank", {}).get("sha256") == expected_bank_hash
                and existing.get("snapshot", {}).get("sha256") == sha256_file(args.output)
                and existing.get("strict_reload") is True
            )
            if not compatible:
                raise RuntimeError("refusing to overwrite incompatible initialization snapshot")
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return
        model_args = parse_structured_factorized_args(
            [
                "--config",
                args.base_config,
                "--factorized_bank_path",
                args.group_bank,
                "--seed",
                str(args.seed),
                "--gpus",
                "1",
            ]
        )
        pl.seed_everything(int(args.seed))
        model = StructuredFactorizedPLModel(model_args)
        state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        atomic_torch_save(
            args.output,
            {
                "state_dict": state_dict,
                "style_shapes": {
                    "weights_only": True,
                    "seed": int(args.seed),
                    "base_config": args.base_config,
                    "group_bank": args.group_bank,
                },
            },
        )
        # Prove strict reload without carrying optimizer/scheduler state.
        reloaded = StructuredFactorizedPLModel(model_args)
        missing, unexpected = reloaded.load_state_dict(
            torch.load(args.output, map_location="cpu")["state_dict"], strict=True
        )
        if missing or unexpected:
            raise RuntimeError("weights-only strict reload failed")
        manifest = {
            "status": "COMPLETE",
            "weights_only": True,
            "seed": int(args.seed),
            "base_config": {
                "path": args.base_config,
                "sha256": expected_base_hash,
            },
            "group_bank": {
                "path": args.group_bank,
                "sha256": expected_bank_hash,
            },
            "snapshot": {
                "path": args.output,
                "sha256": sha256_file(args.output),
            },
            "strict_reload": True,
            "optimizer_state": False,
            "scheduler_state": False,
        }
        atomic_write_json(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

