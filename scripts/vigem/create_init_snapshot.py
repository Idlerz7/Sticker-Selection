#!/usr/bin/env python3
"""Create a strict VIGEM init snapshot from an existing Style Shapes init."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pytorch_lightning as pl
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lvpcm.io import atomic_torch_save
from style_shapes.group_bank import GroupBank
from style_shapes.io import atomic_write_json, command_record, sha256_file
from vigem.config import (
    build_model_args,
    load_legacy_shared_initialization,
    read_config,
    verify_config_contract,
)
from vigem.training import VigemInstanceResidualPLModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", default="artifacts/vigem")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        config = read_config(args.config)
        if os.environ.get("CONDA_DEFAULT_ENV") != str(
            config.get("conda_env", "stickr-select")
        ):
            raise RuntimeError(
                "activate Conda environment %s before creating the snapshot"
                % config.get("conda_env", "stickr-select")
            )
        bank = GroupBank.load(config["group_bank"])
        model_args = build_model_args(config)
        model_args.gpus = 1
        verify_config_contract(config, model_args, bank, require_init=False)
        output = Path(config["init_checkpoint_path"])
        manifest_path = Path(str(output) + ".manifest.json")
        legacy_path = str(config["legacy_init_checkpoint_path"])
        expected = {
            "config": sha256_file(args.config),
            "legacy_init": sha256_file(legacy_path),
            "residual_bundle": sha256_file(config["residual_bundle"]),
            "group_bank": sha256_file(config["group_bank"]),
        }
        if output.exists() or manifest_path.exists():
            if not (output.exists() and manifest_path.exists()):
                raise RuntimeError("refusing incomplete VIGEM init artifact pair")
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if (
                existing.get("input_hashes") != expected
                or existing.get("snapshot", {}).get("sha256")
                != sha256_file(output)
                or existing.get("strict_reload") is not True
            ):
                raise RuntimeError("refusing incompatible VIGEM init snapshot")
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return

        pl.seed_everything(int(model_args.seed))
        model = VigemInstanceResidualPLModel(
            model_args,
            residual_bundle_path=config["residual_bundle"],
            membership_hash=bank.membership_hash,
            defer_residual_load=True,
        )
        load_audit = load_legacy_shared_initialization(model, legacy_path)
        state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            str(output),
            {
                "state_dict": state,
                "vigem": {
                    "weights_only": True,
                    "seed": int(model_args.seed),
                    "legacy_init_checkpoint": legacy_path,
                    "residual_bundle": config["residual_bundle"],
                    "membership_hash": bank.membership_hash,
                },
            },
        )

        pl.seed_everything(int(model_args.seed))
        reloaded = VigemInstanceResidualPLModel(
            model_args,
            residual_bundle_path=config["residual_bundle"],
            membership_hash=bank.membership_hash,
            defer_residual_load=True,
        )
        missing, unexpected = reloaded.load_state_dict(
            torch.load(output, map_location="cpu")["state_dict"], strict=True
        )
        if missing or unexpected:
            raise RuntimeError("new VIGEM snapshot failed strict reload")
        manifest = {
            "status": "COMPLETE",
            "schema_version": "vigem.init_snapshot.v1",
            "seed": int(model_args.seed),
            "dataset": config["dataset"],
            "group_source": config["group_source"],
            "membership_hash": bank.membership_hash,
            "input_hashes": expected,
            "legacy_load_audit": load_audit,
            "snapshot": {
                "path": str(output),
                "sha256": sha256_file(output),
            },
            "strict_reload": True,
            "optimizer_state": False,
            "scheduler_state": False,
        }
        atomic_write_json(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
