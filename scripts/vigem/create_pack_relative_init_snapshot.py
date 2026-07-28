#!/usr/bin/env python3
"""Create a strict setwise init from the existing seed-2021 VIGEM snapshot."""

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
from vigem.pack_config import (
    build_pack_model_args,
    load_vigem_parent_initialization,
    read_pack_config,
    verify_pack_config,
)
from vigem.pack_training import PackRelativeSetwisePLModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", default="artifacts/vigem")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        config = read_pack_config(args.config)
        expected_env = str(config.get("conda_env", "stickr-select"))
        if os.environ.get("CONDA_DEFAULT_ENV") != expected_env:
            raise RuntimeError(
                "activate Conda environment %s before creating init"
                % expected_env
            )
        bank = GroupBank.load(config["group_bank"])
        model_args = build_pack_model_args(config)
        model_args.gpus = 1
        residual_manifest = verify_pack_config(
            config, model_args, bank, require_init=False
        )
        output = Path(config["init_checkpoint_path"])
        manifest_path = Path(str(output) + ".manifest.json")
        parent_path = str(config["parent_vigem_init_checkpoint_path"])
        expected = {
            "config": sha256_file(args.config),
            "parent_vigem_init": sha256_file(parent_path),
            "pack_residual_bundle": sha256_file(
                config["pack_residual_bundle"]
            ),
            "group_bank": sha256_file(config["group_bank"]),
        }
        if output.exists() or manifest_path.exists():
            if not (output.exists() and manifest_path.exists()):
                raise RuntimeError("refusing incomplete setwise init pair")
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if (
                existing.get("input_hashes") != expected
                or existing.get("snapshot", {}).get("sha256")
                != sha256_file(output)
                or existing.get("strict_reload") is not True
            ):
                raise RuntimeError("refusing incompatible setwise init")
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return

        pl.seed_everything(int(model_args.seed))
        model = PackRelativeSetwisePLModel(
            model_args,
            residual_bundle_path=config["pack_residual_bundle"],
            membership_hash=bank.membership_hash,
            defer_residual_load=True,
        )
        load_audit = load_vigem_parent_initialization(model, parent_path)
        state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        parent_state = torch.load(parent_path, map_location="cpu")["state_dict"]
        if not set(parent_state).issubset(state):
            raise RuntimeError("setwise init does not preserve all parent keys")
        for key, value in parent_state.items():
            if not torch.equal(value, state[key]):
                raise RuntimeError("shared initialization changed at %s" % key)
        final_weight = state[
            "model.instance_set_scorer.3.weight"
        ]
        final_bias = state["model.instance_set_scorer.3.bias"]
        if int(torch.count_nonzero(final_weight)) or int(
            torch.count_nonzero(final_bias)
        ):
            raise RuntimeError("setwise scorer final layer must initialize to zero")

        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            output,
            {
                "state_dict": state,
                "vigem_pack_relative": {
                    "weights_only": True,
                    "seed": int(model_args.seed),
                    "parent_vigem_init": parent_path,
                    "pack_residual_bundle": config["pack_residual_bundle"],
                    "vpd_membership_hash": bank.membership_hash,
                    "pack_membership_hash": residual_manifest[
                        "pack_membership_hash"
                    ],
                },
            },
        )

        pl.seed_everything(int(model_args.seed))
        reloaded = PackRelativeSetwisePLModel(
            model_args,
            residual_bundle_path=config["pack_residual_bundle"],
            membership_hash=bank.membership_hash,
            defer_residual_load=True,
        )
        missing, unexpected = reloaded.load_state_dict(
            torch.load(output, map_location="cpu")["state_dict"], strict=True
        )
        if missing or unexpected:
            raise RuntimeError("setwise init failed strict reload")
        manifest = {
            "status": "COMPLETE",
            "schema_version": "vigem.pack_relative_init.v1",
            "seed": int(model_args.seed),
            "dataset": "stickerchat",
            "vpd_membership_hash": bank.membership_hash,
            "pack_membership_hash": residual_manifest[
                "pack_membership_hash"
            ],
            "input_hashes": expected,
            "parent_load_audit": load_audit,
            "shared_tensors_exact": True,
            "setwise_final_layer_zero": True,
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
