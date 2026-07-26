"""Configuration and checkpoint contracts shared by VIGEM entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import yaml

from structured_retrieval_factorized import parse_structured_factorized_args
from structured_retrieval_tokens import _config_mapping_to_argv
from style_shapes.group_bank import GroupBank
from style_shapes.io import sha256_file


NEW_INSTANCE_PREFIXES = (
    "model.instance_query_head.",
    "model.instance_residual_head.",
    "model.instance_temperature_raw",
)


def read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("VIGEM config must be a mapping")
    return value


def build_model_args(config: Mapping[str, Any]):
    overrides = dict(config.get("model_overrides", {}))
    overrides["factorized_bank_path"] = config["group_bank"]
    return parse_structured_factorized_args(
        ["--config", str(config["base_config"])]
        + _config_mapping_to_argv(overrides)
    )


def verify_config_contract(
    config: Mapping[str, Any],
    model_args,
    bank: GroupBank,
    require_init: bool,
) -> None:
    fixed = {
        "seed": 2021,
        "epochs": 10,
        "train_batch_size": 16,
        "lambda_expr": 0.3,
        "lambda_style_proto": 0.4,
    }
    for name, expected in fixed.items():
        actual = getattr(model_args, name)
        if float(actual) != float(expected):
            raise RuntimeError(
                "VIGEM frozen recipe mismatch for %s: %r != %r"
                % (name, actual, expected)
            )
    if str(model_args.factorized_variant) != "minimal":
        raise RuntimeError("VIGEM requires the factorized minimal core")
    if str(config["dataset"]) != str(bank.dataset):
        raise RuntimeError("VIGEM dataset/Group Bank mismatch")
    if str(config["group_source"]) != str(bank.group_source):
        raise RuntimeError("VIGEM group source mismatch")
    if int(config["num_groups"]) != int(bank.num_groups):
        raise RuntimeError("VIGEM Group Bank K mismatch")
    if float(config.get("instance_score_weight", 0.3)) != 0.3:
        raise RuntimeError("instance_score_weight is frozen to 0.3")
    if float(config.get("instance_loss_weight", 0.3)) != 0.3:
        raise RuntimeError("instance_loss_weight is frozen to 0.3")
    residual = Path(str(config["residual_bundle"]))
    residual_manifest = residual.with_suffix(".manifest.json")
    if not residual.exists() or not residual_manifest.exists():
        raise FileNotFoundError(
            "build VIGEM residual assets first: %s" % residual
        )
    with residual_manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("membership_hash") != bank.membership_hash:
        raise RuntimeError("residual manifest membership mismatch")
    if manifest.get("bundle", {}).get("sha256") != sha256_file(residual):
        raise RuntimeError("residual bundle content hash mismatch")
    if str(config["dataset"]) == "dstc":
        if int(model_args.factorized_train_bank_refresh_steps) != 0:
            raise RuntimeError("DSTC bank refresh must be exact (0)")
        if str(model_args.factorized_train_mode) != "legacy_triplet":
            raise RuntimeError("DSTC VIGEM requires legacy triplet candidates")
        if (
            int(model_args.train_same_proto_negatives) != 1
            or int(model_args.train_cross_proto_negatives) != 1
        ):
            raise RuntimeError(
                "DSTC VIGEM freezes one same-group and one cross-group negative"
            )
    else:
        if int(model_args.factorized_train_bank_refresh_steps) != 500:
            raise RuntimeError("StickerChat bank refresh must be 500")
        if str(model_args.factorized_train_mode) != "fixed_same_pack_listwise":
            raise RuntimeError("StickerChat VIGEM requires fixed R10 training")
        if int(model_args.factorized_train_candidate_count) != 10:
            raise RuntimeError("StickerChat VIGEM requires ten candidates")
        if int(model_args.factorized_candidate_forward_chunk_size) not in {
            10,
            5,
            2,
            1,
        }:
            raise RuntimeError("candidate chunk must be 10, 5, 2, or 1")
    if require_init and not Path(str(config["init_checkpoint_path"])).exists():
        raise FileNotFoundError(
            "create the VIGEM initialization snapshot first: %s"
            % config["init_checkpoint_path"]
        )
    if require_init:
        init_path = Path(str(config["init_checkpoint_path"]))
        init_manifest_path = Path(str(init_path) + ".manifest.json")
        if not init_manifest_path.exists():
            raise FileNotFoundError(
                "VIGEM initialization manifest is missing: %s"
                % init_manifest_path
            )
        with init_manifest_path.open("r", encoding="utf-8") as handle:
            init_manifest = json.load(handle)
        compatible_init = (
            init_manifest.get("strict_reload") is True
            and init_manifest.get("membership_hash") == bank.membership_hash
            and init_manifest.get("snapshot", {}).get("sha256")
            == sha256_file(init_path)
            and init_manifest.get("input_hashes", {}).get("residual_bundle")
            == sha256_file(residual)
            and init_manifest.get("input_hashes", {}).get("group_bank")
            == sha256_file(config["group_bank"])
        )
        if not compatible_init:
            raise RuntimeError(
                "VIGEM initialization snapshot is incompatible with config assets"
            )


def load_legacy_shared_initialization(model, checkpoint_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    disallowed_missing = [
        key
        for key in missing
        if not any(
            key == prefix or key.startswith(prefix)
            for prefix in NEW_INSTANCE_PREFIXES
        )
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "legacy initialization mismatch: missing=%s unexpected=%s"
            % (disallowed_missing[:20], unexpected[:20])
        )
    if not missing:
        raise RuntimeError(
            "legacy initialization unexpectedly contained the new VIGEM branch"
        )
    return {
        "missing_new_keys": sorted(missing),
        "unexpected_keys": [],
    }
