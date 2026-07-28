"""Strict contracts for the independent StickerChat pack-relative repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import yaml

from style_shapes.group_bank import GroupBank
from style_shapes.io import sha256_file
from vigem.config import build_model_args
from vigem.pack_relative import PACK_RELATIVE_SCHEMA


SETWISE_PREFIX = "model.instance_set_scorer."


def read_pack_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("pack-relative VIGEM config must be a mapping")
    return value


def verify_pack_config(
    config: Mapping[str, Any],
    model_args,
    bank: GroupBank,
    require_init: bool,
) -> Dict[str, Any]:
    if str(config.get("experiment")) != "vpd_pack_relative_setwise_r10":
        raise RuntimeError("unexpected pack-relative experiment name")
    if str(config.get("dataset")) != "stickerchat":
        raise RuntimeError("pack-relative repair is StickerChat-only")
    if bank.dataset != "stickerchat" or bank.group_source != "vpd_pack":
        raise RuntimeError("pack-relative repair requires StickerChat VPD pack bank")
    fixed = {
        "seed": 2021,
        "epochs": 10,
        "train_batch_size": 16,
        "lambda_expr": 0.3,
        "lambda_style_proto": 0.4,
        "gradient_accumulation_steps": 1,
        "factorized_train_candidate_count": 10,
        "factorized_train_bank_refresh_steps": 500,
    }
    for name, expected in fixed.items():
        actual = getattr(model_args, name)
        if float(actual) != float(expected):
            raise RuntimeError(
                "pack-relative frozen recipe mismatch for %s: %r != %r"
                % (name, actual, expected)
            )
    if str(model_args.factorized_variant) != "minimal":
        raise RuntimeError("pack-relative repair requires minimal core")
    if str(model_args.factorized_train_mode) != "fixed_same_pack_listwise":
        raise RuntimeError("pack-relative repair requires fixed R10 training")
    if int(model_args.factorized_candidate_forward_chunk_size) not in {
        10,
        5,
        2,
        1,
    }:
        raise RuntimeError("candidate chunk must be 10, 5, 2, or 1")
    if float(config.get("instance_score_weight", 0.3)) != 0.3:
        raise RuntimeError("instance score weight is frozen to 0.3")
    if float(config.get("instance_loss_weight", 0.3)) != 0.3:
        raise RuntimeError("instance loss weight is frozen to 0.3")
    if config.get("group_gradient_policy") != "detach_shared_bert":
        raise RuntimeError("Group gradient policy must detach the shared BERT")
    if (getattr(model_args, "per_epoch_eval_test_r10_path", "") or "").strip():
        raise RuntimeError("pack-relative runner uses val_data_path for R10 only")
    if (getattr(model_args, "per_epoch_eval_test_r20_path", "") or "").strip():
        raise RuntimeError("R20 access is forbidden in pack-relative runner")
    required_r10 = (
        "release_val_u_sticker_format_int_with_cand_fixed_same_pack_r10.json",
        "release_test_u_sticker_format_int_with_cand_fixed_same_pack_r10.json",
    )
    if str(model_args.mode) in {"train", "pretrain"} and not str(
        model_args.val_data_path
    ).endswith(required_r10[0]):
        raise RuntimeError("validation must use clean fixed same-pack R10")
    if not str(model_args.test_data_path).endswith(required_r10[1]):
        raise RuntimeError("test must use clean fixed same-pack R10")
    forbidden = ("r20", "global_r20", "with_cand_r20")
    for key, value in config.items():
        if isinstance(value, str) and any(
            marker in value.lower() for marker in forbidden
        ):
            raise RuntimeError("R20 path is forbidden: %s" % key)

    bundle = Path(str(config["pack_residual_bundle"]))
    manifest_path = bundle.with_suffix(".manifest.json")
    if not bundle.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "build pack-relative residual assets first: %s" % bundle
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != PACK_RELATIVE_SCHEMA:
        raise RuntimeError("pack-relative residual schema mismatch")
    if manifest.get("vpd_membership_hash") != bank.membership_hash:
        raise RuntimeError("pack-relative VPD membership mismatch")
    if manifest.get("bundle", {}).get("sha256") != sha256_file(bundle):
        raise RuntimeError("pack-relative bundle content hash mismatch")
    if int(manifest.get("num_stickers", -1)) != 174695:
        raise RuntimeError("unexpected StickerChat catalog size")
    if int(manifest.get("num_packs", -1)) != 3516:
        raise RuntimeError("unexpected original pack count")

    if require_init:
        init = Path(str(config["init_checkpoint_path"]))
        init_manifest_path = Path(str(init) + ".manifest.json")
        if not init.exists() or not init_manifest_path.exists():
            raise FileNotFoundError(
                "create pack-relative initialization first: %s" % init
            )
        with init_manifest_path.open("r", encoding="utf-8") as handle:
            init_manifest = json.load(handle)
        compatible = (
            init_manifest.get("schema_version")
            == "vigem.pack_relative_init.v1"
            and init_manifest.get("strict_reload") is True
            and init_manifest.get("vpd_membership_hash") == bank.membership_hash
            and init_manifest.get("pack_membership_hash")
            == manifest.get("pack_membership_hash")
            and init_manifest.get("snapshot", {}).get("sha256")
            == sha256_file(init)
            and init_manifest.get("input_hashes", {}).get(
                "pack_residual_bundle"
            )
            == sha256_file(bundle)
        )
        if not compatible:
            raise RuntimeError("pack-relative initialization is incompatible")
    return manifest


def load_vigem_parent_initialization(model, checkpoint_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    disallowed = [
        key for key in missing if not key.startswith(SETWISE_PREFIX)
    ]
    if unexpected or disallowed:
        raise RuntimeError(
            "parent VIGEM initialization mismatch: missing=%s unexpected=%s"
            % (disallowed[:20], unexpected[:20])
        )
    if not missing:
        raise RuntimeError("parent VIGEM init unexpectedly contains setwise scorer")
    return {
        "missing_setwise_keys": sorted(missing),
        "unexpected_keys": [],
    }


def build_pack_model_args(config: Mapping[str, Any]):
    return build_model_args(config)
