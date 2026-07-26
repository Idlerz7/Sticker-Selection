#!/usr/bin/env python3
"""Run one real-asset GPU forward/backward/eval smoke for a VIGEM config."""

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

from main import PLDataLoader
from structured_retrieval import load_checkpoint_to_model, move_batch_to_device
from style_shapes.fixed_same_pack import load_fixed_same_pack_runtime
from style_shapes.group_bank import GroupBank
from style_shapes.io import atomic_write_json, command_record
from style_shapes.runtime import resolve_permutation_world_size
from style_shapes.training import StyleShapesDataModule
from vigem.config import (
    build_model_args,
    read_config,
    verify_config_contract,
)
from vigem.training import VigemInstanceResidualPLModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--artifact-root", default="artifacts/vigem")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("smoke requires exactly one visible CUDA GPU")
        config = read_config(args.config)
        if os.environ.get("CONDA_DEFAULT_ENV") != str(
            config.get("conda_env", "stickr-select")
        ):
            raise RuntimeError("activate the configured Conda environment")
        config["model_overrides"] = dict(config.get("model_overrides", {}))
        config["model_overrides"]["gpus"] = 1
        bank = GroupBank.load(config["group_bank"])
        model_args = build_model_args(config)
        formal_batch = model_args.train_batch_size
        verify_config_contract(config, model_args, bank, require_init=True)
        model_args.train_batch_size = 2
        fixed_runtime = None
        if config["dataset"] == "stickerchat":
            fixed_runtime = load_fixed_same_pack_runtime(
                config["fixed_candidates"],
                train_data_path=model_args.train_data_path,
            )
        permutation_path, _ = resolve_permutation_world_size(
            config["permutation_manifest"], 1
        )
        pl.seed_everything(int(model_args.seed))
        model = VigemInstanceResidualPLModel(
            model_args,
            residual_bundle_path=config["residual_bundle"],
            membership_hash=bank.membership_hash,
        )
        load_checkpoint_to_model(
            model, config["init_checkpoint_path"], strict=True
        )
        model = model.cuda()
        datamodule = StyleShapesDataModule(
            model_args,
            model.model.bert_tokenizer,
            permutation_path,
            fixed_candidate_runtime=fixed_runtime,
        )
        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()
        singleton_gold_rows = 0
        if config["dataset"] == "dstc":
            # Exercise the formal DSTC edge case deliberately: singleton VPD
            # groups have a zero residual and must skip Instance CE without
            # skipping the final/group objectives for that row.
            bundle = model.model._ensure_instance_bundle()
            batch = None
            for candidate_batch in train_loader:
                positive_ids = torch.tensor(
                    candidate_batch["img_ids"], dtype=torch.long
                )
                positive_groups = bundle.group_ids.index_select(
                    0, positive_ids
                )
                singleton = bundle.group_sizes.index_select(
                    0, positive_groups
                ).eq(1)
                if bool(singleton.any()) and bool((~singleton).any()):
                    batch = candidate_batch
                    singleton_gold_rows = int(singleton.sum().item())
                    break
            if batch is None:
                raise RuntimeError(
                    "DSTC smoke could not find a mixed singleton/non-singleton batch"
                )
        else:
            batch = next(iter(train_loader))
        batch = move_batch_to_device(batch, torch.device("cuda"))
        model.train()
        if "train_candidate_ids" in batch:
            output = model.model.forward_train_listwise_batch(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                img_ids=batch["img_ids"],
                candidate_ids=batch["train_candidate_ids"],
                gray_mask=batch.get("train_candidate_gray_mask"),
                total_steps=1,
            )
        else:
            output = model.model.forward_train_batch(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                img_ids=batch["img_ids"],
                neg_img_ids=batch["neg_img_ids"],
                total_steps=1,
            )
        output.loss.backward()
        instance_gradients = [
            float(parameter.grad.detach().norm().item())
            for name, parameter in model.model.named_parameters()
            if name.startswith("instance_")
            and parameter.requires_grad
            and parameter.grad is not None
        ]
        if not instance_gradients or max(instance_gradients) <= 0.0:
            raise RuntimeError("VIGEM instance branch received no gradient")
        frozen_legacy_with_grad = [
            name
            for name, parameter in model.model.named_parameters()
            if (
                name.startswith("dialogue_factorizer.expr_query_head")
                or name.startswith("sticker_factorizer.expr_head")
                or name.startswith("expr_match_head")
            )
            and parameter.grad is not None
        ]
        if frozen_legacy_with_grad:
            raise RuntimeError("legacy expression branch unexpectedly got gradients")

        eval_loader = PLDataLoader(
            model_args, model.model.bert_tokenizer
        )
        eval_loader.setup("test")
        eval_batch = move_batch_to_device(
            next(iter(eval_loader.test_dataloader())),
            torch.device("cuda"),
        )
        model.eval()
        model.model.prepare_for_test()
        model.model.prepare_eval_factorization_cache()
        with torch.no_grad():
            _, labels, candidates, debug = model.model.forward_eval_batch(
                input_ids=eval_batch["input_ids"],
                attention_mask=eval_batch["attention_mask"],
                img_ids=eval_batch["img_ids"],
                cands=eval_batch.get("cands"),
                return_debug=True,
                score_breakdown=True,
            )
        if not all(
            torch.isfinite(torch.tensor(debug[key])).all().item()
            for key in (
                "mmbert_score_per_cand",
                "instance_score_per_cand",
                "graph_score_per_cand",
                "final_score_per_cand",
                "final_without_instance_per_cand",
            )
        ):
            raise RuntimeError("non-finite score in VIGEM evaluation smoke")
        result = {
            "status": "SMOKE_COMPLETE",
            "dataset": config["dataset"],
            "group_source": config["group_source"],
            "train_batch_size": int(batch["input_ids"].size(0)),
            "train_loss": float(output.loss.detach().item()),
            "match_loss": float(output.match_loss.detach().item()),
            "group_loss": float(output.group_loss.detach().item()),
            "instance_loss": float(output.instance_loss.detach().item()),
            "instance_gradient_norm_max": max(instance_gradients),
            "legacy_expression_gradients": frozen_legacy_with_grad,
            "temperature": float(
                model.model.instance_temperature.detach().item()
            ),
            "eval_gold": int(labels.item()),
            "eval_candidate_count": len(candidates),
            "instance_score_std": debug["instance_score_stats"]["std"],
            "membership_hash": bank.membership_hash,
            "formal_batch_size_unchanged": int(formal_batch),
            "singleton_gold_rows": int(singleton_gold_rows),
            "smoke_only": True,
        }
        output_path = Path(
            args.output
            or (
                "artifacts/vigem/smoke/%s.json" % config["dataset"]
            )
        )
        atomic_write_json(output_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
