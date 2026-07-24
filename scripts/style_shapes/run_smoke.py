#!/usr/bin/env python3
"""Run one non-formal Style Shapes train step, eval query, and checkpoint reload."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pytorch_lightning as pl
import torch
import yaml
from transformers import AdamW, get_cosine_schedule_with_warmup

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lvpcm.io import atomic_torch_save
from main import PLDataLoader
from structured_retrieval import load_checkpoint_to_model, move_batch_to_device
from structured_retrieval_factorized import parse_structured_factorized_args
from structured_retrieval_tokens import _config_mapping_to_argv
from style_shapes.group_bank import GroupBank
from style_shapes.io import atomic_write_json, command_record, sha256_file
from style_shapes.negative_sampling import (
    load_dual_local_runtime,
    validate_dual_local_trace_record,
)
from style_shapes.training import StyleShapesDataModule, StyleShapesPLModel


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("smoke configuration root must be a mapping")
    return value


def _model_args(config: dict, output: Path, batch_size: int):
    overrides = dict(config.get("model_overrides", {}))
    overrides.update(
        {
            "factorized_bank_path": config["group_bank"],
            "gpus": 1,
            "epochs": 1,
            "train_batch_size": int(batch_size),
            "valtest_batch_size": 1,
            "num_workers": 0,
            "pl_root_dir": str(output / "lightning"),
        }
    )
    return parse_structured_factorized_args(
        ["--config", config["base_config"]] + _config_mapping_to_argv(overrides)
    )


def _optimizer(model, args):
    img_prefixes = ("img_clip", "clip_model", "img_embedding_layer")
    img_params = [
        value
        for name, value in model.model.named_parameters()
        if value.requires_grad and name.startswith(img_prefixes)
    ]
    other_params = [
        value
        for name, value in model.model.named_parameters()
        if value.requires_grad and not name.startswith(img_prefixes)
    ]
    groups = []
    if img_params:
        groups.append({"params": img_params, "lr": args.img_lr})
    if other_params:
        groups.append({"params": other_params, "lr": args.other_lr})
    optimizer = AdamW(
        groups,
        lr=args.other_lr,
        betas=(0.9, 0.98),
        weight_decay=0.2,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=1
    )
    return optimizer, scheduler


def _score_record(labels, debug: dict, membership_hash: str) -> dict:
    candidate_ids = [int(value) for value in debug["candidate_ids_ordered"]]
    final_scores = [float(value) for value in debug["final_score_per_cand"]]
    gold = int(labels.item())
    positive_index = candidate_ids.index(gold)
    order = sorted(
        range(len(final_scores)),
        key=lambda index: (-final_scores[index], candidate_ids[index]),
    )
    return {
        "query_index": 0,
        "candidate_ids": candidate_ids,
        "gold": gold,
        "positive_index": positive_index,
        "base_scores": [float(value) for value in debug["mmbert_score_per_cand"]],
        "instance_scores": [float(value) for value in debug["expr_score_per_cand"]],
        "group_scores": [float(value) for value in debug["graph_score_per_cand"]],
        "final_scores": final_scores,
        "rank": order.index(positive_index) + 1,
        "membership_hash": membership_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    args = parser.parse_args()

    with command_record(args, args.artifact_root):
        if os.environ.get("CONDA_DEFAULT_ENV") != "stickr-select":
            raise RuntimeError("Style Shapes smoke must use Conda environment stickr-select")
        if not torch.cuda.is_available():
            raise RuntimeError("Style Shapes smoke requires one CUDA GPU")
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive")

        config = _read_yaml(args.config)
        output = Path(args.output_dir)
        manifest_path = output / "smoke_manifest.json"
        init_path = Path(config["init_checkpoint_path"])
        bank = GroupBank.load(config["group_bank"])
        expected = {
            "config_sha256": sha256_file(args.config),
            "init_sha256": sha256_file(init_path),
            "membership_hash": bank.membership_hash,
            "batch_size": int(args.batch_size),
        }
        model_args = _model_args(config, output, args.batch_size)
        if str(model_args.factorized_variant) != "minimal":
            raise RuntimeError("smoke requires the v6 minimal core")
        negative_sampler = None
        eligibility_manifest = None
        eligible_source_rows = None
        if config.get("negative_sampling") is not None:
            negative_sampler, eligibility_manifest = load_dual_local_runtime(
                config["negative_sampling"],
                train_data_path=model_args.train_data_path,
                group_bank_path=config["group_bank"],
                bank=bank,
            )
            eligible_source_rows = [
                int(value) for value in eligibility_manifest["eligible_rows"]
            ]
            expected.update(
                {
                    "negative_policy": negative_sampler.policy,
                    "eligibility_manifest_hash": eligibility_manifest["manifest_hash"],
                }
            )
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("status") != "SMOKE_COMPLETE" or any(
                existing.get(key) != value for key, value in expected.items()
            ):
                raise RuntimeError("refusing to overwrite incompatible smoke output")
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("refusing non-empty incomplete smoke output: %s" % output)
        output.mkdir(parents=True, exist_ok=True)
        pl.seed_everything(2021)
        device = torch.device("cuda:0")

        model = StyleShapesPLModel(
            model_args,
            membership_hash=bank.membership_hash,
            negative_sampler=negative_sampler,
        )
        load_checkpoint_to_model(model, str(init_path), strict=True)
        model.to(device)
        model.train()
        if getattr(model.args, "fix_img", False):
            model.model.prepare_for_test()
        model.model.clear_train_factorization_cache()

        data = StyleShapesDataModule(
            model_args,
            model.model.bert_tokenizer,
            config["permutation_manifest"],
            eligible_source_rows=eligible_source_rows,
        )
        data.setup("fit")
        batch = move_batch_to_device(next(iter(data.train_dataloader())), device)

        optimizer, scheduler = _optimizer(model, model_args)
        optimizer.zero_grad()
        started = time.perf_counter()
        result = model.model.forward_train_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            img_ids=batch["img_ids"],
            neg_img_ids=batch["neg_img_ids"],
            global_step=0,
            total_steps=1,
        )
        loss = result.loss
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite smoke loss")
        loss_value = float(loss.detach().cpu().item())
        loss.backward()
        grad_tensors = [
            value.grad.detach()
            for value in model.parameters()
            if value.requires_grad and value.grad is not None
        ]
        grad_nonzero = sum(int(bool(torch.count_nonzero(value).item())) for value in grad_tensors)
        if grad_nonzero <= 0:
            raise RuntimeError("smoke backward produced no non-zero gradients")
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started

        cross, same = model._style_shapes_last_negatives or (None, None)
        if cross is None or same is None:
            raise RuntimeError("smoke did not capture actual group-aware negatives")
        trace = []
        for source_row, positive, fallback, cross_id, same_id in zip(
            batch["source_rows"],
            batch["img_ids"],
            batch["neg_img_ids"],
            cross,
            same,
        ):
            record = {
                "source_row": int(source_row),
                "positive": int(positive),
                "fallback": int(fallback),
                "cross": int(cross_id),
                "same": int(same_id),
                "membership_hash": bank.membership_hash,
            }
            if negative_sampler is not None:
                record.update(
                    {
                        "negative_policy": negative_sampler.policy,
                        "group_top32": int(cross_id),
                        "same_pack": int(same_id),
                        "fallback_used": False,
                    }
                )
                record[negative_sampler.neighbor_trace_field] = int(cross_id)
                validate_dual_local_trace_record(record, negative_sampler)
            trace.append(record)
        atomic_write_json(output / "negative_trace.json", trace)

        checkpoint_path = output / "one_step.ckpt"
        atomic_torch_save(
            checkpoint_path,
            {
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "style_shapes": {
                    "smoke_only": True,
                    "formal_result": False,
                    "config": args.config,
                    "membership_hash": bank.membership_hash,
                },
            },
        )
        del optimizer, scheduler, result, loss, model
        torch.cuda.empty_cache()

        # Match the formal final-evaluation entrypoint rather than leaving the
        # data/model arguments in training mode.
        model_args.mode = "test"
        reloaded = StyleShapesPLModel(
            model_args,
            membership_hash=bank.membership_hash,
        )
        load_checkpoint_to_model(reloaded, str(checkpoint_path), strict=True)
        reloaded.to(device)
        reloaded.eval()
        reloaded.model.prepare_for_test()
        reloaded.model.prepare_eval_factorization_cache()

        eval_data = PLDataLoader(model_args, reloaded.model.bert_tokenizer)
        eval_data.setup("test")
        eval_batch = move_batch_to_device(next(iter(eval_data.test_dataloader())), device)
        eval_started = time.perf_counter()
        with torch.no_grad():
            _, labels, _, debug = reloaded.run_eval_batch(
                eval_batch, return_debug=True, score_breakdown=True
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        eval_seconds = time.perf_counter() - eval_started
        score = _score_record(labels, debug, bank.membership_hash)
        if not all(math.isfinite(value) for value in score["final_scores"]):
            raise RuntimeError("non-finite smoke evaluation scores")
        atomic_write_json(output / "scores" / "query_000000.json", score)

        manifest = {
            "status": "SMOKE_COMPLETE",
            "smoke_only": True,
            "formal_result": False,
            "dataset": config["dataset"],
            "group_source": config["group_source"],
            "negative_policy": (
                negative_sampler.policy
                if negative_sampler is not None
                else "prototype_cross_plus_same"
            ),
            **expected,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "strict_reload": True,
            },
            "train": {
                "batches": 1,
                "rows": len(batch["img_ids"]),
                "loss": loss_value,
                "gradient_tensors_nonzero": grad_nonzero,
                "wall_seconds": train_seconds,
            },
            "eval": {
                "queries": 1,
                "candidate_count": len(score["candidate_ids"]),
                "rank": score["rank"],
                "wall_seconds": eval_seconds,
                "score_path": str(output / "scores" / "query_000000.json"),
            },
        }
        atomic_write_json(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
