#!/usr/bin/env python3
"""One-batch GPU smoke for pack-relative forward/backward/reload/evaluation."""

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
from transformers import AdamW

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lvpcm.io import atomic_torch_save
from main import PLDataLoader
from structured_retrieval import load_checkpoint_to_model, move_batch_to_device
from style_shapes.fixed_same_pack import load_fixed_same_pack_runtime
from style_shapes.group_bank import GroupBank
from style_shapes.io import atomic_write_json, command_record, sha256_file
from style_shapes.training import StyleShapesDataModule
from vigem.pack_config import (
    build_pack_model_args,
    read_pack_config,
    verify_pack_config,
)
from vigem.pack_training import PackRelativeSetwisePLModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-root", default="artifacts/vigem")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        if os.environ.get("CONDA_DEFAULT_ENV") != "stickr-select":
            raise RuntimeError("activate stickr-select before GPU smoke")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("GPU smoke requires exactly one visible GPU")
        config = read_pack_config(args.config)
        output = Path(args.output_dir)
        manifest_path = output / "smoke_manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("status") != "SMOKE_COMPLETE":
                raise RuntimeError("incompatible existing smoke manifest")
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("refusing non-empty incomplete smoke directory")
        output.mkdir(parents=True, exist_ok=True)

        bank = GroupBank.load(config["group_bank"])
        config["model_overrides"] = dict(config["model_overrides"])
        config["model_overrides"].update(
            {
                "gpus": 1,
                "num_workers": 0,
                "pl_root_dir": str(output / "lightning"),
            }
        )
        config["permutation_manifest"] = (
            "artifacts/style_shapes/permutations/"
            "stickerchat_seed2021_ws1.json"
        )
        model_args = build_pack_model_args(config)
        verify_pack_config(config, model_args, bank, require_init=True)
        fixed_runtime = load_fixed_same_pack_runtime(
            config["fixed_candidates"],
            train_data_path=model_args.train_data_path,
        )
        pl.seed_everything(2021)
        device = torch.device("cuda:0")
        model = PackRelativeSetwisePLModel(
            model_args,
            residual_bundle_path=config["pack_residual_bundle"],
            membership_hash=bank.membership_hash,
        )
        load_checkpoint_to_model(
            model, config["init_checkpoint_path"], strict=True
        )
        model.to(device)
        model.train()
        model.model.prepare_for_test()
        model.model.clear_train_factorization_cache()

        data = StyleShapesDataModule(
            model_args,
            model.model.bert_tokenizer,
            config["permutation_manifest"],
            fixed_candidate_runtime=fixed_runtime,
        )
        data.setup("fit")
        raw_batch = next(iter(data.train_dataloader()))
        if int(raw_batch["input_ids"].size(0)) != 16:
            raise RuntimeError("formal batch-16 smoke was not constructed")
        batch = move_batch_to_device(raw_batch, device)

        # Real-model gradient isolation: Group loss may update its own heads and
        # prototypes, but must not reach the shared dialogue BERT.
        model.zero_grad(set_to_none=True)
        setup = model.model._group_and_instance_setup(
            batch["input_ids"],
            batch["attention_mask"],
            batch["img_ids"],
            0,
        )
        group_loss = setup[5]
        group_loss.backward()
        bert_grad = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in model.model.bert.parameters()
            if parameter.grad is not None
        )
        group_head_grad = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in model.model.style_query_head.parameters()
            if parameter.grad is not None
        )
        if bert_grad != 0.0 or group_head_grad <= 0.0:
            raise RuntimeError(
                "real Group gradient isolation failed: bert=%f head=%f"
                % (bert_grad, group_head_grad)
            )
        model.zero_grad(set_to_none=True)
        model.model.clear_train_factorization_cache()

        parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        optimizer = AdamW(
            parameters,
            lr=model_args.other_lr,
            betas=(0.9, 0.98),
            weight_decay=0.2,
        )
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = model.model.forward_train_listwise_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            img_ids=batch["img_ids"],
            candidate_ids=batch["train_candidate_ids"],
            gray_mask=batch["train_candidate_gray_mask"],
            global_step=0,
            total_steps=1,
        )
        if not bool(torch.isfinite(result.loss).item()):
            raise RuntimeError("non-finite pack-relative smoke loss")
        result.loss.backward()
        scorer_grad = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in model.model.instance_set_scorer.parameters()
            if parameter.grad is not None
        )
        if scorer_grad <= 0.0:
            raise RuntimeError("setwise scorer received no gradient")
        optimizer.step()
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        checkpoint = output / "one_step.ckpt"
        atomic_torch_save(
            checkpoint,
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "vigem_pack_relative": {
                    "smoke_only": True,
                    "formal_result": False,
                },
            },
        )
        trace = {
            "source_rows": [int(value) for value in batch["source_rows"]],
            "candidate_ids": result.debug_info["candidate_ids"]
            .detach()
            .cpu()
            .tolist(),
            "gray_mask": result.debug_info["gray_mask"]
            .detach()
            .cpu()
            .tolist(),
            "pack_ids": result.debug_info["pack_ids"]
            .detach()
            .cpu()
            .tolist(),
            "group_shared_bert_detached": result.debug_info[
                "group_shared_bert_detached"
            ],
        }
        atomic_write_json(output / "training_trace.json", trace)
        del optimizer, result, model, setup, group_loss
        torch.cuda.empty_cache()

        model_args.mode = "test"
        reloaded = PackRelativeSetwisePLModel(
            model_args,
            residual_bundle_path=config["pack_residual_bundle"],
            membership_hash=bank.membership_hash,
        )
        load_checkpoint_to_model(reloaded, str(checkpoint), strict=True)
        reloaded.to(device)
        reloaded.eval()
        reloaded.model.prepare_for_test()
        reloaded.model.prepare_eval_factorization_cache()
        eval_data = PLDataLoader(
            model_args, reloaded.model.bert_tokenizer
        )
        eval_data.setup("test")
        eval_batch = move_batch_to_device(
            next(iter(eval_data.test_dataloader())), device
        )
        with torch.no_grad():
            _, labels, candidates, debug = reloaded.run_eval_batch(
                eval_batch, return_debug=True, score_breakdown=True
            )
        final = [float(value) for value in debug["final_score_per_cand"]]
        if len(candidates) != 10 or not all(math.isfinite(v) for v in final):
            raise RuntimeError("invalid smoke R10 evaluation")
        manifest = {
            "status": "SMOKE_COMPLETE",
            "smoke_only": True,
            "formal_result": False,
            "batch_size": 16,
            "candidate_pairs": 160,
            "loss_finite": True,
            "setwise_scorer_gradient_nonzero": True,
            "group_gradient_isolation": {
                "bert_gradient_sum": bert_grad,
                "group_head_gradient_sum": group_head_grad,
            },
            "strict_checkpoint_reload": True,
            "eval_gold": int(labels.item()),
            "eval_candidates": [int(value) for value in candidates],
            "train_seconds": train_seconds,
            "peak_cuda_memory_bytes": peak,
            "config_sha256": sha256_file(args.config),
            "init_sha256": sha256_file(config["init_checkpoint_path"]),
            "pack_residual_sha256": sha256_file(
                config["pack_residual_bundle"]
            ),
            "r20_accessed": False,
        }
        atomic_write_json(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
