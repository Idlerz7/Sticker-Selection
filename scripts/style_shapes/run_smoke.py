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
from style_shapes.fixed_same_pack import (
    flatten_query_major,
    load_fixed_same_pack_runtime,
)
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
        "gray_mask": [
            bool(value)
            for value in debug.get(
                "gray_mask", [value == -1 for value in candidate_ids]
            )
        ],
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
    parser.add_argument(
        "--force-sequence-length",
        type=int,
        default=0,
        help="Smoke-only right padding used to measure a worst-case dialogue length.",
    )
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
        model_args = _model_args(config, output, args.batch_size)
        expected = {
            "config_sha256": sha256_file(args.config),
            "init_sha256": sha256_file(init_path),
            "membership_hash": bank.membership_hash,
            "batch_size": int(args.batch_size),
            "trainer_precision": int(model_args.trainer_precision),
            "forced_sequence_length": int(args.force_sequence_length),
        }
        if str(model_args.factorized_variant) != "minimal":
            raise RuntimeError("smoke requires the v6 minimal core")
        negative_sampler = None
        fixed_candidate_runtime = None
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
        if config.get("fixed_candidates") is not None:
            fixed_candidate_runtime = load_fixed_same_pack_runtime(
                config["fixed_candidates"],
                train_data_path=model_args.train_data_path,
            )
            expected.update(
                {
                    "negative_policy": fixed_candidate_runtime.policy,
                    "fixed_candidate_manifest_hash": (
                        fixed_candidate_runtime.manifest["manifest_hash"]
                    ),
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
            fixed_candidate_runtime=fixed_candidate_runtime,
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
            fixed_candidate_runtime=fixed_candidate_runtime,
        )
        data.setup("fit")
        raw_batch = next(iter(data.train_dataloader()))
        smoke_forced_gray_source = None
        if (
            fixed_candidate_runtime is not None
            and not any(
                any(bool(value) for value in row)
                for row in raw_batch["train_candidate_gray_mask"]
            )
        ):
            gray_rows = torch.nonzero(
                fixed_candidate_runtime.gray_mask.any(dim=1), as_tuple=False
            ).reshape(-1)
            if gray_rows.numel() == 0:
                raise RuntimeError("fixed candidate manifest has no gray smoke row")
            smoke_forced_gray_source = int(gray_rows[0].item())
            source_rows = list(raw_batch["source_rows"])
            source_rows[0] = smoke_forced_gray_source
            raw_batch = data.collate_fn(
                [data.train_dataset[index] for index in source_rows]
            )
        if args.force_sequence_length:
            target_length = int(args.force_sequence_length)
            current_length = int(raw_batch["input_ids"].size(1))
            maximum_length = int(model_args.max_dialogue_length)
            if target_length < current_length or target_length > maximum_length:
                raise ValueError(
                    "forced sequence length must be within [%d,%d], got %d"
                    % (current_length, maximum_length, target_length)
                )
            pad_width = target_length - current_length
            if pad_width:
                batch_rows = int(raw_batch["input_ids"].size(0))
                pad_ids = torch.full(
                    (batch_rows, pad_width),
                    int(model.model.bert_tokenizer.pad_token_id),
                    dtype=raw_batch["input_ids"].dtype,
                )
                pad_mask = torch.zeros(
                    (batch_rows, pad_width),
                    dtype=raw_batch["attention_mask"].dtype,
                )
                raw_batch["input_ids"] = torch.cat(
                    [raw_batch["input_ids"], pad_ids], dim=1
                )
                raw_batch["attention_mask"] = torch.cat(
                    [raw_batch["attention_mask"], pad_mask], dim=1
                )
        batch = move_batch_to_device(raw_batch, device)

        vectorized_serial_max_abs_diff = None
        if fixed_candidate_runtime is not None:
            model.eval()
            with torch.no_grad():
                probe_batch = min(2, len(batch["img_ids"]))
                probe_ids = torch.tensor(
                    batch["train_candidate_ids"][:probe_batch],
                    dtype=torch.long,
                    device=device,
                )
                probe_input = batch["input_ids"][:probe_batch]
                probe_mask = batch["attention_mask"][:probe_batch]
                flat_input, flat_mask, flat_ids = flatten_query_major(
                    probe_input, probe_mask, probe_ids
                )
                bank_h = model.model.get_factorized_bank_img_embs(device)
                flat_id_list = [int(value) for value in flat_ids.cpu().tolist()]
                flat_h = model.model._candidate_embeddings_from_bank(
                    bank_h, flat_id_list
                )
                vectorized = model.model.compute_base_score(
                    model.model._compute_pair_logits(
                        flat_input, flat_mask, flat_id_list, flat_h
                    )
                ).reshape(probe_batch, -1)
                serial_columns = []
                for column in range(probe_ids.size(1)):
                    column_ids = [
                        int(value)
                        for value in probe_ids[:, column].cpu().tolist()
                    ]
                    column_h = model.model._candidate_embeddings_from_bank(
                        bank_h, column_ids
                    )
                    serial_columns.append(
                        model.model.compute_base_score(
                            model.model._compute_pair_logits(
                                probe_input,
                                probe_mask,
                                column_ids,
                                column_h,
                            )
                        )
                    )
                serial = torch.stack(serial_columns, dim=1)
                vectorized_serial_max_abs_diff = float(
                    (vectorized - serial).abs().max().cpu().item()
                )
                # CUDA GEMM kernels may choose a different accumulation order
                # when the effective batch changes from B to B*N. The functions
                # must be equivalent within normal float32 inference tolerance.
                if not torch.allclose(
                    vectorized, serial, rtol=1e-3, atol=1e-3
                ):
                    raise RuntimeError(
                        "vectorized and serial candidate scores differ by %.8f"
                        % vectorized_serial_max_abs_diff
                    )
            model.train()

        optimizer, scheduler = _optimizer(model, model_args)
        optimizer.zero_grad()
        torch.cuda.reset_peak_memory_stats(device)
        use_amp = int(model_args.trainer_precision) == 16
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_amp,
            init_scale=1024.0,
        )
        amp_scale_before = float(scaler.get_scale())
        started = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=use_amp):
            if fixed_candidate_runtime is not None:
                result = model.model.forward_train_listwise_batch(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    img_ids=batch["img_ids"],
                    candidate_ids=batch["train_candidate_ids"],
                    gray_mask=batch["train_candidate_gray_mask"],
                    global_step=0,
                    total_steps=1,
                )
            else:
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
        scaler.scale(loss).backward()
        grad_tensors = [
            value.grad.detach()
            for value in model.parameters()
            if value.requires_grad and value.grad is not None
        ]
        grad_nonzero = sum(int(bool(torch.count_nonzero(value).item())) for value in grad_tensors)
        if grad_nonzero <= 0:
            raise RuntimeError("smoke backward produced no non-zero gradients")
        scaler.step(optimizer)
        scaler.update()
        amp_scale_after = float(scaler.get_scale())
        if use_amp and amp_scale_after < amp_scale_before:
            raise RuntimeError(
                "FP16 smoke overflowed and skipped its optimizer step: "
                "scale %.1f -> %.1f" % (amp_scale_before, amp_scale_after)
            )
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started
        train_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        train_peak_reserved = int(torch.cuda.max_memory_reserved(device))

        trace = []
        if fixed_candidate_runtime is not None:
            debug = result.debug_info
            candidate_rows = debug["candidate_ids"].detach().cpu().tolist()
            gray_rows = debug["gray_mask"].detach().cpu().tolist()
            group_rows = debug["group_scores"].detach().float().cpu().tolist()
            hardest_ids = debug["hardest_expression_ids"].detach().cpu().tolist()
            for source_row, positive, candidates, gray, group, hardest in zip(
                batch["source_rows"],
                batch["img_ids"],
                candidate_rows,
                gray_rows,
                group_rows,
                hardest_ids,
            ):
                if int(candidates[0]) != int(positive):
                    raise RuntimeError("fixed smoke trace gold is misaligned")
                if any(
                    float(group[index]) != 0.0
                    for index, is_gray in enumerate(gray)
                    if bool(is_gray)
                ):
                    raise RuntimeError("gray smoke candidate received nonzero group score")
                trace.append(
                    {
                        "source_row": int(source_row),
                        "positive": int(positive),
                        "negative_policy": fixed_candidate_runtime.policy,
                        "candidate_ids": [int(value) for value in candidates],
                        "gray_mask": [bool(value) for value in gray],
                        "group_scores": [float(value) for value in group],
                        "hardest_expression_id": int(hardest),
                        "membership_hash": bank.membership_hash,
                    }
                )
        else:
            cross, same = model._style_shapes_last_negatives or (None, None)
            if cross is None or same is None:
                raise RuntimeError(
                    "smoke did not capture actual group-aware negatives"
                )
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
            fixed_candidate_runtime=fixed_candidate_runtime,
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
                fixed_candidate_runtime.policy
                if fixed_candidate_runtime is not None
                else (
                    negative_sampler.policy
                    if negative_sampler is not None
                    else "prototype_cross_plus_same"
                )
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
                "amp_enabled": use_amp,
                "amp_scale_before": amp_scale_before,
                "amp_scale_after": amp_scale_after,
                "input_sequence_length": int(batch["input_ids"].size(1)),
                "peak_cuda_memory_allocated_bytes": train_peak_allocated,
                "peak_cuda_memory_reserved_bytes": train_peak_reserved,
                "vectorized_serial_max_abs_diff_eval_mode": (
                    vectorized_serial_max_abs_diff
                ),
                "gray_slots_exercised": sum(
                    sum(bool(value) for value in row)
                    for row in batch.get("train_candidate_gray_mask", [])
                ),
                "forced_gray_source_row": smoke_forced_gray_source,
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
