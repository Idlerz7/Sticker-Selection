#!/usr/bin/env python3
"""Run a real two-step multi-GPU Style Shapes DDP training preflight."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.plugins import DDPPlugin

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from structured_retrieval import DdpStaticGraphCallback, load_checkpoint_to_model
from structured_retrieval_factorized import parse_structured_factorized_args
from structured_retrieval_tokens import _config_mapping_to_argv
from style_shapes.contracts import (
    PILOT_SCHEMA_VERSION,
    SCHEDULER_CONTRACT,
    training_completion_errors,
)
from style_shapes.group_bank import GroupBank
from style_shapes.io import atomic_write_json, command_record, sha256_file
from style_shapes.runtime import resolve_permutation_world_size
from style_shapes.training import StyleShapesDataModule, StyleShapesPLModel


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("preflight configuration root must be a mapping")
    return value


def _build_args(config: dict, output: Path, world_size: int):
    overrides = dict(config.get("model_overrides", {}))
    overrides.update(
        {
            "factorized_bank_path": config["group_bank"],
            "gpus": int(world_size),
            "epochs": 1,
            "num_workers": 0,
            "pl_root_dir": str(output / "lightning"),
        }
    )
    return parse_structured_factorized_args(
        ["--config", config["base_config"]] + _config_mapping_to_argv(overrides)
    )


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    args = parser.parse_args()

    with command_record(args, args.artifact_root):
        if os.environ.get("CONDA_DEFAULT_ENV") != "stickr-select":
            raise RuntimeError("DDP preflight must use Conda environment stickr-select")
        world_size = int(torch.cuda.device_count())
        if world_size <= 0:
            raise RuntimeError("DDP preflight requires at least one visible GPU")
        if args.train_batches <= 0:
            raise ValueError("train-batches must be positive")

        config = _read_yaml(args.config)
        output = Path(args.output_dir)
        manifest_path = output / "preflight_manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if (
                existing.get("status") != "DDP_PREFLIGHT_COMPLETE"
                or int(existing.get("world_size", -1)) != world_size
                or int(existing.get("train_batches", -1)) != int(args.train_batches)
            ):
                raise RuntimeError("incompatible completed DDP preflight")
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return

        permutation_path, permutation = resolve_permutation_world_size(
            config["permutation_manifest"], world_size
        )
        model_args = _build_args(config, output, world_size)
        bank = GroupBank.load(config["group_bank"])
        init_path = config["init_checkpoint_path"]
        pl.seed_everything(int(model_args.seed))

        model = StyleShapesPLModel(
            model_args,
            membership_hash=bank.membership_hash,
            trace_dir=str(output / "negative_trace"),
        )
        load_checkpoint_to_model(model, init_path, strict=True)
        datamodule = StyleShapesDataModule(
            model_args, model.model.bert_tokenizer, permutation_path
        )

        callbacks = []
        if (
            world_size > 1
            and getattr(model_args, "ddp_static_graph_callback", False)
            and getattr(model_args, "bert_gradient_checkpointing", True)
        ):
            callbacks.append(DdpStaticGraphCallback())
        trainer_kwargs = {
            "gpus": world_size,
            "max_epochs": 1,
            "limit_train_batches": int(args.train_batches),
            "limit_val_batches": 0.0,
            "num_sanity_val_steps": 0,
            "accumulate_grad_batches": model_args.gradient_accumulation_steps,
            "default_root_dir": str(output / "lightning"),
            "precision": int(getattr(model_args, "trainer_precision", 32) or 32),
            "replace_sampler_ddp": False,
            "checkpoint_callback": False,
            "logger": False,
        }
        if callbacks:
            trainer_kwargs["callbacks"] = callbacks
        if world_size > 1:
            trainer_kwargs["accelerator"] = "ddp"
            trainer_kwargs["plugins"] = DDPPlugin(
                find_unused_parameters=getattr(
                    model_args, "ddp_find_unused_parameters", False
                )
            )
        trainer = pl.Trainer(**trainer_kwargs)

        started = time.perf_counter()
        trainer.fit(model, datamodule=datamodule)
        wall_seconds = time.perf_counter() - started
        if not trainer.is_global_zero:
            return

        expected_steps = int(model.num_training_steps)
        completed_steps = int(trainer.global_step)
        completion_errors = training_completion_errors(
            interrupted=bool(getattr(trainer, "interrupted", False)),
            completed_optimizer_steps=completed_steps,
            expected_optimizer_steps=expected_steps,
            current_epoch=int(trainer.current_epoch),
            expected_epochs=1,
            trace_dir=str(output / "negative_trace"),
            world_size=world_size,
        )
        if completion_errors:
            raise RuntimeError(
                "DDP preflight completion failure: %s"
                % "; ".join(completion_errors)
            )

        trace_rows = {}
        for rank in range(world_size):
            path = output / "negative_trace" / ("rank_%02d.jsonl" % rank)
            trace_rows[str(rank)] = _line_count(path)
        expected_rows_per_rank = int(args.train_batches) * int(
            model_args.train_batch_size
        )
        if any(value != expected_rows_per_rank for value in trace_rows.values()):
            raise RuntimeError(
                "DDP preflight trace coverage mismatch: %r != %d"
                % (trace_rows, expected_rows_per_rank)
            )

        result = {
            "status": "DDP_PREFLIGHT_COMPLETE",
            "formal_result": False,
            "schema_version": PILOT_SCHEMA_VERSION,
            "scheduler_contract": SCHEDULER_CONTRACT,
            "dataset": config["dataset"],
            "group_source": config["group_source"],
            "world_size": world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "train_batches": int(args.train_batches),
            "train_batch_size": int(model_args.train_batch_size),
            "expected_optimizer_steps": expected_steps,
            "completed_optimizer_steps": completed_steps,
            "trace_rows_by_rank": trace_rows,
            "membership_hash": bank.membership_hash,
            "permutation_manifest": {
                "path": permutation_path,
                "manifest_hash": permutation["manifest_hash"],
                "sha256": sha256_file(permutation_path),
            },
            "init_checkpoint": {
                "path": init_path,
                "sha256": sha256_file(init_path),
            },
            "wall_seconds": wall_seconds,
        }
        atomic_write_json(manifest_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
