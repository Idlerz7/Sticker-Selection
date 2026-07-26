#!/usr/bin/env python3
"""Train or evaluate the independent VIGEM instance-residual model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pytorch_lightning as pl
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from main import (
    PLDataLoader,
    attach_test_log_to_ckpt_version,
    attach_version_log_from_trainer,
)
from structured_retrieval import load_checkpoint_to_model
from style_shapes.contracts import (
    PILOT_SCHEMA_VERSION,
    SCHEDULER_CONTRACT,
    training_completion_errors,
)
from style_shapes.fixed_same_pack import load_fixed_same_pack_runtime
from style_shapes.group_bank import GroupBank
from style_shapes.io import atomic_write_json, command_record, sha256_file
from style_shapes.permutations import load_permutation_manifest
from style_shapes.runtime import resolve_permutation_world_size
from style_shapes.training import (
    StyleShapesDataModule,
    build_final_only_trainer,
)
from vigem.config import (
    build_model_args,
    read_config,
    verify_config_contract,
)
from vigem.training import VigemInstanceResidualPLModel


def _save_checkpoint_atomic(trainer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, dir=str(path.parent)
    )
    os.close(fd)
    try:
        trainer.save_checkpoint(temporary, weights_only=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_strict(model, path: str) -> None:
    load_checkpoint_to_model(model, path, strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-mode", choices=("train", "test"))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--test-data-path")
    parser.add_argument("--run-output-dir")
    parser.add_argument("--artifact-root", default="artifacts/vigem")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        config = read_config(args.config)
        mode = str(args.run_mode or config.get("mode", "train"))
        config["model_overrides"] = dict(config.get("model_overrides", {}))
        if args.test_data_path:
            config["model_overrides"]["test_data_path"] = args.test_data_path
        if args.run_output_dir:
            config["output_dir"] = args.run_output_dir
        checkpoint_path = str(
            args.checkpoint_path or config.get("checkpoint_path", "")
        )
        fixed_runtime = None
        runtime_permutation = None
        if mode == "test":
            config["model_overrides"]["mode"] = "test"
            config["model_overrides"]["gpus"] = 1
            config["model_overrides"]["per_epoch_eval_test_r10_path"] = ""
            config["model_overrides"]["per_epoch_eval_test_r20_path"] = ""
            if not args.run_output_dir:
                stem = Path(
                    config["model_overrides"]["test_data_path"]
                ).stem
                config["output_dir"] = str(
                    Path(config["output_dir"]) / "final_eval" / stem
                )
        else:
            visible_gpus = int(torch.cuda.device_count())
            if visible_gpus <= 0:
                raise RuntimeError("VIGEM formal training requires a visible GPU")
            config["model_overrides"]["gpus"] = visible_gpus
            permutation_path, runtime_permutation = (
                resolve_permutation_world_size(
                    config["permutation_manifest"], visible_gpus
                )
            )
            config["permutation_manifest"] = permutation_path

        expected_env = str(config.get("conda_env", "stickr-select"))
        if os.environ.get("CONDA_DEFAULT_ENV") != expected_env:
            raise RuntimeError(
                "activate Conda environment %s before running VIGEM"
                % expected_env
            )
        bank = GroupBank.load(config["group_bank"])
        model_args = build_model_args(config)
        verify_config_contract(config, model_args, bank, require_init=True)
        if config["dataset"] == "stickerchat" and mode == "train":
            fixed_runtime = load_fixed_same_pack_runtime(
                config["fixed_candidates"],
                train_data_path=model_args.train_data_path,
            )
            if int(runtime_permutation["num_rows"]) != int(
                fixed_runtime.candidate_ids.size(0)
            ):
                raise RuntimeError(
                    "fixed candidates and permutation row counts differ"
                )

        pl.seed_everything(int(model_args.seed))
        output = Path(config["output_dir"])
        per_query_dir = str(output / "scores") if mode == "test" else ""
        completion_path = output / ("%s_manifest.json" % mode)
        if completion_path.exists():
            with completion_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            compatible = (
                existing.get("membership_hash") == bank.membership_hash
                and existing.get("residual_bundle", {}).get("sha256")
                == sha256_file(config["residual_bundle"])
            )
            if mode == "train":
                compatible = compatible and (
                    existing.get("init_checkpoint", {}).get("sha256")
                    == sha256_file(config["init_checkpoint_path"])
                    and existing.get("permutation_manifest", {}).get("sha256")
                    == sha256_file(config["permutation_manifest"])
                )
            else:
                compatible = compatible and (
                    checkpoint_path
                    and existing.get("checkpoint", {}).get("sha256")
                    == sha256_file(checkpoint_path)
                    and existing.get("test_data_path")
                    == model_args.test_data_path
                )
            if not compatible:
                raise RuntimeError(
                    "refusing incompatible completed VIGEM output; "
                    "choose a new --run-output-dir"
                )
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return
        model = VigemInstanceResidualPLModel(
            model_args,
            residual_bundle_path=config["residual_bundle"],
            membership_hash=bank.membership_hash,
            trace_dir=str(output / "training_trace") if mode == "train" else "",
            per_query_dir=per_query_dir,
        )

        if mode == "train":
            init_path = str(config["init_checkpoint_path"])
            _load_strict(model, init_path)
            permutation = load_permutation_manifest(
                config["permutation_manifest"]
            )
            world_size = int(torch.cuda.device_count())
            if int(model_args.gpus) != world_size:
                raise RuntimeError("trainer GPU count differs from visible GPUs")
            if int(permutation["world_size"]) != world_size:
                raise RuntimeError("permutation world size differs from visible GPUs")
            datamodule = StyleShapesDataModule(
                model_args,
                model.model.bert_tokenizer,
                config["permutation_manifest"],
                fixed_candidate_runtime=fixed_runtime,
            )
            trainer = build_final_only_trainer(model_args, for_train=True)
            attach_version_log_from_trainer(model_args, trainer)
            started = time.perf_counter()
            trainer.fit(model, datamodule=datamodule)
            wall_seconds = time.perf_counter() - started
            expected_steps = int(model.num_training_steps)
            completed_steps = int(trainer.global_step)
            completion_errors = training_completion_errors(
                interrupted=bool(getattr(trainer, "interrupted", False)),
                completed_optimizer_steps=completed_steps,
                expected_optimizer_steps=expected_steps,
                current_epoch=int(trainer.current_epoch),
                expected_epochs=int(model_args.epochs),
                trace_dir=str(output / "training_trace"),
                world_size=world_size,
            )
            if completion_errors:
                atomic_write_json(
                    output / "incomplete_run.json",
                    {
                        "status": "TRAIN_INCOMPLETE",
                        "errors": completion_errors,
                        "expected_optimizer_steps": expected_steps,
                        "completed_optimizer_steps": completed_steps,
                    },
                )
                raise RuntimeError(
                    "VIGEM training failed completion contract: %s"
                    % "; ".join(completion_errors)
                )
            final_path = output / "final.ckpt"
            _save_checkpoint_atomic(trainer, final_path)
            result = {
                "status": "TRAIN_COMPLETE",
                "schema_version": "vigem.pilot.v1",
                "base_pilot_schema": PILOT_SCHEMA_VERSION,
                "scheduler_contract": SCHEDULER_CONTRACT,
                "dataset": config["dataset"],
                "group_source": config["group_source"],
                "membership_hash": bank.membership_hash,
                "config": {
                    "path": args.config,
                    "sha256": sha256_file(args.config),
                },
                "residual_bundle": {
                    "path": config["residual_bundle"],
                    "sha256": sha256_file(config["residual_bundle"]),
                },
                "init_checkpoint": {
                    "path": init_path,
                    "sha256": sha256_file(init_path),
                },
                "permutation_manifest": {
                    "path": config["permutation_manifest"],
                    "sha256": sha256_file(config["permutation_manifest"]),
                    "manifest_hash": permutation["manifest_hash"],
                },
                "final_checkpoint": {
                    "path": str(final_path),
                    "sha256": sha256_file(final_path),
                },
                "world_size": world_size,
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
                "expected_optimizer_steps": expected_steps,
                "completed_optimizer_steps": completed_steps,
                "training_wall_seconds": wall_seconds,
                "instance_temperature": float(
                    model.model.instance_temperature.detach().item()
                ),
                "score_formula": "holistic + 0.3*instance + 0.4*group",
                "loss_formula": "final_ce + 0.3*instance_ce + 0.4*group_ce",
                "legacy_expression_and_orth_loss": False,
            }
            if fixed_runtime is not None:
                result["fixed_candidate_manifest"] = {
                    "path": config["fixed_candidates"]["manifest_path"],
                    "sha256": sha256_file(
                        config["fixed_candidates"]["manifest_path"]
                    ),
                    "manifest_hash": fixed_runtime.manifest["manifest_hash"],
                }
        elif mode == "test":
            if not checkpoint_path:
                raise ValueError("--checkpoint-path is required for evaluation")
            if int(model_args.gpus) != 1:
                raise RuntimeError("formal VIGEM evaluation uses one GPU")
            _load_strict(model, checkpoint_path)
            datamodule = PLDataLoader(
                model_args, model.model.bert_tokenizer
            )
            trainer = build_final_only_trainer(model_args, for_train=False)
            attach_test_log_to_ckpt_version(model_args)
            started = time.perf_counter()
            trainer.test(model, datamodule=datamodule)
            wall_seconds = time.perf_counter() - started
            model._write_eval_outputs()
            result = {
                "status": "EVAL_COMPLETE",
                "schema_version": "vigem.eval.v1",
                "dataset": config["dataset"],
                "group_source": config["group_source"],
                "membership_hash": bank.membership_hash,
                "config": {
                    "path": args.config,
                    "sha256": sha256_file(args.config),
                },
                "residual_bundle": {
                    "path": config["residual_bundle"],
                    "sha256": sha256_file(config["residual_bundle"]),
                },
                "checkpoint": {
                    "path": checkpoint_path,
                    "sha256": sha256_file(checkpoint_path),
                },
                "test_data_path": model_args.test_data_path,
                "score_dir": per_query_dir,
                "metrics_path": str(
                    Path(per_query_dir) / "rank_00_metrics.json"
                ),
                "evaluation_wall_seconds": wall_seconds,
                "includes_without_instance_ablation": True,
            }
        else:
            raise ValueError("mode must be train or test")
        atomic_write_json(completion_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
