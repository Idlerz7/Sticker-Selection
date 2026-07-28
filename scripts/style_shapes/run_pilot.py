#!/usr/bin/env python3
"""Run one pre-registered Style Shapes train or single-GPU evaluation job."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from factorized_style_bank import FactorizedStyleBank
from main import PLDataLoader, attach_test_log_to_ckpt_version, attach_version_log_from_trainer
from structured_retrieval import load_checkpoint_to_model
from structured_retrieval_factorized import parse_structured_factorized_args
from structured_retrieval_tokens import _config_mapping_to_argv
from style_shapes.contracts import (
    PILOT_SCHEMA_VERSION,
    SCHEDULER_CONTRACT,
    completed_manifest_uses_current_contract,
    training_completion_errors,
)
from style_shapes.group_bank import GroupBank
from style_shapes.fixed_same_pack import (
    FIXED_SAME_PACK_POLICY,
    load_fixed_same_pack_runtime,
)
from style_shapes.io import atomic_write_json, command_record, sha256_file
from style_shapes.negative_sampling import (
    DUAL_LOCAL_GROUP_SOURCE_BY_POLICY,
    DUAL_LOCAL_POLICIES,
    load_dual_local_runtime,
)
from style_shapes.permutations import load_permutation_manifest
from style_shapes.runtime import resolve_permutation_world_size
from style_shapes.training import (
    StyleShapesDataModule,
    StyleShapesPLModel,
    build_final_only_trainer,
)


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("pilot configuration root must be a mapping")
    return value


def build_model_args(config):
    overrides = dict(config.get("model_overrides", {}))
    overrides["factorized_bank_path"] = config["group_bank"]
    return parse_structured_factorized_args(
        ["--config", config["base_config"]] + _config_mapping_to_argv(overrides)
    )


def verify_contract(config, model_args, bank):
    if str(os.environ.get("CONDA_DEFAULT_ENV", "")) != str(config.get("conda_env", "stickr-select")):
        raise RuntimeError("formal pilot must run in Conda environment stickr-select")
    if str(model_args.factorized_variant) != "minimal":
        raise RuntimeError("Style Shapes formal pilot requires v6 minimal factorized_variant")
    fixed = {
        "seed": 2021,
        "epochs": 10,
        "train_batch_size": 16,
        "lambda_expr": 0.3,
        "lambda_style_proto": 0.4,
        "lambda_orth": 0.5,
    }
    for name, expected in fixed.items():
        actual = getattr(model_args, name)
        if float(actual) != float(expected):
            raise RuntimeError("formal recipe mismatch for %s: %r != %r" % (name, actual, expected))
    allowed_source = config.get("reference_bank_source", config["group_source"])
    if bank.group_source != allowed_source:
        raise RuntimeError(
            "bank source mismatch: %s != %s" % (bank.group_source, allowed_source)
        )
    if int(bank.num_groups) != int(config["num_groups"]):
        raise RuntimeError("bank K mismatch")
    if config["dataset"] == "dstc" and int(model_args.factorized_train_bank_refresh_steps) != 0:
        raise RuntimeError("DSTC requires exact bank refresh 0")
    if config["dataset"] == "stickerchat" and int(model_args.factorized_train_bank_refresh_steps) != 500:
        raise RuntimeError("StickerChat requires bank refresh 500")
    hardware_profile = str(config.get("hardware_profile", "") or "")
    if hardware_profile:
        if hardware_profile != "rtx4090_24gb_fp16":
            raise RuntimeError("unsupported Style Shapes hardware profile")
        if int(model_args.trainer_precision) != 16:
            raise RuntimeError("RTX 4090 24GB profile requires trainer_precision=16")
        if (
            int(model_args.train_batch_size) != 16
            or int(model_args.gradient_accumulation_steps) != 1
        ):
            raise RuntimeError(
                "RTX 4090 24GB profile freezes batch=16 and accumulation=1"
            )
        if int(model_args.factorized_candidate_forward_chunk_size) != 10:
            raise RuntimeError("RTX 4090 24GB profile freezes candidate chunk=10")
        if str(config.get("mode", "train")) == "train":
            visible_names = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if not visible_names or any("4090" not in name for name in visible_names):
                raise RuntimeError(
                    "RTX 4090 24GB profile requires every visible GPU to be a 4090; "
                    "got %s" % visible_names
                )
    if config["dataset"] == "stickerchat" and str(config.get("mode", "train")) == "train":
        if str(model_args.factorized_train_mode) == FIXED_SAME_PACK_POLICY:
            expected_r10 = (
                "stickerchat/processed/"
                "release_val_u_sticker_format_int_with_cand_fixed_same_pack_r10.json"
            )
        else:
            expected_r10 = (
                "stickerchat/processed/"
                "release_val_u_sticker_format_int_with_cand_same_pack_r10.json"
            )
        expected_r20 = (
            "stickerchat/processed/release_val_u_sticker_format_int_with_cand_r20.json"
        )
        if str(model_args.per_epoch_eval_test_r10_path) != expected_r10:
            raise RuntimeError("StickerChat training requires fixed same-pack R10 validation")
        if str(model_args.per_epoch_eval_test_r20_path) != expected_r20:
            raise RuntimeError("StickerChat training requires fixed global-random R20 validation")
    negative_config = config.get("negative_sampling")
    fixed_config = config.get("fixed_candidates")
    if negative_config is not None and fixed_config is not None:
        raise RuntimeError(
            "negative_sampling and fixed_candidates are mutually exclusive"
        )
    if negative_config is not None:
        negative_policy = str(negative_config.get("mode", ""))
        if negative_policy not in DUAL_LOCAL_POLICIES:
            raise RuntimeError("unsupported special negative-sampling policy")
        expected_group_source = DUAL_LOCAL_GROUP_SOURCE_BY_POLICY[negative_policy]
        if (
            config["dataset"] != "stickerchat"
            or config["group_source"] != expected_group_source
        ):
            raise RuntimeError(
                "dual-local policy %s requires StickerChat group source %s"
                % (negative_policy, expected_group_source)
            )
        if int(getattr(model_args, "train_same_proto_negatives", 0)) != 1:
            raise RuntimeError("dual-local negatives require one same-slot negative")
        if int(getattr(model_args, "train_cross_proto_negatives", 0)) != 1:
            raise RuntimeError("dual-local negatives require one cross-slot negative")
        if bool(getattr(model_args, "factorized_train_mmbert_two_way", False)):
            raise RuntimeError("dual-local negatives require the three-candidate match loss")
    if fixed_config is not None:
        if config["dataset"] != "stickerchat":
            raise RuntimeError("fixed same-pack candidates are StickerChat-only")
        if str(fixed_config.get("mode", "")) != FIXED_SAME_PACK_POLICY:
            raise RuntimeError("unsupported fixed-candidate policy")
        if str(model_args.factorized_train_mode) != FIXED_SAME_PACK_POLICY:
            raise RuntimeError(
                "fixed candidate runtime requires factorized_train_mode="
                + FIXED_SAME_PACK_POLICY
            )
        if int(model_args.factorized_train_candidate_count) != 10:
            raise RuntimeError("fixed same-pack training requires 10 candidates")
        if int(model_args.factorized_candidate_forward_chunk_size) not in {
            10,
            5,
            2,
            1,
        }:
            raise RuntimeError(
                "candidate chunk must follow the registered OOM sequence 10/5/2/1"
            )
        if bool(model_args.add_ocr_info):
            raise RuntimeError(
                "gray sentinel candidates require add_ocr_info=false"
            )
    elif str(model_args.factorized_train_mode) != "legacy_triplet":
        raise RuntimeError(
            "non-legacy factorized_train_mode requires fixed_candidates config"
        )


def save_checkpoint_atomic(trainer, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".%s." % target.name, dir=str(target.parent))
    os.close(fd)
    try:
        trainer.save_checkpoint(temporary, weights_only=False)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    parser.add_argument("--run-mode", choices=("train", "test"))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--test-data-path")
    parser.add_argument("--run-output-dir")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        config = read_yaml(args.config)
        config["model_overrides"] = dict(config.get("model_overrides", {}))
        if args.run_mode is not None:
            config["mode"] = args.run_mode
        if args.checkpoint_path is not None:
            config["checkpoint_path"] = args.checkpoint_path
        if args.test_data_path is not None:
            config["model_overrides"]["test_data_path"] = args.test_data_path
        if args.run_output_dir is not None:
            config["output_dir"] = args.run_output_dir
        mode = str(config.get("mode", "train"))
        runtime_permutation = None
        if mode == "test":
            config["model_overrides"]["mode"] = "test"
            config["model_overrides"]["gpus"] = 1
            config["model_overrides"]["per_epoch_eval_test_r10_path"] = ""
            config["model_overrides"]["per_epoch_eval_test_r20_path"] = ""
            if args.run_output_dir is None:
                stem = Path(config["model_overrides"]["test_data_path"]).stem
                config["output_dir"] = str(Path(config["output_dir"]) / "final_eval" / stem)
        else:
            runtime_gpus = int(torch.cuda.device_count())
            if runtime_gpus <= 0:
                raise RuntimeError("formal training requires at least one visible CUDA GPU")
            config["model_overrides"]["gpus"] = runtime_gpus
            runtime_path, runtime_permutation = resolve_permutation_world_size(
                config["permutation_manifest"], runtime_gpus
            )
            config["permutation_manifest"] = runtime_path
        bank = GroupBank.load(config["group_bank"])
        # Exercise the public compatibility path before allocating the model.
        adapted = FactorizedStyleBank.from_json(config["group_bank"])
        if len(adapted.prototypes) != bank.num_groups or len(adapted.records) != len(bank.sticker_ids):
            raise RuntimeError("Group Bank public adapter failed")
        model_args = build_model_args(config)
        verify_contract(config, model_args, bank)
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
            if (
                mode == "train"
                and int(runtime_permutation["num_rows"]) != len(eligible_source_rows)
            ):
                raise RuntimeError(
                    "dual-local permutation rows do not match eligible training rows"
                )
        if config.get("fixed_candidates") is not None:
            fixed_candidate_runtime = load_fixed_same_pack_runtime(
                config["fixed_candidates"],
                train_data_path=model_args.train_data_path,
            )
            if (
                mode == "train"
                and int(runtime_permutation["num_rows"])
                != int(fixed_candidate_runtime.candidate_ids.size(0))
            ):
                raise RuntimeError(
                    "fixed-candidate permutation rows do not match training rows"
                )
        pl.seed_everything(int(model_args.seed))
        output = Path(config["output_dir"])
        mode = str(config.get("mode", "train"))
        per_query_dir = str(output / "scores") if mode == "test" else ""
        init_path = config.get("init_checkpoint_path", "")
        checkpoint_path = config.get("checkpoint_path", "")
        completed_path = output / ("%s_manifest.json" % mode)
        if completed_path.exists():
            with completed_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            compatible = (
                existing.get("membership_hash") == bank.membership_hash
                and existing.get("group_source") == config["group_source"]
            )
            if mode == "train" and compatible:
                compatible = (
                    completed_manifest_uses_current_contract(
                        existing, int(model_args.gpus)
                    )
                    and
                    init_path and Path(init_path).exists()
                    and existing.get("init_checkpoint", {}).get("sha256") == sha256_file(init_path)
                    and existing.get("permutation_manifest", {}).get("sha256")
                    == sha256_file(config["permutation_manifest"])
                )
                if compatible and eligibility_manifest is not None:
                    compatible = (
                        existing.get("negative_policy")
                        == eligibility_manifest["negative_policy"]
                        and existing.get("eligibility_manifest", {}).get("manifest_hash")
                        == eligibility_manifest["manifest_hash"]
                    )
                if compatible and fixed_candidate_runtime is not None:
                    compatible = (
                        existing.get("negative_policy")
                        == FIXED_SAME_PACK_POLICY
                        and existing.get("fixed_candidate_manifest", {}).get(
                            "manifest_hash"
                        )
                        == fixed_candidate_runtime.manifest["manifest_hash"]
                        and int(
                            existing.get("candidate_forward_chunk_size", -1)
                        )
                        == int(model_args.factorized_candidate_forward_chunk_size)
                    )
            if mode == "test" and compatible:
                compatible = (
                    checkpoint_path and Path(checkpoint_path).exists()
                    and existing.get("checkpoint", {}).get("sha256") == sha256_file(checkpoint_path)
                    and existing.get("test_data_path") == model_args.test_data_path
                )
                if compatible and config.get("negative_sampling") is not None:
                    compatible = (
                        existing.get("negative_policy")
                        == str(config["negative_sampling"]["mode"])
                    )
                if compatible and fixed_candidate_runtime is not None:
                    compatible = (
                        existing.get("negative_policy")
                        == FIXED_SAME_PACK_POLICY
                    )
            if not compatible:
                raise RuntimeError(
                    "refusing legacy, interrupted, or incompatible completed pilot "
                    "output; archive it or choose --run-output-dir: %s" % completed_path
                )
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return
        if mode == "train" and (output / "final.ckpt").exists():
            raise RuntimeError("refusing incomplete pilot output with checkpoint but no manifest")
        if mode == "test" and (output / "scores" / "rank_00_scores.json").exists():
            raise RuntimeError("refusing incomplete evaluation output with scores but no manifest")
        model = StyleShapesPLModel(
            model_args,
            membership_hash=bank.membership_hash,
            trace_dir=str(output / "negative_trace") if mode == "train" else "",
            per_query_dir=per_query_dir,
            negative_sampler=negative_sampler,
            fixed_candidate_runtime=fixed_candidate_runtime,
        )
        if mode == "train":
            if not init_path:
                raise ValueError("init_checkpoint_path is required for weights-only initialization")
            load_checkpoint_to_model(model, init_path, strict=True)
            permutation = load_permutation_manifest(config["permutation_manifest"])
            expected_world = int(torch.cuda.device_count())
            if int(model_args.gpus) != expected_world:
                raise RuntimeError("trainer GPU count does not match visible CUDA devices")
            if int(permutation["world_size"]) != expected_world:
                raise RuntimeError("permutation world size does not match visible CUDA devices")
            datamodule = StyleShapesDataModule(
                model_args,
                model.model.bert_tokenizer,
                config["permutation_manifest"],
                eligible_source_rows=eligible_source_rows,
                fixed_candidate_runtime=fixed_candidate_runtime,
            )
            trainer = build_final_only_trainer(model_args, for_train=True)
            attach_version_log_from_trainer(model_args, trainer)
            training_started = time.perf_counter()
            trainer.fit(model, datamodule=datamodule)
            training_wall_seconds = time.perf_counter() - training_started
            expected_optimizer_steps = int(model.num_training_steps)
            completed_optimizer_steps = int(trainer.global_step)
            completion_errors = training_completion_errors(
                interrupted=bool(getattr(trainer, "interrupted", False)),
                completed_optimizer_steps=completed_optimizer_steps,
                expected_optimizer_steps=expected_optimizer_steps,
                current_epoch=int(trainer.current_epoch),
                expected_epochs=int(model_args.epochs),
                trace_dir=str(output / "negative_trace"),
                world_size=int(model_args.gpus),
            )
            if completion_errors:
                incomplete = {
                    "status": "TRAIN_INCOMPLETE",
                    "schema_version": PILOT_SCHEMA_VERSION,
                    "scheduler_contract": SCHEDULER_CONTRACT,
                    "dataset": config["dataset"],
                    "group_source": config["group_source"],
                    "world_size": int(model_args.gpus),
                    "expected_optimizer_steps": expected_optimizer_steps,
                    "completed_optimizer_steps": completed_optimizer_steps,
                    "current_epoch": int(trainer.current_epoch),
                    "expected_epochs": int(model_args.epochs),
                    "errors": completion_errors,
                }
                atomic_write_json(output / "incomplete_run.json", incomplete)
                raise RuntimeError(
                    "formal training did not satisfy completion contract: %s"
                    % "; ".join(completion_errors)
                )
            final_path = output / "final.ckpt"
            save_checkpoint_atomic(trainer, final_path)
            result = {
                "status": "TRAIN_COMPLETE",
                "schema_version": PILOT_SCHEMA_VERSION,
                "scheduler_contract": SCHEDULER_CONTRACT,
                "dataset": config["dataset"],
                "group_source": config["group_source"],
                "membership_hash": bank.membership_hash,
                "world_size": int(model_args.gpus),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "hardware_profile": str(
                    config.get("hardware_profile", "default_fp32")
                ),
                "trainer_precision": int(model_args.trainer_precision),
                "train_batch_size_per_device": int(model_args.train_batch_size),
                "gradient_accumulation_steps": int(
                    model_args.gradient_accumulation_steps
                ),
                "expected_epochs": int(model_args.epochs),
                "completed_epoch": int(trainer.current_epoch),
                "expected_optimizer_steps": expected_optimizer_steps,
                "completed_optimizer_steps": completed_optimizer_steps,
                "init_checkpoint": {"path": init_path, "sha256": sha256_file(init_path)},
                "permutation_manifest": {
                    "path": config["permutation_manifest"],
                    "sha256": sha256_file(config["permutation_manifest"]),
                    "manifest_hash": permutation["manifest_hash"],
                },
                "final_checkpoint": {
                    "path": str(final_path),
                    "sha256": sha256_file(final_path),
                },
                "checkpoint_policy": "final_only",
                "training_wall_seconds": training_wall_seconds,
                "training_performance_dir": str(output / "negative_trace"),
            }
            if eligibility_manifest is not None:
                result.update(
                    {
                        "negative_policy": eligibility_manifest["negative_policy"],
                        "eligible_training_rows": len(eligible_source_rows),
                        "eligibility_manifest": {
                            "path": config["negative_sampling"]["eligibility_manifest"],
                            "sha256": sha256_file(
                                config["negative_sampling"]["eligibility_manifest"]
                            ),
                            "manifest_hash": eligibility_manifest["manifest_hash"],
                        },
                    }
                )
            if fixed_candidate_runtime is not None:
                result.update(
                    {
                        "negative_policy": FIXED_SAME_PACK_POLICY,
                        "fixed_candidate_manifest": {
                            "path": config["fixed_candidates"]["manifest_path"],
                            "sha256": sha256_file(
                                config["fixed_candidates"]["manifest_path"]
                            ),
                            "manifest_hash": fixed_candidate_runtime.manifest[
                                "manifest_hash"
                            ],
                        },
                        "candidate_count": 10,
                        "candidate_forward_chunk_size": int(
                            model_args.factorized_candidate_forward_chunk_size
                        ),
                    }
                )
        elif mode == "test":
            if int(model_args.gpus) != 1:
                raise RuntimeError("formal final evaluation must run on one GPU")
            if not checkpoint_path:
                raise ValueError("checkpoint_path is required for test")
            load_checkpoint_to_model(model, checkpoint_path, strict=True)
            datamodule = PLDataLoader(model_args, model.model.bert_tokenizer)
            trainer = build_final_only_trainer(model_args, for_train=False)
            attach_test_log_to_ckpt_version(model_args)
            evaluation_started = time.perf_counter()
            trainer.test(model, datamodule=datamodule)
            evaluation_wall_seconds = time.perf_counter() - evaluation_started
            result = {
                "status": "EVAL_COMPLETE",
                "dataset": config["dataset"],
                "group_source": config["group_source"],
                "membership_hash": bank.membership_hash,
                "checkpoint": {
                    "path": checkpoint_path,
                    "sha256": sha256_file(checkpoint_path),
                },
                "score_dir": per_query_dir,
                "test_data_path": model_args.test_data_path,
                "evaluation_wall_seconds": evaluation_wall_seconds,
            }
            if config.get("negative_sampling") is not None:
                result["negative_policy"] = str(
                    config["negative_sampling"]["mode"]
                )
            if fixed_candidate_runtime is not None:
                result["negative_policy"] = FIXED_SAME_PACK_POLICY
                result["fixed_candidate_manifest_hash"] = (
                    fixed_candidate_runtime.manifest["manifest_hash"]
                )
        else:
            raise ValueError("mode must be train or test")
        atomic_write_json(output / ("%s_manifest.json" % mode), result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
