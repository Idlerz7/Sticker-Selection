"""Fair-pilot DataModule, trace writer, score exporter, and final-only trainer."""

from __future__ import annotations

import json
import math
import os
import time

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader

from main import PLDataLoader
from structured_retrieval import DdpStaticGraphCallback
from structured_retrieval_factorized import StructuredFactorizedPLModel

from .io import atomic_write_json
from .permutations import (
    FixedEpochDistributedSampler,
    IndexedDataset,
    load_permutation_manifest,
    process_rank,
)


class StyleShapesDataModule(PLDataLoader):
    def __init__(self, args, tokenizer, permutation_path: str):
        super().__init__(args, tokenizer)
        self.permutation_path = str(permutation_path)
        self.permutation_manifest = None

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        if stage in {"fit", None}:
            self.train_dataset = IndexedDataset(self.train_dataset)
            self.permutation_manifest = load_permutation_manifest(
                self.permutation_path, len(self.train_dataset)
            )

    def collate_fn(self, batch):
        value = super().collate_fn(batch)
        if "_style_shapes_source_row" in batch[0]:
            value["source_rows"] = [int(item["_style_shapes_source_row"]) for item in batch]
        return value

    def train_dataloader(self):
        if self.permutation_manifest is None:
            raise RuntimeError("StyleShapesDataModule.setup('fit') must run first")
        world_size = int(self.permutation_manifest["world_size"])
        sampler = FixedEpochDistributedSampler(
            self.train_dataset,
            self.permutation_manifest,
            process_rank(world_size),
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=False,
            collate_fn=self.collate_fn,
        )


def _sharded_training_steps(num_batches, max_epochs, accumulate_grad_batches=1, limit_train_batches=1.0):
    """Optimizer steps when the DataLoader sampler is already rank-sharded."""
    batches = int(num_batches)
    if isinstance(limit_train_batches, int):
        batches = min(batches, int(limit_train_batches))
    else:
        batches = int(float(limit_train_batches) * batches)
    accumulation = max(1, int(accumulate_grad_batches))
    per_epoch = int(math.ceil(max(0, batches) / float(accumulation)))
    return max(1, per_epoch * int(max_epochs))


def _install_negative_trace_hook(owner, factorized_model):
    """Capture the exact negatives returned by the inner factorized scorer."""
    original = factorized_model._resolve_prototype_aware_negatives

    def traced(*args, **kwargs):
        value = original(*args, **kwargs)
        owner._style_shapes_last_negatives = (list(value[0]), list(value[1]))
        return value

    factorized_model._resolve_prototype_aware_negatives = traced


class StyleShapesPLModel(StructuredFactorizedPLModel):
    """Opt-in instrumentation; the scoring and loss implementation stay inherited."""

    def __init__(
        self,
        args,
        membership_hash: str,
        trace_dir: str = "",
        per_query_dir: str = "",
    ):
        self.style_shapes_membership_hash = str(membership_hash)
        self.style_shapes_trace_dir = str(trace_dir or "")
        self.style_shapes_per_query_dir = str(per_query_dir or "")
        self._style_shapes_last_negatives = None
        self._style_shapes_trace_handle = None
        self._style_shapes_trace_partial = None
        self._style_shapes_trace_final = None
        self._style_shapes_query_scores = []
        self._style_shapes_latency_ms = []
        self._style_shapes_refresh_ms = []
        super().__init__(args)
        original_factorization = self.model._compute_bank_factorization

        def timed_factorization(device):
            sample = len(self._style_shapes_refresh_ms) < 32
            if sample and torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter() if sample else None
            value = original_factorization(device)
            if sample:
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                self._style_shapes_refresh_ms.append((time.perf_counter() - started) * 1000.0)
            return value

        self.model._compute_bank_factorization = timed_factorization
        _install_negative_trace_hook(self, self.model)


    @property
    def num_training_steps(self):
        trainer = self.trainer
        if trainer.max_steps is not None and trainer.max_steps > 0:
            return int(trainer.max_steps)
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return super().num_training_steps
        try:
            batches = len(datamodule.train_dataloader())
        except Exception:
            return super().num_training_steps
        return _sharded_training_steps(
            batches,
            trainer.max_epochs,
            trainer.accumulate_grad_batches,
            trainer.limit_train_batches,
        )

    def on_train_start(self):
        result = super().on_train_start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.style_shapes_trace_dir:
            rank = int(getattr(self, "global_rank", 0))
            target = Path(self.style_shapes_trace_dir)
            target.mkdir(parents=True, exist_ok=True)
            self._style_shapes_trace_partial = target / ("rank_%02d.jsonl.partial" % rank)
            self._style_shapes_trace_final = target / ("rank_%02d.jsonl" % rank)
            if self._style_shapes_trace_final.exists():
                raise RuntimeError("refusing to overwrite completed trace: %s" % self._style_shapes_trace_final)
            self._style_shapes_trace_handle = self._style_shapes_trace_partial.open(
                "w", encoding="utf-8"
            )
        return result

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss = super().training_step(batch, batch_idx)
        if self._style_shapes_trace_handle is not None:
            if self._style_shapes_last_negatives is None:
                raise RuntimeError("negative resolver did not expose the actual sampled negatives")
            cross, same = self._style_shapes_last_negatives
            rows = batch["source_rows"]
            positives = batch["img_ids"]
            fallbacks = batch["neg_img_ids"]
            if not (len(rows) == len(positives) == len(fallbacks) == len(cross) == len(same)):
                raise RuntimeError("negative trace batch fields are not aligned")
            for source_row, positive, fallback, cross_id, same_id in zip(
                rows, positives, fallbacks, cross, same
            ):
                record = {
                    "epoch": int(self.current_epoch),
                    "global_step": int(self.global_step),
                    "rank": int(getattr(self, "global_rank", 0)),
                    "source_row": int(source_row),
                    "positive": int(positive),
                    "fallback": int(fallback),
                    "cross": int(cross_id),
                    "same": int(same_id),
                    "membership_hash": self.style_shapes_membership_hash,
                }
                self._style_shapes_trace_handle.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                )
        return loss

    def on_train_epoch_end(self):
        if self._style_shapes_trace_handle is not None:
            self._style_shapes_trace_handle.flush()
            os.fsync(self._style_shapes_trace_handle.fileno())
        parent = getattr(super(), "on_train_epoch_end", None)
        return parent() if parent is not None else None

    def on_train_end(self):
        if self._style_shapes_trace_handle is not None:
            self._style_shapes_trace_handle.flush()
            os.fsync(self._style_shapes_trace_handle.fileno())
            self._style_shapes_trace_handle.close()
            os.replace(self._style_shapes_trace_partial, self._style_shapes_trace_final)
            self._style_shapes_trace_handle = None
        if self.style_shapes_trace_dir:
            values = sorted(self._style_shapes_refresh_ms)
            p95 = values[min(len(values) - 1, int(0.95 * len(values)))] if values else None
            atomic_write_json(
                Path(self.style_shapes_trace_dir) / ("rank_%02d_performance.json" % int(getattr(self, "global_rank", 0))),
                {
                    "membership_hash": self.style_shapes_membership_hash,
                    "refresh_cost_sample_count": len(values),
                    "refresh_cost_mean_ms": sum(values) / len(values) if values else None,
                    "refresh_cost_p95_ms": p95,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    "parameters_total": sum(parameter.numel() for parameter in self.parameters()),
                    "parameters_trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
                },
            )
        parent = getattr(super(), "on_train_end", None)
        return parent() if parent is not None else None

    def run_eval_batch(self, batch, return_debug=False, score_breakdown=False):
        capture = bool(self.style_shapes_per_query_dir)
        if capture and torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter() if capture else None
        output = super().run_eval_batch(
            batch,
            return_debug=return_debug or capture,
            score_breakdown=score_breakdown or capture,
        )
        if capture:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._style_shapes_latency_ms.append((time.perf_counter() - started) * 1000.0)
            scores, labels, candidates, debug = output
            ordered = [int(item) for item in debug["candidate_ids_ordered"]]
            final = [float(item) for item in debug["final_score_per_cand"]]
            order = sorted(range(len(final)), key=lambda index: (-final[index], ordered[index]))
            gold = int(labels.item())
            self._style_shapes_query_scores.append(
                {
                    "query_index": len(self._style_shapes_query_scores),
                    "candidate_ids": ordered,
                    "gold": gold,
                    "positive_index": ordered.index(gold),
                    "base_scores": [float(item) for item in debug["mmbert_score_per_cand"]],
                    "instance_scores": [float(item) for item in debug["expr_score_per_cand"]],
                    "group_scores": [float(item) for item in debug["graph_score_per_cand"]],
                    "final_scores": final,
                    "rank": order.index(ordered.index(gold)) + 1,
                    "membership_hash": self.style_shapes_membership_hash,
                }
            )
            if not return_debug and not score_breakdown:
                return scores, labels, candidates
        return output

    def on_test_epoch_start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return super().on_test_epoch_start()

    def on_test_epoch_end(self):
        if self.style_shapes_per_query_dir and self._style_shapes_query_scores:
            rank = int(getattr(self, "global_rank", 0))
            path = Path(self.style_shapes_per_query_dir) / ("rank_%02d_scores.json" % rank)
            atomic_write_json(path, self._style_shapes_query_scores)
            values = sorted(self._style_shapes_latency_ms)
            p95 = values[min(len(values) - 1, int(0.95 * len(values)))] if values else None
            atomic_write_json(
                Path(self.style_shapes_per_query_dir) / ("rank_%02d_performance.json" % rank),
                {
                    "queries": len(values),
                    "latency_mean_ms": sum(values) / len(values) if values else None,
                    "latency_p95_ms": p95,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    "factorization_cost_sample_ms": self._style_shapes_refresh_ms,
                    "parameters_total": sum(parameter.numel() for parameter in self.parameters()),
                    "parameters_trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
                    "membership_hash": self.style_shapes_membership_hash,
                },
            )
        return super().on_test_epoch_end()


def build_final_only_trainer(args, for_train: bool) -> pl.Trainer:
    kwargs: Dict[str, Any] = {
        "gpus": args.gpus,
        "max_epochs": args.epochs,
        "accumulate_grad_batches": args.gradient_accumulation_steps,
        "default_root_dir": args.pl_root_dir,
        "precision": int(getattr(args, "trainer_precision", 32) or 32),
        "replace_sampler_ddp": False,
    }
    callbacks: list = []
    if (
        for_train
        and args.gpus
        and args.gpus > 1
        and getattr(args, "ddp_static_graph_callback", False)
        and getattr(args, "bert_gradient_checkpointing", True)
    ):
        callbacks.append(DdpStaticGraphCallback())
    if callbacks:
        kwargs["callbacks"] = callbacks
    if for_train:
        kwargs["checkpoint_callback"] = False
    if args.gpus and args.gpus > 1:
        kwargs["accelerator"] = "ddp"
        unused = getattr(args, "ddp_find_unused_parameters", None)
        if unused is not None:
            kwargs["plugins"] = DDPPlugin(find_unused_parameters=unused)
    return pl.Trainer(**kwargs)

