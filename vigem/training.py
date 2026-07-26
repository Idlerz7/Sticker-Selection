"""Lightning wrapper and auditable outputs for the independent VIGEM runner."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch

from main import logger
from metrics import MyAccuracy
from structured_retrieval import (
    attach_per_epoch_dual_test_eval,
    maybe_set_ddp_static_graph,
)
from structured_retrieval_factorized import StructuredFactorizedPLModel
from style_shapes.io import atomic_write_json

from .model import VigemForwardOutput, VigemInstanceResidualStickerModel


def _rank_metrics(ranks: List[int]) -> Dict[str, Any]:
    if not ranks:
        raise ValueError("cannot compute metrics without queries")
    total = float(len(ranks))
    mrr = sum(1.0 / float(rank) for rank in ranks) / total
    return {
        "queries": len(ranks),
        "r_at_1": sum(rank <= 1 for rank in ranks) / total,
        "r_at_2": sum(rank <= 2 for rank in ranks) / total,
        "r_at_5": sum(rank <= 5 for rank in ranks) / total,
        "r_at_10": sum(rank <= 10 for rank in ranks) / total,
        "mrr": mrr,
        "map": mrr,
        "map_equals_mrr": True,
    }


def _sharded_training_steps(
    batches: int,
    epochs: int,
    accumulation: int,
    limit_train_batches: Any,
) -> int:
    value = int(batches)
    if isinstance(limit_train_batches, int):
        value = min(value, int(limit_train_batches))
    else:
        value = int(float(limit_train_batches) * value)
    return max(
        1,
        int(math.ceil(value / float(max(1, int(accumulation)))))
        * int(epochs),
    )


class VigemInstanceResidualPLModel(StructuredFactorizedPLModel):
    """A separate Lightning module; the legacy PL class is never modified."""

    def __init__(
        self,
        args,
        residual_bundle_path: str,
        membership_hash: str,
        trace_dir: str = "",
        per_query_dir: str = "",
        defer_residual_load: bool = False,
    ):
        pl.LightningModule.__init__(self)
        self.args = args
        # VIGEM has its own score/metrics directory.  Never write through the
        # inherited legacy result path, which could overwrite an old run.
        self.args.save_structured_test_outputs = False
        self.model = VigemInstanceResidualStickerModel(
            args,
            residual_bundle_path=residual_bundle_path,
            expected_membership_hash=membership_hash,
            defer_residual_load=defer_residual_load,
        )
        self.model.prepare_imgs(args)

        self.valtest_acc5 = MyAccuracy()
        self.valtest_acc30 = MyAccuracy()
        self.valtest_acc90 = MyAccuracy()
        self.valtest_acc_r10 = MyAccuracy()
        self.valtest_acc_r20 = MyAccuracy()
        self.valtest_map = MyAccuracy()
        self._eval_max_cand_len = 0
        attach_per_epoch_dual_test_eval(self, args)

        self.id2name: Dict[int, str] = {}
        with open(args.id2name_path, encoding="utf-8") as handle:
            raw_names = json.load(handle)
        for key, value in raw_names.items():
            self.id2name[int(key)] = value

        self._style_proto_acc_ema: Optional[float] = None
        self._style_gate_ema: Optional[float] = None
        self._reset_eval_diagnostics()
        self.vigem_membership_hash = str(membership_hash)
        self.vigem_trace_dir = str(trace_dir or "")
        self.vigem_per_query_dir = str(per_query_dir or "")
        self._vigem_trace_handle = None
        self._vigem_trace_partial: Optional[Path] = None
        self._vigem_trace_final: Optional[Path] = None
        self._vigem_last_debug: Optional[Dict[str, Any]] = None
        self._vigem_query_scores: List[Dict[str, Any]] = []
        self._vigem_latency_ms: List[float] = []
        self._vigem_steps: Optional[int] = None
        self._vigem_outputs_written = False

    @property
    def num_training_steps(self):
        trainer = self.trainer
        if trainer.max_steps is not None and trainer.max_steps > 0:
            return int(trainer.max_steps)
        if self._vigem_steps is not None:
            return int(self._vigem_steps)
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return max(1, int(trainer.max_epochs))
        try:
            batches = len(datamodule.train_dataloader())
        except Exception:
            return max(1, int(trainer.max_epochs))
        self._vigem_steps = _sharded_training_steps(
            batches,
            trainer.max_epochs,
            trainer.accumulate_grad_batches,
            trainer.limit_train_batches,
        )
        return int(self._vigem_steps)

    def run_train_batch(self, batch: Dict[str, Any]) -> VigemForwardOutput:
        if "train_candidate_ids" in batch:
            output = self.model.forward_train_listwise_batch(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                img_ids=batch["img_ids"],
                candidate_ids=batch["train_candidate_ids"],
                gray_mask=batch.get("train_candidate_gray_mask"),
                global_step=int(self.global_step),
                total_steps=int(self.num_training_steps),
            )
        else:
            output = self.model.forward_train_batch(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                img_ids=batch["img_ids"],
                neg_img_ids=batch["neg_img_ids"],
                global_step=int(self.global_step),
                total_steps=int(self.num_training_steps),
            )
        self._vigem_last_debug = output.debug_info
        return output

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        del batch_idx
        output = self.run_train_batch(batch)
        self.log("train_loss", output.loss, prog_bar=False)
        self.log("train_match_loss", output.match_loss, prog_bar=False)
        self.log("train_group_loss", output.group_loss, prog_bar=False)
        self.log("train_instance_loss", output.instance_loss, prog_bar=False)
        self.log("total", output.loss.detach(), prog_bar=True, logger=False)
        self.log("match", output.match_loss.detach(), prog_bar=True, logger=False)
        self.log("group", output.group_loss.detach(), prog_bar=True, logger=False)
        self.log(
            "instance", output.instance_loss.detach(), prog_bar=True, logger=False
        )
        self.log(
            "tau_i",
            self.model.instance_temperature.detach(),
            prog_bar=True,
            logger=False,
        )
        proto_acc = float(output.debug_info["proto_acc"].detach().item())
        self._style_proto_acc_ema = self._update_ema(
            self._style_proto_acc_ema, proto_acc
        )
        self.log(
            "pacc",
            torch.tensor(
                self._style_proto_acc_ema,
                device=output.loss.device,
                dtype=torch.float32,
            ),
            prog_bar=True,
            logger=False,
        )
        self._write_train_trace(batch, output.debug_info)
        return output.loss

    def _write_train_trace(
        self, batch: Dict[str, Any], debug: Dict[str, Any]
    ) -> None:
        if self._vigem_trace_handle is None:
            return
        candidates = debug["candidate_ids"].detach().cpu().tolist()
        gray = debug["gray_mask"].detach().cpu().tolist()
        holistic = debug["holistic_scores"].detach().float().cpu().tolist()
        instance = debug["instance_scores"].detach().float().cpu().tolist()
        group = debug["group_scores"].detach().float().cpu().tolist()
        final = debug["final_scores"].detach().float().cpu().tolist()
        without = (
            debug["final_without_instance_scores"]
            .detach()
            .float()
            .cpu()
            .tolist()
        )
        valid = debug["instance_valid_mask"].detach().cpu().tolist()
        rows = batch.get("source_rows", list(range(len(candidates))))
        positives = batch["img_ids"]
        for index, candidate_row in enumerate(candidates):
            record = {
                "epoch": int(self.current_epoch),
                "global_step": int(self.global_step),
                "rank": int(getattr(self, "global_rank", 0)),
                "source_row": int(rows[index]),
                "positive": int(positives[index]),
                "negative_policy": str(debug["negative_policy"]),
                "candidate_ids": [int(value) for value in candidate_row],
                "gray_mask": [bool(value) for value in gray[index]],
                "instance_valid_mask": [
                    bool(value) for value in valid[index]
                ],
                "holistic_scores": [
                    float(value) for value in holistic[index]
                ],
                "instance_scores": [
                    float(value) for value in instance[index]
                ],
                "group_scores": [float(value) for value in group[index]],
                "final_scores": [float(value) for value in final[index]],
                "final_without_instance_scores": [
                    float(value) for value in without[index]
                ],
                "temperature": float(
                    debug["temperature"].detach().float().item()
                ),
                "membership_hash": self.vigem_membership_hash,
            }
            self._vigem_trace_handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"))
                + "\n"
            )

    def on_train_start(self):
        maybe_set_ddp_static_graph(self.trainer)
        cache_path = (
            getattr(self.args, "img_emb_cache_path", None) or ""
        ).strip()
        if getattr(self.args, "fix_img", False) and cache_path and os.path.exists(
            cache_path
        ):
            self.model.prepare_for_test()
        self.model.clear_train_factorization_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.vigem_trace_dir:
            rank = int(getattr(self, "global_rank", 0))
            target = Path(self.vigem_trace_dir)
            target.mkdir(parents=True, exist_ok=True)
            self._vigem_trace_partial = target / (
                "rank_%02d.jsonl.partial" % rank
            )
            self._vigem_trace_final = target / ("rank_%02d.jsonl" % rank)
            if self._vigem_trace_final.exists():
                raise RuntimeError(
                    "refusing to overwrite completed VIGEM trace"
                )
            self._vigem_trace_handle = self._vigem_trace_partial.open(
                "w", encoding="utf-8"
            )
        logger.info(
            "[VIGEM] final=holistic+0.3*instance+0.4*group; "
            "loss=final_ce+0.3*instance_ce+0.4*group_ce; "
            "legacy expression/orthogonal losses disabled"
        )
        return pl.LightningModule.on_train_start(self)

    def on_train_epoch_end(self):
        if self._vigem_trace_handle is not None:
            self._vigem_trace_handle.flush()
            os.fsync(self._vigem_trace_handle.fileno())
        return pl.LightningModule.on_train_epoch_end(self)

    def on_train_end(self):
        if self._vigem_trace_handle is not None:
            self._vigem_trace_handle.flush()
            os.fsync(self._vigem_trace_handle.fileno())
            self._vigem_trace_handle.close()
            os.replace(self._vigem_trace_partial, self._vigem_trace_final)
            self._vigem_trace_handle = None
        if self.vigem_trace_dir:
            atomic_write_json(
                Path(self.vigem_trace_dir)
                / (
                    "rank_%02d_performance.json"
                    % int(getattr(self, "global_rank", 0))
                ),
                {
                    "membership_hash": self.vigem_membership_hash,
                    "temperature": float(
                        self.model.instance_temperature.detach().item()
                    ),
                    "peak_cuda_memory_bytes": (
                        torch.cuda.max_memory_allocated()
                        if torch.cuda.is_available()
                        else 0
                    ),
                    "parameters_total": sum(
                        parameter.numel() for parameter in self.parameters()
                    ),
                    "parameters_trainable": sum(
                        parameter.numel()
                        for parameter in self.parameters()
                        if parameter.requires_grad
                    ),
                },
            )
        return pl.LightningModule.on_train_end(self)

    def run_eval_batch(
        self,
        batch: Dict[str, Any],
        return_debug: bool = False,
        score_breakdown: bool = False,
    ):
        capture = bool(self.vigem_per_query_dir)
        if capture and torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter() if capture else None
        output = self.model.forward_eval_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            img_ids=batch["img_ids"],
            cands=batch.get("cands"),
            return_debug=return_debug or capture,
            score_breakdown=score_breakdown or capture,
        )
        if capture:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._vigem_latency_ms.append(
                (time.perf_counter() - started) * 1000.0
            )
            scores, labels, candidates, debug = output
            del scores
            ordered = [int(value) for value in debug["candidate_ids_ordered"]]
            final = [float(value) for value in debug["final_score_per_cand"]]
            without = [
                float(value)
                for value in debug["final_without_instance_per_cand"]
            ]
            gold = int(labels.item())
            positive_index = ordered.index(gold)
            final_order = sorted(
                range(len(final)), key=lambda idx: (-final[idx], ordered[idx])
            )
            ablated_order = sorted(
                range(len(without)),
                key=lambda idx: (-without[idx], ordered[idx]),
            )
            self._vigem_query_scores.append(
                {
                    "query_index": len(self._vigem_query_scores),
                    "candidate_ids": ordered,
                    "gray_mask": [
                        bool(value) for value in debug["gray_mask"]
                    ],
                    "gold": gold,
                    "positive_index": positive_index,
                    "holistic_scores": [
                        float(value)
                        for value in debug["mmbert_score_per_cand"]
                    ],
                    "instance_scores": [
                        float(value)
                        for value in debug["instance_score_per_cand"]
                    ],
                    "group_scores": [
                        float(value)
                        for value in debug["graph_score_per_cand"]
                    ],
                    "final_scores": final,
                    "final_without_instance_scores": without,
                    "rank": final_order.index(positive_index) + 1,
                    "rank_without_instance": (
                        ablated_order.index(positive_index) + 1
                    ),
                    "temperature": float(debug["temperature"]),
                    "membership_hash": self.vigem_membership_hash,
                }
            )
            if not return_debug and not score_breakdown:
                return output[:3]
        return output

    def on_test_epoch_start(self):
        self._vigem_query_scores = []
        self._vigem_latency_ms = []
        self._vigem_outputs_written = False
        return super().on_test_epoch_start()

    def _write_eval_outputs(self):
        if (
            self._vigem_outputs_written
            or not self.vigem_per_query_dir
            or not self._vigem_query_scores
        ):
            return
        rank = int(getattr(self, "global_rank", 0))
        target = Path(self.vigem_per_query_dir)
        target.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            target / ("rank_%02d_scores.json" % rank),
            self._vigem_query_scores,
        )
        full_ranks = [int(row["rank"]) for row in self._vigem_query_scores]
        ablated_ranks = [
            int(row["rank_without_instance"])
            for row in self._vigem_query_scores
        ]
        metrics = {
            "full": _rank_metrics(full_ranks),
            "without_instance": _rank_metrics(ablated_ranks),
            "delta": {},
            "temperature": float(
                self.model.instance_temperature.detach().item()
            ),
            "membership_hash": self.vigem_membership_hash,
        }
        for key in ("r_at_1", "r_at_2", "r_at_5", "r_at_10", "mrr", "map"):
            metrics["delta"][key] = (
                metrics["full"][key] - metrics["without_instance"][key]
            )
        atomic_write_json(
            target / ("rank_%02d_metrics.json" % rank), metrics
        )
        values = sorted(self._vigem_latency_ms)
        p95 = (
            values[min(len(values) - 1, int(0.95 * len(values)))]
            if values
            else None
        )
        atomic_write_json(
            target / ("rank_%02d_performance.json" % rank),
            {
                "queries": len(values),
                "latency_mean_ms": (
                    sum(values) / len(values) if values else None
                ),
                "latency_p95_ms": p95,
                "peak_cuda_memory_bytes": (
                    torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else 0
                ),
                "parameters_total": sum(
                    parameter.numel() for parameter in self.parameters()
                ),
                "parameters_trainable": sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                ),
                "membership_hash": self.vigem_membership_hash,
            },
        )
        self._vigem_outputs_written = True

    def on_test_epoch_end(self):
        self._write_eval_outputs()
        return super().on_test_epoch_end()
