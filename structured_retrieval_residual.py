import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Work around protobuf C-extension segfault in some environments
# (must be set before importing pytorch_lightning/tensorboard).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import distutils_tensorboard_shim  # noqa: F401

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import HfArgumentParser

from main import PLDataLoader, attach_test_log_to_ckpt_version, attach_version_log_from_trainer, logger
from metrics import MyAccuracy
from structured_retrieval import (
    ProjectionHead,
    StructuredArguments,
    StructuredForwardOutput,
    StructuredPLModel,
    StructuredStickerModel,
    _CAND_EVAL_ONLY_ERR,
    attach_per_epoch_dual_test_eval,
    build_trainer,
    load_checkpoint_to_model,
    nonempty_batch_cands,
)
from structured_retrieval_tokens import (
    _config_mapping_to_argv,
    _deep_merge_dict,
    _extract_yaml_config_paths,
    load_structured_token_yaml_with_extends,
)


def _safe_logit(p: float) -> float:
    p = max(min(float(p), 1.0 - 1e-6), 1e-6)
    return float(torch.logit(torch.tensor(p)).item())


@dataclass
class StructuredResidualArguments(StructuredArguments):
    pl_root_dir: Optional[str] = field(default="logs/structured_residual")
    struct_residual_mode: Optional[str] = field(default="expr")
    residual_condition_mode: Optional[str] = field(default="none")
    struct_residual_use_late_fusion: Optional[bool] = field(default=False)
    struct_residual_compare_plain_eval: Optional[bool] = field(default=True)
    struct_residual_log_interval: Optional[int] = field(default=200)
    residual_init_gain: Optional[float] = field(default=0.1)

    def __post_init__(self):
        super().__post_init__()
        valid_modes = {"none", "expr", "style", "expr_style"}
        valid_condition_modes = {"none", "query_gate"}
        if self.struct_residual_mode not in valid_modes:
            raise ValueError(
                f"Unsupported struct_residual_mode={self.struct_residual_mode}. "
                "Use 'none', 'expr', 'style', or 'expr_style'."
            )
        if self.residual_condition_mode not in valid_condition_modes:
            raise ValueError(
                f"Unsupported residual_condition_mode={self.residual_condition_mode}. "
                "Use 'none' or 'query_gate'."
            )
        if self.struct_residual_log_interval is not None and self.struct_residual_log_interval < 0:
            raise ValueError("struct_residual_log_interval must be >= 0 (0 = off).")
        if not (0.0 <= float(self.residual_init_gain) < 1.0):
            raise ValueError("residual_init_gain must be in [0, 1).")


class StructuredResidualStickerModel(StructuredStickerModel):
    def __init__(self, args: StructuredResidualArguments):
        super().__init__(args)
        self.args = args
        self.expr_residual_proj = ProjectionHead(
            args.structured_hidden_dim, self.bert_hidden_dim, args.structured_dropout
        )
        self.style_residual_proj = ProjectionHead(
            args.structured_hidden_dim, self.bert_hidden_dim, args.structured_dropout
        )
        self.expr_residual_gate = nn.Linear(args.structured_hidden_dim * 2, self.bert_hidden_dim)
        self.style_residual_gate = nn.Linear(args.structured_hidden_dim * 2, self.bert_hidden_dim)
        init_logit = _safe_logit(float(args.residual_init_gain))
        self.expr_residual_gain_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        self.style_residual_gain_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        self._residual_stats_buffer: List[Dict[str, float]] = []

    def _active_residual_names(self) -> List[str]:
        mode = self.args.struct_residual_mode
        if mode == "none":
            return []
        if mode == "expr":
            return ["expr"]
        if mode == "style":
            return ["style"]
        if mode == "expr_style":
            return ["expr", "style"]
        raise ValueError(f"Unsupported struct_residual_mode={mode}")

    def _reset_residual_stats_buffer(self) -> None:
        self._residual_stats_buffer = []

    def _record_residual_stats(self, stats: Dict[str, float]) -> None:
        self._residual_stats_buffer.append(stats)

    def summarize_residual_stats(self) -> Dict[str, float]:
        if not self._residual_stats_buffer:
            return {
                "expr_residual_norm": 0.0,
                "style_residual_norm": 0.0,
                "total_residual_norm": 0.0,
                "sticker_base_norm": 0.0,
                "sticker_mod_norm": 0.0,
                "expr_gain": float(torch.sigmoid(self.expr_residual_gain_logit).detach().item()),
                "style_gain": float(torch.sigmoid(self.style_residual_gain_logit).detach().item()),
            }
        denom = float(len(self._residual_stats_buffer))
        return {
            "expr_residual_norm": sum(x["expr_residual_norm"] for x in self._residual_stats_buffer)
            / denom,
            "style_residual_norm": sum(x["style_residual_norm"] for x in self._residual_stats_buffer)
            / denom,
            "total_residual_norm": sum(x["total_residual_norm"] for x in self._residual_stats_buffer)
            / denom,
            "sticker_base_norm": sum(x["sticker_base_norm"] for x in self._residual_stats_buffer)
            / denom,
            "sticker_mod_norm": sum(x["sticker_mod_norm"] for x in self._residual_stats_buffer)
            / denom,
            "expr_gain": sum(x["expr_gain"] for x in self._residual_stats_buffer) / denom,
            "style_gain": sum(x["style_gain"] for x in self._residual_stats_buffer) / denom,
        }

    def encode_dialogue_query_from_text_emb(
        self, text_emb: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert.bert(
            inputs_embeds=text_emb,
            attention_mask=attention_mask,
            return_dict=True,
        )
        q = outputs.last_hidden_state[:, 0, :]
        return self.query_head(q)

    def _condition_projected_residual(
        self,
        name: str,
        source: torch.Tensor,
        projected: torch.Tensor,
        q: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.args.residual_condition_mode != "query_gate" or q is None:
            return projected
        gate_layer = getattr(self, f"{name}_residual_gate")
        gate = torch.sigmoid(gate_layer(torch.cat([source, q], dim=-1)))
        return projected * gate

    def _residual_gain(self, name: str) -> torch.Tensor:
        return torch.sigmoid(getattr(self, f"{name}_residual_gain_logit"))

    def _build_sticker_residual(
        self,
        img_emb: torch.Tensor,
        base_sticker_token: torch.Tensor,
        q: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        _shared_h, c, a = self.decompose_sticker(img_emb)
        device = img_emb.device
        total = torch.zeros(img_emb.size(0), self.bert_hidden_dim, device=device, dtype=img_emb.dtype)
        stats = {
            "expr_residual_norm": 0.0,
            "style_residual_norm": 0.0,
            "total_residual_norm": 0.0,
            "sticker_base_norm": float(base_sticker_token.detach().norm(dim=-1).mean().item()),
            "sticker_mod_norm": 0.0,
            "expr_gain": float(self._residual_gain("expr").detach().item()),
            "style_gain": float(self._residual_gain("style").detach().item()),
        }

        if "expr" in self._active_residual_names():
            expr_residual = self._condition_projected_residual(
                "expr", a, self.expr_residual_proj(a), q
            )
            expr_residual = self._residual_gain("expr") * expr_residual
            total = total + expr_residual
            stats["expr_residual_norm"] = float(expr_residual.detach().norm(dim=-1).mean().item())

        if "style" in self._active_residual_names():
            style_residual = self._condition_projected_residual(
                "style", c, self.style_residual_proj(c), q
            )
            style_residual = self._residual_gain("style") * style_residual
            total = total + style_residual
            stats["style_residual_norm"] = float(style_residual.detach().norm(dim=-1).mean().item())

        mod_sticker_token = base_sticker_token + total.unsqueeze(1)
        stats["total_residual_norm"] = float(total.detach().norm(dim=-1).mean().item())
        stats["sticker_mod_norm"] = float(mod_sticker_token.detach().norm(dim=-1).mean().item())
        return mod_sticker_token, stats

    def _build_multimodal_inputs(
        self,
        text_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        img_emb: torch.Tensor,
        img_ids: Sequence[int],
        dialogue_q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = text_emb.device
        batch_size = img_emb.size(0)
        base_sticker_token = self._project_sticker_token(img_emb)
        aux_emb = None
        if self.args.add_ocr_info:
            aux_emb = self._build_cls_inputs_emb(img_ids, device)

        q = None
        if self.args.residual_condition_mode == "query_gate" and self.args.struct_residual_mode != "none":
            # Reuse precomputed dialogue_q from encode_dialogue_query when provided (train / eval).
            # Extra bert.bert here + DDP + gradient checkpoint causes "marked as ready twice".
            if dialogue_q is not None:
                q = dialogue_q
            else:
                q = self.encode_dialogue_query_from_text_emb(text_emb, attention_mask)
        sticker_token, residual_stats = self._build_sticker_residual(img_emb, base_sticker_token, q)

        sep_emb = self._build_sep_embeddings(batch_size=batch_size, device=device)
        input_parts = [text_emb]
        extra_len = 2
        if aux_emb is not None:
            input_parts.append(aux_emb)
            extra_len += aux_emb.size(1)
        input_parts.extend([sticker_token, sep_emb])
        input_emb = torch.cat(input_parts, dim=1)
        ones_mask = torch.ones(batch_size, extra_len, device=device)
        full_attention_mask = torch.cat([attention_mask, ones_mask], dim=1)
        token_type_ids = torch.zeros_like(full_attention_mask, dtype=torch.long)
        token_type_ids[:, -extra_len:] = 1
        self._record_residual_stats(residual_stats)
        return input_emb, full_attention_mask, token_type_ids

    def _compute_pair_logits_plain(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_ids: Sequence[int],
        img_emb: torch.Tensor,
    ) -> torch.Tensor:
        text_emb = self._get_text_word_embeddings(input_ids)
        input_emb, full_attention_mask, token_type_ids = StructuredStickerModel._build_multimodal_inputs(
            self,
            text_emb=text_emb,
            attention_mask=attention_mask,
            img_emb=img_emb,
            img_ids=img_ids,
        )
        outputs = self.bert(
            inputs_embeds=input_emb,
            attention_mask=full_attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return outputs.logits

    def fuse_final_score(
        self,
        base_score: torch.Tensor,
        style_score: torch.Tensor,
        expr_score: torch.Tensor,
        style_gate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expr_mod = self.modulate_expression_score(expr_score, style_gate)
        if self.args.struct_fuse_mode == "style_only":
            structured_term = style_score
        elif self.args.struct_fuse_mode == "expr_only":
            structured_term = expr_mod
        else:
            structured_term = style_score + expr_mod

        if self.args.struct_residual_use_late_fusion:
            final_score = base_score + self.args.lambda_fuse * structured_term
        else:
            final_score = base_score
        return final_score, structured_term, expr_mod

    def forward_eval_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_ids: Sequence[int],
        cands: Optional[Sequence[Sequence[int]]] = None,
        return_debug: bool = False,
    ):
        device = input_ids.device
        batch_size = input_ids.size(0)
        if batch_size != 1:
            raise ValueError(
                f"Structured eval currently expects batch_size=1, got {batch_size}."
            )

        q = self.encode_dialogue_query(input_ids, attention_mask)

        use_cands = nonempty_batch_cands(cands)
        if use_cands:
            candidate_ids = [int(x) for x in cands[0]]
            candidate_h = self.all_img_embs[candidate_ids]
        else:
            if getattr(self.args, "candidate_eval_only", False) or getattr(
                self.args, "test_with_cand", False
            ):
                raise ValueError(_CAND_EVAL_ONLY_ERR)
            candidate_ids = list(range(self.args.max_image_id))
            candidate_h = self.all_img_embs

        img_num = candidate_h.size(0)
        expanded_input_ids = input_ids.repeat(img_num, 1)
        expanded_attention_mask = attention_mask.repeat(img_num, 1)

        mod_logits = self._compute_pair_logits(
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            img_ids=candidate_ids,
            img_emb=candidate_h,
            dialogue_q=q.repeat(img_num, 1),
        )
        mod_backbone_score = self.compute_base_score(mod_logits)

        plain_base_score = mod_backbone_score
        plain_logits = None
        if self.args.struct_residual_compare_plain_eval and self.args.struct_residual_mode != "none":
            plain_logits = self._compute_pair_logits_plain(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                img_ids=candidate_ids,
                img_emb=candidate_h,
            )
            plain_base_score = self.compute_base_score(plain_logits)

        candidate_q = q.repeat(img_num, 1)
        _, candidate_c, candidate_a = self.decompose_sticker(candidate_h)
        style_score = self.compute_style_compatibility(candidate_q, candidate_c)
        expr_score = self.compute_expression_compatibility(candidate_q, candidate_a)
        style_gate = self.compute_style_gate(candidate_q, candidate_c)
        final_score, structured_term, expr_mod = self.fuse_final_score(
            mod_backbone_score, style_score, expr_score, style_gate
        )

        rank_scores = final_score.unsqueeze(0)
        labels = torch.tensor(img_ids, dtype=torch.long, device=device)
        if not return_debug:
            return rank_scores, labels, candidate_ids if use_cands else None

        residual_stats = self.summarize_residual_stats()
        residual_delta = final_score - plain_base_score
        if self.args.struct_residual_use_late_fusion:
            ranking_formula = (
                "s_rank = s_backbone(residual_sticker) + lambda_fuse * structured_term; "
                "compare_base = s_backbone(plain_sticker)"
            )
        else:
            ranking_formula = (
                "s_rank = s_backbone(residual_sticker); "
                "compare_base = s_backbone(plain_sticker)"
            )

        eval_debug = {
            "plain_base_score_stats": self._tensor_debug_stats(plain_base_score),
            "rank_score_stats": self._tensor_debug_stats(final_score),
            "residual_delta_score_stats": self._tensor_debug_stats(residual_delta),
            "style_score_stats": self._tensor_debug_stats(style_score),
            "expr_mod_stats": self._tensor_debug_stats(expr_mod),
            "structured_term_stats": self._tensor_debug_stats(structured_term),
            "mod_backbone_score_stats": self._tensor_debug_stats(mod_backbone_score),
            "residual_delta_abs_mean": float(residual_delta.abs().mean().item()),
            "top1_flipped": bool(
                torch.argmax(plain_base_score).item() != torch.argmax(final_score).item()
            ),
            "ranking_score_formula": ranking_formula,
            "candidate_count": int(len(candidate_ids)),
            "uses_late_fusion": bool(self.args.struct_residual_use_late_fusion),
            "struct_residual_mode": self.args.struct_residual_mode,
            "residual_condition_mode": self.args.residual_condition_mode,
            "residual_stats": residual_stats,
            "plain_logits_stats": self._tensor_debug_stats(plain_logits)
            if plain_logits is not None
            else None,
        }
        return rank_scores, labels, candidate_ids if use_cands else None, eval_debug


class StructuredResidualPLModel(StructuredPLModel):
    def __init__(self, args: StructuredResidualArguments):
        pl.LightningModule.__init__(self)
        self.args = args
        self.model = StructuredResidualStickerModel(args)
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
        with open(args.id2name_path, encoding="utf-8") as f:
            raw = json.load(f)
            for k, v in raw.items():
                self.id2name[int(k)] = v

        self._expr_residual_norm_ema: Optional[float] = None
        self._style_residual_norm_ema: Optional[float] = None
        self._total_residual_norm_ema: Optional[float] = None
        self._reset_eval_diagnostics()

    def _reset_eval_diagnostics(self) -> None:
        self.eval_diag_count = 0
        self.eval_plain_base_std_sum = 0.0
        self.eval_rank_score_std_sum = 0.0
        self.eval_residual_delta_std_sum = 0.0
        self.eval_style_score_std_sum = 0.0
        self.eval_expr_mod_std_sum = 0.0
        self.eval_residual_delta_abs_mean_sum = 0.0
        self.eval_top1_flip_count = 0
        self.eval_expr_residual_norm_sum = 0.0
        self.eval_style_residual_norm_sum = 0.0
        self.eval_total_residual_norm_sum = 0.0
        self.eval_sticker_mod_norm_sum = 0.0

    def _update_eval_diagnostics(self, eval_debug: Dict[str, Any]) -> None:
        self.eval_diag_count += 1
        self.eval_plain_base_std_sum += float(eval_debug["plain_base_score_stats"]["std"])
        self.eval_rank_score_std_sum += float(eval_debug["rank_score_stats"]["std"])
        self.eval_residual_delta_std_sum += float(eval_debug["residual_delta_score_stats"]["std"])
        self.eval_style_score_std_sum += float(eval_debug["style_score_stats"]["std"])
        self.eval_expr_mod_std_sum += float(eval_debug["expr_mod_stats"]["std"])
        self.eval_residual_delta_abs_mean_sum += float(eval_debug["residual_delta_abs_mean"])
        self.eval_top1_flip_count += int(bool(eval_debug["top1_flipped"]))
        self.eval_expr_residual_norm_sum += float(eval_debug["residual_stats"]["expr_residual_norm"])
        self.eval_style_residual_norm_sum += float(eval_debug["residual_stats"]["style_residual_norm"])
        self.eval_total_residual_norm_sum += float(eval_debug["residual_stats"]["total_residual_norm"])
        self.eval_sticker_mod_norm_sum += float(eval_debug["residual_stats"]["sticker_mod_norm"])

    def _eval_diagnostics_summary(self) -> Dict[str, float]:
        if self.eval_diag_count <= 0:
            return {}
        denom = float(self.eval_diag_count)
        plain_base_std = self.eval_plain_base_std_sum / denom
        residual_delta_std = self.eval_residual_delta_std_sum / denom
        return {
            "plain_base_std": plain_base_std,
            "rank_score_std": self.eval_rank_score_std_sum / denom,
            "residual_delta_std": residual_delta_std,
            "residual_over_base_ratio": residual_delta_std / max(plain_base_std, 1e-8),
            "style_score_std": self.eval_style_score_std_sum / denom,
            "expr_mod_std": self.eval_expr_mod_std_sum / denom,
            "residual_delta_abs_mean": self.eval_residual_delta_abs_mean_sum / denom,
            "top1_flip_rate": self.eval_top1_flip_count / denom,
            "expr_residual_norm": self.eval_expr_residual_norm_sum / denom,
            "style_residual_norm": self.eval_style_residual_norm_sum / denom,
            "total_residual_norm": self.eval_total_residual_norm_sum / denom,
            "sticker_mod_norm": self.eval_sticker_mod_norm_sum / denom,
        }

    def _update_ema(self, old: Optional[float], value: float, alpha: float = 0.05) -> float:
        if old is None:
            return value
        return alpha * value + (1.0 - alpha) * old

    def on_train_start(self) -> None:
        logger.info(
            "[StructuredResidualV3] mode=%s cond=%s late_fusion=%s compare_plain_eval=%s "
            "residual_init_gain=%.4f",
            self.args.struct_residual_mode,
            self.args.residual_condition_mode,
            "on" if self.args.struct_residual_use_late_fusion else "off",
            bool(self.args.struct_residual_compare_plain_eval),
            float(self.args.residual_init_gain),
        )
        logger.info(
            "[StructuredResidualV3] Residual modulation keeps the legacy sticker slot and "
            "adds projected struct residuals to that slot instead of appending extra tokens."
        )
        return pl.LightningModule.on_train_start(self)

    def run_train_batch(self, batch: Dict[str, Any]) -> StructuredForwardOutput:
        self.model._reset_residual_stats_buffer()
        return self.model.forward_train_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            img_ids=batch["img_ids"],
            neg_img_ids=batch["neg_img_ids"],
            global_step=int(self.global_step),
            total_steps=int(self.num_training_steps),
        )

    def run_eval_batch(self, batch: Dict[str, Any], return_debug: bool = False):
        self.model._reset_residual_stats_buffer()
        return self.model.forward_eval_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            img_ids=batch["img_ids"],
            cands=batch.get("cands"),
            return_debug=return_debug,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.run_train_batch(batch)
        structured_aux = self.args.lambda_struct * (
            outputs.style_neighbor_loss + outputs.orth_loss
        )
        expr_aux = self.args.lambda_expr * outputs.expr_rank_loss
        if self.args.base_only:
            structured_aux = structured_aux * 0.0
            expr_aux = expr_aux * 0.0
        weighted_struct_aux = outputs.aux_warmup * structured_aux
        weighted_expr_aux = outputs.aux_warmup * expr_aux

        self.log("train_loss", outputs.loss, prog_bar=False)
        self.log("train_match_loss", outputs.match_loss, prog_bar=False)
        self.log("train_style_neighbor_loss", outputs.style_neighbor_loss, prog_bar=False)
        self.log("train_orth_loss", outputs.orth_loss, prog_bar=False)
        self.log("train_expr_rank_loss", outputs.expr_rank_loss, prog_bar=False)
        self.log("train_aux_warmup", outputs.aux_warmup, prog_bar=False)
        self.log("train_structured_aux", weighted_struct_aux, prog_bar=False)
        self.log("train_expr_aux", weighted_expr_aux, prog_bar=False)

        residual_stats = self.model.summarize_residual_stats()
        self._expr_residual_norm_ema = self._update_ema(
            self._expr_residual_norm_ema, residual_stats["expr_residual_norm"]
        )
        self._style_residual_norm_ema = self._update_ema(
            self._style_residual_norm_ema, residual_stats["style_residual_norm"]
        )
        self._total_residual_norm_ema = self._update_ema(
            self._total_residual_norm_ema, residual_stats["total_residual_norm"]
        )

        pb_full = self.args.train_prog_bar_mode == "full"
        dev = outputs.loss.device
        self.log("total", outputs.loss.detach(), prog_bar=True, logger=False)
        self.log("match", outputs.match_loss.detach(), prog_bar=True, logger=False)
        self.log("style", outputs.style_neighbor_loss.detach(), prog_bar=pb_full, logger=False)
        self.log("orth", outputs.orth_loss.detach(), prog_bar=pb_full, logger=False)
        self.log("expr", outputs.expr_rank_loss.detach(), prog_bar=pb_full, logger=False)
        self.log("saux", weighted_struct_aux.detach(), prog_bar=pb_full, logger=False)
        self.log("eaux", weighted_expr_aux.detach(), prog_bar=pb_full, logger=False)
        self.log("wt", outputs.aux_warmup.detach(), prog_bar=pb_full, logger=False)
        self.log(
            "eres",
            torch.tensor(self._expr_residual_norm_ema, device=dev, dtype=torch.float32),
            prog_bar=True,
            logger=False,
        )
        self.log(
            "sres",
            torch.tensor(self._style_residual_norm_ema, device=dev, dtype=torch.float32),
            prog_bar=True,
            logger=False,
        )
        self.log(
            "rmod",
            torch.tensor(self._total_residual_norm_ema, device=dev, dtype=torch.float32),
            prog_bar=True,
            logger=False,
        )

        log_iv = int(self.args.struct_residual_log_interval or 0)
        if log_iv > 0 and int(self.global_step) % log_iv == 0:
            logger.info(
                "[StructuredResidualTrain] step=%d mode=%s cond=%s late_fusion=%s "
                "expr_residual_norm=%.4f style_residual_norm=%.4f total_residual_norm=%.4f "
                "sticker_mod_norm=%.4f expr_gain=%.4f style_gain=%.4f",
                int(self.global_step),
                self.args.struct_residual_mode,
                self.args.residual_condition_mode,
                "on" if self.args.struct_residual_use_late_fusion else "off",
                residual_stats["expr_residual_norm"],
                residual_stats["style_residual_norm"],
                residual_stats["total_residual_norm"],
                residual_stats["sticker_mod_norm"],
                residual_stats["expr_gain"],
                residual_stats["style_gain"],
            )

        return outputs.loss

    def on_validation_epoch_start(self) -> None:
        self.model.prepare_for_test()
        self._reset_eval_diagnostics()
        if getattr(self, "_per_epoch_dual_test_eval", False):
            self._eval_max_cand_pair = [0, 0]
        else:
            self._eval_max_cand_len = 0
        return pl.LightningModule.on_validation_epoch_start(self)

    def on_test_epoch_start(self) -> None:
        self.model.prepare_for_test()
        self._reset_eval_diagnostics()
        if getattr(self, "_per_epoch_dual_test_eval", False):
            self._eval_max_cand_pair = [0, 0]
        else:
            self._eval_max_cand_len = 0
        return pl.LightningModule.on_test_epoch_start(self)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        return StructuredPLModel.validation_step(self, batch, batch_idx, dataloader_idx)

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.validation_step(batch, batch_idx, 0)

    def on_validation_epoch_end(self) -> None:
        self._structured_eval_epoch_log_and_reset_metrics()
        diag = self._eval_diagnostics_summary()
        if diag:
            logger.info(
                "[StructuredResidualScale] "
                f"epoch={self.current_epoch} "
                f"mode={self.args.struct_residual_mode} "
                f"cond={self.args.residual_condition_mode} "
                f"late_fusion={'on' if self.args.struct_residual_use_late_fusion else 'off'} "
                f"n_batches={self.eval_diag_count} "
                f"plain_base_std={diag['plain_base_std']:.4f} "
                f"rank_std={diag['rank_score_std']:.4f} "
                f"delta_std={diag['residual_delta_std']:.4f} "
                f"delta/base={diag['residual_over_base_ratio']:.4f} "
                f"style_std={diag['style_score_std']:.4f} "
                f"expr_mod_std={diag['expr_mod_std']:.4f} "
                f"delta_abs={diag['residual_delta_abs_mean']:.4f} "
                f"top1_flip={diag['top1_flip_rate']:.4f} "
                f"expr_res={diag['expr_residual_norm']:.4f} "
                f"style_res={diag['style_residual_norm']:.4f} "
                f"total_res={diag['total_residual_norm']:.4f} "
                f"sticker_mod={diag['sticker_mod_norm']:.4f}"
            )
        self._reset_eval_diagnostics()
        return pl.LightningModule.on_validation_epoch_end(self)


def run_structured_residual_main(args: StructuredResidualArguments) -> None:
    if args.local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    pl.seed_everything(args.seed)
    if args.debug_smoke_test:
        logger.warning(
            "[StructuredResidualV3] debug_smoke_test is not wired for the new entrypoint; "
            "running the normal train/test path instead."
        )

    if args.mode in {"train", "pretrain"}:
        model = StructuredResidualPLModel(args)
        if args.ckpt_path:
            load_checkpoint_to_model(
                model, args.ckpt_path, strict=args.strict_checkpoint_load
            )
        datamodule = PLDataLoader(args, model.model.bert_tokenizer)
        trainer = build_trainer(args, for_train=True)
        attach_version_log_from_trainer(args, trainer)
        trainer.fit(model, datamodule=datamodule)
        return

    if args.mode in {"test", "gen"}:
        model = StructuredResidualPLModel(args)
        if not args.ckpt_path:
            raise ValueError("ckpt_path is required for test/gen mode.")
        load_checkpoint_to_model(
            model, args.ckpt_path, strict=args.strict_checkpoint_load
        )
        datamodule = PLDataLoader(args, model.model.bert_tokenizer)
        trainer = build_trainer(args, for_train=False)
        attach_test_log_to_ckpt_version(args)
        trainer.test(model, datamodule=datamodule)
        return

    raise ValueError(
        f"Unsupported mode={args.mode} for structured residual script. Use train/pretrain/test/gen."
    )


def parse_structured_residual_args(
    argv: Optional[List[str]] = None,
) -> StructuredResidualArguments:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_paths, rest = _extract_yaml_config_paths(argv)
    merged: Dict[str, Any] = {}
    for p in config_paths:
        merged = _deep_merge_dict(merged, load_structured_token_yaml_with_extends(p))
    prefix = _config_mapping_to_argv(merged)
    parser = HfArgumentParser(StructuredResidualArguments)
    (args,) = parser.parse_args_into_dataclasses(
        args=prefix + rest,
        look_for_args_file=False,
    )
    return args
