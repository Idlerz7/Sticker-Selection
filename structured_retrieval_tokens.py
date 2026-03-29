import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

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


@dataclass
class StructuredTokenArguments(StructuredArguments):
    pl_root_dir: Optional[str] = field(default="logs/structured_tokens")
    struct_token_mode: Optional[str] = field(default="expr")
    token_condition_mode: Optional[str] = field(default="none")
    struct_token_use_late_fusion: Optional[bool] = field(default=False)
    struct_token_compare_plain_eval: Optional[bool] = field(default=True)
    struct_token_log_interval: Optional[int] = field(default=200)
    # When True (default), append img_ff(CLIP) sticker_token before struct tokens (legacy layout).
    # When False and struct_token_mode != none, multimodal tail is only struct token(s) + SEP (no raw CLIP slot).
    struct_token_include_sticker_token: Optional[bool] = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        valid_token_modes = {"none", "expr", "style", "expr_style"}
        valid_condition_modes = {"none", "query_gate"}
        if self.struct_token_mode not in valid_token_modes:
            raise ValueError(
                f"Unsupported struct_token_mode={self.struct_token_mode}. "
                "Use 'none', 'expr', 'style', or 'expr_style'."
            )
        if self.token_condition_mode not in valid_condition_modes:
            raise ValueError(
                f"Unsupported token_condition_mode={self.token_condition_mode}. "
                "Use 'none' or 'query_gate'."
            )
        if self.struct_token_log_interval is not None and self.struct_token_log_interval < 0:
            raise ValueError("struct_token_log_interval must be >= 0 (0 = off).")
        if self.struct_token_mode == "none" and not self.struct_token_include_sticker_token:
            raise ValueError(
                "struct_token_include_sticker_token=False requires struct_token_mode != none "
                "(otherwise the multimodal tail would have no image-derived tokens)."
            )


class StructuredTokenStickerModel(StructuredStickerModel):
    def __init__(self, args: StructuredTokenArguments):
        super().__init__(args)
        self.args = args
        self.expr_token_proj = ProjectionHead(
            args.structured_hidden_dim, self.bert_hidden_dim, args.structured_dropout
        )
        self.style_token_proj = ProjectionHead(
            args.structured_hidden_dim, self.bert_hidden_dim, args.structured_dropout
        )
        self.expr_token_gate = nn.Linear(args.structured_hidden_dim * 2, self.bert_hidden_dim)
        self.style_token_gate = nn.Linear(args.structured_hidden_dim * 2, self.bert_hidden_dim)
        self._token_stats_buffer: List[Dict[str, float]] = []

    def _include_sticker_slot_in_sequence(self) -> bool:
        if self.args.struct_token_mode == "none":
            return True
        return bool(self.args.struct_token_include_sticker_token)

    def _active_token_names(self) -> List[str]:
        mode = self.args.struct_token_mode
        if mode == "none":
            return []
        if mode == "expr":
            return ["expr"]
        if mode == "style":
            return ["style"]
        if mode == "expr_style":
            return ["expr", "style"]
        raise ValueError(f"Unsupported struct_token_mode={mode}")

    def _reset_token_stats_buffer(self) -> None:
        self._token_stats_buffer = []

    def _record_token_stats(self, stats: Dict[str, float]) -> None:
        self._token_stats_buffer.append(stats)

    def summarize_token_stats(self) -> Dict[str, float]:
        if not self._token_stats_buffer:
            return {
                "expr_token_norm": 0.0,
                "style_token_norm": 0.0,
                "active_token_count": 0.0,
            }
        denom = float(len(self._token_stats_buffer))
        return {
            "expr_token_norm": sum(x["expr_token_norm"] for x in self._token_stats_buffer) / denom,
            "style_token_norm": sum(x["style_token_norm"] for x in self._token_stats_buffer) / denom,
            "active_token_count": sum(x["active_token_count"] for x in self._token_stats_buffer)
            / denom,
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

    def _condition_projected_token(
        self,
        name: str,
        source: torch.Tensor,
        projected: torch.Tensor,
        q: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.args.token_condition_mode != "query_gate" or q is None:
            return projected
        gate_layer = getattr(self, f"{name}_token_gate")
        gate = torch.sigmoid(gate_layer(torch.cat([source, q], dim=-1)))
        return projected * gate

    def _build_struct_tokens(
        self, img_emb: torch.Tensor, q: Optional[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        # `decompose_sticker` returns a shared latent plus style/expression branches.
        # We do not expose the shared latent as a "user token" in the non-personalized V2 path.
        _shared_h, c, a = self.decompose_sticker(img_emb)
        tokens: List[torch.Tensor] = []
        stats = {
            "expr_token_norm": 0.0,
            "style_token_norm": 0.0,
            "active_token_count": 0.0,
        }

        if "expr" in self._active_token_names():
            expr_token = self._condition_projected_token(
                "expr", a, self.expr_token_proj(a), q
            )
            tokens.append(expr_token.unsqueeze(1))
            stats["expr_token_norm"] = float(expr_token.detach().norm(dim=-1).mean().item())
            stats["active_token_count"] += 1.0

        if "style" in self._active_token_names():
            style_token = self._condition_projected_token(
                "style", c, self.style_token_proj(c), q
            )
            tokens.append(style_token.unsqueeze(1))
            stats["style_token_norm"] = float(style_token.detach().norm(dim=-1).mean().item())
            stats["active_token_count"] += 1.0

        return tokens, stats

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
        include_sticker = self._include_sticker_slot_in_sequence()
        sticker_token = self._project_sticker_token(img_emb) if include_sticker else None
        aux_emb = None
        if self.args.add_ocr_info:
            aux_emb = self._build_cls_inputs_emb(img_ids, device)

        q = None
        if self.args.token_condition_mode == "query_gate" and self.args.struct_token_mode != "none":
            q = self.encode_dialogue_query_from_text_emb(text_emb, attention_mask)
        struct_tokens, token_stats = self._build_struct_tokens(img_emb, q=q)

        sep_emb = self._build_sep_embeddings(batch_size=batch_size, device=device)
        input_parts = [text_emb]
        extra_len = len(struct_tokens) + 1 + (1 if sticker_token is not None else 0)
        if aux_emb is not None:
            input_parts.append(aux_emb)
            extra_len += aux_emb.size(1)
        if sticker_token is not None:
            input_parts.append(sticker_token)
        input_parts.extend(struct_tokens)
        input_parts.append(sep_emb)
        input_emb = torch.cat(input_parts, dim=1)
        ones_mask = torch.ones(batch_size, extra_len, device=device)
        full_attention_mask = torch.cat([attention_mask, ones_mask], dim=1)
        token_type_ids = torch.zeros_like(full_attention_mask, dtype=torch.long)
        token_type_ids[:, -extra_len:] = 1
        self._record_token_stats(token_stats)
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

        if self.args.struct_token_use_late_fusion:
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

        token_backbone_logits = self._compute_pair_logits(
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            img_ids=candidate_ids,
            img_emb=candidate_h,
        )
        token_backbone_score = self.compute_base_score(token_backbone_logits)

        plain_base_score = token_backbone_score
        plain_logits = None
        if self.args.struct_token_compare_plain_eval and self.args.struct_token_mode != "none":
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
            token_backbone_score, style_score, expr_score, style_gate
        )

        rank_scores = final_score.unsqueeze(0)
        labels = torch.tensor(img_ids, dtype=torch.long, device=device)
        if not return_debug:
            return rank_scores, labels, candidate_ids if use_cands else None

        token_stats = self.summarize_token_stats()
        token_delta = final_score - plain_base_score
        if self.args.struct_token_use_late_fusion:
            ranking_formula = (
                "s_rank = s_backbone(tokens) + lambda_fuse * structured_term; "
                "compare_base = s_backbone(no_extra_tokens)"
            )
        else:
            ranking_formula = (
                "s_rank = s_backbone(tokens_only); "
                "compare_base = s_backbone(no_extra_tokens)"
            )

        eval_debug = {
            "plain_base_score_stats": self._tensor_debug_stats(plain_base_score),
            "rank_score_stats": self._tensor_debug_stats(final_score),
            "token_delta_score_stats": self._tensor_debug_stats(token_delta),
            "style_score_stats": self._tensor_debug_stats(style_score),
            "expr_mod_stats": self._tensor_debug_stats(expr_mod),
            "structured_term_stats": self._tensor_debug_stats(structured_term),
            "token_backbone_score_stats": self._tensor_debug_stats(token_backbone_score),
            "token_delta_abs_mean": float(token_delta.abs().mean().item()),
            "top1_flipped": bool(
                torch.argmax(plain_base_score).item() != torch.argmax(final_score).item()
            ),
            "ranking_score_formula": ranking_formula,
            "candidate_count": int(len(candidate_ids)),
            "uses_late_fusion": bool(self.args.struct_token_use_late_fusion),
            "struct_token_include_sticker_token": bool(self.args.struct_token_include_sticker_token),
            "struct_token_mode": self.args.struct_token_mode,
            "token_condition_mode": self.args.token_condition_mode,
            "token_stats": token_stats,
            "plain_logits_stats": self._tensor_debug_stats(plain_logits)
            if plain_logits is not None
            else None,
        }
        return rank_scores, labels, candidate_ids if use_cands else None, eval_debug


class StructuredTokenPLModel(StructuredPLModel):
    def __init__(self, args: StructuredTokenArguments):
        pl.LightningModule.__init__(self)
        self.args = args
        self.model = StructuredTokenStickerModel(args)
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

        self._expr_token_norm_ema: Optional[float] = None
        self._style_token_norm_ema: Optional[float] = None
        self._reset_eval_diagnostics()

    def _reset_eval_diagnostics(self) -> None:
        self.eval_diag_count = 0
        self.eval_plain_base_std_sum = 0.0
        self.eval_rank_score_std_sum = 0.0
        self.eval_token_delta_std_sum = 0.0
        self.eval_style_score_std_sum = 0.0
        self.eval_expr_mod_std_sum = 0.0
        self.eval_token_delta_abs_mean_sum = 0.0
        self.eval_top1_flip_count = 0
        self.eval_expr_token_norm_sum = 0.0
        self.eval_style_token_norm_sum = 0.0

    def _update_eval_diagnostics(self, eval_debug: Dict[str, Any]) -> None:
        self.eval_diag_count += 1
        self.eval_plain_base_std_sum += float(eval_debug["plain_base_score_stats"]["std"])
        self.eval_rank_score_std_sum += float(eval_debug["rank_score_stats"]["std"])
        self.eval_token_delta_std_sum += float(eval_debug["token_delta_score_stats"]["std"])
        self.eval_style_score_std_sum += float(eval_debug["style_score_stats"]["std"])
        self.eval_expr_mod_std_sum += float(eval_debug["expr_mod_stats"]["std"])
        self.eval_token_delta_abs_mean_sum += float(eval_debug["token_delta_abs_mean"])
        self.eval_top1_flip_count += int(bool(eval_debug["top1_flipped"]))
        self.eval_expr_token_norm_sum += float(eval_debug["token_stats"]["expr_token_norm"])
        self.eval_style_token_norm_sum += float(eval_debug["token_stats"]["style_token_norm"])

    def _eval_diagnostics_summary(self) -> Dict[str, float]:
        if self.eval_diag_count <= 0:
            return {}
        denom = float(self.eval_diag_count)
        plain_base_std = self.eval_plain_base_std_sum / denom
        token_delta_std = self.eval_token_delta_std_sum / denom
        return {
            "plain_base_std": plain_base_std,
            "rank_score_std": self.eval_rank_score_std_sum / denom,
            "token_delta_std": token_delta_std,
            "token_over_base_ratio": token_delta_std / max(plain_base_std, 1e-8),
            "style_score_std": self.eval_style_score_std_sum / denom,
            "expr_mod_std": self.eval_expr_mod_std_sum / denom,
            "token_delta_abs_mean": self.eval_token_delta_abs_mean_sum / denom,
            "top1_flip_rate": self.eval_top1_flip_count / denom,
            "expr_token_norm": self.eval_expr_token_norm_sum / denom,
            "style_token_norm": self.eval_style_token_norm_sum / denom,
        }

    def _update_ema(self, old: Optional[float], value: float, alpha: float = 0.05) -> float:
        if old is None:
            return value
        return alpha * value + (1.0 - alpha) * old

    def on_train_start(self) -> None:
        logger.info(
            "[StructuredTokenV2] mode=%s cond=%s late_fusion=%s compare_plain_eval=%s "
            "include_sticker_token=%s",
            self.args.struct_token_mode,
            self.args.token_condition_mode,
            "on" if self.args.struct_token_use_late_fusion else "off",
            bool(self.args.struct_token_compare_plain_eval),
            bool(self.args.struct_token_include_sticker_token),
        )
        logger.info(
            "[StructuredTokenV2] Compact tqdm shows token norms (etok/stok). "
            "Aux losses keep the same train_* logging as the reference run."
        )
        return pl.LightningModule.on_train_start(self)

    def run_train_batch(self, batch: Dict[str, Any]) -> StructuredForwardOutput:
        self.model._reset_token_stats_buffer()
        return self.model.forward_train_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            img_ids=batch["img_ids"],
            neg_img_ids=batch["neg_img_ids"],
            global_step=int(self.global_step),
            total_steps=int(self.num_training_steps),
        )

    def run_eval_batch(self, batch: Dict[str, Any], return_debug: bool = False):
        self.model._reset_token_stats_buffer()
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

        token_stats = self.model.summarize_token_stats()
        self._expr_token_norm_ema = self._update_ema(
            self._expr_token_norm_ema, token_stats["expr_token_norm"]
        )
        self._style_token_norm_ema = self._update_ema(
            self._style_token_norm_ema, token_stats["style_token_norm"]
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
            "etok",
            torch.tensor(self._expr_token_norm_ema, device=dev, dtype=torch.float32),
            prog_bar=True,
            logger=False,
        )
        self.log(
            "stok",
            torch.tensor(self._style_token_norm_ema, device=dev, dtype=torch.float32),
            prog_bar=True,
            logger=False,
        )

        log_iv = int(self.args.struct_token_log_interval or 0)
        if log_iv > 0 and int(self.global_step) % log_iv == 0:
            logger.info(
                "[StructuredTokenTrain] step=%d mode=%s cond=%s late_fusion=%s "
                "expr_token_norm=%.4f style_token_norm=%.4f",
                int(self.global_step),
                self.args.struct_token_mode,
                self.args.token_condition_mode,
                "on" if self.args.struct_token_use_late_fusion else "off",
                token_stats["expr_token_norm"],
                token_stats["style_token_norm"],
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
            sticker_slot = (
                "on"
                if self.args.struct_token_mode == "none" or self.args.struct_token_include_sticker_token
                else "off"
            )
            logger.info(
                "[StructuredTokenScale] "
                f"epoch={self.current_epoch} "
                f"mode={self.args.struct_token_mode} "
                f"cond={self.args.token_condition_mode} "
                f"sticker_slot={sticker_slot} "
                f"late_fusion={'on' if self.args.struct_token_use_late_fusion else 'off'} "
                f"n_batches={self.eval_diag_count} "
                f"plain_base_std={diag['plain_base_std']:.4f} "
                f"rank_std={diag['rank_score_std']:.4f} "
                f"delta_std={diag['token_delta_std']:.4f} "
                f"delta/base={diag['token_over_base_ratio']:.4f} "
                f"style_std={diag['style_score_std']:.4f} "
                f"expr_mod_std={diag['expr_mod_std']:.4f} "
                f"delta_abs={diag['token_delta_abs_mean']:.4f} "
                f"top1_flip={diag['top1_flip_rate']:.4f} "
                f"expr_tok={diag['expr_token_norm']:.4f} "
                f"style_tok={diag['style_token_norm']:.4f}"
            )
        self._reset_eval_diagnostics()
        return pl.LightningModule.on_validation_epoch_end(self)


def run_structured_tokens_main(args: StructuredTokenArguments) -> None:
    if args.local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    pl.seed_everything(args.seed)
    if args.debug_smoke_test:
        logger.warning(
            "[StructuredTokenV2] debug_smoke_test is not wired for the new entrypoint; "
            "running the normal train/test path instead."
        )

    if args.mode in {"train", "pretrain"}:
        model = StructuredTokenPLModel(args)
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
        model = StructuredTokenPLModel(args)
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
        f"Unsupported mode={args.mode} for structured token script. Use train/pretrain/test/gen."
    )


# Keys allowed in YAML but not passed to HfArgumentParser (metadata / inheritance).
_YAML_META_KEYS = frozenset({"extends", "description", "name"})


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in _YAML_META_KEYS:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_structured_token_yaml_with_extends(
    path: str, _seen: Optional[set] = None
) -> Dict[str, Any]:
    path = os.path.abspath(path)
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Circular YAML extends chain involving {path}")
    _seen.add(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    extends = raw.get("extends")
    base: Dict[str, Any] = {}
    if extends:
        ext_path = extends if os.path.isabs(extends) else os.path.join(os.path.dirname(path), extends)
        base = load_structured_token_yaml_with_extends(ext_path, _seen)
    _seen.remove(path)
    merged = _deep_merge_dict(base, raw)
    merged.pop("extends", None)
    return merged


def _extract_yaml_config_paths(argv: List[str]) -> Tuple[List[str], List[str]]:
    configs: List[str] = []
    rest: List[str] = []
    i = 0
    n = len(argv)
    while i < n:
        if argv[i] == "--config":
            i += 1
            if i >= n or argv[i].startswith("-"):
                raise ValueError("--config requires at least one YAML path")
            while i < n and not argv[i].startswith("-"):
                configs.append(argv[i])
                i += 1
        else:
            rest.append(argv[i])
            i += 1
    return configs, rest


def _yaml_value_to_argv(key: str, value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return [f"--{key}", "true" if value else "false"]
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:  # NaN
            raise ValueError(f"Invalid float for --{key}")
        return [f"--{key}", str(value)]
    if isinstance(value, str):
        return [f"--{key}", value]
    if isinstance(value, dict):
        raise ValueError(
            f"Nested mapping for field '{key}' is not supported as a CLI flag; "
            "flatten the key or use a single-level dict merged at root."
        )
    return [f"--{key}", str(value)]


def _config_mapping_to_argv(cfg: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for key in sorted(cfg.keys()):
        if key in _YAML_META_KEYS:
            continue
        argv.extend(_yaml_value_to_argv(key, cfg[key]))
    return argv


def parse_structured_token_args(argv: Optional[List[str]] = None) -> StructuredTokenArguments:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_paths, rest = _extract_yaml_config_paths(argv)
    merged: Dict[str, Any] = {}
    for p in config_paths:
        merged = _deep_merge_dict(merged, load_structured_token_yaml_with_extends(p))
    prefix = _config_mapping_to_argv(merged)
    parser = HfArgumentParser(StructuredTokenArguments)
    (args,) = parser.parse_args_into_dataclasses(
        args=prefix + rest,
        look_for_args_file=False,
    )
    return args
