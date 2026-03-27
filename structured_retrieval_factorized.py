import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Work around protobuf C-extension segfault in some environments
# (must be set before importing pytorch_lightning/tensorboard).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from transformers import HfArgumentParser

from factorized_style_bank import FactorizedStyleBank
from main import (
    PLDataLoader,
    attach_test_log_to_ckpt_version,
    attach_version_log_from_trainer,
    logger,
)
from metrics import MyAccuracy
from structured_retrieval import (
    ProjectionHead,
    ScalarMatchHead,
    StructuredArguments,
    StructuredPLModel,
    StructuredStickerModel,
    _CAND_EVAL_ONLY_ERR,
    attach_per_epoch_dual_test_eval,
    build_trainer,
    load_checkpoint_to_model,
    maybe_set_ddp_static_graph,
    nonempty_batch_cands,
    normalize_sticker_id,
)
from structured_retrieval_tokens import (
    _config_mapping_to_argv,
    _deep_merge_dict,
    _extract_yaml_config_paths,
    load_structured_token_yaml_with_extends,
)


@dataclass
class StructuredFactorizedArguments(StructuredArguments):
    pl_root_dir: Optional[str] = field(default="logs/structured_factorized")
    structured_result_dir: Optional[str] = field(default="./result/structured_factorized")

    factorized_bank_path: Optional[str] = field(default="./factorized_style_bank.json")
    factorized_bank_source: Optional[str] = field(default="pseudo_labels")
    factorized_pseudo_label_path: Optional[str] = field(
        default=(
            "./pseudo_labels/"
            "sticker_identity_style_labels_from_data_meme_set_model_gemini-3-pro-preview_date_20260321.jsonl"
        )
    )
    factorized_style_metadata_path: Optional[str] = field(default="")
    factorized_min_fine_proto_size: Optional[int] = field(default=2)
    factorized_min_coarse_proto_size: Optional[int] = field(default=2)

    factorized_style_recall_topk: Optional[int] = field(default=32)
    factorized_style_proto_topk: Optional[int] = field(default=4)
    factorized_proto_expand_per_proto: Optional[int] = field(default=12)
    factorized_mmbert_branch_topk: Optional[int] = field(default=16)
    factorized_candidate_union_topk: Optional[int] = field(default=64)
    factorized_use_mmbert_branch: Optional[bool] = field(default=True)
    factorized_eval_compare_mmbert: Optional[bool] = field(default=True)

    factorized_fusion_hidden_dim: Optional[int] = field(default=32)
    factorized_proto_logit_scale: Optional[float] = field(default=10.0)
    factorized_log_interval: Optional[int] = field(default=200)
    factorized_train_bank_refresh_steps: Optional[int] = field(default=0)
    # true = one MM-BERT forward at batch 3B; false = three forwards at batch B (default, matches older runs).
    factorized_fused_train_mmbert: Optional[bool] = field(default=False)
    # Turn off only for debugging speed; training memory is much higher without checkpointing.
    bert_gradient_checkpointing: Optional[bool] = field(default=True)
    # First N global steps: log mean CUDA section times for forward_train_batch (ms/step).
    # Env STICKER_FACTORIZED_PROFILE_STEPS overrides if larger.
    factorized_profile_train_steps: Optional[int] = field(default=0)
    # 2-class match loss (pos vs one neg): skips one MM-BERT forward per sample vs full 3-way CE.
    factorized_train_mmbert_two_way: Optional[bool] = field(default=False)
    # Which negative is scored against pos when two_way: "same" (same-proto neg) or "cross" (cross-proto neg).
    factorized_train_two_way_neg: Optional[str] = field(default="same")

    lambda_style_proto: Optional[float] = field(default=0.4)
    train_same_proto_negatives: Optional[int] = field(default=1)
    train_cross_proto_negatives: Optional[int] = field(default=1)

    def __post_init__(self):
        super().__post_init__()
        positive_int_fields = {
            "factorized_style_recall_topk": self.factorized_style_recall_topk,
            "factorized_style_proto_topk": self.factorized_style_proto_topk,
            "factorized_proto_expand_per_proto": self.factorized_proto_expand_per_proto,
            "factorized_mmbert_branch_topk": self.factorized_mmbert_branch_topk,
            "factorized_candidate_union_topk": self.factorized_candidate_union_topk,
            "factorized_fusion_hidden_dim": self.factorized_fusion_hidden_dim,
            "factorized_min_fine_proto_size": self.factorized_min_fine_proto_size,
            "factorized_min_coarse_proto_size": self.factorized_min_coarse_proto_size,
            "train_same_proto_negatives": self.train_same_proto_negatives,
            "train_cross_proto_negatives": self.train_cross_proto_negatives,
        }
        for field_name, value in positive_int_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{field_name} must be > 0.")
        if float(self.lambda_style_proto) < 0:
            raise ValueError("lambda_style_proto must be >= 0.")
        if float(self.factorized_proto_logit_scale) <= 0:
            raise ValueError("factorized_proto_logit_scale must be > 0.")
        if self.factorized_log_interval is not None and int(self.factorized_log_interval) < 0:
            raise ValueError("factorized_log_interval must be >= 0 (0 = off).")
        if (
            self.factorized_train_bank_refresh_steps is not None
            and int(self.factorized_train_bank_refresh_steps) < 0
        ):
            raise ValueError("factorized_train_bank_refresh_steps must be >= 0 (0 = exact per-step).")
        if self.bert_gradient_checkpointing is None:
            self.bert_gradient_checkpointing = True
        if self.factorized_fused_train_mmbert is None:
            self.factorized_fused_train_mmbert = False
        if self.factorized_profile_train_steps is not None and int(self.factorized_profile_train_steps) < 0:
            raise ValueError("factorized_profile_train_steps must be >= 0.")
        if self.factorized_train_mmbert_two_way is None:
            self.factorized_train_mmbert_two_way = False
        tw = str(self.factorized_train_two_way_neg or "same").strip().lower()
        if tw not in {"same", "cross"}:
            raise ValueError("factorized_train_two_way_neg must be 'same' or 'cross'.")
        self.factorized_train_two_way_neg = tw
        if self.factorized_bank_source not in {"pseudo_labels", "style_metadata"}:
            raise ValueError(
                "factorized_bank_source must be one of {'pseudo_labels', 'style_metadata'}."
            )


@dataclass
class StructuredFactorizedForwardOutput:
    loss: torch.Tensor
    match_loss: torch.Tensor
    style_proto_loss: torch.Tensor
    style_neighbor_loss: torch.Tensor
    orth_loss: torch.Tensor
    expr_rank_loss: torch.Tensor
    aux_warmup: torch.Tensor
    pos_fused_score: torch.Tensor
    cross_fused_score: torch.Tensor
    same_fused_score: torch.Tensor
    style_usage_gate: torch.Tensor
    debug_info: Optional[Dict[str, Any]] = None


class StructuredFactorizedStickerModel(StructuredStickerModel):
    def __init__(self, args: StructuredFactorizedArguments):
        super().__init__(args)
        self.args = args
        self.style_query_head = ProjectionHead(
            self.bert_hidden_dim, args.structured_hidden_dim, args.structured_dropout
        )
        self.expr_query_head = ProjectionHead(
            self.bert_hidden_dim, args.structured_hidden_dim, args.structured_dropout
        )
        self.style_usage_gate_head = nn.Sequential(
            nn.Linear(args.structured_hidden_dim * 2, args.structured_hidden_dim),
            nn.Tanh(),
            nn.Dropout(args.structured_dropout),
            nn.Linear(args.structured_hidden_dim, 1),
        )
        self.prototype_match_head = ScalarMatchHead(
            args.structured_hidden_dim, args.structured_dropout
        )
        # Factorized training runs BERT many times per step (query encoder + 3×pair); checkpointing
        # cuts activation VRAM. One fused 3×B forward is worse for attention memory than three B-sized passes.
        if getattr(args, "bert_gradient_checkpointing", True):
            if hasattr(self.bert, "gradient_checkpointing_enable"):
                self.bert.gradient_checkpointing_enable()
                logger.info(
                    "[StructuredFactorized] BERT gradient checkpointing enabled (lower VRAM, slower step)."
                )
            else:
                self.bert.config.gradient_checkpointing = True
                logger.info(
                    "[StructuredFactorized] Set config.gradient_checkpointing=True on BERT."
                )
        else:
            logger.info(
                "[StructuredFactorized] BERT gradient checkpointing disabled "
                "(faster steps, much higher VRAM; avoid fused 3× MM-BERT if OOM)."
            )
        self.fusion_head = nn.Sequential(
            nn.Linear(5, args.factorized_fusion_hidden_dim),
            nn.Tanh(),
            nn.Dropout(args.structured_dropout),
            nn.Linear(args.factorized_fusion_hidden_dim, 1),
        )
        self._factorized_bank_rng = torch.Generator().manual_seed(int(args.seed))
        self._python_bank_rng = random.Random(int(args.seed))
        self.style_bank = self._load_or_build_style_bank()
        self._bank_img_embs_cpu: Optional[torch.Tensor] = None
        self._bank_img_embs_by_device: Dict[str, torch.Tensor] = {}
        self._eval_bank_factorization_cache: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self._eval_bank_factorization_device: Optional[str] = None
        self._train_bank_factorization_cache: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self._train_bank_factorization_device: Optional[str] = None
        self._train_bank_factorization_step: Optional[int] = None
        self._proto_reduce_cache_by_device: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._sticker_to_proto_cpu = self._build_sticker_to_proto_cpu()
        self._train_profile_sums: Optional[Dict[str, float]] = None
        self._train_profile_count: int = 0

    def _factorized_profile_limit(self) -> int:
        a = int(getattr(self.args, "factorized_profile_train_steps", 0) or 0)
        try:
            e = int(os.environ.get("STICKER_FACTORIZED_PROFILE_STEPS", "0") or "0")
        except ValueError:
            e = 0
        return max(a, e)

    def _accumulate_train_profile_ms(
        self,
        global_step: int,
        limit: int,
        ms_enc: float,
        ms_mid: float,
        ms_mmbert: float,
        ms_tail: float,
    ) -> None:
        if self._train_profile_sums is None:
            self._train_profile_sums = {"enc": 0.0, "mid": 0.0, "mmbert": 0.0, "tail": 0.0}
        self._train_profile_sums["enc"] += ms_enc
        self._train_profile_sums["mid"] += ms_mid
        self._train_profile_sums["mmbert"] += ms_mmbert
        self._train_profile_sums["tail"] += ms_tail
        self._train_profile_count += 1
        if global_step == limit - 1 and self._train_profile_count > 0:
            n = float(self._train_profile_count)
            s = self._train_profile_sums
            tot = (s["enc"] + s["mid"] + s["mmbert"] + s["tail"]) / n
            logger.info(
                "[StructuredFactorizedProfile] mean ms/step over steps 0..%d (n=%d): "
                "encode_style_expr=%.2f bank_proto_neg_emb=%.2f mmbert=%.2f tail_rest=%.2f total=%.2f",
                limit - 1,
                self._train_profile_count,
                s["enc"] / n,
                s["mid"] / n,
                s["mmbert"] / n,
                s["tail"] / n,
                tot,
            )
            self._train_profile_sums = None
            self._train_profile_count = 0

    def _load_or_build_style_bank(self) -> FactorizedStyleBank:
        bank_path = (self.args.factorized_bank_path or "").strip()
        if bank_path and os.path.exists(bank_path):
            logger.info("[StructuredFactorized] Loading style bank from %s", bank_path)
            return FactorizedStyleBank.from_json(bank_path)
        if self.args.factorized_bank_source == "style_metadata":
            logger.info(
                "[StructuredFactorized] Building style bank from style metadata=%s and neighbors=%s",
                self.args.factorized_style_metadata_path,
                self.args.style_neighbors_path,
            )
            bank = FactorizedStyleBank.from_style_metadata(
                style_metadata_path=self.args.factorized_style_metadata_path,
                style_neighbors_path=self.args.style_neighbors_path,
                max_image_id=self.args.max_image_id,
            )
        else:
            logger.info(
                "[StructuredFactorized] Building style bank from pseudo labels=%s and neighbors=%s",
                self.args.factorized_pseudo_label_path,
                self.args.style_neighbors_path,
            )
            bank = FactorizedStyleBank.from_assets(
                pseudo_label_path=self.args.factorized_pseudo_label_path,
                style_neighbors_path=self.args.style_neighbors_path,
                max_image_id=self.args.max_image_id,
                min_fine_proto_size=self.args.factorized_min_fine_proto_size,
                min_coarse_proto_size=self.args.factorized_min_coarse_proto_size,
            )
        if bank_path:
            bank.save_json(bank_path)
            logger.info("[StructuredFactorized] Saved style bank to %s", bank_path)
        return bank

    def _ensure_bank_img_embs_cpu(self) -> torch.Tensor:
        if self._bank_img_embs_cpu is not None:
            return self._bank_img_embs_cpu
        cache_path = (getattr(self.args, "img_emb_cache_path", None) or "").strip()
        if cache_path and os.path.exists(cache_path):
            self._bank_img_embs_cpu = torch.load(cache_path, map_location="cpu").float()
            return self._bank_img_embs_cpu
        self.prepare_for_test()
        if not hasattr(self, "all_img_embs") or self.all_img_embs is None:
            raise RuntimeError("Unable to prepare image embeddings for factorized style bank.")
        self._bank_img_embs_cpu = self.all_img_embs.detach().cpu().float()
        return self._bank_img_embs_cpu

    def _build_sticker_to_proto_cpu(self) -> torch.Tensor:
        num_stickers = int(self.args.max_image_id)
        mapping = torch.empty(num_stickers, dtype=torch.long)
        for record in self.style_bank.records:
            mapping[int(record.sticker_id)] = int(record.proto_id)
        return mapping

    def _get_sticker_to_proto(self, device: torch.device) -> torch.Tensor:
        cache_key = self._device_cache_key(device)
        cached = getattr(self, "_sticker_to_proto_by_device", {}).get(cache_key)
        if cached is not None:
            return cached
        if not hasattr(self, "_sticker_to_proto_by_device"):
            self._sticker_to_proto_by_device: Dict[str, torch.Tensor] = {}
        mapping = self._sticker_to_proto_cpu.to(device)
        self._sticker_to_proto_by_device[cache_key] = mapping
        return mapping

    def _get_proto_reduce_tensors(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = self._device_cache_key(device)
        cached = self._proto_reduce_cache_by_device.get(cache_key)
        if cached is not None:
            return cached
        member_ids: List[int] = []
        proto_ids: List[int] = []
        proto_counts: List[float] = []
        proto_density: List[float] = []
        for proto in self.style_bank.prototypes:
            pid = int(proto.proto_id)
            members = [int(x) for x in proto.member_ids]
            member_ids.extend(members)
            proto_ids.extend([pid] * len(members))
            proto_counts.append(float(len(members)))
            proto_density.append(float(proto.proto_density))
        tensors = (
            torch.tensor(member_ids, dtype=torch.long, device=device),
            torch.tensor(proto_ids, dtype=torch.long, device=device),
            torch.tensor(proto_counts, dtype=torch.float32, device=device),
            torch.tensor(proto_density, dtype=torch.float32, device=device),
        )
        self._proto_reduce_cache_by_device[cache_key] = tensors
        return tensors

    @staticmethod
    def _device_cache_key(device: torch.device) -> str:
        dev = torch.device(device)
        return dev.type if dev.index is None else f"{dev.type}:{dev.index}"

    def get_factorized_bank_img_embs(self, device: torch.device) -> torch.Tensor:
        cache_key = self._device_cache_key(device)
        cached = self._bank_img_embs_by_device.get(cache_key)
        if cached is not None:
            return cached
        live_bank = getattr(self, "all_img_embs", None)
        if live_bank is not None:
            target_device = torch.device(device)
            if live_bank.device != target_device:
                live_bank = live_bank.to(target_device)
                self.all_img_embs = live_bank
            self._bank_img_embs_by_device[cache_key] = live_bank
            return live_bank
        bank = self._ensure_bank_img_embs_cpu().to(device)
        self._bank_img_embs_by_device[cache_key] = bank
        return bank

    def clear_eval_factorization_cache(self) -> None:
        self._eval_bank_factorization_cache = None
        self._eval_bank_factorization_device = None

    def clear_train_factorization_cache(self) -> None:
        self._train_bank_factorization_cache = None
        self._train_bank_factorization_device = None
        self._train_bank_factorization_step = None

    def prepare_eval_factorization_cache(self, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = next(self.bert.parameters()).device
        cache_key = self._device_cache_key(device)
        if (
            self._eval_bank_factorization_cache is not None
            and self._eval_bank_factorization_device == cache_key
        ):
            return
        with torch.no_grad():
            self._eval_bank_factorization_cache = self._compute_bank_factorization(device)
        self._eval_bank_factorization_device = cache_key

    def _get_eval_or_fresh_bank_factorization(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = self._device_cache_key(device)
        if (
            self._eval_bank_factorization_cache is not None
            and self._eval_bank_factorization_device == cache_key
        ):
            return self._eval_bank_factorization_cache
        return self._compute_bank_factorization(device)

    def _get_train_or_fresh_bank_factorization(
        self, device: torch.device, global_step: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        refresh_steps = int(getattr(self.args, "factorized_train_bank_refresh_steps", 0) or 0)
        if refresh_steps <= 0:
            return self._compute_bank_factorization(device)
        cache_key = self._device_cache_key(device)
        should_refresh = (
            self._train_bank_factorization_cache is None
            or self._train_bank_factorization_device != cache_key
            or self._train_bank_factorization_step is None
            or int(global_step) < int(self._train_bank_factorization_step)
            or (int(global_step) - int(self._train_bank_factorization_step)) >= refresh_steps
        )
        if should_refresh:
            with torch.no_grad():
                self._train_bank_factorization_cache = self._compute_bank_factorization(device)
            self._train_bank_factorization_device = cache_key
            self._train_bank_factorization_step = int(global_step)
        return self._train_bank_factorization_cache

    def encode_style_expr_queries(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_emb = self._get_text_word_embeddings(input_ids)
        outputs = self.bert.bert(
            inputs_embeds=text_emb,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        q_style = self.style_query_head(cls)
        q_expr = self.expr_query_head(cls)
        style_usage_gate = torch.sigmoid(
            self.style_usage_gate_head(torch.cat([q_style, q_expr], dim=-1))
        ).squeeze(-1)
        return q_style, q_expr, style_usage_gate

    def _compute_proto_vectors(
        self, style_bank_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.style_bank.prototypes:
            zero_proto = style_bank_c.new_zeros((1, style_bank_c.size(-1)))
            zero_density = style_bank_c.new_zeros((1,))
            return zero_proto, zero_density
        member_ids, proto_ids, proto_counts, proto_density = self._get_proto_reduce_tensors(
            style_bank_c.device
        )
        member_feats = style_bank_c.index_select(0, member_ids)
        proto_vectors = style_bank_c.new_zeros((len(self.style_bank.prototypes), style_bank_c.size(-1)))
        proto_vectors.index_add_(0, proto_ids, member_feats)
        proto_vectors = proto_vectors / proto_counts.to(dtype=style_bank_c.dtype).unsqueeze(-1).clamp_min(1.0)
        return proto_vectors, proto_density.to(dtype=style_bank_c.dtype)

    def _compute_bank_factorization(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_h = self.get_factorized_bank_img_embs(device)
        _, style_bank_c, style_bank_a = self.decompose_sticker(all_h)
        proto_vectors, proto_density = self._compute_proto_vectors(style_bank_c)
        return all_h, style_bank_c, style_bank_a, proto_vectors, proto_density

    def _compute_proto_logits(
        self, q_style: torch.Tensor, proto_vectors: torch.Tensor
    ) -> torch.Tensor:
        if proto_vectors.ndim == 1:
            proto_vectors = proto_vectors.unsqueeze(0)
        norm_q = F.normalize(q_style, dim=-1)
        norm_p = F.normalize(proto_vectors, dim=-1)
        scale = float(self.args.factorized_proto_logit_scale)
        return scale * torch.matmul(norm_q, norm_p.transpose(0, 1))

    def _gather_proto_scores_for_batch(
        self,
        proto_logits: torch.Tensor,
        proto_density: torch.Tensor,
        sticker_ids: Sequence[int],
    ) -> torch.Tensor:
        sticker_to_proto = self._get_sticker_to_proto(proto_logits.device)
        sticker_idx = torch.tensor(list(sticker_ids), dtype=torch.long, device=proto_logits.device)
        proto_ids = sticker_to_proto.index_select(0, sticker_idx)
        batch_idx = torch.arange(proto_logits.size(0), device=proto_logits.device)
        gathered = proto_logits[batch_idx, proto_ids]
        return gathered * proto_density.index_select(0, proto_ids)

    def _gather_proto_scores_for_candidates(
        self,
        proto_logit_row: torch.Tensor,
        proto_density: torch.Tensor,
        candidate_ids: Sequence[int],
    ) -> torch.Tensor:
        sticker_to_proto = self._get_sticker_to_proto(proto_logit_row.device)
        sticker_idx = torch.tensor(list(candidate_ids), dtype=torch.long, device=proto_logit_row.device)
        proto_ids = sticker_to_proto.index_select(0, sticker_idx)
        return proto_logit_row.index_select(0, proto_ids) * proto_density.index_select(0, proto_ids)

    def _fuse_scores(
        self,
        mmbert_score: torch.Tensor,
        style_score: torch.Tensor,
        expr_score: torch.Tensor,
        graph_score: torch.Tensor,
        style_usage_gate: torch.Tensor,
    ) -> torch.Tensor:
        gate = style_usage_gate
        while gate.ndim < mmbert_score.ndim:
            gate = gate.unsqueeze(-1)
        gate = gate.expand_as(mmbert_score)
        stacked = torch.stack(
            [mmbert_score, style_score, expr_score, graph_score, gate], dim=-1
        )
        fused = self.fusion_head(stacked).squeeze(-1)
        if self.args.base_only:
            return mmbert_score
        return fused

    def _resolve_cross_proto_negatives(
        self, pos_ids: Sequence[int], fallback_neg_ids: Sequence[int]
    ) -> List[int]:
        resolved: List[int] = []
        for pos_id, neg_id in zip(pos_ids, fallback_neg_ids):
            pos_proto = self.style_bank.proto_id_of(int(pos_id))
            neg_proto = self.style_bank.proto_id_of(int(neg_id))
            if pos_proto != neg_proto:
                resolved.append(int(neg_id))
                continue
            sampled = self.style_bank.sample_cross_proto_negative(
                int(pos_id), self._python_bank_rng, exclude_ids=[int(neg_id)]
            )
            resolved.append(int(sampled) if sampled is not None else int(neg_id))
        return resolved

    def _resolve_same_proto_negatives(
        self, pos_ids: Sequence[int], fallback_neg_ids: Sequence[int]
    ) -> List[int]:
        resolved: List[int] = []
        for pos_id, neg_id in zip(pos_ids, fallback_neg_ids):
            sampled = self.style_bank.sample_same_proto_negative(
                int(pos_id), self._python_bank_rng, exclude_ids=[int(neg_id)]
            )
            resolved.append(int(sampled) if sampled is not None else int(neg_id))
        return resolved

    def _compute_style_neighbor_loss_with_meta(
        self, pos_img_ids: Sequence[int], c_i: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        debug_meta: Dict[str, Any] = {
            "valid_style_neighbor_pairs": 0,
            "total_positive_samples": len(pos_img_ids),
            "missing_neighbor_img_ids": [],
            "sampled_neighbor_pairs_preview": [],
        }
        if self.args.lambda_struct <= 0 or self.args.style_neighbor_topk <= 0:
            return c_i.new_zeros(()), debug_meta

        valid_indices: List[int] = []
        valid_neighbor_ids: List[int] = []
        sampled_pairs: List[Tuple[int, int]] = []

        for idx, img_id in enumerate(pos_img_ids):
            sampled = self.neighbor_store.sample(
                sticker_id=img_id,
                topk=self.args.style_neighbor_topk,
                sampling_mode=self.args.style_sampling_mode,
                rng=self._neighbor_rng,
            )
            if sampled is None:
                debug_meta["missing_neighbor_img_ids"].append(int(img_id))
                continue
            normalized = normalize_sticker_id(sampled)
            if isinstance(normalized, int):
                valid_indices.append(idx)
                valid_neighbor_ids.append(normalized)
                sampled_pairs.append((int(img_id), normalized))
            else:
                debug_meta["missing_neighbor_img_ids"].append(int(img_id))

        sample_loss = c_i.new_zeros(c_i.size(0))
        if not valid_neighbor_ids:
            return sample_loss.mean(), debug_meta

        neighbor_h = self.get_emb_by_imgids(valid_neighbor_ids).to(c_i.device)
        _, neighbor_c, _ = self.decompose_sticker(neighbor_h)
        valid_c = c_i[torch.tensor(valid_indices, device=c_i.device, dtype=torch.long)]
        sample_loss[torch.tensor(valid_indices, device=c_i.device, dtype=torch.long)] = (
            1.0 - self._cosine_similarity_from_raw(valid_c, neighbor_c)
        )
        debug_meta["valid_style_neighbor_pairs"] = len(valid_neighbor_ids)
        debug_meta["sampled_neighbor_pairs_preview"] = sampled_pairs[:8]
        return sample_loss.mean(), debug_meta

    def _compute_mmbert_score_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_ids: Sequence[int],
        candidate_h: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._compute_pair_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            img_ids=candidate_ids,
            img_emb=candidate_h,
        )
        return self.compute_base_score(logits)

    def _compute_mmbert_scores_train_triplet(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_ids: Sequence[int],
        pos_h: torch.Tensor,
        cross_ids: Sequence[int],
        cross_h: torch.Tensor,
        same_ids: Sequence[int],
        same_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single multimodal BERT forward for pos / cross / same (same text, three image sets)."""
        b = int(input_ids.size(0))
        if (
            len(pos_ids) != b
            or len(cross_ids) != b
            or len(same_ids) != b
            or pos_h.size(0) != b
            or cross_h.size(0) != b
            or same_h.size(0) != b
        ):
            raise ValueError("triplet MM-BERT expects aligned batch sizes for ids and embeddings")
        tri_in = input_ids.repeat(3, 1)
        tri_mask = attention_mask.repeat(3, 1)
        tri_h = torch.cat([pos_h, cross_h, same_h], dim=0)
        tri_ids = list(pos_ids) + list(cross_ids) + list(same_ids)
        logits = self._compute_pair_logits(
            input_ids=tri_in,
            attention_mask=tri_mask,
            img_ids=tri_ids,
            img_emb=tri_h,
        )
        base = self.compute_base_score(logits)
        return base[:b], base[b : 2 * b], base[2 * b :]

    def _compute_mmbert_scores_train_pair(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ids_a: Sequence[int],
        h_a: torch.Tensor,
        ids_b: Sequence[int],
        h_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single multimodal BERT forward for two image sets (e.g. pos + same neg)."""
        b = int(input_ids.size(0))
        if (
            len(ids_a) != b
            or len(ids_b) != b
            or h_a.size(0) != b
            or h_b.size(0) != b
        ):
            raise ValueError("pair MM-BERT expects aligned batch sizes for ids and embeddings")
        dbl_in = input_ids.repeat(2, 1)
        dbl_mask = attention_mask.repeat(2, 1)
        dbl_h = torch.cat([h_a, h_b], dim=0)
        dbl_ids = list(ids_a) + list(ids_b)
        logits = self._compute_pair_logits(
            input_ids=dbl_in,
            attention_mask=dbl_mask,
            img_ids=dbl_ids,
            img_emb=dbl_h,
        )
        base = self.compute_base_score(logits)
        return base[:b], base[b:]

    def _compute_candidate_factor_scores(
        self,
        q_style: torch.Tensor,
        q_expr: torch.Tensor,
        candidate_c: torch.Tensor,
        candidate_a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        style_score = self.compute_style_compatibility(q_style, candidate_c)
        expr_score = self.compute_expression_compatibility(q_expr, candidate_a)
        return style_score, expr_score

    def forward_train_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_ids: Sequence[int],
        neg_img_ids: Sequence[int],
        global_step: int = 0,
        total_steps: Optional[int] = None,
    ) -> StructuredFactorizedForwardOutput:
        device = input_ids.device
        prof_lim = self._factorized_profile_limit()
        do_prof = prof_lim > 0 and global_step < prof_lim and device.type == "cuda"
        if do_prof:
            pe = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            torch.cuda.synchronize()
            pe[0].record()

        q_style, q_expr, style_usage_gate = self.encode_style_expr_queries(
            input_ids, attention_mask
        )

        if do_prof:
            pe[1].record()

        bank_all_h, _, _, proto_vectors, proto_density = self._get_train_or_fresh_bank_factorization(
            device, global_step
        )
        proto_logits = self._compute_proto_logits(q_style, proto_vectors)
        sticker_to_proto = self._get_sticker_to_proto(device)
        pos_idx = torch.tensor(list(img_ids), dtype=torch.long, device=device)
        pos_proto_ids = sticker_to_proto.index_select(0, pos_idx)
        style_proto_loss = F.cross_entropy(proto_logits, pos_proto_ids)
        proto_acc = (torch.argmax(proto_logits, dim=-1) == pos_proto_ids).float().mean()

        cross_neg_ids = self._resolve_cross_proto_negatives(img_ids, neg_img_ids)
        same_neg_ids = self._resolve_same_proto_negatives(img_ids, cross_neg_ids)

        two_way = bool(getattr(self.args, "factorized_train_mmbert_two_way", False))
        tw_neg = str(getattr(self.args, "factorized_train_two_way_neg", "same") or "same").lower()

        pos_h = self.get_emb_by_imgids(img_ids).to(device)
        if two_way and tw_neg == "same":
            cross_h = None
            same_h = self.get_emb_by_imgids(same_neg_ids).to(device)
        else:
            cross_h = self.get_emb_by_imgids(cross_neg_ids).to(device)
            same_h = self.get_emb_by_imgids(same_neg_ids).to(device)

        if do_prof:
            pe[2].record()

        use_fused = getattr(self.args, "factorized_fused_train_mmbert", False)
        if not two_way:
            if use_fused:
                pos_mmbert, cross_mmbert, same_mmbert = self._compute_mmbert_scores_train_triplet(
                    input_ids,
                    attention_mask,
                    img_ids,
                    pos_h,
                    cross_neg_ids,
                    cross_h,
                    same_neg_ids,
                    same_h,
                )
            else:
                pos_mmbert = self._compute_mmbert_score_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    candidate_ids=img_ids,
                    candidate_h=pos_h,
                )
                cross_mmbert = self._compute_mmbert_score_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    candidate_ids=cross_neg_ids,
                    candidate_h=cross_h,
                )
                same_mmbert = self._compute_mmbert_score_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    candidate_ids=same_neg_ids,
                    candidate_h=same_h,
                )
        elif tw_neg == "same":
            if use_fused:
                pos_mmbert, same_mmbert = self._compute_mmbert_scores_train_pair(
                    input_ids,
                    attention_mask,
                    img_ids,
                    pos_h,
                    same_neg_ids,
                    same_h,
                )
            else:
                pos_mmbert = self._compute_mmbert_score_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    candidate_ids=img_ids,
                    candidate_h=pos_h,
                )
                same_mmbert = self._compute_mmbert_score_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    candidate_ids=same_neg_ids,
                    candidate_h=same_h,
                )
            cross_mmbert = pos_mmbert.new_zeros(pos_mmbert.shape)
        else:
            if use_fused:
                pos_mmbert, cross_mmbert = self._compute_mmbert_scores_train_pair(
                    input_ids,
                    attention_mask,
                    img_ids,
                    pos_h,
                    cross_neg_ids,
                    cross_h,
                )
            else:
                pos_mmbert = self._compute_mmbert_score_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    candidate_ids=img_ids,
                    candidate_h=pos_h,
                )
                cross_mmbert = self._compute_mmbert_score_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    candidate_ids=cross_neg_ids,
                    candidate_h=cross_h,
                )
            same_mmbert = pos_mmbert.new_zeros(pos_mmbert.shape)

        if do_prof:
            pe[3].record()

        _, pos_c, pos_a = self.decompose_sticker(pos_h)
        if cross_h is not None:
            _, cross_c, cross_a = self.decompose_sticker(cross_h)
        else:
            cross_c = pos_c.new_zeros(pos_c.shape)
            cross_a = pos_a.new_zeros(pos_a.shape)
        _, same_c, same_a = self.decompose_sticker(same_h)

        pos_style, pos_expr = self._compute_candidate_factor_scores(
            q_style, q_expr, pos_c, pos_a
        )
        cross_style, cross_expr = self._compute_candidate_factor_scores(
            q_style, q_expr, cross_c, cross_a
        )
        same_style, same_expr = self._compute_candidate_factor_scores(
            q_style, q_expr, same_c, same_a
        )

        pos_graph = self._gather_proto_scores_for_batch(proto_logits, proto_density, img_ids)
        if cross_h is not None:
            cross_graph = self._gather_proto_scores_for_batch(
                proto_logits, proto_density, cross_neg_ids
            )
        else:
            cross_graph = pos_graph.new_zeros(pos_graph.shape)
        same_graph = self._gather_proto_scores_for_batch(
            proto_logits, proto_density, same_neg_ids
        )

        pos_fused = self._fuse_scores(
            pos_mmbert, pos_style, pos_expr, pos_graph, style_usage_gate
        )
        if not two_way:
            cross_fused = self._fuse_scores(
                cross_mmbert, cross_style, cross_expr, cross_graph, style_usage_gate
            )
            same_fused = self._fuse_scores(
                same_mmbert, same_style, same_expr, same_graph, style_usage_gate
            )
        elif tw_neg == "same":
            cross_fused = pos_mmbert.new_zeros(pos_mmbert.shape)
            same_fused = self._fuse_scores(
                same_mmbert, same_style, same_expr, same_graph, style_usage_gate
            )
        else:
            cross_fused = self._fuse_scores(
                cross_mmbert, cross_style, cross_expr, cross_graph, style_usage_gate
            )
            same_fused = pos_mmbert.new_zeros(pos_mmbert.shape)

        if not two_way:
            train_logits = torch.stack([pos_fused, cross_fused, same_fused], dim=-1)
        elif tw_neg == "same":
            train_logits = torch.stack([pos_fused, same_fused], dim=-1)
        else:
            train_logits = torch.stack([pos_fused, cross_fused], dim=-1)
        train_labels = torch.zeros(train_logits.size(0), dtype=torch.long, device=device)
        match_loss = F.cross_entropy(train_logits, train_labels)

        style_neighbor_loss, style_neighbor_meta = self._compute_style_neighbor_loss_with_meta(
            img_ids, pos_c
        )
        orth_loss = self.compute_orth_loss(pos_c, pos_a)
        expr_rank_loss = self.compute_expr_rank_loss(pos_expr, same_expr)
        aux_warmup = self.compute_aux_warmup(
            global_step=global_step, total_steps=total_steps
        ).to(device=device)

        structured_aux = self.args.lambda_struct * (style_neighbor_loss + orth_loss)
        expr_aux = self.args.lambda_expr * expr_rank_loss
        style_proto_aux = float(self.args.lambda_style_proto) * style_proto_loss
        if self.args.base_only:
            structured_aux = structured_aux * 0.0
            expr_aux = expr_aux * 0.0
            style_proto_aux = style_proto_aux * 0.0
        total_loss = match_loss + aux_warmup * (structured_aux + expr_aux + style_proto_aux)

        # One small tensor for training_step: a single .cpu() sync per step instead of many .item() calls.
        train_scalars = torch.stack(
            [
                proto_acc.reshape(()),
                style_usage_gate.mean().reshape(()),
                (pos_fused - same_fused).mean().reshape(()),
                pos_mmbert.mean().reshape(()),
                cross_mmbert.mean().reshape(()),
                same_mmbert.mean().reshape(()),
                pos_style.mean().reshape(()),
                same_expr.mean().reshape(()),
            ],
            dim=0,
        )
        debug_info = {
            "_train_scalars": train_scalars,
            "cross_neg_ids_preview": [int(x) for x in cross_neg_ids[:8]],
            "same_neg_ids_preview": [int(x) for x in same_neg_ids[:8]],
            "style_neighbor_meta": style_neighbor_meta,
        }

        if do_prof:
            pe[4].record()
            torch.cuda.synchronize()
            self._accumulate_train_profile_ms(
                global_step,
                prof_lim,
                pe[0].elapsed_time(pe[1]),
                pe[1].elapsed_time(pe[2]),
                pe[2].elapsed_time(pe[3]),
                pe[3].elapsed_time(pe[4]),
            )

        return StructuredFactorizedForwardOutput(
            loss=total_loss,
            match_loss=match_loss,
            style_proto_loss=style_proto_loss,
            style_neighbor_loss=style_neighbor_loss,
            orth_loss=orth_loss,
            expr_rank_loss=expr_rank_loss,
            aux_warmup=aux_warmup,
            pos_fused_score=pos_fused,
            cross_fused_score=cross_fused,
            same_fused_score=same_fused,
            style_usage_gate=style_usage_gate,
            debug_info=debug_info,
        )

    def _build_candidate_union(
        self,
        candidate_ids: Sequence[int],
        style_score: torch.Tensor,
        mmbert_score: torch.Tensor,
        proto_logits_row: torch.Tensor,
    ) -> Tuple[List[int], Dict[str, Any]]:
        style_topk = min(len(candidate_ids), int(self.args.factorized_style_recall_topk))
        mmbert_topk = min(len(candidate_ids), int(self.args.factorized_mmbert_branch_topk))
        proto_topk = min(len(self.style_bank.prototypes), int(self.args.factorized_style_proto_topk))

        style_rank_idx = torch.topk(style_score, k=style_topk).indices.tolist()
        style_branch_ids = [int(candidate_ids[idx]) for idx in style_rank_idx]

        top_proto_ids = torch.topk(proto_logits_row, k=proto_topk).indices.tolist()
        proto_branch_ids = self.style_bank.expand_top_prototypes(
            top_proto_ids,
            allowed_ids=candidate_ids,
            per_proto_limit=int(self.args.factorized_proto_expand_per_proto),
        )

        mmbert_branch_ids: List[int] = []
        if self.args.factorized_use_mmbert_branch:
            mmbert_rank_idx = torch.topk(mmbert_score, k=mmbert_topk).indices.tolist()
            mmbert_branch_ids = [int(candidate_ids[idx]) for idx in mmbert_rank_idx]

        ordered_union: List[int] = []
        seen = set()
        for group in (style_branch_ids, proto_branch_ids, mmbert_branch_ids):
            for sticker_id in group:
                if sticker_id in seen:
                    continue
                ordered_union.append(int(sticker_id))
                seen.add(int(sticker_id))
                if len(ordered_union) >= int(self.args.factorized_candidate_union_topk):
                    break
            if len(ordered_union) >= int(self.args.factorized_candidate_union_topk):
                break

        branch_meta = {
            "style_branch_ids": style_branch_ids,
            "proto_branch_ids": proto_branch_ids,
            "mmbert_branch_ids": mmbert_branch_ids,
            "top_proto_ids": [int(x) for x in top_proto_ids],
        }
        return ordered_union, branch_meta

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
                f"Structured factorized eval currently expects batch_size=1, got {batch_size}."
            )

        q_style, q_expr, style_usage_gate = self.encode_style_expr_queries(
            input_ids, attention_mask
        )
        bank_all_h, style_bank_c, style_bank_a, proto_vectors, proto_density = (
            self._get_eval_or_fresh_bank_factorization(device)
        )

        use_cands = nonempty_batch_cands(cands)
        if use_cands:
            candidate_ids = [int(x) for x in cands[0]]
            candidate_idx = torch.tensor(candidate_ids, dtype=torch.long, device=device)
            candidate_h = bank_all_h.index_select(0, candidate_idx)
            candidate_c = style_bank_c.index_select(0, candidate_idx)
            candidate_a = style_bank_a.index_select(0, candidate_idx)
        else:
            if getattr(self.args, "candidate_eval_only", False) or getattr(
                self.args, "test_with_cand", False
            ):
                raise ValueError(_CAND_EVAL_ONLY_ERR)
            candidate_ids = list(range(self.args.max_image_id))
            candidate_h = bank_all_h
            candidate_c = style_bank_c
            candidate_a = style_bank_a

        img_num = candidate_h.size(0)
        expanded_input_ids = input_ids.repeat(img_num, 1)
        expanded_attention_mask = attention_mask.repeat(img_num, 1)

        mmbert_score = self._compute_mmbert_score_batch(
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            candidate_ids=candidate_ids,
            candidate_h=candidate_h,
        )

        candidate_q_style = q_style.repeat(img_num, 1)
        candidate_q_expr = q_expr.repeat(img_num, 1)
        style_score, expr_score = self._compute_candidate_factor_scores(
            candidate_q_style, candidate_q_expr, candidate_c, candidate_a
        )

        proto_logits_row = self._compute_proto_logits(q_style, proto_vectors).squeeze(0)
        graph_score = self._gather_proto_scores_for_candidates(
            proto_logits_row, proto_density, candidate_ids
        )

        union_ids, branch_meta = self._build_candidate_union(
            candidate_ids=candidate_ids,
            style_score=style_score,
            mmbert_score=mmbert_score,
            proto_logits_row=proto_logits_row,
        )
        candidate_to_idx = {int(sticker_id): idx for idx, sticker_id in enumerate(candidate_ids)}
        union_idx = torch.tensor(
            [candidate_to_idx[int(sticker_id)] for sticker_id in union_ids],
            dtype=torch.long,
            device=device,
        )
        union_fused = self._fuse_scores(
            mmbert_score.index_select(0, union_idx),
            style_score.index_select(0, union_idx),
            expr_score.index_select(0, union_idx),
            graph_score.index_select(0, union_idx),
            style_usage_gate.repeat(union_idx.numel()),
        )

        fallback_floor = float(mmbert_score.detach().min().item()) - 100.0
        final_score = mmbert_score.new_full((img_num,), fallback_floor)
        final_score.index_copy_(0, union_idx, union_fused)

        rank_scores = final_score.unsqueeze(0)
        labels = torch.tensor(img_ids, dtype=torch.long, device=device)
        if not return_debug:
            return rank_scores, labels, candidate_ids if use_cands else None

        gold_id = int(img_ids[0])
        eval_debug = {
            "uses_factorized_ranking": True,
            "candidate_count": int(len(candidate_ids)),
            "union_size": int(len(union_ids)),
            "gold_in_union": bool(gold_id in set(union_ids)),
            "gold_in_style_branch": bool(gold_id in set(branch_meta["style_branch_ids"])),
            "gold_in_proto_branch": bool(gold_id in set(branch_meta["proto_branch_ids"])),
            "gold_in_mmbert_branch": bool(gold_id in set(branch_meta["mmbert_branch_ids"])),
            "mmbert_score_stats": self._tensor_debug_stats(mmbert_score),
            "final_score_stats": self._tensor_debug_stats(final_score),
            "style_score_stats": self._tensor_debug_stats(style_score),
            "expr_score_stats": self._tensor_debug_stats(expr_score),
            "graph_score_stats": self._tensor_debug_stats(graph_score),
            "style_usage_gate_stats": self._tensor_debug_stats(style_usage_gate),
            "fused_minus_mmbert_abs_mean": float((final_score - mmbert_score).abs().mean().item()),
            "top1_flipped_vs_mmbert": bool(
                torch.argmax(final_score).item() != torch.argmax(mmbert_score).item()
            ),
            "ranking_score_formula": (
                "s_rank = FusionMLP([s_mmbert, s_style, s_expr, s_graph, g_use_style]) "
                "after candidate union from style/prototype/mmbert branches"
            ),
        }
        return rank_scores, labels, candidate_ids if use_cands else None, eval_debug


class StructuredFactorizedPLModel(StructuredPLModel):
    def __init__(self, args: StructuredFactorizedArguments):
        pl.LightningModule.__init__(self)
        self.args = args
        self.model = StructuredFactorizedStickerModel(args)
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

        self._style_proto_acc_ema: Optional[float] = None
        self._style_gate_ema: Optional[float] = None
        self._reset_eval_diagnostics()

    def _update_ema(self, old: Optional[float], value: float, alpha: float = 0.05) -> float:
        if old is None:
            return value
        return alpha * value + (1.0 - alpha) * old

    def _reset_eval_diagnostics(self) -> None:
        self.eval_diag_count = 0
        self.eval_mmbert_score_std_sum = 0.0
        self.eval_final_score_std_sum = 0.0
        self.eval_style_score_std_sum = 0.0
        self.eval_expr_score_std_sum = 0.0
        self.eval_graph_score_std_sum = 0.0
        self.eval_union_size_sum = 0.0
        self.eval_union_rate_sum = 0.0
        self.eval_gold_in_union_count = 0
        self.eval_gold_in_style_branch_count = 0
        self.eval_gold_in_proto_branch_count = 0
        self.eval_top1_flip_count = 0
        self.eval_delta_abs_mean_sum = 0.0

    def _update_eval_diagnostics(self, eval_debug: Dict[str, Any]) -> None:
        self.eval_diag_count += 1
        self.eval_mmbert_score_std_sum += float(eval_debug["mmbert_score_stats"]["std"])
        self.eval_final_score_std_sum += float(eval_debug["final_score_stats"]["std"])
        self.eval_style_score_std_sum += float(eval_debug["style_score_stats"]["std"])
        self.eval_expr_score_std_sum += float(eval_debug["expr_score_stats"]["std"])
        self.eval_graph_score_std_sum += float(eval_debug["graph_score_stats"]["std"])
        self.eval_union_size_sum += float(eval_debug["union_size"])
        self.eval_union_rate_sum += float(eval_debug["union_size"]) / max(
            float(eval_debug["candidate_count"]), 1.0
        )
        self.eval_gold_in_union_count += int(bool(eval_debug["gold_in_union"]))
        self.eval_gold_in_style_branch_count += int(bool(eval_debug["gold_in_style_branch"]))
        self.eval_gold_in_proto_branch_count += int(bool(eval_debug["gold_in_proto_branch"]))
        self.eval_top1_flip_count += int(bool(eval_debug["top1_flipped_vs_mmbert"]))
        self.eval_delta_abs_mean_sum += float(eval_debug["fused_minus_mmbert_abs_mean"])

    def _eval_diagnostics_summary(self) -> Dict[str, float]:
        if self.eval_diag_count <= 0:
            return {}
        denom = float(self.eval_diag_count)
        return {
            "mmbert_score_std": self.eval_mmbert_score_std_sum / denom,
            "final_score_std": self.eval_final_score_std_sum / denom,
            "style_score_std": self.eval_style_score_std_sum / denom,
            "expr_score_std": self.eval_expr_score_std_sum / denom,
            "graph_score_std": self.eval_graph_score_std_sum / denom,
            "union_size": self.eval_union_size_sum / denom,
            "union_rate": self.eval_union_rate_sum / denom,
            "gold_in_union_rate": self.eval_gold_in_union_count / denom,
            "gold_in_style_branch_rate": self.eval_gold_in_style_branch_count / denom,
            "gold_in_proto_branch_rate": self.eval_gold_in_proto_branch_count / denom,
            "top1_flip_rate": self.eval_top1_flip_count / denom,
            "delta_abs_mean": self.eval_delta_abs_mean_sum / denom,
        }

    def on_train_start(self) -> None:
        maybe_set_ddp_static_graph(self.trainer)
        cache_path = (getattr(self.args, "img_emb_cache_path", None) or "").strip()
        if getattr(self.args, "fix_img", False) and cache_path and os.path.exists(cache_path):
            self.model.prepare_for_test()
        self.model.clear_train_factorization_cache()
        prof_lim = self.model._factorized_profile_limit()
        logger.info(
            "[StructuredFactorized] style_recall_topk=%s proto_topk=%s proto_expand=%s "
            "mmbert_branch_topk=%s union_topk=%s lambda_style_proto=%.3f train_bank_refresh=%s "
            "fused_train_mmbert=%s bert_grad_ckpt=%s train_mmbert_two_way=%s two_way_neg=%s "
            "max_dialogue_length=%s profile_steps=%s",
            self.args.factorized_style_recall_topk,
            self.args.factorized_style_proto_topk,
            self.args.factorized_proto_expand_per_proto,
            self.args.factorized_mmbert_branch_topk,
            self.args.factorized_candidate_union_topk,
            float(self.args.lambda_style_proto),
            int(getattr(self.args, "factorized_train_bank_refresh_steps", 0) or 0),
            bool(getattr(self.args, "factorized_fused_train_mmbert", False)),
            bool(getattr(self.args, "bert_gradient_checkpointing", True)),
            bool(getattr(self.args, "factorized_train_mmbert_two_way", False)),
            str(getattr(self.args, "factorized_train_two_way_neg", "same")),
            int(getattr(self.args, "max_dialogue_length", None) or 490),
            prof_lim,
        )
        if prof_lim > 0:
            logger.info(
                "[StructuredFactorized] CUDA train profiling enabled for steps 0..%d "
                "(factorized_profile_train_steps or STICKER_FACTORIZED_PROFILE_STEPS).",
                prof_lim - 1,
            )
        logger.info(
            "[StructuredFactorized] Ranking uses factorized candidate union "
            "(style direct branch + prototype expansion + optional MM-BERT branch) "
            "followed by a learned fusion head."
        )
        return pl.LightningModule.on_train_start(self)

    def run_train_batch(self, batch: Dict[str, Any]) -> StructuredFactorizedForwardOutput:
        return self.model.forward_train_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            img_ids=batch["img_ids"],
            neg_img_ids=batch["neg_img_ids"],
            global_step=int(self.global_step),
            total_steps=int(self.num_training_steps),
        )

    def run_eval_batch(self, batch: Dict[str, Any], return_debug: bool = False):
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
        style_proto_aux = float(self.args.lambda_style_proto) * outputs.style_proto_loss
        if self.args.base_only:
            structured_aux = structured_aux * 0.0
            expr_aux = expr_aux * 0.0
            style_proto_aux = style_proto_aux * 0.0
        weighted_struct_aux = outputs.aux_warmup * structured_aux
        weighted_expr_aux = outputs.aux_warmup * expr_aux
        weighted_style_proto_aux = outputs.aux_warmup * style_proto_aux

        self.log("train_loss", outputs.loss, prog_bar=False)
        self.log("train_match_loss", outputs.match_loss, prog_bar=False)
        self.log("train_style_proto_loss", outputs.style_proto_loss, prog_bar=False)
        self.log("train_style_neighbor_loss", outputs.style_neighbor_loss, prog_bar=False)
        self.log("train_orth_loss", outputs.orth_loss, prog_bar=False)
        self.log("train_expr_rank_loss", outputs.expr_rank_loss, prog_bar=False)
        self.log("train_aux_warmup", outputs.aux_warmup, prog_bar=False)
        self.log("train_structured_aux", weighted_struct_aux, prog_bar=False)
        self.log("train_expr_aux", weighted_expr_aux, prog_bar=False)
        self.log("train_style_proto_aux", weighted_style_proto_aux, prog_bar=False)

        ts_list: Optional[List[float]] = None
        proto_acc = 0.0
        gate_mean = 0.0
        if outputs.debug_info is not None and "_train_scalars" in outputs.debug_info:
            ts_list = outputs.debug_info["_train_scalars"].detach().float().cpu().tolist()
            proto_acc = float(ts_list[0])
            gate_mean = float(ts_list[1])
        self._style_proto_acc_ema = self._update_ema(self._style_proto_acc_ema, proto_acc)
        self._style_gate_ema = self._update_ema(self._style_gate_ema, gate_mean)

        pb_full = self.args.train_prog_bar_mode == "full"
        dev = outputs.loss.device
        self.log("total", outputs.loss.detach(), prog_bar=True, logger=False)
        self.log("match", outputs.match_loss.detach(), prog_bar=True, logger=False)
        self.log("proto", outputs.style_proto_loss.detach(), prog_bar=True, logger=False)
        self.log("style", outputs.style_neighbor_loss.detach(), prog_bar=pb_full, logger=False)
        self.log("orth", outputs.orth_loss.detach(), prog_bar=pb_full, logger=False)
        self.log("expr", outputs.expr_rank_loss.detach(), prog_bar=pb_full, logger=False)
        self.log("saux", weighted_struct_aux.detach(), prog_bar=pb_full, logger=False)
        self.log("eaux", weighted_expr_aux.detach(), prog_bar=pb_full, logger=False)
        self.log("paux", weighted_style_proto_aux.detach(), prog_bar=pb_full, logger=False)
        self.log("wt", outputs.aux_warmup.detach(), prog_bar=pb_full, logger=False)
        self.log(
            "pacc",
            torch.tensor(self._style_proto_acc_ema, device=dev, dtype=torch.float32),
            prog_bar=True,
            logger=False,
        )
        self.log(
            "gate",
            torch.tensor(self._style_gate_ema, device=dev, dtype=torch.float32),
            prog_bar=True,
            logger=False,
        )

        log_iv = int(self.args.factorized_log_interval or 0)
        if (
            log_iv > 0
            and int(self.global_step) % log_iv == 0
            and outputs.debug_info is not None
            and ts_list is not None
        ):
            logger.info(
                "[StructuredFactorizedTrain] step=%d proto_acc=%.4f gate=%.4f "
                "cross_neg=%s same_neg=%s margin=%.4f",
                int(self.global_step),
                float(ts_list[0]),
                float(ts_list[1]),
                outputs.debug_info["cross_neg_ids_preview"],
                outputs.debug_info["same_neg_ids_preview"],
                float(ts_list[2]),
            )

        return outputs.loss

    def on_validation_epoch_start(self) -> None:
        self.model.prepare_for_test()
        self.model.prepare_eval_factorization_cache()
        self._reset_eval_diagnostics()
        if getattr(self, "_per_epoch_dual_test_eval", False):
            self._eval_max_cand_pair = [0, 0]
        else:
            self._eval_max_cand_len = 0
        return pl.LightningModule.on_validation_epoch_start(self)

    def on_test_epoch_start(self) -> None:
        self.model.prepare_for_test()
        self.model.prepare_eval_factorization_cache()
        self._reset_eval_diagnostics()
        if getattr(self, "_per_epoch_dual_test_eval", False):
            self._eval_max_cand_pair = [0, 0]
        else:
            self._eval_max_cand_len = 0
        return pl.LightningModule.on_test_epoch_start(self)

    def on_validation_epoch_end(self) -> None:
        self._structured_eval_epoch_log_and_reset_metrics()
        diag = self._eval_diagnostics_summary()
        if diag:
            logger.info(
                "[StructuredFactorizedScale] "
                f"epoch={self.current_epoch} "
                f"n_batches={self.eval_diag_count} "
                f"mmbert_std={diag['mmbert_score_std']:.4f} "
                f"rank_std={diag['final_score_std']:.4f} "
                f"style_std={diag['style_score_std']:.4f} "
                f"expr_std={diag['expr_score_std']:.4f} "
                f"graph_std={diag['graph_score_std']:.4f} "
                f"union_size={diag['union_size']:.2f} "
                f"union_rate={diag['union_rate']:.4f} "
                f"gold_union={diag['gold_in_union_rate']:.4f} "
                f"gold_style={diag['gold_in_style_branch_rate']:.4f} "
                f"gold_proto={diag['gold_in_proto_branch_rate']:.4f} "
                f"delta_abs={diag['delta_abs_mean']:.4f} "
                f"top1_flip={diag['top1_flip_rate']:.4f}"
            )
        self._reset_eval_diagnostics()
        self.model.clear_eval_factorization_cache()
        return pl.LightningModule.on_validation_epoch_end(self)

    def on_test_epoch_end(self) -> None:
        self.model.clear_eval_factorization_cache()
        return pl.LightningModule.on_test_epoch_end(self)


def run_structured_factorized_main(args: StructuredFactorizedArguments) -> None:
    if args.local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(args, "allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(getattr(args, "allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(getattr(args, "cudnn_benchmark", True))

    pl.seed_everything(args.seed)

    if args.mode in {"train", "pretrain"}:
        model = StructuredFactorizedPLModel(args)
        datamodule = PLDataLoader(args, model.model.bert_tokenizer)
        trainer = build_trainer(args, for_train=True)
        attach_version_log_from_trainer(args, trainer)
        # Full resume (optimizer, schedulers, epoch, global_step): pass ckpt to Lightning.
        # load_checkpoint_to_model() only restores weights and would restart the LR schedule.
        if args.ckpt_path:
            trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
        else:
            trainer.fit(model, datamodule=datamodule)
        return

    if args.mode in {"test", "gen"}:
        model = StructuredFactorizedPLModel(args)
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
        f"Unsupported mode={args.mode} for structured factorized script. Use train/pretrain/test/gen."
    )


def parse_structured_factorized_args(
    argv: Optional[List[str]] = None,
) -> StructuredFactorizedArguments:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_paths, rest = _extract_yaml_config_paths(argv)
    merged: Dict[str, Any] = {}
    for p in config_paths:
        merged = _deep_merge_dict(merged, load_structured_token_yaml_with_extends(p))
    prefix = _config_mapping_to_argv(merged)
    parser = HfArgumentParser(StructuredFactorizedArguments)
    (args,) = parser.parse_args_into_dataclasses(
        args=prefix + rest,
        look_for_args_file=False,
    )
    return args
