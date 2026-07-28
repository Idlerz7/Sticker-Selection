"""VIGEM model with explicit group-centred instance evidence.

This module subclasses the published factorized implementation but never patches
it.  Holistic MM-BERT and group-prototype helpers are inherited unchanged.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from structured_retrieval import ProjectionHead, nonempty_batch_cands
from structured_retrieval_factorized import StructuredFactorizedStickerModel
from style_shapes.fixed_same_pack import (
    GRAY_SENTINEL_ID,
    flatten_query_major,
    listwise_match_loss,
)

from .residuals import InstanceResidualBundle, masked_instance_listwise_loss


@dataclass
class VigemForwardOutput:
    loss: torch.Tensor
    match_loss: torch.Tensor
    group_loss: torch.Tensor
    instance_loss: torch.Tensor
    pos_final_score: torch.Tensor
    cross_final_score: torch.Tensor
    same_final_score: torch.Tensor
    debug_info: Dict[str, Any]


def _inverse_bounded_sigmoid(
    value: float, lower: float, upper: float
) -> float:
    ratio = (float(value) - float(lower)) / (float(upper) - float(lower))
    if not 0.0 < ratio < 1.0:
        raise ValueError("initial temperature must lie strictly inside its bounds")
    return math.log(ratio / (1.0 - ratio))


class VigemInstanceResidualStickerModel(StructuredFactorizedStickerModel):
    """Minimal factorized scorer with a new, isolated instance branch."""

    def __init__(
        self,
        args,
        residual_bundle_path: str,
        expected_membership_hash: str,
        instance_dim: int = 256,
        instance_score_weight: float = 0.3,
        instance_loss_weight: float = 0.3,
        temperature_min: float = 0.5,
        temperature_max: float = 20.0,
        temperature_init: float = 10.0,
        defer_residual_load: bool = False,
    ):
        super().__init__(args)
        if self.uses_full_variant():
            raise ValueError("VIGEM instance residual requires the v6 minimal core")
        if int(instance_dim) != int(args.structured_hidden_dim):
            raise ValueError(
                "instance_dim must equal structured_hidden_dim for the frozen recipe"
            )
        if float(instance_score_weight) != 0.3:
            raise ValueError("VIGEM instance score weight is frozen to 0.3")
        if float(instance_loss_weight) != 0.3:
            raise ValueError("VIGEM instance loss weight is frozen to 0.3")
        if not 0.0 < float(temperature_min) < float(temperature_max):
            raise ValueError("invalid instance temperature bounds")

        self.instance_residual_bundle_path = str(residual_bundle_path)
        self.instance_expected_membership_hash = str(expected_membership_hash)
        self.instance_bundle: Optional[InstanceResidualBundle] = None
        if defer_residual_load:
            manifest_path = Path(residual_bundle_path).with_suffix(
                ".manifest.json"
            )
            with manifest_path.open("r", encoding="utf-8") as handle:
                residual_manifest = json.load(handle)
            if residual_manifest.get("membership_hash") != str(
                expected_membership_hash
            ):
                raise ValueError("deferred residual membership hash mismatch")
            if int(residual_manifest.get("num_stickers", -1)) != int(
                args.max_image_id
            ):
                raise ValueError("deferred residual catalog size mismatch")
        else:
            self._ensure_instance_bundle()
        self.instance_score_weight = float(instance_score_weight)
        self.instance_loss_weight = float(instance_loss_weight)
        self.instance_temperature_min = float(temperature_min)
        self.instance_temperature_max = float(temperature_max)

        self.instance_query_head = ProjectionHead(
            self.bert_hidden_dim, int(instance_dim), args.structured_dropout
        )
        self.instance_residual_head = ProjectionHead(
            768, int(instance_dim), args.structured_dropout
        )
        self.instance_query_norm = nn.LayerNorm(
            int(instance_dim), elementwise_affine=False
        )
        self.instance_residual_norm = nn.LayerNorm(
            int(instance_dim), elementwise_affine=False
        )
        raw_temperature = _inverse_bounded_sigmoid(
            temperature_init, temperature_min, temperature_max
        )
        self.instance_temperature_raw = nn.Parameter(
            torch.tensor(raw_temperature, dtype=torch.float32)
        )

        # These are retained in the state dict for legacy checkpoint compatibility,
        # but are not part of VIGEM scoring or training.
        legacy_instance_modules = (
            self.dialogue_factorizer.expr_query_head,
            self.sticker_factorizer.expr_head,
            self.expr_match_head,
        )
        for module in legacy_instance_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def _ensure_instance_bundle(self) -> InstanceResidualBundle:
        if self.instance_bundle is None:
            bundle = InstanceResidualBundle.load(
                self.instance_residual_bundle_path,
                expected_membership_hash=self.instance_expected_membership_hash,
            )
            if int(bundle.ids.numel()) != int(self.args.max_image_id):
                raise ValueError(
                    "residual catalog size does not match max_image_id"
                )
            self.instance_bundle = bundle
        return self.instance_bundle

    @property
    def instance_temperature(self) -> torch.Tensor:
        span = self.instance_temperature_max - self.instance_temperature_min
        return self.instance_temperature_min + span * torch.sigmoid(
            self.instance_temperature_raw
        )

    def encode_group_and_instance_queries(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_emb = self._get_text_word_embeddings(input_ids)
        outputs = self.bert.bert(
            inputs_embeds=text_emb,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        q_group = self.style_query_head(cls_hidden)
        q_instance = self.instance_query_norm(
            self.instance_query_head(cls_hidden)
        )
        return q_group, q_instance

    def _residual_rows(
        self,
        candidate_ids: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_ids = candidate_ids.detach().cpu().long().reshape(-1)
        bundle = self._ensure_instance_bundle()
        real = flat_ids.ne(GRAY_SENTINEL_ID)
        invalid = flat_ids[real][
            (flat_ids[real] < 0)
            | (flat_ids[real] >= int(bundle.ids.numel()))
        ]
        if int(invalid.numel()):
            raise IndexError(
                "candidate ID outside residual catalog: %s"
                % invalid[:8].tolist()
            )
        safe = flat_ids.clone()
        safe[~real] = 0
        residual = bundle.residuals.index_select(0, safe).to(
            device=device, non_blocking=True
        )
        group_ids = bundle.group_ids.index_select(0, safe).to(
            device=device, non_blocking=True
        )
        group_sizes = bundle.group_sizes.index_select(
            0, group_ids.detach().cpu()
        ).to(device=device, non_blocking=True)
        real_device = real.to(device=device, non_blocking=True)
        residual = residual.masked_fill(~real_device.unsqueeze(-1), 0.0)
        group_ids = group_ids.masked_fill(~real_device, -1)
        group_sizes = group_sizes.masked_fill(~real_device, 0)
        return residual, group_ids, group_sizes, real_device

    def compute_instance_scores(
        self,
        q_instance: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = tuple(candidate_ids.shape)
        residual, group_ids, group_sizes, real = self._residual_rows(
            candidate_ids, q_instance.device
        )
        projected = self.instance_residual_norm(
            self.instance_residual_head(residual)
        )
        flat_query = q_instance.reshape(-1, q_instance.size(-1))
        if int(flat_query.size(0)) != int(projected.size(0)):
            raise ValueError("query and candidate residual rows are not aligned")
        score = self.instance_temperature * F.cosine_similarity(
            flat_query, projected, dim=-1
        )
        informative = real & group_sizes.gt(1) & residual.norm(dim=-1).gt(1e-12)
        score = score.masked_fill(~informative, 0.0)
        return (
            score.reshape(original_shape),
            group_ids.reshape(original_shape),
            group_sizes.reshape(original_shape),
            informative.reshape(original_shape),
        )

    def _compute_final_score_vigem(
        self,
        holistic_score: torch.Tensor,
        instance_score: torch.Tensor,
        group_score: torch.Tensor,
    ) -> torch.Tensor:
        if bool(self.args.base_only):
            return holistic_score
        return (
            holistic_score
            + self.instance_score_weight * instance_score
            + float(self.args.lambda_style_proto) * group_score
        )

    def _group_and_instance_setup(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positive_ids: Sequence[int],
        global_step: int,
    ):
        device = input_ids.device
        q_group, q_instance = self.encode_group_and_instance_queries(
            input_ids, attention_mask
        )
        bank_all_h, _, style_bank_a, proto_vectors, _ = (
            self._get_train_or_fresh_bank_factorization(device, global_step)
        )
        proto_logits, _, group_loss, proto_acc = self._compute_proto_supervision(
            q_style=q_group,
            proto_vectors=proto_vectors,
            pos_ids=positive_ids,
            device=device,
        )
        return (
            q_group,
            q_instance,
            bank_all_h,
            style_bank_a,
            proto_logits,
            group_loss,
            proto_acc,
        )

    def forward_train_listwise_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_ids: Sequence[int],
        candidate_ids: Sequence[Sequence[int]],
        gray_mask: Optional[Sequence[Sequence[bool]]] = None,
        global_step: int = 0,
        total_steps: Optional[int] = None,
    ) -> VigemForwardOutput:
        del total_steps
        device = input_ids.device
        batch_size = int(input_ids.size(0))
        candidate_count = int(self.args.factorized_train_candidate_count)
        candidates = torch.tensor(candidate_ids, dtype=torch.long, device=device)
        if tuple(candidates.shape) != (batch_size, candidate_count):
            raise ValueError("fixed candidates must have shape [B,10]")
        positives = torch.tensor(img_ids, dtype=torch.long, device=device)
        if not torch.equal(candidates[:, 0], positives):
            raise ValueError("candidate 0 must be the gold sticker")
        computed_gray = candidates.eq(GRAY_SENTINEL_ID)
        if computed_gray[:, 0].any():
            raise ValueError("gold candidate cannot be gray")
        if gray_mask is not None:
            supplied = torch.tensor(gray_mask, dtype=torch.bool, device=device)
            if not torch.equal(supplied, computed_gray):
                raise ValueError("gray mask is not aligned with candidate IDs")

        (
            _,
            q_instance,
            bank_all_h,
            _,
            proto_logits,
            group_loss,
            proto_acc,
        ) = self._group_and_instance_setup(
            input_ids, attention_mask, img_ids, global_step
        )
        chunk_size = min(
            candidate_count,
            int(self.args.factorized_candidate_forward_chunk_size),
        )
        holistic_chunks: List[torch.Tensor] = []
        instance_chunks: List[torch.Tensor] = []
        group_chunks: List[torch.Tensor] = []
        final_chunks: List[torch.Tensor] = []
        group_id_chunks: List[torch.Tensor] = []
        informative_chunks: List[torch.Tensor] = []
        try:
            for start in range(0, candidate_count, chunk_size):
                stop = min(candidate_count, start + chunk_size)
                width = stop - start
                ids_chunk = candidates[:, start:stop]
                flat_input, flat_mask, flat_ids = flatten_query_major(
                    input_ids, attention_mask, ids_chunk
                )
                flat_id_list = [int(item) for item in flat_ids.detach().cpu()]
                candidate_h = self._candidate_embeddings_from_bank(
                    bank_all_h, flat_id_list
                )
                logits = self._compute_pair_logits(
                    input_ids=flat_input,
                    attention_mask=flat_mask,
                    img_ids=flat_id_list,
                    img_emb=candidate_h,
                )
                holistic = self.compute_base_score(logits)
                flat_q = q_instance.repeat_interleave(width, dim=0)
                instance, group_ids, _, informative = self.compute_instance_scores(
                    flat_q, flat_ids
                )
                flat_proto_logits = proto_logits.repeat_interleave(width, dim=0)
                group = self._gather_proto_scores_for_batch(
                    flat_proto_logits, flat_id_list
                )
                final = self._compute_final_score_vigem(
                    holistic, instance, group
                )
                holistic_chunks.append(holistic.reshape(batch_size, width))
                instance_chunks.append(instance.reshape(batch_size, width))
                group_chunks.append(group.reshape(batch_size, width))
                final_chunks.append(final.reshape(batch_size, width))
                group_id_chunks.append(group_ids.reshape(batch_size, width))
                informative_chunks.append(informative.reshape(batch_size, width))
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "VIGEM fixed R10 OOM with query batch=%d and candidate chunk=%d; "
                "retry explicitly with chunk 5, 2, or 1"
                % (batch_size, chunk_size)
            ) from exc

        holistic_scores = torch.cat(holistic_chunks, dim=1)
        instance_scores = torch.cat(instance_chunks, dim=1)
        group_scores = torch.cat(group_chunks, dim=1)
        final_scores = torch.cat(final_chunks, dim=1)
        candidate_groups = torch.cat(group_id_chunks, dim=1)
        informative = torch.cat(informative_chunks, dim=1)
        positive_groups = candidate_groups[:, :1]
        same_group = candidate_groups.eq(positive_groups)
        instance_valid = informative & same_group & ~computed_gray
        instance_valid[:, 0] = informative[:, 0]
        real_negative = ~computed_gray[:, 1:]
        if not bool(
            (same_group[:, 1:] | ~real_negative).all()
        ):
            raise RuntimeError(
                "StickerChat fixed same-pack candidates crossed a VPD pack group"
            )

        match_loss = listwise_match_loss(final_scores)
        instance_loss, eligible = masked_instance_listwise_loss(
            instance_scores, instance_valid
        )
        total_loss = (
            match_loss
            + float(self.args.lambda_style_proto) * group_loss
            + self.instance_loss_weight * instance_loss
        )
        zero = final_scores.new_zeros(batch_size)
        debug = {
            "negative_policy": "fixed_same_pack_listwise",
            "candidate_ids": candidates.detach(),
            "gray_mask": computed_gray.detach(),
            "holistic_scores": holistic_scores.detach(),
            "instance_scores": instance_scores.detach(),
            "group_scores": group_scores.detach(),
            "final_scores": final_scores.detach(),
            "final_without_instance_scores": (
                holistic_scores
                + float(self.args.lambda_style_proto) * group_scores
            ).detach(),
            "instance_valid_mask": instance_valid.detach(),
            "instance_eligible_rows": eligible.detach(),
            "candidate_forward_chunk_size": int(chunk_size),
            "proto_acc": proto_acc.detach(),
            "temperature": self.instance_temperature.detach(),
        }
        return VigemForwardOutput(
            loss=total_loss,
            match_loss=match_loss,
            group_loss=group_loss,
            instance_loss=instance_loss,
            pos_final_score=final_scores[:, 0],
            cross_final_score=final_scores[:, 1],
            same_final_score=zero,
            debug_info=debug,
        )

    def forward_train_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_ids: Sequence[int],
        neg_img_ids: Sequence[int],
        global_step: int = 0,
        total_steps: Optional[int] = None,
    ) -> VigemForwardOutput:
        del total_steps
        device = input_ids.device
        (
            _,
            q_instance,
            bank_all_h,
            style_bank_a,
            proto_logits,
            group_loss,
            proto_acc,
        ) = self._group_and_instance_setup(
            input_ids, attention_mask, img_ids, global_step
        )
        # The registered DSTC recipe uses pools of one.  Supplying q_instance
        # preserves the existing resolver/RNG while avoiding a live legacy head.
        cross_ids, same_ids, negative_meta = (
            self._resolve_prototype_aware_negatives(
                q_expr=q_instance.detach(),
                pos_ids=img_ids,
                fallback_neg_ids=neg_img_ids,
                style_bank_a=style_bank_a,
            )
        )
        pos_h = self.get_emb_by_imgids(img_ids).to(device)
        cross_h = self.get_emb_by_imgids(cross_ids).to(device)
        same_h = self.get_emb_by_imgids(same_ids).to(device)
        if bool(getattr(self.args, "factorized_fused_train_mmbert", False)):
            pos_holistic, cross_holistic, same_holistic = (
                self._compute_mmbert_scores_train_triplet(
                    input_ids,
                    attention_mask,
                    img_ids,
                    pos_h,
                    cross_ids,
                    cross_h,
                    same_ids,
                    same_h,
                )
            )
        else:
            pos_holistic = self._compute_mmbert_score_batch(
                input_ids, attention_mask, img_ids, pos_h
            )
            cross_holistic = self._compute_mmbert_score_batch(
                input_ids, attention_mask, cross_ids, cross_h
            )
            same_holistic = self._compute_mmbert_score_batch(
                input_ids, attention_mask, same_ids, same_h
            )

        pos_tensor = torch.tensor(img_ids, dtype=torch.long, device=device)
        cross_tensor = torch.tensor(cross_ids, dtype=torch.long, device=device)
        same_tensor = torch.tensor(same_ids, dtype=torch.long, device=device)
        pos_instance, pos_groups, _, pos_info = (
            self.compute_instance_scores(q_instance, pos_tensor)
        )
        cross_instance, _, _, _ = self.compute_instance_scores(
            q_instance, cross_tensor
        )
        same_instance, same_groups, _, same_info = self.compute_instance_scores(
            q_instance, same_tensor
        )
        pos_group_score = self._gather_proto_scores_for_batch(
            proto_logits, img_ids
        )
        cross_group_score = self._gather_proto_scores_for_batch(
            proto_logits, cross_ids
        )
        same_group_score = self._gather_proto_scores_for_batch(
            proto_logits, same_ids
        )
        pos_final = self._compute_final_score_vigem(
            pos_holistic, pos_instance, pos_group_score
        )
        cross_final = self._compute_final_score_vigem(
            cross_holistic, cross_instance, cross_group_score
        )
        same_final = self._compute_final_score_vigem(
            same_holistic, same_instance, same_group_score
        )
        logits = torch.stack([pos_final, cross_final, same_final], dim=-1)
        labels = torch.zeros(
            int(logits.size(0)), dtype=torch.long, device=device
        )
        match_loss = F.cross_entropy(logits, labels)
        local_scores = torch.stack([pos_instance, same_instance], dim=-1)
        local_valid = torch.stack(
            [
                pos_info,
                same_info
                & same_groups.eq(pos_groups)
                & same_tensor.ne(pos_tensor),
            ],
            dim=-1,
        )
        instance_loss, eligible = masked_instance_listwise_loss(
            local_scores, local_valid
        )
        total_loss = (
            match_loss
            + float(self.args.lambda_style_proto) * group_loss
            + self.instance_loss_weight * instance_loss
        )
        debug = {
            "negative_policy": "prototype_cross_plus_same",
            "candidate_ids": torch.stack(
                [pos_tensor, cross_tensor, same_tensor], dim=-1
            ).detach(),
            "gray_mask": torch.zeros_like(
                torch.stack([pos_tensor, cross_tensor, same_tensor], dim=-1),
                dtype=torch.bool,
            ),
            "holistic_scores": torch.stack(
                [pos_holistic, cross_holistic, same_holistic], dim=-1
            ).detach(),
            "instance_scores": torch.stack(
                [pos_instance, cross_instance, same_instance], dim=-1
            ).detach(),
            "group_scores": torch.stack(
                [pos_group_score, cross_group_score, same_group_score], dim=-1
            ).detach(),
            "final_scores": logits.detach(),
            "final_without_instance_scores": torch.stack(
                [
                    pos_holistic
                    + float(self.args.lambda_style_proto) * pos_group_score,
                    cross_holistic
                    + float(self.args.lambda_style_proto) * cross_group_score,
                    same_holistic
                    + float(self.args.lambda_style_proto) * same_group_score,
                ],
                dim=-1,
            ).detach(),
            "instance_valid_mask": local_valid.detach(),
            "instance_eligible_rows": eligible.detach(),
            "cross_neg_ids_preview": [int(value) for value in cross_ids[:8]],
            "same_neg_ids_preview": [int(value) for value in same_ids[:8]],
            "negative_meta": negative_meta,
            "proto_acc": proto_acc.detach(),
            "temperature": self.instance_temperature.detach(),
        }
        return VigemForwardOutput(
            loss=total_loss,
            match_loss=match_loss,
            group_loss=group_loss,
            instance_loss=instance_loss,
            pos_final_score=pos_final,
            cross_final_score=cross_final,
            same_final_score=same_final,
            debug_info=debug,
        )

    @staticmethod
    def _stats(value: torch.Tensor) -> Dict[str, float]:
        flat = value.detach().float().reshape(-1)
        return {
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
        }

    def forward_eval_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_ids: Sequence[int],
        cands: Optional[Sequence[Sequence[int]]] = None,
        return_debug: bool = False,
        score_breakdown: bool = False,
    ):
        device = input_ids.device
        if int(input_ids.size(0)) != 1:
            raise ValueError("VIGEM evaluation expects batch_size=1")
        q_group, q_instance = self.encode_group_and_instance_queries(
            input_ids, attention_mask
        )
        bank_all_h, _, _, proto_vectors, _ = (
            self._get_eval_or_fresh_bank_factorization(device)
        )
        use_candidates = nonempty_batch_cands(cands)
        if not use_candidates:
            if getattr(self.args, "candidate_eval_only", False) or getattr(
                self.args, "test_with_cand", False
            ):
                raise ValueError("candidate-only evaluation requires candidates")
            candidate_ids = list(range(int(self.args.max_image_id)))
        else:
            candidate_ids = [int(value) for value in cands[0]]
        candidate_tensor = torch.tensor(
            candidate_ids, dtype=torch.long, device=device
        )
        candidate_h = self._candidate_embeddings_from_bank(
            bank_all_h, candidate_ids
        )
        candidate_count = len(candidate_ids)
        holistic = self._compute_mmbert_score_batch(
            input_ids=input_ids.repeat(candidate_count, 1),
            attention_mask=attention_mask.repeat(candidate_count, 1),
            candidate_ids=candidate_ids,
            candidate_h=candidate_h,
        )
        repeated_instance_query = q_instance.repeat(candidate_count, 1)
        instance, _, _, _ = self.compute_instance_scores(
            repeated_instance_query, candidate_tensor
        )
        proto_logits = self._compute_proto_logits(
            q_group, proto_vectors
        ).squeeze(0)
        group = self._gather_proto_scores_for_candidates(
            proto_logits, candidate_ids
        )
        final_without_instance = (
            holistic + float(self.args.lambda_style_proto) * group
        )
        final = self._compute_final_score_vigem(holistic, instance, group)
        rank_scores = final.unsqueeze(0)
        labels = torch.tensor(img_ids, dtype=torch.long, device=device)
        if not return_debug and not score_breakdown:
            return (
                rank_scores,
                labels,
                candidate_ids if use_candidates else None,
            )
        debug = {
            "uses_factorized_ranking": True,
            "candidate_count": candidate_count,
            "mmbert_score_stats": self._stats(holistic),
            "final_score_stats": self._stats(final),
            # Compatibility aliases for inherited diagnostics.
            "expr_score_stats": self._stats(instance),
            "graph_score_stats": self._stats(group),
            "instance_score_stats": self._stats(instance),
            "fused_minus_mmbert_abs_mean": float(
                (final - holistic).abs().mean().item()
            ),
            "top1_flipped_vs_mmbert": bool(
                torch.argmax(final).item() != torch.argmax(holistic).item()
            ),
            "ranking_score_formula": (
                "s_final = s_holistic + 0.3*s_instance_residual "
                "+ 0.4*s_group"
            ),
            "temperature": float(self.instance_temperature.detach().item()),
        }
        if score_breakdown:
            debug.update(
                {
                    "candidate_ids_ordered": candidate_ids,
                    "gray_mask": [
                        int(value) == GRAY_SENTINEL_ID
                        for value in candidate_ids
                    ],
                    "mmbert_score_per_cand": holistic.detach()
                    .float()
                    .cpu()
                    .tolist(),
                    "instance_score_per_cand": instance.detach()
                    .float()
                    .cpu()
                    .tolist(),
                    # Compatibility alias for tooling expecting old naming.
                    "expr_score_per_cand": instance.detach()
                    .float()
                    .cpu()
                    .tolist(),
                    "graph_score_per_cand": group.detach()
                    .float()
                    .cpu()
                    .tolist(),
                    "final_score_per_cand": final.detach()
                    .float()
                    .cpu()
                    .tolist(),
                    "final_without_instance_per_cand": final_without_instance.detach()
                    .float()
                    .cpu()
                    .tolist(),
                }
            )
        return (
            rank_scores,
            labels,
            candidate_ids if use_candidates else None,
            debug,
        )
