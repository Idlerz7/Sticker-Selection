"""Independent pack-relative, setwise Instance branch for StickerChat VIGEM."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from style_shapes.fixed_same_pack import (
    GRAY_SENTINEL_ID,
    flatten_query_major,
    listwise_match_loss,
)
from structured_retrieval import nonempty_batch_cands
from vigem.model import VigemForwardOutput, VigemInstanceResidualStickerModel
from vigem.pack_relative import PackRelativeResidualBundle
from vigem.residuals import masked_instance_listwise_loss


class PackRelativeSetwiseStickerModel(VigemInstanceResidualStickerModel):
    """Use original-pack residuals and permutation-equivariant R10 context."""

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
        super().__init__(
            args,
            residual_bundle_path=residual_bundle_path,
            expected_membership_hash=expected_membership_hash,
            instance_dim=instance_dim,
            instance_score_weight=instance_score_weight,
            instance_loss_weight=instance_loss_weight,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
            temperature_init=temperature_init,
            defer_residual_load=defer_residual_load,
        )
        dim = int(instance_dim)
        self.instance_set_scorer = nn.Sequential(
            nn.Linear(dim * 5, dim),
            nn.Tanh(),
            nn.Dropout(args.structured_dropout),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.instance_set_scorer[-1].weight)
        nn.init.zeros_(self.instance_set_scorer[-1].bias)

    def _ensure_instance_bundle(self) -> PackRelativeResidualBundle:
        if self.instance_bundle is None:
            bundle = PackRelativeResidualBundle.load(
                self.instance_residual_bundle_path,
                expected_vpd_membership_hash=(
                    self.instance_expected_membership_hash
                ),
            )
            if int(bundle.ids.numel()) != int(self.args.max_image_id):
                raise ValueError(
                    "pack-relative catalog size does not match max_image_id"
                )
            self.instance_bundle = bundle
        return self.instance_bundle

    def encode_group_and_instance_queries(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detach only Group supervision from the shared dialogue BERT."""
        text_emb = self._get_text_word_embeddings(input_ids)
        outputs = self.bert.bert(
            inputs_embeds=text_emb,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        q_group = self.style_query_head(cls_hidden.detach())
        q_instance = self.instance_query_norm(
            self.instance_query_head(cls_hidden)
        )
        return q_group, q_instance

    def _pack_rows(
        self, candidate_ids: torch.Tensor, device: torch.device
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        original_shape = tuple(candidate_ids.shape)
        flat = candidate_ids.detach().cpu().long().reshape(-1)
        bundle = self._ensure_instance_bundle()
        real = flat.ne(GRAY_SENTINEL_ID)
        invalid = flat[real][
            (flat[real] < 0) | (flat[real] >= int(bundle.ids.numel()))
        ]
        if int(invalid.numel()):
            raise IndexError(
                "candidate ID outside pack-relative catalog: %s"
                % invalid[:8].tolist()
            )
        safe = flat.clone()
        safe[~real] = 0
        residuals = bundle.residuals.index_select(0, safe).to(
            device=device, non_blocking=True
        )
        pack_ids = bundle.pack_ids.index_select(0, safe)
        pack_sizes = bundle.pack_sizes.index_select(0, pack_ids)
        vpd_ids = bundle.vpd_group_ids.index_select(0, safe)
        vpd_sizes = bundle.vpd_group_sizes.index_select(0, vpd_ids)
        real_device = real.to(device=device, non_blocking=True)
        residuals = residuals.to(device=device, non_blocking=True)
        pack_ids = pack_ids.to(device=device, non_blocking=True)
        pack_sizes = pack_sizes.to(device=device, non_blocking=True)
        vpd_ids = vpd_ids.to(device=device, non_blocking=True)
        vpd_sizes = vpd_sizes.to(device=device, non_blocking=True)
        residuals = residuals.masked_fill(
            ~real_device.unsqueeze(-1), 0.0
        )
        pack_ids = pack_ids.masked_fill(~real_device, -1)
        pack_sizes = pack_sizes.masked_fill(~real_device, 0)
        vpd_ids = vpd_ids.masked_fill(~real_device, -1)
        vpd_sizes = vpd_sizes.masked_fill(~real_device, 0)
        return (
            residuals.reshape(*original_shape, 768),
            pack_ids.reshape(original_shape),
            pack_sizes.reshape(original_shape),
            vpd_ids.reshape(original_shape),
            vpd_sizes.reshape(original_shape),
            real_device.reshape(original_shape),
        )

    def compute_setwise_instance_scores(
        self,
        q_instance: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if candidate_ids.ndim != 2:
            raise ValueError("setwise candidates must have shape [B,N]")
        if int(q_instance.size(0)) != int(candidate_ids.size(0)):
            raise ValueError("query/candidate batch size mismatch")
        (
            residuals,
            pack_ids,
            pack_sizes,
            vpd_ids,
            _,
            real,
        ) = self._pack_rows(candidate_ids, q_instance.device)
        batch, width = candidate_ids.shape
        projected = self.instance_residual_norm(
            self.instance_residual_head(residuals.reshape(-1, 768))
        ).reshape(batch, width, -1)
        projected = projected.masked_fill(~real.unsqueeze(-1), 0.0)
        denominator = real.sum(dim=1, keepdim=True).clamp_min(1).to(
            projected.dtype
        )
        context = projected.sum(dim=1, keepdim=True) / denominator.unsqueeze(-1)
        query = q_instance.unsqueeze(1).expand(-1, width, -1)
        features = torch.cat(
            [
                query,
                projected,
                projected - context,
                query * projected,
                torch.abs(query - projected),
            ],
            dim=-1,
        )
        raw = self.instance_set_scorer(features).squeeze(-1)
        scores = self.instance_temperature * torch.tanh(raw)
        informative = (
            real
            & pack_sizes.gt(1)
            & residuals.norm(dim=-1).gt(1e-12)
        )
        scores = scores.masked_fill(~informative, 0.0)
        return {
            "scores": scores,
            "pack_ids": pack_ids,
            "pack_sizes": pack_sizes,
            "vpd_group_ids": vpd_ids,
            "real_mask": real,
            "informative_mask": informative,
            "context": context.squeeze(1),
            "projected_residuals": projected,
        }

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
        instance_output = self.compute_setwise_instance_scores(
            q_instance, candidates
        )
        instance_scores = instance_output["scores"]
        candidate_packs = instance_output["pack_ids"]
        candidate_groups = instance_output["vpd_group_ids"]
        informative = instance_output["informative_mask"]
        real_negative = ~computed_gray[:, 1:]
        same_pack = candidate_packs.eq(candidate_packs[:, :1])
        if not bool((same_pack[:, 1:] | ~real_negative).all()):
            raise RuntimeError("fixed R10 candidates crossed an original pack")
        same_group = candidate_groups.eq(candidate_groups[:, :1])
        if not bool((same_group[:, 1:] | ~real_negative).all()):
            raise RuntimeError("fixed R10 candidates crossed a VPD group")

        chunk_size = min(
            candidate_count,
            int(self.args.factorized_candidate_forward_chunk_size),
        )
        holistic_chunks: List[torch.Tensor] = []
        group_chunks: List[torch.Tensor] = []
        final_chunks: List[torch.Tensor] = []
        try:
            for start in range(0, candidate_count, chunk_size):
                stop = min(candidate_count, start + chunk_size)
                width = stop - start
                ids_chunk = candidates[:, start:stop]
                flat_input, flat_mask, flat_ids = flatten_query_major(
                    input_ids, attention_mask, ids_chunk
                )
                flat_id_list = [
                    int(item) for item in flat_ids.detach().cpu().tolist()
                ]
                candidate_h = self._candidate_embeddings_from_bank(
                    bank_all_h, flat_id_list
                )
                logits = self._compute_pair_logits(
                    input_ids=flat_input,
                    attention_mask=flat_mask,
                    img_ids=flat_id_list,
                    img_emb=candidate_h,
                )
                holistic = self.compute_base_score(logits).reshape(
                    batch_size, width
                )
                flat_proto_logits = proto_logits.repeat_interleave(width, dim=0)
                group = self._gather_proto_scores_for_batch(
                    flat_proto_logits, flat_id_list
                ).reshape(batch_size, width)
                local_instance = instance_scores[:, start:stop]
                final = self._compute_final_score_vigem(
                    holistic, local_instance, group
                )
                holistic_chunks.append(holistic)
                group_chunks.append(group)
                final_chunks.append(final)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "pack-relative fixed R10 OOM with query batch=%d and "
                "candidate chunk=%d; retry explicitly with chunk 5, 2, or 1"
                % (batch_size, chunk_size)
            ) from exc

        holistic_scores = torch.cat(holistic_chunks, dim=1)
        group_scores = torch.cat(group_chunks, dim=1)
        final_scores = torch.cat(final_chunks, dim=1)
        instance_valid = informative & same_pack & ~computed_gray
        instance_valid[:, 0] = informative[:, 0]
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
            "negative_policy": "fixed_same_pack_listwise_pack_relative",
            "candidate_ids": candidates.detach(),
            "gray_mask": computed_gray.detach(),
            "pack_ids": candidate_packs.detach(),
            "vpd_group_ids": candidate_groups.detach(),
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
            "set_context_norm": instance_output["context"]
            .norm(dim=-1)
            .detach(),
            "group_shared_bert_detached": True,
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
            raise ValueError("pack-relative evaluation expects batch_size=1")
        q_group, q_instance = self.encode_group_and_instance_queries(
            input_ids, attention_mask
        )
        bank_all_h, _, _, proto_vectors, _ = (
            self._get_eval_or_fresh_bank_factorization(device)
        )
        if not nonempty_batch_cands(cands):
            raise ValueError("pack-relative evaluation is fixed-candidate only")
        candidate_ids = [int(value) for value in cands[0]]
        candidate_tensor = torch.tensor(
            [candidate_ids], dtype=torch.long, device=device
        )
        candidate_h = self._candidate_embeddings_from_bank(
            bank_all_h, candidate_ids
        )
        count = len(candidate_ids)
        holistic = self._compute_mmbert_score_batch(
            input_ids=input_ids.repeat(count, 1),
            attention_mask=attention_mask.repeat(count, 1),
            candidate_ids=candidate_ids,
            candidate_h=candidate_h,
        )
        instance_output = self.compute_setwise_instance_scores(
            q_instance, candidate_tensor
        )
        instance = instance_output["scores"].squeeze(0)
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
            return rank_scores, labels, candidate_ids
        debug: Dict[str, Any] = {
            "uses_factorized_ranking": True,
            "candidate_count": count,
            "mmbert_score_stats": self._stats(holistic),
            "final_score_stats": self._stats(final),
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
                "holistic + 0.3*pack_relative_setwise_instance + 0.4*group"
            ),
            "temperature": float(self.instance_temperature.detach().item()),
            "group_shared_bert_detached": True,
        }
        if score_breakdown:
            debug.update(
                {
                    "candidate_ids_ordered": candidate_ids,
                    "gray_mask": [
                        value == GRAY_SENTINEL_ID for value in candidate_ids
                    ],
                    "pack_ids": instance_output["pack_ids"]
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .tolist(),
                    "vpd_group_ids": instance_output["vpd_group_ids"]
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .tolist(),
                    "mmbert_score_per_cand": holistic.detach()
                    .float()
                    .cpu()
                    .tolist(),
                    "instance_score_per_cand": instance.detach()
                    .float()
                    .cpu()
                    .tolist(),
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
                    "final_without_instance_per_cand": (
                        final_without_instance.detach().float().cpu().tolist()
                    ),
                    "set_context_norm": float(
                        instance_output["context"].norm(dim=-1).item()
                    ),
                }
            )
        return rank_scores, labels, candidate_ids, debug
