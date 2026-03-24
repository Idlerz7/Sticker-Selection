import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Work around protobuf C-extension segfault in some environments
# (must be set before importing pytorch_lightning/tensorboard).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import transformers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW, HfArgumentParser

from main import (
    Arguments as LegacyArguments,
    Model as LegacyModel,
    PLDataLoader,
    attach_test_log_to_ckpt_version,
    attach_version_log_from_trainer,
    logger,
)
from metrics import MyAccuracy
from utils import try_create_dir


StickerId = Union[int, str]


def normalize_sticker_id(raw_id: Any) -> StickerId:
    if raw_id is None:
        return ""
    text = str(raw_id).strip()
    if text == "":
        return ""
    if text.isdigit():
        return int(text)
    return text


@dataclass
class StructuredArguments(LegacyArguments):
    pl_root_dir: Optional[str] = field(default="logs/structured")
    structured_result_dir: Optional[str] = field(default="./result/structured")
    style_neighbors_path: Optional[str] = field(default="./style_neighbors.json")
    structured_hidden_dim: Optional[int] = field(default=256)
    structured_dropout: Optional[float] = field(default=0.1)
    alpha_style_neighbor: Optional[float] = field(default=0.1)
    beta_orth: Optional[float] = field(default=0.05)
    init_style_match_weight: Optional[float] = field(default=0.5)
    init_expr_match_weight: Optional[float] = field(default=0.5)
    style_neighbor_topk: Optional[int] = field(default=5)
    style_sampling_mode: Optional[str] = field(default="random_topk")
    strict_checkpoint_load: Optional[bool] = field(default=False)
    save_structured_test_outputs: Optional[bool] = field(default=True)
    debug_smoke_test: Optional[bool] = field(default=False)
    debug_smoke_train_steps: Optional[int] = field(default=3)
    debug_smoke_eval_steps: Optional[int] = field(default=1)
    debug_smoke_log_every_step: Optional[bool] = field(default=True)
    base_only: Optional[bool] = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        # Structured experiments explicitly disable all legacy personalization paths.
        self.use_visual_personalization_token = False
        self.use_visual_history_attention = False
        if self.style_sampling_mode not in {"top1", "random_topk"}:
            raise ValueError(
                f"Unsupported style_sampling_mode={self.style_sampling_mode}. "
                "Use 'top1' or 'random_topk'."
            )
        if self.structured_hidden_dim <= 0:
            raise ValueError("structured_hidden_dim must be > 0.")
        if self.style_neighbor_topk < 0:
            raise ValueError("style_neighbor_topk must be >= 0.")
        if self.debug_smoke_train_steps <= 0:
            raise ValueError("debug_smoke_train_steps must be > 0.")
        if self.debug_smoke_eval_steps <= 0:
            raise ValueError("debug_smoke_eval_steps must be > 0.")


@dataclass
class StructuredForwardOutput:
    loss: torch.Tensor
    match_loss: torch.Tensor
    style_neighbor_loss: torch.Tensor
    orth_loss: torch.Tensor
    pos_fused_logits: torch.Tensor
    neg_fused_logits: torch.Tensor
    pos_style_score: torch.Tensor
    pos_expr_score: torch.Tensor
    neg_style_score: torch.Tensor
    neg_expr_score: torch.Tensor
    debug_info: Optional[Dict[str, Any]] = None


class StyleNeighborStore:
    def __init__(self, neighbor_map: Dict[StickerId, List[StickerId]]):
        self.neighbor_map = neighbor_map

    @classmethod
    def from_json(cls, path: str) -> "StyleNeighborStore":
        if not path:
            logger.warning("No style_neighbors_path provided; style neighbor loss disabled.")
            return cls({})
        if not os.path.exists(path):
            logger.warning(
                f"style neighbors file not found at {path}; style neighbor loss disabled."
            )
            return cls({})

        with open(path, "r", encoding="utf-8") as f:
            raw_obj = json.load(f)

        if isinstance(raw_obj, dict):
            records = raw_obj.get("neighbors", [])
        elif isinstance(raw_obj, list):
            records = raw_obj
        else:
            raise ValueError(f"Unsupported style neighbors JSON structure in {path}.")

        neighbor_map: Dict[StickerId, List[StickerId]] = {}
        for item in records:
            item_id = normalize_sticker_id(item.get("id", ""))
            if item_id == "":
                continue
            neighbors = item.get("neighbors", [])
            normalized_neighbors: List[StickerId] = []
            for neighbor in neighbors:
                neighbor_id = normalize_sticker_id(neighbor.get("id", ""))
                if neighbor_id == "" or neighbor_id == item_id:
                    continue
                normalized_neighbors.append(neighbor_id)
            neighbor_map[item_id] = normalized_neighbors

        logger.info(
            f"Loaded style neighbors: items={len(neighbor_map)} from {path}"
        )
        return cls(neighbor_map)

    def get_topk(self, sticker_id: StickerId, topk: int) -> List[StickerId]:
        if topk <= 0:
            return []
        normalized_id = normalize_sticker_id(sticker_id)
        return self.neighbor_map.get(normalized_id, [])[:topk]

    def sample(
        self,
        sticker_id: StickerId,
        topk: int,
        sampling_mode: str,
        rng: random.Random,
    ) -> Optional[StickerId]:
        candidates = self.get_topk(sticker_id, topk)
        if not candidates:
            return None
        if sampling_mode == "top1":
            return candidates[0]
        if sampling_mode == "random_topk":
            return rng.choice(candidates)
        raise ValueError(f"Unsupported sampling_mode={sampling_mode}")


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        hidden_dim = max(out_dim, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedStickerHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarMatchHead(nn.Module):
    def __init__(self, feature_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1)).squeeze(-1)


def softplus_inverse_scalar(value: float) -> torch.Tensor:
    value = max(float(value), 1e-6)
    tensor_value = torch.tensor(value, dtype=torch.float32)
    return torch.log(torch.expm1(tensor_value))


class StructuredStickerModel(LegacyModel):
    """
    Reuses the original text/image backbones from `main.Model`,
    while re-organizing the retrieval logic into a cleaner structured version.
    """

    def __init__(self, args: StructuredArguments):
        super().__init__(args)
        self.args = args
        self.query_head = ProjectionHead(
            self.bert_hidden_dim, args.structured_hidden_dim, args.structured_dropout
        )
        self.shared_sticker_head = SharedStickerHead(
            self.visual_feat_dim, args.structured_hidden_dim, args.structured_dropout
        )
        self.style_head = ProjectionHead(
            args.structured_hidden_dim, args.structured_hidden_dim, args.structured_dropout
        )
        self.expr_head = ProjectionHead(
            args.structured_hidden_dim, args.structured_hidden_dim, args.structured_dropout
        )
        self.style_match_head = ScalarMatchHead(
            args.structured_hidden_dim, args.structured_dropout
        )
        self.expr_match_head = ScalarMatchHead(
            args.structured_hidden_dim, args.structured_dropout
        )
        self.style_match_weight = nn.Parameter(
            softplus_inverse_scalar(float(args.init_style_match_weight))
        )
        self.expr_match_weight = nn.Parameter(
            softplus_inverse_scalar(float(args.init_expr_match_weight))
        )
        self.loss_fct = nn.CrossEntropyLoss()
        self.neighbor_store = StyleNeighborStore.from_json(args.style_neighbors_path)
        self._neighbor_rng = random.Random(args.seed)

    def get_emb_by_imgids(self, img_ids: Sequence[int]) -> torch.Tensor:
        """
        A robust version of image embedding lookup that supports all legacy modes.
        """
        if self.args.model_choice == "use_clip_repo":
            img_objs = [self.get_image_obj(int(img_id)) for img_id in img_ids]
            device = next(self.clip_model.parameters()).device
            img_batch = torch.stack(img_objs, dim=0).to(device)
            return self.clip_model.encode_image(img_batch)

        if self.args.model_choice == "use_img_id":
            device = self.img_embedding_layer.weight.device
            ids = torch.tensor(list(img_ids), dtype=torch.long, device=device)
            return self.img_embedding_layer(ids)

        # fix_img 且已有预计算 embedding 时，直接查表
        if (
            getattr(self.args, "fix_img", False)
            and hasattr(self, "all_img_embs")
            and self.all_img_embs is not None
        ):
            device = getattr(
                self.img_clip, "_target_device", next(self.bert.parameters()).device
            )
            ids_t = torch.tensor(list(img_ids), dtype=torch.long, device=device)
            return self.all_img_embs[ids_t]

        img_objs = [self.get_image_obj(int(img_id)) for img_id in img_ids]
        img_tokens = self.img_clip.tokenize(img_objs)
        if hasattr(self.img_clip, "model"):
            clip_device = next(self.img_clip.model.parameters()).device
        elif hasattr(self.img_clip, "_target_device"):
            clip_device = self.img_clip._target_device
        else:
            clip_device = next(self.bert.parameters()).device
        img_tokens = {
            k: v.to(clip_device) if hasattr(v, "to") else v
            for k, v in img_tokens.items()
        }
        return self.img_clip.forward(img_tokens)["sentence_embedding"]

    def prepare_for_test(self):
        logger.info("prepare_for_test! (structured override)")
        with torch.no_grad():
            if self.args.model_choice == "use_clip_repo":
                img_objs = [self.id2img[idx] for idx in range(self.args.max_image_id)]
                device = next(self.clip_model.parameters()).device
                img = torch.stack(img_objs, dim=0).to(device)
                self.clip_model.eval()
                self.all_img_embs = self.clip_model.encode_image(img)
                return

            if self.args.model_choice == "use_img_id":
                img_ids = list(range(self.args.max_image_id))
                device = self.img_embedding_layer.weight.device
                self.all_img_embs = self.img_embedding_layer(
                    torch.tensor(img_ids, dtype=torch.long, device=device)
                )
                return

            cache_path = (getattr(self.args, "img_emb_cache_path", None) or "").strip()
            if cache_path and os.path.exists(cache_path):
                logger.info(f"Loading precomputed embeddings from {cache_path}")
                self.all_img_embs = torch.load(cache_path, map_location="cpu")
                device = getattr(
                    self.img_clip, "_target_device", next(self.bert.parameters()).device
                )
                self.all_img_embs = self.all_img_embs.to(device)
                assert self.all_img_embs.size(0) == self.args.max_image_id
                return

            img_objs = [self.get_image_obj(idx) for idx in range(self.args.max_image_id)]
            img_tokens = self.img_clip.tokenize(img_objs)
            if hasattr(self.img_clip, "model"):
                clip_device = next(self.img_clip.model.parameters()).device
            elif hasattr(self.img_clip, "_target_device"):
                clip_device = self.img_clip._target_device
            else:
                clip_device = next(self.bert.parameters()).device
            img_tokens = {
                k: v.to(clip_device) if hasattr(v, "to") else v
                for k, v in img_tokens.items()
            }
            self.img_clip.eval()
            self.all_img_embs = self.img_clip.forward(img_tokens)["sentence_embedding"]
            if cache_path:
                d = os.path.dirname(cache_path)
                if d:
                    os.makedirs(d, exist_ok=True)
                torch.save(self.all_img_embs.cpu(), cache_path)
                logger.info(f"Saved embeddings to {cache_path}")

    def _get_text_word_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.bert.bert.embeddings.word_embeddings(input_ids)

    def _build_sep_embeddings(self, batch_size: int, device: torch.device) -> torch.Tensor:
        sep_id = torch.tensor(
            self.bert_tokenizer.sep_token_id, device=device, dtype=torch.long
        )
        sep_emb = self.bert.bert.embeddings.word_embeddings(sep_id)
        return sep_emb.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)

    def _build_cls_inputs_emb(
        self, img_ids: Sequence[int], device: torch.device
    ) -> torch.Tensor:
        cls_inputs_ts = []
        for img_id in img_ids:
            _, _, cls_inputs = self.get_input_output_imglabel_by_imgid(int(img_id))
            cls_inputs_ts.append(cls_inputs)
        cls_inputs_ts = torch.tensor(cls_inputs_ts, device=device, dtype=torch.long)
        return self.bert.bert.embeddings.word_embeddings(cls_inputs_ts)

    def _project_sticker_token(self, img_emb: torch.Tensor) -> torch.Tensor:
        return self.img_ff(img_emb).unsqueeze(1)

    def _build_multimodal_inputs(
        self,
        text_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        img_emb: torch.Tensor,
        img_ids: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = text_emb.device
        batch_size = img_emb.size(0)
        sticker_token = self._project_sticker_token(img_emb)
        aux_emb = None
        if self.args.add_ocr_info:
            aux_emb = self._build_cls_inputs_emb(img_ids, device)

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
        return input_emb, full_attention_mask, token_type_ids

    def encode_dialogue_query(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # q is a dialogue-only query representation.
        # It is extracted from pure text without any candidate sticker token,
        # so s_style / s_expr are conditioned on dialogue semantics only.
        text_emb = self._get_text_word_embeddings(input_ids)
        outputs = self.bert.bert(
            inputs_embeds=text_emb,
            attention_mask=attention_mask,
            return_dict=True,
        )
        q = outputs.last_hidden_state[:, 0, :]
        return self.query_head(q)

    def decompose_sticker(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.shared_sticker_head(h)
        return self.style_head(u), self.expr_head(u)

    def score_query_to_components(
        self, q: torch.Tensor, c: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        style_score = self.style_match_head(q, c)
        expr_score = self.expr_match_head(q, a)
        return style_score, expr_score

    def get_nonnegative_match_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return F.softplus(self.style_match_weight), F.softplus(self.expr_match_weight)

    def fuse_logits(
        self,
        logits: torch.Tensor,
        style_score: torch.Tensor,
        expr_score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.args.base_only:
            delta = torch.zeros_like(style_score)
            fused_logits = logits
            return fused_logits, delta
        style_w, expr_w = self.get_nonnegative_match_weights()
        delta = style_w * style_score + expr_w * expr_score
        fused_logits = torch.stack(
            [logits[:, 0] - delta, logits[:, 1] + delta], dim=-1
        )
        return fused_logits, delta

    def _compute_pair_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_ids: Sequence[int],
        img_emb: torch.Tensor,
    ) -> torch.Tensor:
        text_emb = self._get_text_word_embeddings(input_ids)
        input_emb, full_attention_mask, token_type_ids = self._build_multimodal_inputs(
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

    def _cosine_similarity_from_raw(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        left_norm = F.normalize(left, dim=-1)
        right_norm = F.normalize(right, dim=-1)
        return torch.sum(left_norm * right_norm, dim=-1)

    def _compute_style_neighbor_loss_with_meta(
        self, pos_img_ids: Sequence[int], c_i: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        debug_meta: Dict[str, Any] = {
            "valid_style_neighbor_pairs": 0,
            "total_positive_samples": len(pos_img_ids),
            "missing_neighbor_img_ids": [],
            "sampled_neighbor_pairs_preview": [],
        }
        if self.args.alpha_style_neighbor <= 0 or self.args.style_neighbor_topk <= 0:
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

        neighbor_h = self.get_emb_by_imgids(valid_neighbor_ids)
        neighbor_c, _ = self.decompose_sticker(neighbor_h)
        valid_c = c_i[torch.tensor(valid_indices, device=c_i.device, dtype=torch.long)]
        if self.args.debug_smoke_test:
            assert (
                valid_c.size(0) == neighbor_c.size(0)
            ), "style neighbor c_i / c_j batch size mismatch"
        sample_loss[torch.tensor(valid_indices, device=c_i.device, dtype=torch.long)] = (
            1.0 - self._cosine_similarity_from_raw(valid_c, neighbor_c)
        )
        debug_meta["valid_style_neighbor_pairs"] = len(valid_neighbor_ids)
        debug_meta["sampled_neighbor_pairs_preview"] = sampled_pairs[:8]
        return sample_loss.mean(), debug_meta

    def compute_orth_loss(self, c_i: torch.Tensor, a_i: torch.Tensor) -> torch.Tensor:
        return torch.pow(self._cosine_similarity_from_raw(c_i, a_i), 2).mean()

    def _tensor_debug_stats(self, x: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            x_float = x.detach().float()
            flat = x_float.reshape(-1)
            norm_mean = (
                x_float.norm(dim=-1).mean().item() if x_float.ndim >= 2 else x_float.abs().mean().item()
            )
            return {
                "shape": tuple(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "mean": float(flat.mean().item()),
                "std": float(flat.std(unbiased=False).item()),
                "min": float(flat.min().item()),
                "max": float(flat.max().item()),
                "l2_norm_mean": float(norm_mean),
                "has_nan": bool(torch.isnan(x_float).any().item()),
                "has_inf": bool(torch.isinf(x_float).any().item()),
            }

    def forward_train_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_ids: Sequence[int],
        neg_img_ids: Sequence[int],
    ) -> StructuredForwardOutput:
        device = input_ids.device
        batch_size = input_ids.size(0)
        pos_labels = torch.ones(len(img_ids), dtype=torch.long, device=device)
        neg_labels = torch.zeros(len(neg_img_ids), dtype=torch.long, device=device)

        # q is dialogue-only and computed once from pure text.
        q = self.encode_dialogue_query(input_ids, attention_mask)
        if self.args.debug_smoke_test:
            assert q.shape[0] == batch_size, "q batch size mismatch"

        pos_h = self.get_emb_by_imgids(img_ids)
        neg_h = self.get_emb_by_imgids(neg_img_ids)

        # Base logits come from the original candidate-aware multimodal BERT path.
        pos_logits = self._compute_pair_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            img_ids=img_ids,
            img_emb=pos_h,
        )
        neg_logits = self._compute_pair_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            img_ids=neg_img_ids,
            img_emb=neg_h,
        )

        pos_u = self.shared_sticker_head(pos_h)
        neg_u = self.shared_sticker_head(neg_h)
        pos_c = self.style_head(pos_u)
        neg_c = self.style_head(neg_u)
        pos_a = self.expr_head(pos_u)
        neg_a = self.expr_head(neg_u)
        if self.args.debug_smoke_test:
            assert pos_c.shape == pos_a.shape, "pos c/a shape mismatch"
            assert neg_c.shape == neg_a.shape, "neg c/a shape mismatch"

        pos_style_score, pos_expr_score = self.score_query_to_components(q, pos_c, pos_a)
        neg_style_score, neg_expr_score = self.score_query_to_components(q, neg_c, neg_a)
        if self.args.debug_smoke_test:
            assert (
                pos_style_score.shape == pos_expr_score.shape
            ), "pos style/expr score shape mismatch"
            assert (
                neg_style_score.shape == neg_expr_score.shape
            ), "neg style/expr score shape mismatch"

        pos_fused_logits, pos_delta = self.fuse_logits(
            pos_logits, pos_style_score, pos_expr_score
        )
        neg_fused_logits, neg_delta = self.fuse_logits(
            neg_logits, neg_style_score, neg_expr_score
        )
        if self.args.debug_smoke_test:
            assert pos_fused_logits.shape[-1] == 2, "pos fused logits last dim must be 2"
            assert neg_fused_logits.shape[-1] == 2, "neg fused logits last dim must be 2"

        pos_loss = self.loss_fct(pos_fused_logits, pos_labels)
        neg_loss = self.loss_fct(neg_fused_logits, neg_labels)
        match_loss = 0.5 * (pos_loss + neg_loss)

        style_neighbor_loss, style_neighbor_meta = self._compute_style_neighbor_loss_with_meta(
            img_ids, pos_c
        )
        orth_loss = self.compute_orth_loss(pos_c, pos_a)
        total_loss = (
            match_loss
            + self.args.alpha_style_neighbor * style_neighbor_loss
            + self.args.beta_orth * orth_loss
        )
        if self.args.debug_smoke_test:
            assert not torch.isnan(total_loss).any(), "final_loss has NaN"
            assert not torch.isinf(total_loss).any(), "final_loss has Inf"

        debug_info = None
        if self.args.debug_smoke_test:
            style_w, expr_w = self.get_nonnegative_match_weights()
            pos_logit_delta = (pos_fused_logits - pos_logits).detach()
            neg_logit_delta = (neg_fused_logits - neg_logits).detach()
            debug_info = {
                "batch_size": batch_size,
                "input_ids_shape": tuple(input_ids.shape),
                "attention_mask_shape": tuple(attention_mask.shape),
                "pos_img_ids": [int(x) for x in img_ids],
                "neg_img_ids": [int(x) for x in neg_img_ids],
                "q_stats": self._tensor_debug_stats(q),
                "pos_h_stats": self._tensor_debug_stats(pos_h),
                "neg_h_stats": self._tensor_debug_stats(neg_h),
                "pos_u_stats": self._tensor_debug_stats(pos_u),
                "neg_u_stats": self._tensor_debug_stats(neg_u),
                "pos_c_stats": self._tensor_debug_stats(pos_c),
                "neg_c_stats": self._tensor_debug_stats(neg_c),
                "pos_a_stats": self._tensor_debug_stats(pos_a),
                "neg_a_stats": self._tensor_debug_stats(neg_a),
                "pos_style_score_stats": self._tensor_debug_stats(pos_style_score),
                "neg_style_score_stats": self._tensor_debug_stats(neg_style_score),
                "pos_expr_score_stats": self._tensor_debug_stats(pos_expr_score),
                "neg_expr_score_stats": self._tensor_debug_stats(neg_expr_score),
                "style_weight_raw": float(self.style_match_weight.detach().item()),
                "expr_weight_raw": float(self.expr_match_weight.detach().item()),
                "style_weight_softplus": float(style_w.detach().item()),
                "expr_weight_softplus": float(expr_w.detach().item()),
                "pos_delta_stats": self._tensor_debug_stats(pos_delta),
                "neg_delta_stats": self._tensor_debug_stats(neg_delta),
                "pos_logits_stats": self._tensor_debug_stats(pos_logits),
                "neg_logits_stats": self._tensor_debug_stats(neg_logits),
                "pos_fused_logits_stats": self._tensor_debug_stats(pos_fused_logits),
                "neg_fused_logits_stats": self._tensor_debug_stats(neg_fused_logits),
                "pos_logit_delta_stats": self._tensor_debug_stats(pos_logit_delta),
                "neg_logit_delta_stats": self._tensor_debug_stats(neg_logit_delta),
                "pos_logit_delta_abs_mean": float(pos_logit_delta.abs().mean().item()),
                "neg_logit_delta_abs_mean": float(neg_logit_delta.abs().mean().item()),
                "pos_match_loss": float(pos_loss.detach().item()),
                "neg_match_loss": float(neg_loss.detach().item()),
                "match_loss": float(match_loss.detach().item()),
                "style_neighbor_loss": float(style_neighbor_loss.detach().item()),
                "orth_loss": float(orth_loss.detach().item()),
                "final_loss": float(total_loss.detach().item()),
                "style_neighbor_meta": style_neighbor_meta,
            }

        return StructuredForwardOutput(
            loss=total_loss,
            match_loss=match_loss,
            style_neighbor_loss=style_neighbor_loss,
            orth_loss=orth_loss,
            pos_fused_logits=pos_fused_logits,
            neg_fused_logits=neg_fused_logits,
            pos_style_score=pos_style_score,
            pos_expr_score=pos_expr_score,
            neg_style_score=neg_style_score,
            neg_expr_score=neg_expr_score,
            debug_info=debug_info,
        )

    def forward_eval_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_ids: Sequence[int],
        cands: Optional[Sequence[Sequence[int]]] = None,
        return_debug: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, Optional[List[int]]],
        Tuple[torch.Tensor, torch.Tensor, Optional[List[int]], Dict[str, Any]],
    ]:
        device = input_ids.device
        batch_size = input_ids.size(0)
        if batch_size != 1:
            raise ValueError(
                f"Structured eval currently expects batch_size=1, got {batch_size}."
            )

        q = self.encode_dialogue_query(input_ids, attention_mask)

        if cands:
            candidate_ids = [int(x) for x in cands[0]]
            candidate_h = self.all_img_embs[candidate_ids]
        else:
            candidate_ids = list(range(self.args.max_image_id))
            candidate_h = self.all_img_embs

        img_num = candidate_h.size(0)
        expanded_input_ids = input_ids.repeat(img_num, 1)
        expanded_attention_mask = attention_mask.repeat(img_num, 1)

        # s_base is the original candidate-aware score from the legacy multimodal
        # BERT branch, where dialogue text is paired with each candidate sticker.
        base_logits = self._compute_pair_logits(
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            img_ids=candidate_ids,
            img_emb=candidate_h,
        )

        candidate_q = q.repeat(img_num, 1)
        candidate_c, candidate_a = self.decompose_sticker(candidate_h)
        # s_style / s_expr are compatibility scores between the dialogue-only
        # query q and the candidate's decomposed style / expression components.
        style_score, expr_score = self.score_query_to_components(
            candidate_q, candidate_c, candidate_a
        )
        fused_logits, _ = self.fuse_logits(base_logits, style_score, expr_score)
        # Evaluation ranking explicitly uses fused logits.
        final_scores = (fused_logits[:, 1] - fused_logits[:, 0]).unsqueeze(0)
        labels = torch.tensor(img_ids, dtype=torch.long, device=device)
        if not return_debug:
            return final_scores, labels, candidate_ids if cands else None
        eval_debug = {
            "uses_fused_logits_for_ranking": True,
            "base_logits_stats": self._tensor_debug_stats(base_logits),
            "fused_logits_stats": self._tensor_debug_stats(fused_logits),
            "final_scores_stats": self._tensor_debug_stats(final_scores),
            "style_score_stats": self._tensor_debug_stats(style_score),
            "expr_score_stats": self._tensor_debug_stats(expr_score),
            "fused_minus_base_abs_mean": float(
                (fused_logits - base_logits).abs().mean().item()
            ),
            "ranking_score_formula": "fused_logits[:,1] - fused_logits[:,0]",
            "candidate_count": int(len(candidate_ids)),
        }
        return final_scores, labels, candidate_ids if cands else None, eval_debug


class StructuredPLModel(pl.LightningModule):
    def __init__(self, args: StructuredArguments):
        super().__init__()
        self.args = args
        self.model = StructuredStickerModel(args)
        self.model.prepare_imgs(args)

        self.valtest_acc5 = MyAccuracy()
        self.valtest_acc30 = MyAccuracy()
        self.valtest_acc90 = MyAccuracy()
        self.valtest_map = MyAccuracy()

        self.id2name: Dict[int, str] = {}
        with open("./data/id2name.json", encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            self.id2name[int(k)] = v

    def _debug_assert_tensor_finite(self, name: str, x: torch.Tensor) -> None:
        if torch.isnan(x).any():
            raise ValueError(f"{name} contains NaN")
        if torch.isinf(x).any():
            raise ValueError(f"{name} contains Inf")

    def _format_stats_line(self, name: str, stats: Dict[str, Any]) -> str:
        return (
            f"{name}: shape={stats['shape']} dtype={stats['dtype']} device={stats['device']} "
            f"mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"norm={stats['l2_norm_mean']:.4f} nan={stats['has_nan']} inf={stats['has_inf']}"
        )

    def _log_debug_train_summary(self, debug_info: Dict[str, Any], step: int) -> None:
        logger.info(
            "[DebugTrainSummary] "
            f"step={step} batch_size={debug_info['batch_size']} "
            f"style_w={debug_info['style_weight_softplus']:.4f} "
            f"expr_w={debug_info['expr_weight_softplus']:.4f} "
            f"delta_pos_mean={debug_info['pos_delta_stats']['mean']:.4f} "
            f"delta_neg_mean={debug_info['neg_delta_stats']['mean']:.4f} "
            f"match={debug_info['match_loss']:.4f} "
            f"style={debug_info['style_neighbor_loss']:.4f} "
            f"orth={debug_info['orth_loss']:.4f} final={debug_info['final_loss']:.4f} "
            f"valid_style_pairs={debug_info['style_neighbor_meta']['valid_style_neighbor_pairs']}/"
            f"{debug_info['style_neighbor_meta']['total_positive_samples']}"
        )
        logger.info("[DebugTrain] " + self._format_stats_line("q", debug_info["q_stats"]))
        logger.info("[DebugTrain] " + self._format_stats_line("pos_u", debug_info["pos_u_stats"]))
        logger.info("[DebugTrain] " + self._format_stats_line("pos_c", debug_info["pos_c_stats"]))
        logger.info("[DebugTrain] " + self._format_stats_line("pos_a", debug_info["pos_a_stats"]))
        logger.info(
            "[DebugTrain] "
            + self._format_stats_line("pos_style_score", debug_info["pos_style_score_stats"])
        )
        logger.info(
            "[DebugTrain] "
            + self._format_stats_line("pos_expr_score", debug_info["pos_expr_score_stats"])
        )
        if debug_info["style_neighbor_meta"]["missing_neighbor_img_ids"]:
            logger.info(
                "[DebugTrain] missing_neighbor_img_ids="
                f"{debug_info['style_neighbor_meta']['missing_neighbor_img_ids'][:8]}"
            )
        if debug_info["style_neighbor_meta"]["sampled_neighbor_pairs_preview"]:
            logger.info(
                "[DebugTrain] sampled_neighbor_pairs_preview="
                f"{debug_info['style_neighbor_meta']['sampled_neighbor_pairs_preview']}"
            )

    def run_train_batch(self, batch: Dict[str, Any]) -> StructuredForwardOutput:
        return self.model.forward_train_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            img_ids=batch["img_ids"],
            neg_img_ids=batch["neg_img_ids"],
        )

    def run_eval_batch(
        self, batch: Dict[str, Any], return_debug: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, Optional[List[int]]],
        Tuple[torch.Tensor, torch.Tensor, Optional[List[int]], Dict[str, Any]],
    ]:
        return self.model.forward_eval_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            img_ids=batch["img_ids"],
            cands=batch.get("cands"),
            return_debug=return_debug,
        )

    def compute_acc(self, sorted_idx: torch.Tensor, labels: torch.Tensor, k: int) -> Tuple[torch.Tensor, int]:
        idx = sorted_idx[:, :k]
        labels = labels.unsqueeze(-1).expand_as(idx)
        cor_ts = torch.eq(idx, labels).any(dim=-1)
        cor = cor_ts.sum()
        total = cor_ts.numel()
        return cor, total

    def compute_map(self, sorted_idx: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        labels = labels.unsqueeze(-1)
        matches = sorted_idx == labels
        _, idx = matches.nonzero(as_tuple=True)
        reciprocal_rank = 1.0 / (idx + 1)
        return torch.sum(reciprocal_rank), labels.size(0)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.run_train_batch(batch)
        style_w, expr_w = self.model.get_nonnegative_match_weights()
        self.log("train_loss", outputs.loss)
        self.log("train_match_loss", outputs.match_loss)
        self.log("train_style_neighbor_loss", outputs.style_neighbor_loss)
        self.log("train_orth_loss", outputs.orth_loss)
        if self.args.base_only:
            self.log("train_style_match_weight", torch.zeros_like(style_w))
            self.log("train_expr_match_weight", torch.zeros_like(expr_w))
        else:
            self.log("train_style_match_weight", style_w)
            self.log("train_expr_match_weight", expr_w)
        if self.args.debug_smoke_test and outputs.debug_info is not None:
            if self.args.debug_smoke_log_every_step:
                self._log_debug_train_summary(outputs.debug_info, int(batch_idx))
            self._debug_assert_tensor_finite("train_loss", outputs.loss)
        return outputs.loss

    def on_validation_epoch_start(self) -> None:
        self.model.prepare_for_test()
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self) -> None:
        self.model.prepare_for_test()
        return super().on_test_epoch_start()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if self.args.debug_smoke_test:
            scores, labels, cands, eval_debug = self.run_eval_batch(batch, return_debug=True)
            logger.info(
                "[DebugEval] "
                f"batch_idx={batch_idx} uses_fused={eval_debug['uses_fused_logits_for_ranking']} "
                f"candidate_count={eval_debug['candidate_count']} "
                f"formula={eval_debug['ranking_score_formula']}"
            )
            logger.info(
                "[DebugEval] fused_minus_base_abs_mean="
                f"{eval_debug['fused_minus_base_abs_mean']:.6f}"
            )
        else:
            scores, labels, cands = self.run_eval_batch(batch)
        _, idx = torch.sort(scores, dim=-1, descending=True)

        if cands is not None:
            return_preds = torch.tensor(cands)[idx.squeeze(0)].tolist()
        else:
            return_preds = torch.arange(idx.size(1))[idx.squeeze(0)].tolist()
        return_label = labels.item()

        if cands:
            for cand_idx, cand in enumerate(cands):
                if int(cand) == labels.item():
                    labels[0] = cand_idx
                    break

        cor1, tot1 = self.compute_acc(idx, labels, 1)
        cor2, tot2 = self.compute_acc(idx, labels, 2)
        cor5, tot5 = self.compute_acc(idx, labels, 5)
        map_sum, map_tot = self.compute_map(idx, labels)

        self.valtest_acc5.update(cor1, tot1)
        self.valtest_acc30.update(cor2, tot2)
        self.valtest_acc90.update(cor5, tot5)
        self.valtest_map.update(map_sum, map_tot)

        metrics = [
            (cor1 / tot1).item(),
            (cor2 / tot2).item(),
            (cor5 / tot5).item(),
            (map_sum / map_tot).item(),
        ]
        return metrics, return_preds, return_label

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        eval_mode = "R10(candidates)" if self.args.test_with_cand else "Rall(all-stickers)"
        logger.info(
            f"\n[StructuredEval] epoch={self.current_epoch} mode={eval_mode} total={self.valtest_acc5.total} "
            f"r@1={self.valtest_acc5.compute():.4f} r@2={self.valtest_acc30.compute():.4f} "
            f"r@5={self.valtest_acc90.compute():.4f} mrr={self.valtest_map.compute():.4f}"
        )
        self.valtest_acc5.reset()
        self.valtest_acc30.reset()
        self.valtest_acc90.reset()
        self.valtest_map.reset()
        return super().on_validation_epoch_end()

    def test_epoch_end(self, outputs):
        self.on_validation_epoch_end()
        if not outputs or not self.args.save_structured_test_outputs:
            return

        metrics_cols = ["r@1", "r@2", "r@5", "mrr"]
        metrics_obj = {k: [] for k in metrics_cols}
        pred_obj = []
        for metrics, pred, answer in outputs:
            pred_obj.append({"pred": pred, "answer": answer})
            for idx, val in enumerate(metrics):
                metrics_obj[metrics_cols[idx]].append(val)

        try_create_dir(self.args.structured_result_dir)
        test_stem = os.path.splitext(os.path.basename(self.args.test_data_path))[0]
        metrics_path = os.path.join(
            self.args.structured_result_dir,
            f"{test_stem}_{self.args.test_with_cand}_structured_metrics.json",
        )
        pred_path = os.path.join(
            self.args.structured_result_dir,
            f"{test_stem}_{self.args.test_with_cand}_structured_pred.json",
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_obj, f, ensure_ascii=False, indent=2)
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(pred_obj, f, ensure_ascii=False, indent=2)

        sample = pred_obj[0] if pred_obj else {}
        logger.info(
            f"[StructuredTestSummary] samples={len(outputs)} metrics_file={metrics_path} pred_file={pred_path}"
        )
        logger.info(
            f"[StructuredTestPreview] answer={sample.get('answer')} top10_pred={sample.get('pred', [])[:10]}"
        )

    @property
    def num_training_steps(self) -> int:
        if (self.trainer.max_steps is not None) and (self.trainer.max_steps > 0):
            return self.trainer.max_steps

        estimated_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if estimated_steps is not None and estimated_steps > 0:
            return int(estimated_steps)

        batches = getattr(self.trainer, "num_training_batches", None)
        if not isinstance(batches, int) or batches <= 0:
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None:
                try:
                    batches = len(dm.train_dataloader())
                except Exception:
                    batches = None

        if batches is None or batches <= 0:
            return max(1, int(self.trainer.max_epochs))

        limit_batches = self.trainer.limit_train_batches
        if isinstance(limit_batches, int):
            batches = min(batches, limit_batches)
        else:
            batches = int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = max(1, self.trainer.accumulate_grad_batches * num_devices)
        return max(1, (batches // effective_accum) * self.trainer.max_epochs)

    def configure_optimizers(self):
        logger.info(
            f"img lr:{self.args.img_lr}, fix text:{self.args.fix_text}, fix img:{self.args.fix_img}"
        )
        img_prefixes = ("img_clip", "clip_model", "img_embedding_layer")
        img_params = [
            var
            for name, var in self.model.named_parameters()
            if var.requires_grad and name.startswith(img_prefixes)
        ]
        other_params = [
            var
            for name, var in self.model.named_parameters()
            if var.requires_grad and (not name.startswith(img_prefixes))
        ]
        param_groups = []
        if img_params:
            param_groups.append({"params": img_params, "lr": self.args.img_lr})
        if other_params:
            param_groups.append({"params": other_params, "lr": self.args.other_lr})

        optimizer = AdamW(
            param_groups,
            lr=self.args.other_lr,
            betas=(0.9, 0.98),
            weight_decay=0.2,
        )
        total_steps = max(1, self.num_training_steps)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        logger.info(
            f"use scheduler! total_steps={total_steps}, warmup_steps=0"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def load_checkpoint_to_model(
    model: StructuredPLModel, ckpt_path: str, strict: bool
) -> None:
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    logger.info(
        f"load checkpoint from {ckpt_path} strict={strict} "
        f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
    )
    if missing:
        logger.info(f"missing keys (first 20): {missing[:20]}")
    if unexpected:
        logger.info(f"unexpected keys (first 20): {unexpected[:20]}")


def build_trainer(args: StructuredArguments, for_train: bool) -> pl.Trainer:
    trainer_kwargs: Dict[str, Any] = {
        "gpus": args.gpus,
        "max_epochs": args.epochs,
        "accumulate_grad_batches": args.gradient_accumulation_steps,
        "default_root_dir": args.pl_root_dir,
    }
    if for_train:
        trainer_kwargs["callbacks"] = [ModelCheckpoint(save_top_k=-1, verbose=True)]
    if args.gpus and args.gpus > 1:
        trainer_kwargs["accelerator"] = "ddp"
    return pl.Trainer(**trainer_kwargs)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _module_grad_summary(module: nn.Module) -> Dict[str, float]:
    grad_norms: List[float] = []
    none_count = 0
    total_count = 0
    for p in module.parameters():
        total_count += 1
        if p.grad is None:
            none_count += 1
            continue
        grad_norms.append(float(p.grad.detach().norm().item()))
    nonzero = [x for x in grad_norms if x > 0.0]
    return {
        "params_total": float(total_count),
        "params_grad_none": float(none_count),
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
        "grad_nonzero_count": float(len(nonzero)),
    }


def run_debug_smoke_test(args: StructuredArguments) -> None:
    logger.info("[DebugSmoke] Starting structured debug smoke test.")
    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpus and args.gpus > 0) else "cpu")
    logger.info(f"[DebugSmoke] device={device}")

    model = StructuredPLModel(args).to(device)
    if args.ckpt_path:
        load_checkpoint_to_model(model, args.ckpt_path, strict=args.strict_checkpoint_load)

    datamodule = PLDataLoader(args, model.model.bert_tokenizer)
    datamodule.setup("fit")
    datamodule.setup("test")
    train_loader = datamodule.train_dataloader()
    eval_loader = datamodule.val_dataloader() or datamodule.test_dataloader()
    train_iter = _cycle_loader(train_loader)
    eval_iter = _cycle_loader(eval_loader)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.other_lr, betas=(0.9, 0.98), weight_decay=0.2)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=max(1, args.debug_smoke_train_steps)
    )

    model.train()
    for step in range(args.debug_smoke_train_steps):
        raw_batch = next(train_iter)
        batch = move_batch_to_device(raw_batch, device)
        optimizer.zero_grad()

        output = model.run_train_batch(batch)
        debug_info = output.debug_info or {}
        loss = output.loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError(f"[DebugSmoke] invalid loss at step={step}: {loss.item()}")

        logger.info(
            "[DebugSmokeTrainStep] "
            f"step={step} input_ids={tuple(batch['input_ids'].shape)} "
            f"attention_mask={tuple(batch['attention_mask'].shape)} "
            f"pos_ids={batch['img_ids'][:4]} neg_ids={batch['neg_img_ids'][:4]} "
            f"loss={float(loss.detach().item()):.6f}"
        )
        if args.add_ocr_info:
            logger.info(
                "[DebugSmokeTrainStep] aux_ocr_enabled=True "
                f"(built in forward from img_ids; batch_size={len(batch['img_ids'])})"
            )
        if debug_info and args.debug_smoke_log_every_step:
            model._log_debug_train_summary(debug_info, step=step)

        tracked_param = model.model.style_match_weight
        before = tracked_param.detach().clone()

        loss.backward()

        grad_report = {
            "shared_sticker_head": _module_grad_summary(model.model.shared_sticker_head),
            "style_head": _module_grad_summary(model.model.style_head),
            "expr_head": _module_grad_summary(model.model.expr_head),
            "query_head": _module_grad_summary(model.model.query_head),
            "style_match_head": _module_grad_summary(model.model.style_match_head),
            "expr_match_head": _module_grad_summary(model.model.expr_match_head),
            "bert_classifier_head": _module_grad_summary(model.model.bert.classifier),
            "bert_backbone_sample_layer0": _module_grad_summary(model.model.bert.bert.encoder.layer[0]),
            "bert_backbone_sample_layer_last": _module_grad_summary(model.model.bert.bert.encoder.layer[-1]),
        }
        for module_name, report in grad_report.items():
            logger.info(
                "[DebugGrad] "
                f"{module_name} mean={report['grad_norm_mean']:.6f} max={report['grad_norm_max']:.6f} "
                f"none={int(report['params_grad_none'])}/{int(report['params_total'])} "
                f"nonzero={int(report['grad_nonzero_count'])}"
            )
            if report["grad_nonzero_count"] == 0:
                logger.warning(f"[DebugGradWarning] zero gradients for module={module_name}")

        optimizer.step()
        scheduler.step()
        after = tracked_param.detach().clone()
        param_delta = float((after - before).abs().mean().item())
        logger.info(
            f"[DebugStepUpdate] step={step} style_match_weight_delta_abs_mean={param_delta:.8f} "
            f"lr={scheduler.get_last_lr()[0]:.8e}"
        )

    model.eval()
    model.model.prepare_for_test()
    with torch.no_grad():
        for eval_step in range(args.debug_smoke_eval_steps):
            raw_eval_batch = next(eval_iter)
            eval_batch = move_batch_to_device(raw_eval_batch, device)
            scores, labels, cands, eval_debug = model.run_eval_batch(
                eval_batch, return_debug=True
            )
            if eval_debug["uses_fused_logits_for_ranking"] is not True:
                raise ValueError("[DebugSmoke] evaluation does not use fused logits.")
            logger.info(
                "[DebugSmokeEvalStep] "
                f"step={eval_step} scores_shape={tuple(scores.shape)} "
                f"label={int(labels.item())} "
                f"cands={len(cands) if cands is not None else model.args.max_image_id} "
                f"fused_minus_base_abs_mean={eval_debug['fused_minus_base_abs_mean']:.6f} "
                f"formula={eval_debug['ranking_score_formula']}"
            )

    logger.info("[DebugSmoke] Completed successfully.")


def run_structured_main(args: StructuredArguments) -> None:
    if args.local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    pl.seed_everything(args.seed)

    if args.debug_smoke_test:
        run_debug_smoke_test(args)
        return

    if args.mode in {"train", "pretrain"}:
        model = StructuredPLModel(args)
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
        model = StructuredPLModel(args)
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
        f"Unsupported mode={args.mode} for structured script. Use train/pretrain/test/gen."
    )


def parse_structured_args() -> StructuredArguments:
    parser = HfArgumentParser(StructuredArguments)
    args, = parser.parse_args_into_dataclasses()
    return args
