import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Work around protobuf C-extension segfault in some environments
# (must be set before importing pytorch_lightning/tensorboard).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import distutils_tensorboard_shim  # noqa: F401

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import transformers
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import AdamW, HfArgumentParser

from main import (
    Arguments as LegacyArguments,
    Model as LegacyModel,
    PLDataLoader,
    attach_test_log_to_ckpt_version,
    attach_version_log_from_trainer,
    logger,
    save_final_checkpoint_from_trainer,
)
from metrics import MyAccuracy
from utils import try_create_dir


StickerId = Union[int, str]


def eval_retrieval_mode_label(test_with_cand: bool, max_cand_len: int) -> str:
    """Human-readable eval protocol tag for logs (R10/R20/Rall)."""
    if not test_with_cand:
        return "Rall(all-stickers)"
    if max_cand_len > 0:
        return f"R{max_cand_len}(candidates)"
    return "Rcand(candidates)"


def nonempty_batch_cands(cands: Optional[Sequence[Sequence[int]]]) -> bool:
    """True iff collate passed one row with at least one candidate (not [[]])."""
    return (
        cands is not None
        and len(cands) == 1
        and len(cands[0]) > 0
    )


_CAND_EVAL_ONLY_ERR = (
    "Eval batch has missing or empty `cands` while candidate_eval_only/test_with_cand "
    "is set; refusing silent full-bank ranking."
)


def structured_eval_metric_bundle() -> Dict[str, MyAccuracy]:
    """One eval split (R10 or R20 test) worth of MyAccuracy counters."""
    return {
        "acc5": MyAccuracy(),
        "acc30": MyAccuracy(),
        "acc90": MyAccuracy(),
        "acc_r10": MyAccuracy(),
        "acc_r20": MyAccuracy(),
        "map": MyAccuracy(),
    }


def attach_per_epoch_dual_test_eval(module: Any, args: Any) -> None:
    """
    If both paths are set, each training epoch runs validation on test R10 then test R20
    (no val split). See PLDataLoader.val_dataloader list return.
    """
    p10 = (getattr(args, "per_epoch_eval_test_r10_path", None) or "").strip()
    p20 = (getattr(args, "per_epoch_eval_test_r20_path", None) or "").strip()
    module._per_epoch_dual_test_eval = bool(p10 and p20)
    if module._per_epoch_dual_test_eval:
        module._eval_bundles = (
            structured_eval_metric_bundle(),
            structured_eval_metric_bundle(),
        )
        module._eval_max_cand_pair = [0, 0]
    else:
        module._eval_bundles = None


def _with_cand_data_path_if_exists(path: Optional[str]) -> Optional[str]:
    """If `path` has a known *_with_cand* counterpart on disk, return it; else None."""
    if not path:
        return None
    suffix_map = (
        ("validation_pair.json", "validation_pair_with_cand.json"),
        ("u_sticker_val_split.json", "u_sticker_val_split_with_cand.json"),
        (
            "release_val_u_sticker_format_int_with_cand.json",
            "release_val_u_sticker_format_int_with_cand_r10.json",
        ),
    )
    for old_suf, new_suf in suffix_map:
        if path.endswith(old_suf):
            cand_path = path[: -len(old_suf)] + new_suf
            if os.path.isfile(cand_path):
                return cand_path
            return None
    return None


def effective_lambda_expr_rank_loss_weight(args: Any) -> float:
    """Weight for expr *rank* auxiliary loss only.

    If ``lambda_expr_rank_loss`` is None (default), use ``lambda_expr`` — same as legacy behavior.
    If set (e.g. 0), fused scores still use ``lambda_expr`` in factorized minimal; only the rank
    auxiliary term changes.
    """
    raw = getattr(args, "lambda_expr_rank_loss", None)
    if raw is None:
        return float(getattr(args, "lambda_expr", 0.0) or 0.0)
    return float(raw)


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
    # More aggressive defaults so the structured branch is not numerically drowned out
    # by the legacy MM-BERT base score.
    lambda_fuse: Optional[float] = field(default=3.0)
    lambda_struct: Optional[float] = field(default=0.5)
    lambda_expr: Optional[float] = field(default=0.3)
    # Optional weight for expr *rank* auxiliary loss only. None = use lambda_expr (backward compatible).
    # Set to 0.0 to disable expr rank supervision while keeping lambda_expr on fused scores (factorized).
    lambda_expr_rank_loss: Optional[float] = field(default=None)
    warmup_ratio: Optional[float] = field(default=0.1)
    expr_margin: Optional[float] = field(default=0.1)
    style_neighbor_topk: Optional[int] = field(default=5)
    style_sampling_mode: Optional[str] = field(default="random_topk")
    expr_gate_mode: Optional[str] = field(default="sigmoid")
    # Which terms enter lambda_fuse * structured_term (besides s_base).
    # full: s_style + expr_mod; style_only: s_style; expr_only: expr_mod only.
    struct_fuse_mode: Optional[str] = field(default="full")
    strict_checkpoint_load: Optional[bool] = field(default=False)
    save_structured_test_outputs: Optional[bool] = field(default=True)
    debug_smoke_test: Optional[bool] = field(default=False)
    debug_smoke_train_steps: Optional[int] = field(default=3)
    debug_smoke_eval_steps: Optional[int] = field(default=1)
    debug_smoke_log_every_step: Optional[bool] = field(default=True)
    debug_compare_base_only: Optional[bool] = field(default=False)
    base_only: Optional[bool] = field(default=False)
    # Log how much base vs structured parts contribute to the fused match score (training batches).
    # 0 disables. Example: 200 -> one line every 200 optimizer steps.
    train_score_decomp_log_interval: Optional[int] = field(default=200)
    # compact: tqdm shows total, match, s_b, pm only (fits narrow terminals). full: all loss fields on bar.
    train_prog_bar_mode: Optional[str] = field(default="compact")
    trainer_precision: Optional[int] = field(default=32)
    allow_tf32: Optional[bool] = field(default=True)
    cudnn_benchmark: Optional[bool] = field(default=True)
    # Multiple BERT forwards per step (query + pair logits); checkpointing trades speed for VRAM.
    bert_gradient_checkpointing: Optional[bool] = field(default=True)
    # DDP only: None = PyTorch Lightning default (find_unused_parameters=True).
    # False can be used with DdpStaticGraphCallback when that callback is enabled.
    ddp_find_unused_parameters: Optional[bool] = field(default=None)
    # DDP multi-GPU: if True, register DdpStaticGraphCallback (helps some PT+checkpoint builds).
    # Default False: transformers 4.10 only sets config.gradient_checkpointing; combining static
    # graph with that reentrant checkpoint can cause EngineBase.run_backward NULL — keep off unless needed.
    ddp_static_graph_callback: Optional[bool] = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        if self.bert_gradient_checkpointing is None:
            self.bert_gradient_checkpointing = True
        if self.ddp_static_graph_callback is None:
            self.ddp_static_graph_callback = False
        # Structured experiments explicitly disable all legacy personalization paths.
        self.use_visual_personalization_token = False
        self.use_visual_history_attention = False
        if self.style_sampling_mode not in {"top1", "random_topk"}:
            raise ValueError(
                f"Unsupported style_sampling_mode={self.style_sampling_mode}. "
                "Use 'top1' or 'random_topk'."
            )
        if self.expr_gate_mode not in {"sigmoid", "none", "floor_half"}:
            raise ValueError(
                f"Unsupported expr_gate_mode={self.expr_gate_mode}. "
                "Use 'sigmoid', 'none', or 'floor_half'."
            )
        if self.struct_fuse_mode not in {"full", "style_only", "expr_only"}:
            raise ValueError(
                f"Unsupported struct_fuse_mode={self.struct_fuse_mode}. "
                "Use 'full', 'style_only', or 'expr_only'."
            )
        if self.structured_hidden_dim <= 0:
            raise ValueError("structured_hidden_dim must be > 0.")
        if self.lambda_fuse < 0:
            raise ValueError("lambda_fuse must be >= 0.")
        if self.lambda_struct < 0:
            raise ValueError("lambda_struct must be >= 0.")
        if self.lambda_expr < 0:
            raise ValueError("lambda_expr must be >= 0.")
        if self.lambda_expr_rank_loss is not None and float(self.lambda_expr_rank_loss) < 0:
            raise ValueError("lambda_expr_rank_loss must be >= 0 when set.")
        if self.expr_margin < 0:
            raise ValueError("expr_margin must be >= 0.")
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError("warmup_ratio must be in [0, 1].")
        if self.style_neighbor_topk < 0:
            raise ValueError("style_neighbor_topk must be >= 0.")
        if self.debug_smoke_train_steps <= 0:
            raise ValueError("debug_smoke_train_steps must be > 0.")
        if self.debug_smoke_eval_steps <= 0:
            raise ValueError("debug_smoke_eval_steps must be > 0.")
        if self.train_score_decomp_log_interval is not None and self.train_score_decomp_log_interval < 0:
            raise ValueError("train_score_decomp_log_interval must be >= 0 (0 = off).")
        if self.train_prog_bar_mode is None:
            self.train_prog_bar_mode = "compact"
        if self.train_prog_bar_mode not in {"compact", "full"}:
            raise ValueError("train_prog_bar_mode must be 'compact' or 'full'.")
        if self.trainer_precision not in {16, 32, 64}:
            raise ValueError("trainer_precision must be one of {16, 32, 64}.")
        # Training / pretrain validation always uses R10 (candidates), not full-corpus Rall.
        if self.mode in {"train", "pretrain"}:
            self.test_with_cand = True
            pe10 = (getattr(self, "per_epoch_eval_test_r10_path", None) or "").strip()
            pe20 = (getattr(self, "per_epoch_eval_test_r20_path", None) or "").strip()
            if pe10 and pe20:
                self.val_data_path = ""
                logger.info(
                    "[StructuredArgs] per-epoch eval on test R10+R20 only; val_data_path cleared."
                )
            else:
                old_val = self.val_data_path
                upgraded = _with_cand_data_path_if_exists(old_val)
                if upgraded is not None:
                    logger.info(
                        "[StructuredArgs] Training val uses fixed candidates: val_data_path %s -> %s",
                        old_val,
                        upgraded,
                    )
                    self.val_data_path = upgraded
                    if self.test_data_path == old_val:
                        self.test_data_path = upgraded


@dataclass
class StructuredForwardOutput:
    loss: torch.Tensor
    match_loss: torch.Tensor
    style_neighbor_loss: torch.Tensor
    orth_loss: torch.Tensor
    expr_rank_loss: torch.Tensor
    aux_warmup: torch.Tensor
    pos_fused_logits: torch.Tensor
    neg_fused_logits: torch.Tensor
    pos_style_score: torch.Tensor
    pos_expr_score: torch.Tensor
    neg_style_score: torch.Tensor
    neg_expr_score: torch.Tensor
    pos_style_gate: torch.Tensor
    neg_style_gate: torch.Tensor
    pos_expr_mod: torch.Tensor
    neg_expr_mod: torch.Tensor
    debug_info: Optional[Dict[str, Any]] = None
    # Detached scalars: what magnitudes drive pos/neg fused scores in this batch.
    score_decomp: Optional[Dict[str, float]] = None


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
class StructuredStickerModel(LegacyModel):
    """
    Reuses the original text/image backbones from `main.Model`,
    while keeping a unified dialogue query and a structured sticker-side
    decomposition in a low-hyperparameter configuration.
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
        self.style_gate_head = ScalarMatchHead(
            args.structured_hidden_dim, args.structured_dropout
        )
        self.loss_fct = nn.CrossEntropyLoss()
        self.neighbor_store = StyleNeighborStore.from_json(args.style_neighbors_path)
        self._neighbor_rng = random.Random(args.seed)
        if getattr(args, "bert_gradient_checkpointing", True):
            if hasattr(self.bert, "gradient_checkpointing_enable"):
                try:
                    self.bert.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    logger.info(
                        "[Structured] BERT gradient checkpointing enabled "
                        "(use_reentrant=False; lower VRAM; slower step)."
                    )
                except TypeError:
                    self.bert.gradient_checkpointing_enable()
                    logger.info(
                        "[Structured] BERT gradient checkpointing enabled (lower VRAM; slower step)."
                    )
            else:
                self.bert.config.gradient_checkpointing = True
                logger.info(
                    "[Structured] Set config.gradient_checkpointing=True on BERT."
                )

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

            batch_size = int(getattr(self.args, "prepare_batch_size", 512) or 512)
            workers = int(getattr(self.args, "prepare_workers", 8) or 8)
            if hasattr(self.img_clip, "model"):
                clip_device = next(self.img_clip.model.parameters()).device
            elif hasattr(self.img_clip, "_target_device"):
                clip_device = self.img_clip._target_device
            else:
                clip_device = next(self.bert.parameters()).device
            self.img_clip.eval()
            all_embs: List[torch.Tensor] = []

            def load_one(sticker_id: int):
                return sticker_id, self.get_image_obj(sticker_id)

            for start in range(0, self.args.max_image_id, batch_size):
                end = min(start + batch_size, self.args.max_image_id)
                ids_batch = list(range(start, end))
                img_objs = [None] * len(ids_batch)
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(load_one, idx): idx for idx in ids_batch}
                    for future in as_completed(futures):
                        idx, obj = future.result()
                        img_objs[idx - start] = obj
                img_tokens = self.img_clip.tokenize(img_objs)
                img_tokens = {
                    k: v.to(clip_device) if hasattr(v, "to") else v
                    for k, v in img_tokens.items()
                }
                all_embs.append(self.img_clip.forward(img_tokens)["sentence_embedding"])
            self.all_img_embs = torch.cat(all_embs, dim=0)
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
        dialogue_q: Optional[torch.Tensor] = None,
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
        # Unified dialogue query from pure text only.
        text_emb = self._get_text_word_embeddings(input_ids)
        outputs = self.bert.bert(
            inputs_embeds=text_emb,
            attention_mask=attention_mask,
            return_dict=True,
        )
        q = outputs.last_hidden_state[:, 0, :]
        return self.query_head(q)

    def decompose_sticker(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Shared sticker base -> style-oriented stable component + response-oriented
        # expression component. No hard normalization is applied here.
        u = self.shared_sticker_head(h)
        c = self.style_head(u)
        a = self.expr_head(u)
        return u, c, a

    def compute_style_compatibility(self, q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.style_match_head(q, c)

    def compute_expression_compatibility(
        self, q: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        return self.expr_match_head(q, a)

    def compute_style_gate(self, q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Style-conditioned expression modulation gate in (0, 1).
        return torch.sigmoid(self.style_gate_head(q, c))

    def modulate_expression_score(
        self, expr_score: torch.Tensor, style_gate: torch.Tensor
    ) -> torch.Tensor:
        if self.args.expr_gate_mode == "none":
            return expr_score
        if self.args.expr_gate_mode == "floor_half":
            return (0.5 + 0.5 * style_gate) * expr_score
        return style_gate * expr_score

    def describe_expr_gate_mode(self) -> str:
        if self.args.expr_gate_mode == "none":
            return "expr_mod = s_expr"
        if self.args.expr_gate_mode == "floor_half":
            return "expr_mod = (0.5 + 0.5 * style_gate) * s_expr"
        return "expr_mod = style_gate * s_expr"

    def expr_mod_subformula(self) -> str:
        if self.args.expr_gate_mode == "none":
            return "s_expr"
        if self.args.expr_gate_mode == "floor_half":
            return "(0.5 + 0.5 * style_gate) * s_expr"
        return "style_gate * s_expr"

    def structured_term_formula(self) -> str:
        em = self.expr_mod_subformula()
        if self.args.struct_fuse_mode == "style_only":
            return "s_style"
        if self.args.struct_fuse_mode == "expr_only":
            return em
        return f"s_style + ({em})"

    def compute_base_score(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, 1] - logits[:, 0]

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
        if self.args.base_only:
            final_score = base_score
        else:
            final_score = base_score + self.args.lambda_fuse * structured_term
        return final_score, structured_term, expr_mod

    def score_to_logits(self, final_score: torch.Tensor) -> torch.Tensor:
        return torch.stack([-final_score, final_score], dim=-1)

    def _compute_pair_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_ids: Sequence[int],
        img_emb: torch.Tensor,
        dialogue_q: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_emb = self._get_text_word_embeddings(input_ids)
        input_emb, full_attention_mask, token_type_ids = self._build_multimodal_inputs(
            text_emb=text_emb,
            attention_mask=attention_mask,
            img_emb=img_emb,
            img_ids=img_ids,
            dialogue_q=dialogue_q,
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
        # Weakly supervised style neighborhood only constrains the
        # style-oriented stable component c_i.
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

        neighbor_h = self.get_emb_by_imgids(valid_neighbor_ids)
        _, neighbor_c, _ = self.decompose_sticker(neighbor_h)
        valid_c = c_i[torch.tensor(valid_indices, device=c_i.device, dtype=torch.long)]
        if self.args.debug_smoke_test:
            assert (
                valid_c.size(0) == neighbor_c.size(0)
            ), "style neighbor c_i / c_j batch size mismatch"
        idx_t = torch.tensor(valid_indices, device=c_i.device, dtype=torch.long)
        per_neighbor = 1.0 - self._cosine_similarity_from_raw(valid_c, neighbor_c)
        sample_loss[idx_t] = per_neighbor.to(dtype=sample_loss.dtype)
        debug_meta["valid_style_neighbor_pairs"] = len(valid_neighbor_ids)
        debug_meta["sampled_neighbor_pairs_preview"] = sampled_pairs[:8]
        return sample_loss.mean(), debug_meta

    def compute_orth_loss(self, c_i: torch.Tensor, a_i: torch.Tensor) -> torch.Tensor:
        return torch.pow(self._cosine_similarity_from_raw(c_i, a_i), 2).mean()

    def compute_expr_rank_loss(
        self, pos_expr_mod: torch.Tensor, neg_expr_mod: torch.Tensor
    ) -> torch.Tensor:
        """Margin rank on gated expression scores (same as used in fuse_final_score)."""
        margin = float(self.args.expr_margin)
        return F.relu(margin - pos_expr_mod + neg_expr_mod).mean()

    def compute_aux_warmup(
        self, global_step: int, total_steps: Optional[int]
    ) -> torch.Tensor:
        if total_steps is None or total_steps <= 0:
            total_steps = 1
        warmup_steps = int(float(self.args.warmup_ratio) * float(total_steps))
        if warmup_steps <= 0:
            return torch.tensor(1.0, dtype=torch.float32)
        return torch.tensor(
            min(1.0, float(global_step) / float(warmup_steps)), dtype=torch.float32
        )

    def _batch_match_score_decomp(
        self,
        pos_base_score: torch.Tensor,
        neg_base_score: torch.Tensor,
        pos_style_score: torch.Tensor,
        neg_style_score: torch.Tensor,
        pos_expr_mod: torch.Tensor,
        neg_expr_mod: torch.Tensor,
        pos_structured_term: torch.Tensor,
        neg_structured_term: torch.Tensor,
        pos_final_score: torch.Tensor,
        neg_final_score: torch.Tensor,
    ) -> Dict[str, float]:
        """Scalar summary: which terms dominate s_final for the contrastive match objective."""
        with torch.no_grad():
            lam = float(self.args.lambda_fuse)

            def pair_mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
                x = torch.cat([a.detach().float().reshape(-1), b.detach().float().reshape(-1)])
                return float(x.abs().mean().item())

            base_mabs = pair_mean_abs(pos_base_score, neg_base_score)
            rank_margin = float(
                (pos_final_score - neg_final_score).detach().float().mean().item()
            )

            if self.args.base_only:
                return {
                    "base_mabs": base_mabs,
                    "lam_struct_mabs": 0.0,
                    "lam_style_mabs": 0.0,
                    "lam_expr_mod_mabs": 0.0,
                    "struct_over_base": 0.0,
                    "rank_margin_mean": rank_margin,
                }

            lam_struct_mabs = pair_mean_abs(
                lam * pos_structured_term, lam * neg_structured_term
            )
            mode = self.args.struct_fuse_mode
            if mode == "full":
                lam_style_mabs = pair_mean_abs(lam * pos_style_score, lam * neg_style_score)
                lam_expr_mabs = pair_mean_abs(lam * pos_expr_mod, lam * neg_expr_mod)
            elif mode == "style_only":
                lam_style_mabs = pair_mean_abs(lam * pos_style_score, lam * neg_style_score)
                lam_expr_mabs = 0.0
            else:
                lam_style_mabs = 0.0
                lam_expr_mabs = pair_mean_abs(lam * pos_expr_mod, lam * neg_expr_mod)

            struct_over_base = lam_struct_mabs / max(base_mabs, 1e-8)
            return {
                "base_mabs": base_mabs,
                "lam_struct_mabs": lam_struct_mabs,
                "lam_style_mabs": lam_style_mabs,
                "lam_expr_mod_mabs": lam_expr_mabs,
                "struct_over_base": struct_over_base,
                "rank_margin_mean": rank_margin,
            }

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
        global_step: int = 0,
        total_steps: Optional[int] = None,
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
            dialogue_q=q,
        )
        neg_logits = self._compute_pair_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            img_ids=neg_img_ids,
            img_emb=neg_h,
            dialogue_q=q,
        )

        pos_u, pos_c, pos_a = self.decompose_sticker(pos_h)
        neg_u, neg_c, neg_a = self.decompose_sticker(neg_h)
        if self.args.debug_smoke_test:
            assert pos_c.shape == pos_a.shape, "pos c/a shape mismatch"
            assert neg_c.shape == neg_a.shape, "neg c/a shape mismatch"

        pos_style_score = self.compute_style_compatibility(q, pos_c)
        neg_style_score = self.compute_style_compatibility(q, neg_c)
        pos_expr_score = self.compute_expression_compatibility(q, pos_a)
        neg_expr_score = self.compute_expression_compatibility(q, neg_a)
        pos_style_gate = self.compute_style_gate(q, pos_c)
        neg_style_gate = self.compute_style_gate(q, neg_c)
        if self.args.debug_smoke_test:
            assert (
                pos_style_score.shape == pos_expr_score.shape
            ), "pos style/expr score shape mismatch"
            assert (
                neg_style_score.shape == neg_expr_score.shape
            ), "neg style/expr score shape mismatch"

        pos_base_score = self.compute_base_score(pos_logits)
        neg_base_score = self.compute_base_score(neg_logits)
        pos_final_score, pos_structured_term, pos_expr_mod = self.fuse_final_score(
            pos_base_score, pos_style_score, pos_expr_score, pos_style_gate
        )
        neg_final_score, neg_structured_term, neg_expr_mod = self.fuse_final_score(
            neg_base_score, neg_style_score, neg_expr_score, neg_style_gate
        )
        pos_fused_logits = self.score_to_logits(pos_final_score)
        neg_fused_logits = self.score_to_logits(neg_final_score)
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
        # Align with inference: rank loss on gated expression (style_gate * expr_score), not raw expr_score.
        expr_rank_loss = self.compute_expr_rank_loss(pos_expr_mod, neg_expr_mod)
        aux_warmup = self.compute_aux_warmup(global_step=global_step, total_steps=total_steps).to(
            device=device
        )
        structured_aux = self.args.lambda_struct * (style_neighbor_loss + orth_loss)
        expr_aux = effective_lambda_expr_rank_loss_weight(self.args) * expr_rank_loss
        if self.args.base_only:
            structured_aux = structured_aux * 0.0
            expr_aux = expr_aux * 0.0
        total_loss = match_loss + aux_warmup * (structured_aux + expr_aux)
        if self.args.debug_smoke_test:
            assert not torch.isnan(total_loss).any(), "final_loss has NaN"
            assert not torch.isinf(total_loss).any(), "final_loss has Inf"

        debug_info = None
        if self.args.debug_smoke_test:
            pos_score_delta = (pos_final_score - pos_base_score).detach()
            neg_score_delta = (neg_final_score - neg_base_score).detach()
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
                "pos_base_score_stats": self._tensor_debug_stats(pos_base_score),
                "neg_base_score_stats": self._tensor_debug_stats(neg_base_score),
                "pos_style_score_stats": self._tensor_debug_stats(pos_style_score),
                "neg_style_score_stats": self._tensor_debug_stats(neg_style_score),
                "pos_expr_score_stats": self._tensor_debug_stats(pos_expr_score),
                "neg_expr_score_stats": self._tensor_debug_stats(neg_expr_score),
                "pos_style_gate_stats": self._tensor_debug_stats(pos_style_gate),
                "neg_style_gate_stats": self._tensor_debug_stats(neg_style_gate),
                "pos_expr_mod_stats": self._tensor_debug_stats(pos_expr_mod),
                "neg_expr_mod_stats": self._tensor_debug_stats(neg_expr_mod),
                "pos_structured_term_stats": self._tensor_debug_stats(pos_structured_term),
                "neg_structured_term_stats": self._tensor_debug_stats(neg_structured_term),
                "lambda_fuse": float(self.args.lambda_fuse),
                "lambda_struct": float(self.args.lambda_struct),
                "lambda_expr": float(self.args.lambda_expr),
                "aux_warmup": float(aux_warmup.detach().item()),
                "pos_logits_stats": self._tensor_debug_stats(pos_logits),
                "neg_logits_stats": self._tensor_debug_stats(neg_logits),
                "pos_fused_logits_stats": self._tensor_debug_stats(pos_fused_logits),
                "neg_fused_logits_stats": self._tensor_debug_stats(neg_fused_logits),
                "pos_score_delta_stats": self._tensor_debug_stats(pos_score_delta),
                "neg_score_delta_stats": self._tensor_debug_stats(neg_score_delta),
                "pos_score_delta_abs_mean": float(pos_score_delta.abs().mean().item()),
                "neg_score_delta_abs_mean": float(neg_score_delta.abs().mean().item()),
                "pos_match_loss": float(pos_loss.detach().item()),
                "neg_match_loss": float(neg_loss.detach().item()),
                "match_loss": float(match_loss.detach().item()),
                "style_neighbor_loss": float(style_neighbor_loss.detach().item()),
                "orth_loss": float(orth_loss.detach().item()),
                "expr_rank_loss": float(expr_rank_loss.detach().item()),
                "final_loss": float(total_loss.detach().item()),
                "pos_style_score_mean": float(pos_style_score.detach().mean().item()),
                "neg_style_score_mean": float(neg_style_score.detach().mean().item()),
                "pos_expr_score_mean": float(pos_expr_score.detach().mean().item()),
                "neg_expr_score_mean": float(neg_expr_score.detach().mean().item()),
                "pos_style_gate_mean": float(pos_style_gate.detach().mean().item()),
                "neg_style_gate_mean": float(neg_style_gate.detach().mean().item()),
                "style_neighbor_meta": style_neighbor_meta,
            }

        score_decomp = self._batch_match_score_decomp(
            pos_base_score,
            neg_base_score,
            pos_style_score,
            neg_style_score,
            pos_expr_mod,
            neg_expr_mod,
            pos_structured_term,
            neg_structured_term,
            pos_final_score,
            neg_final_score,
        )

        return StructuredForwardOutput(
            loss=total_loss,
            match_loss=match_loss,
            style_neighbor_loss=style_neighbor_loss,
            orth_loss=orth_loss,
            expr_rank_loss=expr_rank_loss,
            aux_warmup=aux_warmup,
            pos_fused_logits=pos_fused_logits,
            neg_fused_logits=neg_fused_logits,
            pos_style_score=pos_style_score,
            pos_expr_score=pos_expr_score,
            neg_style_score=neg_style_score,
            neg_expr_score=neg_expr_score,
            pos_style_gate=pos_style_gate,
            neg_style_gate=neg_style_gate,
            pos_expr_mod=pos_expr_mod,
            neg_expr_mod=neg_expr_mod,
            debug_info=debug_info,
            score_decomp=score_decomp,
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

        # s_base is the original candidate-aware score from the legacy multimodal
        # BERT branch, where dialogue text is paired with each candidate sticker.
        base_logits = self._compute_pair_logits(
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            img_ids=candidate_ids,
            img_emb=candidate_h,
        )

        candidate_q = q.repeat(img_num, 1)
        _, candidate_c, candidate_a = self.decompose_sticker(candidate_h)
        style_score = self.compute_style_compatibility(candidate_q, candidate_c)
        expr_score = self.compute_expression_compatibility(candidate_q, candidate_a)
        style_gate = self.compute_style_gate(candidate_q, candidate_c)
        base_score = self.compute_base_score(base_logits)
        final_score, structured_term, expr_mod = self.fuse_final_score(
            base_score, style_score, expr_score, style_gate
        )
        fused_logits = self.score_to_logits(final_score)
        final_scores = final_score.unsqueeze(0)
        labels = torch.tensor(img_ids, dtype=torch.long, device=device)
        if not return_debug:
            return final_scores, labels, candidate_ids if use_cands else None
        eval_debug = {
            "uses_fused_logits_for_ranking": not self.args.base_only,
            "base_logits_stats": self._tensor_debug_stats(base_logits),
            "base_score_stats": self._tensor_debug_stats(base_score),
            "fused_logits_stats": self._tensor_debug_stats(fused_logits),
            "final_scores_stats": self._tensor_debug_stats(final_scores),
            "style_score_stats": self._tensor_debug_stats(style_score),
            "expr_score_stats": self._tensor_debug_stats(expr_score),
            "style_gate_stats": self._tensor_debug_stats(style_gate),
            "expr_mod_stats": self._tensor_debug_stats(expr_mod),
            "structured_term_stats": self._tensor_debug_stats(structured_term),
            "scaled_structured_term_stats": self._tensor_debug_stats(
                (float(self.args.lambda_fuse) * structured_term)
            ),
            "fused_minus_base_abs_mean": float((final_score - base_score).abs().mean().item()),
            "top1_flipped": bool(torch.argmax(base_score).item() != torch.argmax(final_score).item()),
            "base_only_scores_stats": self._tensor_debug_stats(base_score.unsqueeze(0)),
            "lambda_fuse": float(self.args.lambda_fuse),
            "expr_gate_mode": self.args.expr_gate_mode,
            "struct_fuse_mode": self.args.struct_fuse_mode,
            "ranking_score_formula": (
                f"s_final = s_base + lambda_fuse * ({self.structured_term_formula()})"
                if not self.args.base_only
                else "s_final = s_base (base_only analysis)"
            ),
            "candidate_count": int(len(candidate_ids)),
        }
        return final_scores, labels, candidate_ids if use_cands else None, eval_debug


class StructuredPLModel(pl.LightningModule):
    def __init__(self, args: StructuredArguments):
        super().__init__()
        self.args = args
        self.model = StructuredStickerModel(args)
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
        self._reset_eval_diagnostics()
        self._match_score_sb_ema: Optional[float] = None
        self._match_score_margin_ema: Optional[float] = None

    def _reset_eval_diagnostics(self) -> None:
        self.eval_diag_count = 0
        self.eval_base_score_std_sum = 0.0
        self.eval_style_score_std_sum = 0.0
        self.eval_expr_mod_std_sum = 0.0
        self.eval_structured_term_std_sum = 0.0
        self.eval_scaled_structured_term_std_sum = 0.0
        self.eval_struct_over_base_ratio_sum = 0.0
        self.eval_fused_minus_base_abs_mean_sum = 0.0
        self.eval_top1_flip_count = 0

    def _update_eval_diagnostics(self, eval_debug: Dict[str, Any]) -> None:
        base_std = float(eval_debug["base_score_stats"]["std"])
        style_std = float(eval_debug["style_score_stats"]["std"])
        expr_mod_std = float(eval_debug["expr_mod_stats"]["std"])
        structured_std = float(eval_debug["structured_term_stats"]["std"])
        scaled_structured_std = float(eval_debug["scaled_structured_term_stats"]["std"])
        ratio = scaled_structured_std / max(base_std, 1e-8)

        self.eval_diag_count += 1
        self.eval_base_score_std_sum += base_std
        self.eval_style_score_std_sum += style_std
        self.eval_expr_mod_std_sum += expr_mod_std
        self.eval_structured_term_std_sum += structured_std
        self.eval_scaled_structured_term_std_sum += scaled_structured_std
        self.eval_struct_over_base_ratio_sum += ratio
        self.eval_fused_minus_base_abs_mean_sum += float(eval_debug["fused_minus_base_abs_mean"])
        self.eval_top1_flip_count += int(bool(eval_debug["top1_flipped"]))

    def _eval_diagnostics_summary(self) -> Dict[str, float]:
        if self.eval_diag_count <= 0:
            return {}
        denom = float(self.eval_diag_count)
        return {
            "base_score_std": self.eval_base_score_std_sum / denom,
            "style_score_std": self.eval_style_score_std_sum / denom,
            "expr_mod_std": self.eval_expr_mod_std_sum / denom,
            "structured_term_std": self.eval_structured_term_std_sum / denom,
            "scaled_structured_term_std": self.eval_scaled_structured_term_std_sum / denom,
            "struct_over_base_ratio": self.eval_struct_over_base_ratio_sum / denom,
            "fused_minus_base_abs_mean": self.eval_fused_minus_base_abs_mean_sum / denom,
            "top1_flip_rate": self.eval_top1_flip_count / denom,
        }

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
            f"lambda_fuse={debug_info['lambda_fuse']:.4f} "
            f"lambda_struct={debug_info['lambda_struct']:.4f} "
            f"lambda_expr={debug_info['lambda_expr']:.4f} "
            f"w_t={debug_info['aux_warmup']:.4f} "
            f"delta_pos_mean={debug_info['pos_score_delta_stats']['mean']:.4f} "
            f"delta_neg_mean={debug_info['neg_score_delta_stats']['mean']:.4f} "
            f"match={debug_info['match_loss']:.4f} "
            f"style={debug_info['style_neighbor_loss']:.4f} "
            f"orth={debug_info['orth_loss']:.4f} "
            f"expr_rank={debug_info['expr_rank_loss']:.4f} "
            f"final={debug_info['final_loss']:.4f} "
            f"valid_style_pairs={debug_info['style_neighbor_meta']['valid_style_neighbor_pairs']}/"
            f"{debug_info['style_neighbor_meta']['total_positive_samples']}"
        )
        logger.info("[DebugTrain] " + self._format_stats_line("q", debug_info["q_stats"]))
        logger.info("[DebugTrain] " + self._format_stats_line("pos_u", debug_info["pos_u_stats"]))
        logger.info("[DebugTrain] " + self._format_stats_line("pos_c", debug_info["pos_c_stats"]))
        logger.info("[DebugTrain] " + self._format_stats_line("pos_a", debug_info["pos_a_stats"]))
        logger.info(
            "[DebugTrain] "
            + self._format_stats_line("pos_base_score", debug_info["pos_base_score_stats"])
        )
        logger.info(
            "[DebugTrain] "
            + self._format_stats_line("pos_style_score", debug_info["pos_style_score_stats"])
        )
        logger.info(
            "[DebugTrain] "
            + self._format_stats_line("pos_expr_score", debug_info["pos_expr_score_stats"])
        )
        logger.info(
            "[DebugTrain] "
            + self._format_stats_line("pos_style_gate", debug_info["pos_style_gate_stats"])
        )
        logger.info(
            "[DebugTrain] "
            + self._format_stats_line("pos_expr_mod", debug_info["pos_expr_mod_stats"])
        )
        logger.info(
            "[DebugTrain] "
            f"pos_style_score_mean={debug_info['pos_style_score_mean']:.4f} "
            f"neg_style_score_mean={debug_info['neg_style_score_mean']:.4f} "
            f"pos_expr_score_mean={debug_info['pos_expr_score_mean']:.4f} "
            f"neg_expr_score_mean={debug_info['neg_expr_score_mean']:.4f} "
            f"pos_style_gate_mean={debug_info['pos_style_gate_mean']:.4f} "
            f"neg_style_gate_mean={debug_info['neg_style_gate_mean']:.4f}"
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
            global_step=int(self.global_step),
            total_steps=int(self.num_training_steps),
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

    def on_train_start(self) -> None:
        """One-time legend for tqdm + [MatchScoreDecomp] fields (see train.log if terminal wraps)."""
        sys.stderr.write("\n")
        logger.info(
            "[TrainMetricsLegend] Match objective uses fused score s_final (per sample): "
            "s_base = BERT logit margin (pos class - neg class); "
            "structured_term from struct_fuse_mode (e.g. full => s_style + expr_mod); "
            "s_final = s_base + lambda_fuse * structured_term (unless base_only)."
        )
        logger.info(
            "[TrainMetricsLegend] tqdm (compact): "
            "total=full loss; match=CE on fused pos/neg logits; "
            "s_b=EMA( mean|lambda_fuse*structured_term| / mean|s_base| ); "
            "pm=EMA( mean(s_final_pos - s_final_neg) ). "
            "Higher pm => on average positive sticker scores above negative in this batch."
        )
        logger.info(
            "[TrainMetricsLegend] [MatchScoreDecomp] every train_score_decomp_log_interval steps: "
            "instant batch stats (not EMA). "
            "|s_base|~=mean abs s_base over pos+neg; "
            "lambda|struct|~=mean abs(lambda_fuse*structured_term); "
            "ratio~=lambda|struct|/|s_base|; "
            "lambda|style| / lambda|expr_mod| split the structured term when mode=full. "
            "Other losses (style neighbor, orth, expr rank) are aux; they shape embeddings, "
            "not add directly into s_final."
        )
        if self.args.train_prog_bar_mode == "compact":
            logger.info(
                "[TrainMetricsLegend] style/orth/expr/saux/eaux/wt are logged to TensorBoard as "
                "train_* only (compact bar). Use --train_prog_bar_mode full to show them on tqdm."
            )
        return super().on_train_start()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.run_train_batch(batch)
        structured_aux = self.args.lambda_struct * (
            outputs.style_neighbor_loss + outputs.orth_loss
        )
        expr_aux = effective_lambda_expr_rank_loss_weight(self.args) * outputs.expr_rank_loss
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

        # tqdm: default compact bar so narrow terminals still show score-decomp fields (s_b, pm).
        # Full losses stay in train_* (TensorBoard). Use --train_prog_bar_mode full for the old bar.
        _pb = self.args.train_prog_bar_mode == "full"
        self.log("total", outputs.loss.detach(), prog_bar=True, logger=False)
        self.log("match", outputs.match_loss.detach(), prog_bar=True, logger=False)
        self.log("style", outputs.style_neighbor_loss.detach(), prog_bar=_pb, logger=False)
        self.log("orth", outputs.orth_loss.detach(), prog_bar=_pb, logger=False)
        self.log("expr", outputs.expr_rank_loss.detach(), prog_bar=_pb, logger=False)
        self.log("saux", weighted_struct_aux.detach(), prog_bar=_pb, logger=False)
        self.log("eaux", weighted_expr_aux.detach(), prog_bar=_pb, logger=False)
        self.log("wt", outputs.aux_warmup.detach(), prog_bar=_pb, logger=False)

        decomp = outputs.score_decomp
        if decomp is not None:
            a = 0.05
            sb = decomp["struct_over_base"]
            pm = decomp["rank_margin_mean"]
            if self._match_score_sb_ema is None:
                self._match_score_sb_ema = sb
                self._match_score_margin_ema = pm
            else:
                self._match_score_sb_ema = a * sb + (1.0 - a) * self._match_score_sb_ema
                self._match_score_margin_ema = a * pm + (1.0 - a) * self._match_score_margin_ema
            dev = outputs.loss.device
            self.log(
                "s_b",
                torch.tensor(self._match_score_sb_ema, device=dev, dtype=torch.float32),
                prog_bar=True,
                logger=False,
            )
            self.log(
                "pm",
                torch.tensor(self._match_score_margin_ema, device=dev, dtype=torch.float32),
                prog_bar=True,
                logger=False,
            )

        log_iv = int(self.args.train_score_decomp_log_interval or 0)
        if (
            log_iv > 0
            and decomp is not None
            and int(self.global_step) % log_iv == 0
        ):
            # Avoid glueing log lines to the tqdm carriage-return line.
            sys.stderr.write("\n")
            st = int(self.global_step)
            if self.args.base_only:
                logger.info("[MatchScoreDecomp] step=%d  (base_only: s_final = s_base)", st)
                logger.info(
                    "  |s_base|~%.4f  mean(s_pos-s_neg)~%.4f",
                    decomp["base_mabs"],
                    decomp["rank_margin_mean"],
                )
            else:
                logger.info(
                    "[MatchScoreDecomp] step=%d  s_final = s_base + lambda_fuse * (%s)",
                    st,
                    self.model.structured_term_formula(),
                )
                logger.info(
                    "  lambda_fuse=%.3f  |s_base|~%.4f  lambda|struct|~%.4f  "
                    "ratio~%.4f  mean(s_pos-s_neg)~%.4f",
                    float(self.args.lambda_fuse),
                    decomp["base_mabs"],
                    decomp["lam_struct_mabs"],
                    decomp["struct_over_base"],
                    decomp["rank_margin_mean"],
                )
                logger.info(
                    "  lambda|style|~%.4f  lambda|expr_mod|~%.4f",
                    decomp["lam_style_mabs"],
                    decomp["lam_expr_mod_mabs"],
                )

        if self.args.debug_smoke_test and outputs.debug_info is not None:
            if self.args.debug_smoke_log_every_step:
                self._log_debug_train_summary(outputs.debug_info, int(batch_idx))
            self._debug_assert_tensor_finite("train_loss", outputs.loss)
        return outputs.loss

    def on_validation_epoch_start(self) -> None:
        self.model.prepare_for_test()
        self._reset_eval_diagnostics()
        if getattr(self, "_per_epoch_dual_test_eval", False):
            self._eval_max_cand_pair = [0, 0]
        else:
            self._eval_max_cand_len = 0
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self) -> None:
        self.model.prepare_for_test()
        self._reset_eval_diagnostics()
        if getattr(self, "_per_epoch_dual_test_eval", False):
            self._eval_max_cand_pair = [0, 0]
        else:
            self._eval_max_cand_len = 0
        return super().on_test_epoch_start()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        scores, labels, cands, eval_debug = self.run_eval_batch(batch, return_debug=True)
        # Sanity check runs ~2 batches only; scale stats there are not comparable to full val.
        tr = getattr(self, "trainer", None)
        if tr is None or not getattr(tr, "sanity_checking", False):
            if not getattr(self, "_per_epoch_dual_test_eval", False) or int(dataloader_idx) == 0:
                self._update_eval_diagnostics(eval_debug)
        if self.args.debug_smoke_test or self.args.debug_compare_base_only:
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
            logger.info(
                "[DebugEval] "
                + self._format_stats_line("style_gate", eval_debug["style_gate_stats"])
            )
            logger.info(
                "[DebugEval] "
                + self._format_stats_line("expr_mod", eval_debug["expr_mod_stats"])
            )
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

        if getattr(self, "_per_epoch_dual_test_eval", False):
            di = int(dataloader_idx)
            b = self._eval_bundles[di]
            b["acc5"].update(cor1, tot1)
            b["acc30"].update(cor2, tot2)
            b["acc90"].update(cor5, tot5)
            b["map"].update(map_sum, map_tot)
            if cands:
                nc = len(cands)
                self._eval_max_cand_pair[di] = max(self._eval_max_cand_pair[di], nc)
                if nc >= 10:
                    c10, t10 = self.compute_acc(idx, labels, 10)
                    b["acc_r10"].update(c10, t10)
                if nc >= 20:
                    c20, t20 = self.compute_acc(idx, labels, 20)
                    b["acc_r20"].update(c20, t20)
        else:
            self.valtest_acc5.update(cor1, tot1)
            self.valtest_acc30.update(cor2, tot2)
            self.valtest_acc90.update(cor5, tot5)
            self.valtest_map.update(map_sum, map_tot)
            if cands:
                nc = len(cands)
                self._eval_max_cand_len = max(self._eval_max_cand_len, nc)
                if nc >= 10:
                    c10, t10 = self.compute_acc(idx, labels, 10)
                    self.valtest_acc_r10.update(c10, t10)
                if nc >= 20:
                    c20, t20 = self.compute_acc(idx, labels, 20)
                    self.valtest_acc_r20.update(c20, t20)

        metrics = [
            (cor1 / tot1).item(),
            (cor2 / tot2).item(),
            (cor5 / tot5).item(),
            (map_sum / map_tot).item(),
        ]
        return metrics, return_preds, return_label

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.validation_step(batch, batch_idx, 0)

    def _structured_eval_epoch_log_and_reset_metrics(self) -> None:
        if getattr(self, "_per_epoch_dual_test_eval", False):
            for di, tag in enumerate(["test-R10", "test-R20"]):
                b = self._eval_bundles[di]
                if int(b["acc5"].total) == 0:
                    for _k, m in b.items():
                        m.reset()
                    continue
                nc = int(self._eval_max_cand_pair[di])
                mode = eval_retrieval_mode_label(bool(self.args.test_with_cand), nc)
                extra_r = ""
                if int(b["acc_r10"].total) > 0:
                    extra_r += f" r@10={b['acc_r10'].compute():.4f}"
                if int(b["acc_r20"].total) > 0:
                    extra_r += f" r@20={b['acc_r20'].compute():.4f}"
                logger.info(
                    f"\n[StructuredEval] epoch={self.current_epoch} split={tag} mode={mode} "
                    f"total={b['acc5'].total} "
                    f"r@1={b['acc5'].compute():.4f} r@2={b['acc30'].compute():.4f} "
                    f"r@5={b['acc90'].compute():.4f}{extra_r} mrr={b['map'].compute():.4f}"
                )
                for _k, m in b.items():
                    m.reset()
        else:
            eval_mode = eval_retrieval_mode_label(
                bool(self.args.test_with_cand), int(self._eval_max_cand_len)
            )
            extra_r = ""
            if int(self.valtest_acc_r10.total) > 0:
                extra_r += f" r@10={self.valtest_acc_r10.compute():.4f}"
            if int(self.valtest_acc_r20.total) > 0:
                extra_r += f" r@20={self.valtest_acc_r20.compute():.4f}"
            logger.info(
                f"\n[StructuredEval] epoch={self.current_epoch} mode={eval_mode} total={self.valtest_acc5.total} "
                f"r@1={self.valtest_acc5.compute():.4f} r@2={self.valtest_acc30.compute():.4f} "
                f"r@5={self.valtest_acc90.compute():.4f}{extra_r} mrr={self.valtest_map.compute():.4f}"
            )
            self.valtest_acc5.reset()
            self.valtest_acc30.reset()
            self.valtest_acc90.reset()
            self.valtest_acc_r10.reset()
            self.valtest_acc_r20.reset()
            self.valtest_map.reset()

    def on_validation_epoch_end(self) -> None:
        self._structured_eval_epoch_log_and_reset_metrics()
        diag = self._eval_diagnostics_summary()
        if diag:
            logger.info(
                "[StructuredScale] "
                f"epoch={self.current_epoch} "
                f"n_batches={self.eval_diag_count} "
                f"base_std={diag['base_score_std']:.4f} "
                f"style_std={diag['style_score_std']:.4f} "
                f"expr_mod_std={diag['expr_mod_std']:.4f} "
                f"struct_std={diag['structured_term_std']:.4f} "
                f"scaled_struct_std={diag['scaled_structured_term_std']:.4f} "
                f"struct/base={diag['struct_over_base_ratio']:.4f} "
                f"delta_abs={diag['fused_minus_base_abs_mean']:.4f} "
                f"top1_flip={diag['top1_flip_rate']:.4f}"
            )
        elif (
            not getattr(self, "_per_epoch_dual_test_eval", False)
            and int(self.valtest_acc5.total) > 0
            and int(self.valtest_acc5.total) < 64
        ):
            logger.info(
                "[StructuredScale] skipped (Lightning sanity check uses few batches; "
                "scale line appears after full validation with n_batches≈val set size)"
            )
        self._reset_eval_diagnostics()
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


def maybe_set_ddp_static_graph(trainer: Optional[pl.Trainer]) -> None:
    """Enable DDP static graph for gradient checkpoint + reentrant backward (idempotent).

    BERT ``gradient_checkpointing`` under multi-GPU DDP otherwise may raise
    ``RuntimeError: Expected to mark a variable ready only once``.
    """
    if trainer is None or getattr(trainer, "world_size", 1) <= 1:
        return
    plugins: List[Any] = []
    ttp = getattr(trainer, "training_type_plugin", None)
    if ttp is not None:
        plugins.append(ttp)
    acc = getattr(trainer, "accelerator", None)
    if acc is not None:
        ttp2 = getattr(acc, "training_type_plugin", None)
        if ttp2 is not None and ttp2 not in plugins:
            plugins.append(ttp2)
    for plugin in plugins:
        wrapped = getattr(plugin, "_model", None)
        if wrapped is None or not isinstance(wrapped, DistributedDataParallel):
            continue
        if getattr(wrapped, "static_graph", False):
            return
        if hasattr(wrapped, "_set_static_graph"):
            wrapped._set_static_graph()
            return


class DdpStaticGraphCallback(Callback):
    """Trainer callback wrapper for :func:`maybe_set_ddp_static_graph`."""

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        maybe_set_ddp_static_graph(trainer)


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
        "precision": int(getattr(args, "trainer_precision", 32) or 32),
    }
    callbacks: List[Callback] = []
    if for_train:
        callbacks.append(ModelCheckpoint(save_top_k=-1, verbose=True))
        if (
            args.gpus
            and args.gpus > 1
            and getattr(args, "ddp_static_graph_callback", False)
            and getattr(args, "bert_gradient_checkpointing", True)
        ):
            callbacks.append(DdpStaticGraphCallback())
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks
    if args.gpus and args.gpus > 1:
        trainer_kwargs["accelerator"] = "ddp"
        fu = getattr(args, "ddp_find_unused_parameters", None)
        if fu is not None:
            trainer_kwargs["plugins"] = DDPPlugin(find_unused_parameters=fu)
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

        output = model.model.forward_train_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            img_ids=batch["img_ids"],
            neg_img_ids=batch["neg_img_ids"],
            global_step=step,
            total_steps=max(1, args.debug_smoke_train_steps),
        )
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

        tracked_param = next(model.model.style_gate_head.parameters())
        before = tracked_param.detach().clone()

        loss.backward()

        grad_report = {
            "shared_sticker_head": _module_grad_summary(model.model.shared_sticker_head),
            "style_head": _module_grad_summary(model.model.style_head),
            "expr_head": _module_grad_summary(model.model.expr_head),
            "query_head": _module_grad_summary(model.model.query_head),
            "style_match_head": _module_grad_summary(model.model.style_match_head),
            "expr_match_head": _module_grad_summary(model.model.expr_match_head),
            "style_gate_head": _module_grad_summary(model.model.style_gate_head),
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
            f"[DebugStepUpdate] step={step} tracked_param_delta_abs_mean={param_delta:.8f} "
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
            if (not args.base_only) and eval_debug["uses_fused_logits_for_ranking"] is not True:
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
        save_final_checkpoint_from_trainer(trainer, args)
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
