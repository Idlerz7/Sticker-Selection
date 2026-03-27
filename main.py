import os
import random
# Work around protobuf C-extension segfault on some environments.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import time

from logging import log
import logging
from posixpath import join
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from torch._C import device
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, HfArgumentParser, BertTokenizer, CLIPModel as HFCLIPModel, CLIPProcessor
import transformers
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
import hashlib
from transformers.utils.dummy_pt_objects import LogitsProcessor
from utils import get_logger, top_filtering, try_create_dir, Timer
import json
from copy import deepcopy
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device, cos_sim

from metrics import MyAccuracy
import clip

logger = get_logger(__name__)
_run_file_handler = None


def attach_run_log_file(log_file, args=None):
    """
    绑定“本次运行专属”的文件日志句柄，避免不同实验互相覆盖日志。

    输入：
    - log_file (str): 目标日志文件路径，通常位于 lightning 的 version 目录下。
    - args (Optional[Arguments]): 当前运行参数对象；仅用于在日志中打印上下文提示。

    输出：
    - 无显式返回值。

    副作用：
    - 会关闭并替换全局 `_run_file_handler`；
    - 会向 `logger` 追加新的 `FileHandler`；
    - 后续 `logger.info/debug` 将写入该文件。
    """
    global _run_file_handler
    log_dir = os.path.dirname(log_file)
    if log_dir:
        try_create_dir(log_dir)
    if _run_file_handler is not None:
        logger.removeHandler(_run_file_handler)
        try:
            _run_file_handler.close()
        except Exception:
            pass
    file_handler = logging.FileHandler(log_file, mode='w')
    if len(logger.handlers) > 0 and logger.handlers[0].formatter is not None:
        file_handler.setFormatter(logger.handlers[0].formatter)
    _run_file_handler = file_handler
    logger.addHandler(file_handler)
    logger.info(f"run log file: {log_file}")
    if args is not None:
        logger.info("run args attached to this version log")


def attach_version_log_from_trainer(args, trainer):
    """
    根据 `trainer` 的实际日志目录，为当前任务绑定版本级日志文件。

    输入：
    - args (Arguments): 运行参数，至少需要 `mode` 与 `pl_root_dir`。
    - trainer (pl.Trainer): Lightning 训练器实例，优先读取其 `log_dir`。

    输出：
    - 无显式返回值。

    行为说明：
    - 若 `trainer.log_dir` 可用，日志写到 `<log_dir>/<mode>.log`；
    - 否则回退到 `<pl_root_dir>/lightning_logs/<mode>.log`。
    """
    # In PL 1.5 + DDP, accessing trainer.log_dir too early may trigger
    # distributed broadcast before process group init.
    # So we guard it and fallback to a stable local path.
    try:
        log_dir = trainer.log_dir
    except Exception as e:
        logger.warning(
            f"trainer.log_dir unavailable before fit/test starts, fallback to root lightning_logs: {e}")
        log_dir = None
    if not log_dir:
        log_dir = os.path.join(args.pl_root_dir, 'lightning_logs')
    log_file = os.path.join(log_dir, f'{args.mode}.log')
    attach_run_log_file(log_file, args=args)


def attach_test_log_to_ckpt_version(args):
    """
    测试模式专用：把日志写到被测 checkpoint 所属的 version 目录下。

    输入：
    - args (Arguments): 至少包含 ckpt_path / test_data_path / test_with_cand / mode。

    输出：
    - 无显式返回值。

    日志路径规则：
    - version 目录 = ckpt_path 向上两级（checkpoints/ -> version_x/）；
    - 若推断失败则回退到 pl_root_dir/lightning_logs/；
    - 文件名格式：test_<数据集名>_<rall|r10>_<YYYYMMDD-HHMMSS>.log
    """
    # 从 ckpt_path 文件路径向上两级推出 version 目录
    # 例：.../version_6/checkpoints/epoch=9-step=264469.ckpt
    #       -> dirname -> .../version_6/checkpoints
    #       -> dirname -> .../version_6
    version_dir = None
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        checkpoints_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
        candidate = os.path.dirname(checkpoints_dir)
        if os.path.isdir(candidate):
            version_dir = candidate

    if version_dir is None:
        version_dir = os.path.join(args.pl_root_dir, 'lightning_logs')

    # 构造辨识度高的文件名
    dataset_stem = os.path.splitext(os.path.basename(
        getattr(args, 'test_data_path', '') or 'unknown'))[0]
    eval_tag = 'r10' if getattr(args, 'test_with_cand', False) else 'rall'
    ts = time.strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(version_dir, f'test_{dataset_stem}_{eval_tag}_{ts}.log')
    attach_run_log_file(log_file, args=args)


def attach_manual_log_file(args):
    """
    为不走 Lightning Trainer 的模式创建手工日志文件（带时间戳）。

    输入：
    - args (Arguments): 运行参数，至少需要 `mode` 与 `pl_root_dir`。

    输出：
    - 无显式返回值。

    行为说明：
    - 文件路径格式：`<pl_root_dir>/manual_logs/<mode>-<timestamp>.log`；
    - 每次运行文件名不同，便于回溯历史调试过程。
    """
    ts = time.strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(args.pl_root_dir, 'manual_logs', f'{args.mode}-{ts}.log')
    attach_run_log_file(log_file, args=args)


class BertModel(BertForSequenceClassification):
    def __init__(self, config):
        """
        初始化二分类 BERT 模型封装。

        输入：
        - config: HuggingFace BERT 配置对象。

        输出：
        - 无显式返回值（构造函数）。

        关键行为：
        - 强制 `config.num_labels = 2`，对应主任务“正例=1/负例=0”的二分类定义；
        - 其他结构与 `BertForSequenceClassification` 一致。
        """
        config.num_labels = 2
        super().__init__(config)


def _extract_speaker_from_text(text):
    """
    从一句文本前缀中提取说话人标签。

    输入：
    - text (str): 形如 `[speaker1]...` 或 `[speaker2]...` 的文本。

    输出：
    - str: `[speaker1]` / `[speaker2]` / `unknown_speaker`。

    用途：
    - 该返回值会参与 user_key 构造；
    - 当数据缺失真实 `user_id` 时，用于 fallback 用户标识生成。
    """
    if not text:
        return 'unknown_speaker'
    if text.startswith('[speaker1]'):
        return '[speaker1]'
    if text.startswith('[speaker2]'):
        return '[speaker2]'
    return 'unknown_speaker'


def _build_speaker_special_tokens(max_speaker_id):
    """
    构建 speaker 特殊 token 列表：['[speaker1]', ..., f'[speaker{N}]']。
    """
    n = int(max_speaker_id) if max_speaker_id is not None else 2
    n = max(2, n)
    return [f"[speaker{i}]" for i in range(1, n + 1)]


def _resolve_user_key(sample, dialog, fallback_index):
    """
    解析单个样本的“用户主键” `user_key`，用于个性化映射与历史缓存索引。

    输入：
    - sample (dict): 当前样本字典（可能包含 `user_id` / `dialogue_id`）。
    - dialog (list): 对话列表（通常 `sample['dialog']`）。
    - fallback_index (int): 样本索引（当前版本未直接使用，保留兼容位）。

    输出：
    - str: 可稳定用于 `user_to_index` 的用户键。

    解析优先级：
    1) 直接使用 `sample['user_id']`（最可靠）；
    2) 使用 `dialogue_id + 最后一句speaker` 组合；
    3) 老数据回退到 pseudo id（`md5(first_text) + speaker`）。

    注意：
    - pseudo id 仅用于兼容缺字段旧数据，不等价真实用户身份；
    - 这是 visual-only 版本中用户级建模的基础假设之一。
    """
    user_id = sample.get('user_id', None)
    if user_id is not None:
        return str(user_id)
    dialogue_id = sample.get('dialogue_id', None)
    if dialogue_id is not None and len(dialog) > 0:
        speaker = _extract_speaker_from_text(dialog[-1].get('text', ''))
        return f"{dialogue_id}::{speaker}"
    # Backward-compatible fallback for old *_pair.json without real user_id.
    # Current visual-only version may use this pseudo user key in training.
    # This pseudo id is deterministic, but not guaranteed to align across all
    # samples from the same real user.
    first_text = ''
    if len(dialog) > 0:
        first_text = dialog[0].get('text', '')
    speaker = _extract_speaker_from_text(dialog[-1].get('text', '')) if len(
        dialog) > 0 else 'unknown_speaker'
    digest = hashlib.md5(first_text.encode('utf-8')).hexdigest()[:12]
    return f"pseudo::{digest}::{speaker}"


class VisualHistoryAttention(torch.nn.Module):
    """可学习的视觉历史注意力池化模块。"""

    def __init__(self, input_dim, hidden_dim):
        """
        初始化视觉历史注意力池化层参数。

        输入：
        - input_dim (int): 单条历史视觉向量维度 D（例如 512）。
        - hidden_dim (int): 注意力打分中间层维度。

        输出：
        - 无显式返回值（构造函数）。

        参数化形式：
        - `W_v1: D -> hidden_dim`
        - `W_v2: hidden_dim -> 1`
        - 与论文式 `score_i = W_v2(tanh(W_v1(v_i)))` 对齐。
        """
        super().__init__()
        self.w_v1 = torch.nn.Linear(input_dim, hidden_dim)
        self.w_v2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, history_feats, history_mask=None):
        """
        对批量用户历史视觉特征做注意力聚合，输出每个用户的视觉偏好向量。

        输入：
        - history_feats (Tensor): `[B, K, D]`，B 为 batch，K 为历史长度，D 为特征维度。
        - history_mask (Optional[Tensor]): `[B, K]`，True 表示有效历史，False 为 padding。

        输出：
        - pooled (Tensor): `[B, D]`，每个样本的聚合视觉向量。

        计算流程：
        1) 对每个历史项计算 score；
        2) 用 mask 屏蔽无效位；
        3) softmax 得到权重；
        4) 对 `history_feats` 加权求和。

        数值与边界处理：
        - 当 `K=0` 直接返回零向量；
        - mask 后重新归一化并 `clamp_min`，避免分母为 0；
        - 对“全 padding”的行最终强制为零向量。
        """
        # history_feats: [B, K, D]
        if history_feats.size(1) == 0:
            return torch.zeros(history_feats.size(0), history_feats.size(2), device=history_feats.device, dtype=history_feats.dtype)
        scores = self.w_v2(torch.tanh(self.w_v1(history_feats))).squeeze(-1)
        if history_mask is not None:
            scores = scores.masked_fill(~history_mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        if history_mask is not None:
            attn = attn * history_mask.float()
            denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            attn = attn / denom
        pooled = torch.bmm(attn.unsqueeze(1), history_feats).squeeze(1)
        if history_mask is not None:
            has_history = history_mask.any(dim=-1, keepdim=True).float()
            pooled = pooled * has_history
        return pooled


class UserVisualFusion(torch.nn.Module):
    """将 user_id embedding 与视觉偏好向量进行门控融合。"""

    def __init__(self, num_users, feature_dim):
        """
        初始化用户门控融合模块。

        输入：
        - num_users (int): 用户词表大小（等于 `len(user_to_index)`）。
        - feature_dim (int): 用户向量维度（与 visual 向量维度一致）。

        输出：
        - 无显式返回值（构造函数）。

        组件：
        - `user_embedding`: 从 user index 查表得到 `u_id`；
        - `gate`: 线性层，输入 `[u_id; v_visual]` 输出逐维门控系数。
        """
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, feature_dim)
        self.gate = torch.nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, visual_pref, user_indices):
        """
        计算门控融合后的最终用户表示。

        输入：
        - visual_pref (Tensor): `[B, D]`，来自历史贴纸聚合的视觉偏好。
        - user_indices (Tensor): `[B]`，每个样本的用户索引。

        输出：
        - Tensor `[B, D]`: `u_final`，融合后的用户向量。

        公式：
        - `u_id = Emb(user_indices)`
        - `g = sigmoid(W_g [u_id ; visual_pref])`
        - `u_final = g * u_id + (1 - g) * visual_pref`

        解释：
        - 无历史用户时 `visual_pref≈0`，模型自然退化为更依赖 `u_id`；
        - 有历史时则由门控自动平衡“用户ID先验”与“历史视觉偏好”。
        """
        user_pref = self.user_embedding(user_indices)
        gate = torch.sigmoid(self.gate(torch.cat([user_pref, visual_pref], dim=-1)))
        return gate * user_pref + (1.0 - gate) * visual_pref


def _hf_clip_pil_rgb(img):
    """Pickle-friendly RGB convert for HF CLIP torchvision Compose (DataLoader workers)."""
    return img.convert('RGB')


class HFClipSentenceEncoder(torch.nn.Module):
    """本地 HuggingFace CLIP 的回退编码器封装。"""

    def __init__(self, model_path, local_files_only=True, device=None):
        """
        初始化 HF-CLIP 编码器，当 `SentenceTransformer` 路径加载失败时作为回退方案。

        输入：
        - model_path (str): 本地 CLIP checkpoint 路径。
        - local_files_only (bool): 是否严格离线加载。
        - device (Optional[str|torch.device]): 强制设备；默认 cuda（若可用）否则 cpu。

        输出：
        - 无显式返回值（构造函数）。

        兼容与稳定性处理：
        - 自动兼容 `.../0_CLIPModel/config.json` 的目录结构；
        - 使用 `torchvision` 预处理图像，规避慢速 ndarray->tensor 路径。
        """
        super().__init__()
        resolved_model_path = model_path
        if not os.path.exists(os.path.join(resolved_model_path, 'config.json')):
            nested = os.path.join(resolved_model_path, '0_CLIPModel')
            if os.path.exists(os.path.join(nested, 'config.json')):
                resolved_model_path = nested
        self.model = HFCLIPModel.from_pretrained(
            resolved_model_path, local_files_only=local_files_only)
        self.processor = CLIPProcessor.from_pretrained(
            resolved_model_path, local_files_only=local_files_only)
        if device is not None:
            self._target_device = torch.device(device)
        else:
            self._target_device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        if self._target_device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                f'HFClipSentenceEncoder requested {self._target_device} but torch.cuda.is_available() is False '
                '(install CUDA build of PyTorch, or pass device=cpu).')
        self.model.to(self._target_device)
        # Use torchvision preprocessing directly to avoid the slow
        # "list of numpy.ndarrays -> tensor" path in transformers feature extractor.
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            _hf_clip_pil_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def tokenize(self, items):
        """
        将混合列表输入（图像与文本）转换为 CLIP 前向所需张量字典。

        输入：
        - items (list): 元素可以是 `str`（文本）或图像对象/图像张量。

        输出：
        - features (dict):
          - `pixel_values`（若有图像）
          - `input_ids` / `attention_mask`（若有文本）
          - `image_text_info`（0=图像, 1=文本，用于还原顺序）

        说明：
        - 该函数只做打包，不做模型前向；
        - 输出会传给 `forward()` 统一编码并对齐回原顺序。
        """
        image_features = []
        text_inputs = []
        image_text_info = []
        for item in items:
            if isinstance(item, str):
                text_inputs.append(item)
                image_text_info.append(1)
            else:
                if isinstance(item, torch.Tensor):
                    image_features.append(item.unsqueeze(0))
                else:
                    image_features.append(self.preprocess(item).unsqueeze(0))
                image_text_info.append(0)

        features = {'image_text_info': image_text_info}
        if image_features:
            features['pixel_values'] = torch.cat(image_features, dim=0)
        if text_inputs:
            text_tokens = self.processor.tokenizer(
                text_inputs, return_tensors='pt', padding=True, truncation=True)
            features['input_ids'] = text_tokens['input_ids']
            features['attention_mask'] = text_tokens['attention_mask']
        return features

    def forward(self, features):
        """
        对 tokenize 后的 `features` 做编码，并按原输入顺序返回统一 embedding。

        输入：
        - features (dict): `tokenize()` 的输出字典。

        输出：
        - dict: `{'sentence_embedding': Tensor[N, D]}`。

        处理逻辑：
        - 图像与文本分别走各自 CLIP 编码接口；
        - 再根据 `image_text_info` 把两路结果交错回原顺序；
        - 输出统一 float tensor，供上层模块直接使用。
        """
        sentence_embedding = []
        img_embs = None
        txt_embs = None
        if 'pixel_values' in features:
            with torch.no_grad():
                pv = features['pixel_values'].to(
                    self._target_device, non_blocking=True)
                img_embs = self.model.get_image_features(pixel_values=pv)
        if 'input_ids' in features:
            with torch.no_grad():
                ids = features['input_ids'].to(
                    self._target_device, non_blocking=True)
                mask = features.get('attention_mask')
                if mask is not None:
                    mask = mask.to(self._target_device, non_blocking=True)
                txt_embs = self.model.get_text_features(
                    input_ids=ids, attention_mask=mask)

        img_i, txt_i = 0, 0
        for input_type in features['image_text_info']:
            if input_type == 0:
                sentence_embedding.append(img_embs[img_i])
                img_i += 1
            else:
                sentence_embedding.append(txt_embs[txt_i])
                txt_i += 1
        return {'sentence_embedding': torch.stack(sentence_embedding).float()}


class Model(torch.nn.Module):
    def __init__(self, args):
        """
        初始化项目主模型（文本主干 + 视觉编码 + 辅助头 + 个性化组件）。

        输入：
        - args (Arguments): 运行配置对象，决定模型模式、路径、任务开关与超参。

        输出：
        - 无显式返回值（构造函数）。

        初始化内容：
        - 文本侧：加载 BERT 分类模型与 tokenizer；
        - 视觉侧：按 `model_choice` 选择 CLIP / img_id embedding；
        - 任务头：主分类头（BERT 自带）+ 可选 emotion/mlm 头；
        - 数据字典：OCR、id2name、图像路径缓存；
        - 个性化：`user_to_index`、历史缓存、attention/fusion/proj（按开关启用）。

        关键约束：
        - 当前版本仅实现 visual-only personalization；
        - semantic/emotion personalization 在此版本明确不接入数据流。
        """
        # config.num_labels = 307
        # super().__init__(config)
        super().__init__()
        # self.segment_embedding = torch.nn.Embedding(2, config.hidden_size) # 0: speaker1 1: speaker2
        self.args = args
        logger.info(self.args.model_choice)
        self.temperature = torch.nn.Parameter(torch.tensor(args.init_temp))
        if self.args.local_files_only and (not os.path.exists(args.bert_pretrain_path)):
            raise FileNotFoundError(
                f"Local bert model path not found: {args.bert_pretrain_path}")
        self.bert = BertModel.from_pretrained(
            args.bert_pretrain_path, local_files_only=args.local_files_only)
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_pretrain_path, local_files_only=args.local_files_only)
        self.visual_feat_dim = 512
        self.bert_hidden_dim = self.bert.config.hidden_size
        special_tokens_dict = {
            'additional_special_tokens': _build_speaker_special_tokens(
                self.args.speaker_token_max_id
            )
        }
        self.bert_tokenizer.add_special_tokens(special_tokens_dict)
        self.bert.resize_token_embeddings(len(self.bert_tokenizer))
        if self.args.model_choice == 'use_clip_repo':
            logger.info('use clip repo')
            self.clip_model, self.clip_process = clip.load(
                "ViT-B/32", download_root=args.clip_download_root)
        elif self.args.model_choice == 'use_img_id':
            logger.info('use img id')
            if self.args.local_files_only and (not os.path.exists(args.text_clip_pretrain_path)):
                raise FileNotFoundError(
                    f"Local sentence-transformer path not found: {args.text_clip_pretrain_path}")
            self.text_clip = SentenceTransformer(
                args.text_clip_pretrain_path)
            self.img_embedding_layer = torch.nn.Embedding(
                args.max_image_id, self.visual_feat_dim)
        else:
            logger.info('use image clip')
            if self.args.local_files_only and (not os.path.exists(args.img_pretrain_path)):
                raise FileNotFoundError(
                    f"Local sentence-transformer path not found: {args.img_pretrain_path}")
            try:
                self.img_clip = SentenceTransformer(args.img_pretrain_path)
            except Exception as e:
                logger.warning(
                    f"SentenceTransformer load failed, fallback to HF CLIP: {e}")
                self.img_clip = HFClipSentenceEncoder(
                    args.img_pretrain_path, local_files_only=args.local_files_only)
            # self.text_clip = SentenceTransformer(
            #     'clip-ViT-B-32-multilingual-v1')
            if args.fix_text:
                self.bert.eval()
                for p in self.bert.parameters():
                    p.requires_grad = False
            if args.fix_img:
                self.img_clip.eval()
                for p in self.img_clip.parameters():
                    p.requires_grad = False

        # self.text_clip = SentenceTransformer('clip-ViT-B-32')
        self.text_ff = torch.nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim)
        self.img_ff = torch.nn.Linear(self.visual_feat_dim, self.bert_hidden_dim)
        if args.add_emotion_task:
            self.emotion_head = torch.nn.Linear(self.bert_hidden_dim, args.max_emotion_id)
        if args.add_predict_context_task or args.add_predict_img_label_task:
            self.mlm_head = torch.nn.Linear(self.bert_hidden_dim, len(self.bert_tokenizer))
        # self.text_clip_ff = torch.nn.Linear(512, 512)
        with open(self.args.ocr_path, encoding='utf-8') as f:
            a = json.load(f)

        self.id2ocr = {}
        for k, v in a.items():
            self.id2ocr[int(k)] = v['ocr']

        self.id2name = {}
        id2name_path = getattr(self.args, 'id2name_path', './data/id2name.json')
        with open(id2name_path, encoding='utf-8') as f:
            a = json.load(f)
            for k, v in a.items():
                self.id2name[int(k)] = v

        self.predict_id2ocr = {}
        self.user_to_index = {'__UNK__': 0}
        self.user_history_imgids = {}
        self.history_img_emb_cache = None
        if self.args.use_visual_personalization_token:
            self._init_visual_personalization()

    def _build_user_visual_history_cache(self):
        """
        从训练数据构建“用户 -> 历史正例贴纸ID列表”的静态缓存。

        输入：
        - 无显式参数（内部读取 `self.args.train_data_path`）。

        输出：
        - 无显式返回值。

        副作用：
        - 更新 `self.user_to_index`（字符串 user_key 到行号）；
        - 更新 `self.user_history_imgids`（行号到历史 img_id 列表）。

        细节说明：
        - 仅收集每条样本最后一句中的正例 `img_id`；
        - 按 `user_key` 聚合，并保留最后 `visual_history_max_len` 条；
        - 建库阶段不做样本级 LOO 去泄漏（在 forward 阶段动态做）。
        """
        path = self.args.train_data_path
        if not path or (not os.path.exists(path)):
            logger.warning(
                f"visual personalization disabled cache build, missing train data path: {path}")
            return

        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        user_hist = defaultdict(list)
        for idx, sample in enumerate(data):
            dialog = sample.get('dialog', [])
            if len(dialog) == 0:
                continue
            target = dialog[-1]
            img_id = target.get('img_id', None)
            if img_id is None:
                continue
            user_key = _resolve_user_key(sample, dialog, idx)
            # 这里先构建“完整静态历史”（不过滤当前样本），
            # 真正的样本级 leave-one-out 会在 forward 阶段按 batch 动态执行。
            # 这样可避免在离线建库时引入复杂时序逻辑，也便于训练/测试复用同一份缓存。
            user_hist[user_key].append(int(img_id))

        max_len = self.args.visual_history_max_len
        # 按 user_key 排序，保证 user_to_index 映射稳定可复现。
        # 这样不同运行中只要数据一致，user embedding 的行号就不会漂移。
        for user_key in sorted(user_hist.keys()):
            ids = user_hist[user_key]
            self.user_to_index[user_key] = len(self.user_to_index)
            self.user_history_imgids[self.user_to_index[user_key]] = ids[-max_len:]
        logger.info(
            f"visual history cache built, users:{len(self.user_to_index)-1}, max_len:{max_len}")

    def _save_user_history_cache(self):
        """
        将用户映射与历史缓存持久化到 JSON 文件。

        输入：
        - 无显式参数（使用 `self.args.user_history_cache_path`）。

        输出：
        - 无显式返回值。

        持久化字段：
        - `user_to_index`
        - `user_history_imgids`
        - `visual_history_max_len`（记录构建时配置）

        目的：
        - 让训练与测试共享同一 user 索引空间，避免 embedding 行号错位。
        """
        path = self.args.user_history_cache_path
        if not path:
            return
        out_dir = os.path.dirname(path)
        if out_dir:
            try_create_dir(out_dir)
        payload = {
            "user_to_index": self.user_to_index,
            "user_history_imgids": {str(k): v for k, v in self.user_history_imgids.items()},
            "visual_history_max_len": self.args.visual_history_max_len,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"saved user history cache: {path}")

    def _load_user_history_cache(self):
        """
        从磁盘加载 `user_to_index` 与历史缓存到内存。

        输入：
        - 无显式参数（读取 `self.args.user_history_cache_path`）。

        输出：
        - bool: 成功加载返回 `True`，失败/缺失返回 `False`。

        副作用：
        - 覆盖当前内存中的 `self.user_to_index` / `self.user_history_imgids`。

        安全处理：
        - 强制键值类型标准化（str/int）；
        - 若缺失 `__UNK__` 自动补齐为 0。
        """
        path = self.args.user_history_cache_path
        if not path or (not os.path.exists(path)):
            return False
        with open(path, encoding='utf-8') as f:
            payload = json.load(f)
        user_to_index = payload.get("user_to_index", None)
        history = payload.get("user_history_imgids", None)
        if not isinstance(user_to_index, dict) or not isinstance(history, dict):
            logger.warning(f"invalid user history cache format: {path}")
            return False
        self.user_to_index = {str(k): int(v) for k, v in user_to_index.items()}
        if '__UNK__' not in self.user_to_index:
            self.user_to_index['__UNK__'] = 0
        self.user_history_imgids = {}
        for k, ids in history.items():
            self.user_history_imgids[int(k)] = [int(x) for x in ids]
        logger.info(
            f"loaded user history cache: {path}, users:{len(self.user_to_index)-1}")
        return True

    def _init_visual_personalization(self):
        """
        初始化 visual-only 个性化分支的全部组件。

        输入：
        - 无显式参数。

        输出：
        - 无显式返回值。

        执行流程：
        1) 先尝试加载 `user_history_cache`；
        2) 若加载失败：训练模式重建并保存；测试/生成模式直接报错；
        3) 初始化历史注意力池化、用户融合模块、token 投影层。

        设计意图：
        - 强制训练/测试使用同一份 user 映射；
        - 明确当前版本是 visual-only，不引入 semantic/emotion 分支。
        """
        # 当前版本仅实现 visual-only 个性化：
        #   1) 用户历史视觉偏好向量
        #   2) user_id embedding 门控融合
        #   3) 作为 [USER] token 早融合进 BERT
        # semantic / emotion 个性化故意不在本版本实现。
        loaded = self._load_user_history_cache()
        if not loaded:
            self._build_user_visual_history_cache()
            if self.args.mode in ['train', 'pretrain']:
                self._save_user_history_cache()
            else:
                raise FileNotFoundError(
                    f"user_history_cache_path not found for mode={self.args.mode}: {self.args.user_history_cache_path}. "
                    "Use the same cache generated during training to keep user_to_index aligned.")
        hidden_dim = self.args.visual_personalization_hidden_dim
        self.visual_history_attention = VisualHistoryAttention(
            input_dim=self.visual_feat_dim, hidden_dim=hidden_dim)
        if not self.args.use_visual_history_attention:
            logger.info("use_visual_history_attention=False, fallback to masked mean pooling.")
        self.user_visual_fusion = UserVisualFusion(
            num_users=len(self.user_to_index), feature_dim=self.visual_feat_dim)
        # 两个投影层都映射到 BERT hidden size：
        # user_token_proj: 用户偏好向量 -> [USER] token
        # sticker_token_proj: 贴纸视觉向量 -> [STICKER] token
        self.user_token_proj = torch.nn.Linear(self.visual_feat_dim, self.bert_hidden_dim)
        self.sticker_token_proj = torch.nn.Linear(self.visual_feat_dim, self.bert_hidden_dim)

    def _get_user_indices(self, user_ids, device):
        """
        将一批字符串 `user_id` 转成 embedding 可用的 long 索引张量。

        输入：
        - user_ids (list[str]): batch 内每个样本的用户ID字符串。
        - device: 目标设备（CPU/CUDA）。

        输出：
        - Tensor[B] (dtype=long): 每个样本对应的用户行号。

        规则：
        - 未命中映射的用户统一回退到 `__UNK__`（索引 0）。
        """
        idx = [self.user_to_index.get(str(u), 0) for u in user_ids]
        return torch.tensor(idx, dtype=torch.long, device=device)

    def _get_history_img_emb_cache(self, device):
        """
        懒加载并返回“全贴纸视觉 embedding 缓存”。

        输入：
        - device: 目标计算设备。

        输出：
        - Tensor[max_image_id, D]: 每个贴纸ID的视觉向量表。

        行为：
        - 首次调用时按 `model_choice` 编码所有贴纸并缓存；
        - 后续直接复用缓存，避免每步重复编码历史贴纸。
        """
        if self.history_img_emb_cache is None:
            with torch.no_grad():
                all_ids = list(range(self.args.max_image_id))
                if self.args.model_choice == 'use_img_id':
                    img_ids_ts = torch.tensor(all_ids, dtype=torch.long, device=device)
                    embs = self.img_embedding_layer(img_ids_ts)
                elif self.args.model_choice == 'use_clip_repo':
                    img_objs = [self.get_image_obj(i) for i in all_ids]
                    clip_device = next(self.clip_model.parameters()).device
                    img = torch.stack(img_objs, dim=0).to(clip_device)
                    self.clip_model.eval()
                    embs = self.clip_model.encode_image(img).to(device)
                else:
                    embs = self.get_emb_by_imgids(all_ids)
                self.history_img_emb_cache = embs.detach()
        return self.history_img_emb_cache.to(device)

    def _get_user_visual_preference(self, user_indices, current_img_ids, device):
        """
        计算 batch 内每个样本的用户视觉偏好向量 `V_visual`。

        输入：
        - user_indices (Tensor[B]): 用户索引张量。
        - current_img_ids (Optional[list[int]]): 当前样本正例贴纸ID，用于 LOO 过滤。
        - device: 目标设备。

        输出：
        - Tensor[B, D]: 每个样本的视觉偏好向量。

        核心流程：
        1) 由 `user_indices` 查该用户历史 img_id 列表；
        2) 若提供 `current_img_ids`，对每个样本执行 LOO 过滤；
        3) 截断到 `visual_history_max_len`；
        4) 从全贴纸 embedding 表索引历史特征；
        5) 注意力池化（或 masked mean）得到 `V_visual`。

        LOO 策略说明：
        - 当前实现采用“删除所有与当前正例 img_id 相同的历史项”；
        - 这是更保守且实现简单的防泄漏策略。

        边界行为：
        - 用户无历史或过滤后为空时，返回零向量（冷启动由 user_id embedding 承担）。
        """
        batch_size = user_indices.size(0)
        if (len(self.user_history_imgids) == 0) or (self.args.visual_history_max_len <= 0):
            return torch.zeros(batch_size, self.visual_feat_dim, device=device)
        max_len = self.args.visual_history_max_len
        history_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        history_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        current_img_ids = current_img_ids if current_img_ids is not None else [None] * batch_size
        for row, uid in enumerate(user_indices.tolist()):
            ids = self.user_history_imgids.get(uid, [])
            if len(ids) == 0:
                continue
            current_img_id = current_img_ids[row]
            # Leave-one-out（样本级）：
            # 只对“当前样本”移除自己的正例 img_id，且这里采用“移除全部同 id 项”的策略。
            # 这样实现简单且最安全，避免同图重复历史造成信息泄漏。
            if current_img_id is not None:
                ids = [hid for hid in ids if hid != int(current_img_id)]
            ids = ids[-max_len:]
            if len(ids) == 0:
                continue
            n = len(ids)
            history_ids[row, :n] = torch.tensor(ids, dtype=torch.long, device=device)
            history_mask[row, :n] = True
        all_embs = self._get_history_img_emb_cache(device)
        history_feats = all_embs[history_ids]
        if self.args.use_visual_history_attention:
            visual_pref = self.visual_history_attention(
                history_feats=history_feats, history_mask=history_mask)
        else:
            denom = history_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            visual_pref = (history_feats * history_mask.unsqueeze(-1).float()).sum(dim=1) / denom
            visual_pref = visual_pref * history_mask.any(dim=-1, keepdim=True).float()
        return visual_pref

    def _build_user_token(self, user_ids, current_img_ids, device):
        """
        构建个性化输入中的 `[USER]` token embedding。

        输入：
        - user_ids (list[str]): 当前 batch 用户ID列表。
        - current_img_ids (Optional[list[int]]): 当前样本正例贴纸ID（用于 LOO）。
        - device: 目标设备。

        输出：
        - Tensor[B, 1, H]: 与 BERT hidden size 对齐的用户 token 向量。

        处理链路：
        - `user_ids -> user_indices`
        - `user_indices -> V_visual`（历史聚合）
        - `V_visual + user_embedding -> u_final`（门控融合）
        - `u_final -> user_token_proj -> [USER] token`
        """
        user_indices = self._get_user_indices(user_ids, device)
        visual_pref = self._get_user_visual_preference(user_indices, current_img_ids, device)
        fused_user = self.user_visual_fusion(visual_pref=visual_pref, user_indices=user_indices)
        return self.user_token_proj(fused_user).unsqueeze(1)

    def _get_loo_current_img_ids(self, img_ids, test):
        """
        统一决定当前场景下 LOO 过滤是否启用，以及传什么 `current_img_ids`。

        输入：
        - img_ids (list[int] or None): 当前样本正例贴纸ID列表。
        - test (bool): 是否处于测试路径。

        输出：
        - None 或原始 `img_ids`。

        规则：
        - 训练阶段：默认启用 LOO（--loo_filter_in_train=true），
          设为 false 则不过滤（历史包含当前正例）；
        - 测试阶段：默认关闭（--loo_filter_in_test=false），
          设为 true 则强制过滤。
        """
        if test:
            # 测试路径：默认不过滤，可由 --loo_filter_in_test 强制开启。
            if not self.args.loo_filter_in_test:
                return None
        else:
            # 训练路径：默认过滤，可由 --loo_filter_in_train=false 关闭。
            if not self.args.loo_filter_in_train:
                return None
        return img_ids

    def _compose_personalized_inputs(self, text_emb, attention_mask, sticker_token, user_token, aux_emb=None):
        """
        按 visual-only 个性化模板拼接 `inputs_embeds` 与配套掩码。

        输入：
        - text_emb (Tensor[B, L, H]): 对话文本 embedding（含 CLS/SEP）。
        - attention_mask (Tensor[B, L]): 文本 attention mask。
        - sticker_token (Tensor[B, 1, H]): 候选贴纸 token embedding。
        - user_token (Tensor[B, 1, H]): 用户偏好 token embedding。
        - aux_emb (Optional[Tensor[B, A, H]]): OCR/辅助序列 embedding（可选）。

        输出：
        - input_emb (Tensor[B, L', H]): 拼接后的最终输入 embedding。
        - full_attention_mask (Tensor[B, L']): 与 `input_emb` 对齐的 mask。
        - token_type_ids (Tensor[B, L']): 文本段/扩展段分段标记。

        模板：
        - `[CLS] [USER] text... [SEP] [AUX可选] [STICKER] [SEP]`

        注意：
        - 本函数同时负责补“trailing [SEP]”；
        - 三个输出长度必须严格一致，否则 BERT 前向会报错。
        """
        # 个性化输入拼接模板（早融合）：
        # [CLS] [USER] 文本... [SEP] [AUX可选] [STICKER] [SEP]
        # 注意三件事必须同步：
        # 1) input_emb 的拼接顺序
        # 2) attention_mask 的长度和位置
        # 3) token_type_ids 的分段标记
        cls_emb = text_emb[:, :1, :]
        text_rest = text_emb[:, 1:, :]
        sep_emb = self.bert.bert.embeddings.word_embeddings(
            torch.tensor(self.bert_tokenizer.sep_token_id, device=text_emb.device, dtype=torch.long)
        ).unsqueeze(0).unsqueeze(0).repeat(text_emb.size(0), 1, 1)
        input_parts = [cls_emb, user_token, text_rest]
        mask_parts = [attention_mask[:, :1], torch.ones(text_emb.size(0), 1, device=attention_mask.device), attention_mask[:, 1:]]
        if aux_emb is not None:
            input_parts.append(aux_emb)
            mask_parts.append(torch.ones(text_emb.size(0), aux_emb.size(1), device=attention_mask.device))
        input_parts.append(sticker_token)
        mask_parts.append(torch.ones(text_emb.size(0), 1, device=attention_mask.device))
        input_parts.append(sep_emb)
        mask_parts.append(torch.ones(text_emb.size(0), 1, device=attention_mask.device))
        input_emb = torch.cat(input_parts, dim=1)
        full_attention_mask = torch.cat(mask_parts, dim=1)
        token_type_ids = torch.zeros_like(full_attention_mask, dtype=torch.long)
        token_type_ids[:, -2:] = 1
        if aux_emb is not None:
            token_type_ids[:, -(aux_emb.size(1) + 2):] = 1
        return input_emb, full_attention_mask, token_type_ids

    def get_image_obj(self, id):
        """
        根据贴纸ID读取图像并执行预处理，必要时写入内存缓存。

        输入：
        - id (int): 贴纸ID。

        输出：
        - 预处理后的图像对象/张量（取决于 `self.img_process` 实现）。

        行为：
        - 若已缓存则直接返回；
        - 否则从 `id2imgpath` 读取文件，抽取帧并预处理后返回。
        """
        if id in self.id2img:
            return self.id2img[id]

        img_file_name = self.id2imgpath[id]
        img_dir = self.args.img_dir
        path = os.path.join(img_dir, img_file_name)
        from img_utils import pick_single_frame
        with open(path, 'rb') as fin:
            data = fin.read()
        img_obj = self.img_process(pick_single_frame(data))
        if self.args.mode != 'pretrain':
            self.id2img[id] = img_obj
        return img_obj

    def prepare_for_test(self):
        """
        在验证/测试开始前，预先计算“全部贴纸候选”的视觉 embedding。

        输入：
        - 无显式参数。

        输出：
        - 无显式返回值；副作用是写入 `self.all_img_embs`。

        用途：
        - Rall：直接对全库打分；
        - R10：从全库中按 cand 索引切片打分。
        """
        logger.info("prepare_for_test!")
        with torch.no_grad():
            if self.args.model_choice == 'use_clip_repo':
                img_objs = []
                for id in tqdm(range(self.args.max_image_id), desc="load_imgs"):
                    img_obj = self.id2img[id]
                    img_objs.append(img_obj)
                device = next(self.clip_model.parameters()).device
                img = torch.stack(img_objs, dim=0).to(device)
                self.clip_model.eval()
                self.all_img_embs = self.clip_model.encode_image(img)

            elif self.args.model_choice == 'use_img_id':
                img_ids = list(range(self.args.max_image_id))
                self.all_img_embs = self.img_embedding_layer(torch.tensor(
                    img_ids, dtype=torch.long, device=self.text_clip.device))

            else:
                cache_path = (self.args.img_emb_cache_path or '').strip()
                if cache_path and os.path.exists(cache_path):
                    logger.info(f"Loading precomputed embeddings from {cache_path}")
                    self.all_img_embs = torch.load(cache_path, map_location='cpu')
                    device = self.img_clip._target_device
                    self.all_img_embs = self.all_img_embs.to(device)
                    assert self.all_img_embs.size(0) == self.args.max_image_id, \
                        f"Cache shape {self.all_img_embs.shape} vs max_image_id {self.args.max_image_id}"
                else:
                    # Batched encoding with parallel image loading for higher GPU utilization
                    batch_size = self.args.prepare_batch_size
                    workers = self.args.prepare_workers
                    device = self.img_clip._target_device
                    self.img_clip.eval()
                    all_embs = []

                    def load_one(id):
                        return id, self.get_image_obj(id)

                    for start in tqdm(range(0, self.args.max_image_id, batch_size),
                                     desc="encode_all_stickers"):
                        end = min(start + batch_size, self.args.max_image_id)
                        ids_batch = list(range(start, end))
                        img_objs = [None] * len(ids_batch)
                        with ThreadPoolExecutor(max_workers=workers) as ex:
                            futures = {ex.submit(load_one, i): i for i in ids_batch}
                            for f in as_completed(futures):
                                idx, obj = f.result()
                                img_objs[idx - start] = obj
                        img_tokens = self.img_clip.tokenize(img_objs)
                        img_tokens = batch_to_device(img_tokens, device)
                        embs = self.img_clip.forward(img_tokens)['sentence_embedding']
                        all_embs.append(embs)
                    self.all_img_embs = torch.cat(all_embs, dim=0)
                    if cache_path:
                        d = os.path.dirname(cache_path)
                        if d:
                            os.makedirs(d, exist_ok=True)
                        torch.save(self.all_img_embs.cpu(), cache_path)
                        logger.info(f"Saved embeddings to {cache_path}")

    def get_emb_by_imgids(self, img_ids):
        """
        根据一组贴纸ID获取对应视觉 embedding。

        输入：
        - img_ids (list[int]): 待编码贴纸ID列表。

        输出：
        - Tensor[B, D]: 贴纸视觉向量。
        """
        # fix_img 且已有预计算 embedding 时，直接查表，避免训练时重复加载图像
        if (self.args.fix_img and hasattr(self, 'all_img_embs') and
                self.all_img_embs is not None):
            device = self.img_clip._target_device
            ids_t = torch.tensor(img_ids, dtype=torch.long, device=device)
            return self.all_img_embs[ids_t]
        img_objs = []
        for i, id in enumerate(img_ids):
            img_obj = self.get_image_obj(id)
            img_objs.append(img_obj)
        img_tokens = self.img_clip.tokenize(img_objs)

        img_tokens = batch_to_device(
            img_tokens, self.img_clip._target_device)

        img_emb = self.img_clip.forward(
            img_tokens)['sentence_embedding']
        return img_emb

    def get_input_output_imglabel_by_imgid(self, img_id):
        """
        为“图像标签预测（OCR相关）辅助任务”构造输入和监督目标。

        输入：
        - img_id (int): 当前贴纸ID。

        输出：
        - tuple:
          - mask_predict_inputs (list[int]): 模型输入中的 mask 序列；
          - mask_predict_outputs (list[int]): 监督标签（无监督位为 -100）；
          - cls_inputs (list[int]): 分类侧附加输入序列。

        规则：
        - 有 OCR 文本：输出含 CLS 起始与 OCR token；
        - 无 OCR 文本：输出全 -100 / 全 mask，训练时自然忽略。
        """
        max_len = self.args.max_img_label_mask_num
        self.tokenizer = self.bert_tokenizer
        mask_id = self.tokenizer.mask_token_id
        cls_id = self.tokenizer.cls_token_id
        mask_predict_inputs = [mask_id] * max_len
        ocr = self.id2ocr[img_id]
        if ocr != '':
            ids = self.tokenizer.encode(ocr, add_special_tokens=False)
            # mask_predict_outputs = [mask_id] * \
            #     (max_len - len(ids) - 1) + [cls_id] + ids
            mask_predict_outputs = [-100] * \
                (max_len - len(ids) - 1) + [cls_id] + ids
            cls_inputs = [mask_id] * (max_len - len(ids)) + ids
        else:
            mask_predict_outputs = [-100] * max_len
            cls_inputs = [mask_id] * max_len
        return mask_predict_inputs, mask_predict_outputs, cls_inputs

    def check_has_ocr(self, img_id):
        """
        判断某个贴纸是否存在 OCR 文本标注。

        输入：
        - img_id (int): 贴纸ID。

        输出：
        - bool: 有 OCR 返回 True，否则 False。
        """
        ocr = self.id2ocr[img_id]
        if ocr == '':
            return False
        return True

    def update_predict_ocr(self, img_id, img_label, logits=None):
        """
        更新图像标签预测统计，并可选打印 top-k token 调试信息。

        输入：
        - img_id (int): 贴纸ID。
        - img_label (str): 当前预测得到的标签字符串。
        - logits (Optional[Tensor]): 若提供，将额外打印 top-k token。

        输出：
        - 无显式返回值。

        副作用：
        - 更新 `self.predict_id2ocr[img_id]` 计数器。
        """
        if img_id not in self.predict_id2ocr:
            self.predict_id2ocr[img_id] = Counter()

        self.predict_id2ocr[img_id][img_label] += 1
        logger.debug(
            f"img_id: {img_id}, true img label: {self.id2name[img_id]}, pred img_label: {img_label}")
        if logits is not None:
            logits = logits.squeeze(0)
            _, indices = torch.topk(logits, k=5, dim=-1)
            for idx in indices:
                logger.debug(
                    f"tokens:{self.tokenizer.convert_ids_to_tokens(idx)}")

    def predict_img_label(self, input_ids, attention_mask, batch_idx=1, img_ids=None):
        """
        执行“预测贴纸文本标签/OCR”的辅助任务前向（调试模式）。

        输入：
        - input_ids (Tensor[B, L]): 文本 token ids。
        - attention_mask (Tensor[B, L]): 文本 mask。
        - batch_idx (int): 批次索引（日志用途）。
        - img_ids (list[int]): 当前样本贴纸ID（此模式下期望 batch=1）。

        输出：
        - 无显式返回值（结果通过 `update_predict_ocr` 记录）。

        说明：
        - 该模式用于诊断“仅凭上下文+图像是否能还原贴纸文字语义”；
        - 不参与主任务训练损失。
        """
        device = input_ids.device
        img_emb = self.get_emb_by_imgids(img_ids)
        mask_predict_inputs_ts = []
        assert len(img_ids) == 1
        if self.check_has_ocr(img_ids[0]):
            return
        for img_id in img_ids:
            mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                img_id)
            mask_predict_inputs_ts.append(mask_predict_inputs)
        mask_predict_inputs_ts = torch.tensor(
            mask_predict_inputs_ts, device=device, dtype=torch.long)
        mask_predict_inputs_emb = self.bert.bert.embeddings.word_embeddings(
            mask_predict_inputs_ts)
        text_emb = self.bert.bert.embeddings.word_embeddings(input_ids)
        batch_size = input_ids.size(0)

        sep_emb = self.bert.bert.embeddings.word_embeddings(
            torch.tensor(self.bert_tokenizer.sep_token_id, device=device, dtype=torch.long)).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        img_emb = self.img_ff(img_emb).unsqueeze(1)
        # logger.debug(f"{text_emb.size()}, {mask_predict_inputs_emb.size()}, {img_emb.size()}, {sep_emb.size()}")
        mask_input_emb = torch.cat(
            [text_emb, mask_predict_inputs_emb, img_emb, sep_emb], dim=1)

        extra_len = mask_input_emb.size(1) - text_emb.size(1)
        ones_mask2 = torch.ones(batch_size, extra_len, device=device)
        attention_mask = torch.cat(
            [attention_mask, ones_mask2], dim=1)

        token_type_ids = torch.zeros_like(
            attention_mask, dtype=torch.long)
        token_type_ids[:, -extra_len:] = 1

        mask_res = self.bert(inputs_embeds=mask_input_emb,
                             attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
        mask_hidden_states = mask_res.hidden_states
        start = text_emb.size(1)
        end = text_emb.size(1) + mask_predict_inputs_emb.size(1)

        mask_hidden_states = mask_hidden_states[-1][:, start:end, :]
        logits = self.mlm_head(mask_hidden_states)
        # logger.debug(logits.size())
        tokenizer = self.bert_tokenizer
        ids = logits.argmax(-1)[0]
        # res = tokenizer(ids[0])
        # logger.debug(res)
        generated_ids = None
        cls_id = tokenizer.cls_token_id
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] == cls_id:
                if i == len(ids) - 1:
                    break
                generated_ids = ids[i+1:]
                break

        if generated_ids is not None:
            img_label = ''.join(tokenizer.convert_ids_to_tokens(generated_ids))
            self.update_predict_ocr(img_ids[0], img_label, logits)

    def test_img_emotion(self, input_ids, attention_mask, img_ids):
        """
        执行情绪分类调试前向。

        输入：
        - input_ids (Tensor[B, L]): 对话文本 token ids。
        - attention_mask (Tensor[B, L]): 对话文本 mask。
        - img_ids (list[int]): 当前贴纸ID。

        输出：
        - Tensor[B]: 预测情绪类别索引（argmax）。

        依赖：
        - 需 `add_ocr_info=True`，因为该路径按“文本+OCR+贴纸”模板拼接输入。
        """
        img_emb = self.get_emb_by_imgids(img_ids)
        device = attention_mask.device
        batch_size = img_emb.size(0)
        sep_emb = self.bert.bert.embeddings.word_embeddings(
            torch.tensor(self.bert_tokenizer.sep_token_id, device=device, dtype=torch.long)).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        # logger.debug(text_emb.size())
        text_emb = self.bert.bert.embeddings.word_embeddings(input_ids)
        img_emb = self.img_ff(img_emb).unsqueeze(1)
        assert self.args.add_ocr_info is True
        if self.args.add_ocr_info:
            # neg_mask_predict_inputs, neg_mask_predict_outputs, neg_cls_inputs = self.get_input_output_imglabel_by_imgid(neg_img_id)
            # mask_predict_inputs_ts = []
            # mask_predict_outputs_ts = []
            cls_inputs_ts = []

            for img_id in img_ids:
                mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                    img_id)
                # mask_predict_inputs_ts.append(mask_predict_inputs)
                # mask_predict_outputs_ts.append(mask_predict_outputs)
                cls_inputs_ts.append(cls_inputs)

            # mask_predict_inputs_ts = torch.tensor(
            #     mask_predict_inputs_ts, device=device, dtype=torch.long)
            # mask_predict_outputs_ts = torch.tensor(
            #     mask_predict_outputs_ts, device=device, dtype=torch.long)
            cls_inputs_ts = torch.tensor(
                cls_inputs_ts, device=device, dtype=torch.long)

            # mask_predict_inputs_emb = self.bert.bert.embeddings.word_embeddings(
            #     mask_predict_inputs_ts)
            cls_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                cls_inputs_ts)

            input_emb = torch.cat(
                [text_emb, cls_inputs_emb, img_emb, sep_emb], dim=1)  # format

            extra_len = input_emb.size(1) - text_emb.size(1)
            ones_mask2 = torch.ones(batch_size, extra_len, device=device)
            addocr_attention_mask = torch.cat(
                [attention_mask, ones_mask2], dim=1)
            labels = torch.ones(
                batch_size, dtype=torch.long, device=device)

            token_type_ids = torch.zeros_like(
                addocr_attention_mask, dtype=torch.long)
            token_type_ids[:, -extra_len:] = 1
            res = self.bert(inputs_embeds=input_emb,
                            attention_mask=addocr_attention_mask, labels=labels, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
            hidden_states = res.hidden_states  # len=13, embedding + 12 layer
            hidden_states = hidden_states[-1][:, -2, :]
            logits = self.emotion_head(hidden_states)
            return logits.argmax(-1)

    def test_gradient(self, input_ids, attention_mask, img_ids):
        """
        执行梯度可解释性调试前向，返回 token 级梯度强度。

        输入：
        - input_ids (Tensor[B, L]): 对话 token ids。
        - attention_mask (Tensor[B, L]): 对话 mask。
        - img_ids (list[int]): 贴纸ID列表。

        输出：
        - tuple:
          - embeddings_grad (Tensor[B, L', H]): embedding 层梯度；
          - success (bool): 当前样本是否预测正确。

        用途：
        - 观察模型对哪些 token 更敏感，辅助误差分析与解释。
        """
        img_emb = self.get_emb_by_imgids(img_ids)
        device = attention_mask.device
        batch_size = img_emb.size(0)
        sep_emb = self.bert.bert.embeddings.word_embeddings(
            torch.tensor(self.bert_tokenizer.sep_token_id, device=device, dtype=torch.long)).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        # logger.debug(text_emb.size())
        text_emb = self.bert.bert.embeddings.word_embeddings(input_ids)
        img_emb = self.img_ff(img_emb).unsqueeze(1)
        # print(type(self.bert))
        assert self.args.add_ocr_info is True
        if self.args.add_ocr_info:
            # neg_mask_predict_inputs, neg_mask_predict_outputs, neg_cls_inputs = self.get_input_output_imglabel_by_imgid(neg_img_id)
            # mask_predict_inputs_ts = []
            # mask_predict_outputs_ts = []
            cls_inputs_ts = []

            for img_id in img_ids:
                mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                    img_id)
                # mask_predict_inputs_ts.append(mask_predict_inputs)
                # mask_predict_outputs_ts.append(mask_predict_outputs)
                cls_inputs_ts.append(cls_inputs)

            # mask_predict_inputs_ts = torch.tensor(
            #     mask_predict_inputs_ts, device=device, dtype=torch.long)
            # mask_predict_outputs_ts = torch.tensor(
            #     mask_predict_outputs_ts, device=device, dtype=torch.long)
            cls_inputs_ts = torch.tensor(
                cls_inputs_ts, device=device, dtype=torch.long)

            # mask_predict_inputs_emb = self.bert.bert.embeddings.word_embeddings(
            #     mask_predict_inputs_ts)
            cls_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                cls_inputs_ts)

            input_emb = torch.cat(
                [text_emb, cls_inputs_emb, img_emb, sep_emb], dim=1)  # format

            extra_len = input_emb.size(1) - text_emb.size(1)
            ones_mask2 = torch.ones(batch_size, extra_len, device=device)
            addocr_attention_mask = torch.cat(
                [attention_mask, ones_mask2], dim=1)
            labels = torch.ones(
                batch_size, dtype=torch.long, device=device)

            token_type_ids = torch.zeros_like(
                addocr_attention_mask, dtype=torch.long)
            token_type_ids[:, -extra_len:] = 1
            res = self.bert(inputs_embeds=input_emb,
                            attention_mask=addocr_attention_mask, labels=labels, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
            logits = res.logits
            success = (logits.argmax(-1) == labels).item()
            hidden_states = res.hidden_states  # len=13, embedding + 12 layer
            # hidden_states = hidden_states[-1][:, -2, :]
            embeddings = hidden_states[0]
            embeddings.retain_grad()
            loss = res.loss
            loss.backward()
            # print(embeddings.size())
            # print(embeddings.grad)
            # print(embeddings.grad.size())
            # ret_grad = embeddings.grad.detach().clone()
            # print(torch.autograd.grad(loss, embeddings, retain_graph=True))
            # logits = self.emotion_head(hidden_states)
            # return logits.argmax(-1)
            self.zero_grad() # 不会清空中间变量的grad
            # for name, v in self.named_parameters():
            #     print(name)
            # self.bert.zero_grad()
            # print(embeddings.grad)
            # print(embeddings.grad.size())
            return embeddings.grad, success
            

    def forward(self, input_ids, attention_mask, batch_idx=1, neg_img_ids=None, img_ids=None, emotion_ids=None, user_ids=None, img_tokens=None, neg_img_tokens=None,
                test=False, cands=None, mask_context_input_ids=None, mask_context_output_ids=None, mask_context_attention_mask=None):
        """
        模型统一前向入口：覆盖训练、测试以及若干调试/辅助模式。

        输入（核心）：
        - input_ids (Tensor[B, L]): 文本 token ids。
        - attention_mask (Tensor[B, L]): 文本 mask。
        - batch_idx (int): 批次索引（日志调试用）。
        - neg_img_ids (list[int] or None): 训练负例贴纸ID。
        - img_ids (list[int] or None): 正例贴纸ID（训练）或真实标签ID（测试）。
        - emotion_ids (list[int] or None): 情绪标签（若启用 emotion 任务）。
        - user_ids (list[str] or None): 用户ID（个性化分支使用）。
        - test (bool): `False=训练路径`，`True=评测路径`。
        - cands (list or None): R10 候选列表（每样本一组候选贴纸ID）。
        - mask_context_*: 上下文预测辅助任务张量。

        输出：
        - 训练主路径：`(final_loss,)`；
        - 测试主路径：`(logits, labels, cands)`；
        - 调试模式：返回对应分支的预测/梯度结果。

        关键逻辑：
        - 主任务始终保持“正负二分类 CE 平均”不变；
        - 个性化开关仅影响输入拼接，不改损失定义；
        - R10 与 Rall 共用测试逻辑，仅候选集合来源不同。
        """

        if self.args.mode == 'predict_img_label':
            return self.predict_img_label(input_ids=input_ids, attention_mask=attention_mask, batch_idx=batch_idx, img_ids=img_ids)

        if self.args.mode == 'predict_emotion':
            return self.test_img_emotion(input_ids=input_ids, attention_mask=attention_mask, img_ids=img_ids)

        if self.args.mode == 'test_gradient':
            return self.test_gradient(input_ids=input_ids, attention_mask=attention_mask, img_ids=img_ids)

        if not test:
            # 训练路径：每条样本构造一对 (正例贴纸, 负例贴纸)，
            # 仍按原论文/原实现的二分类训练，不改任务定义。
            img_emb = self.get_emb_by_imgids(img_ids)
            neg_img_emb = self.get_emb_by_imgids(neg_img_ids)
        else:
            # 测试路径：一次前向对候选集合打分（assert batch_size==1 见下）。
            # - 非空 cand_ids：R10/R20（仅对 these ids 取 all_img_embs）
            # - 无候选且未强制候选评测：Rall（全库，显存极大）
            # 注意：collate 可能产生 cands=[[]]（JSON 里 cand: []）。旧代码 `if cands:` 为真，
            # 取 cands[0] 后空列表在下一分支被当成 falsy，误走 Rall，一次性 BERT batch≈max_image_id → OOM。
            cand_ids = None
            if cands is not None:
                assert len(cands) == 1, "val/test ranking expects batch_size==1"
                cand_ids = cands[0]
            if cand_ids is not None and len(cand_ids) > 0:
                img_emb = self.all_img_embs[cand_ids]
                if self.args.add_ocr_info:
                    cls_inputs_ts = []
                    for img_id in cand_ids:
                        mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                            img_id)
                        cls_inputs_ts.append(cls_inputs)
                    cls_inputs_ts = torch.tensor(
                        cls_inputs_ts, device=self.all_img_embs.device, dtype=torch.long)
                    cls_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                        cls_inputs_ts)
                cands_ret = cand_ids
            else:
                if getattr(self.args, "candidate_eval_only", False) or self.args.test_with_cand:
                    raise ValueError(
                        "Eval batch has missing or empty `cand` while candidate_eval_only/test_with_cand "
                        "is set. Falling back to full-bank ranking would run MM-BERT on ~max_image_id "
                        "candidates and typically CUDA OOM. Fix data: non-empty cand in JSON, or valid "
                        "neg_img_id so PLDataset can build a local candidate list."
                    )
                img_emb = self.all_img_embs
                if self.args.add_ocr_info:
                    cls_inputs_ts = []
                    for img_id in range(self.args.max_image_id):
                        mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                            img_id)
                        cls_inputs_ts.append(cls_inputs)
                    cls_inputs_ts = torch.tensor(
                        cls_inputs_ts, device=self.all_img_embs.device, dtype=torch.long)
                    cls_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                        cls_inputs_ts)
                cands_ret = None
        # logger.info(f'unique img ids num:{len(list(set(img_ids)))}')

        if not test:
            device = attention_mask.device
            batch_size = img_emb.size(0)
            sep_emb = self.bert.bert.embeddings.word_embeddings(
                torch.tensor(self.bert_tokenizer.sep_token_id, device=device, dtype=torch.long)).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
            # logger.debug(text_emb.size())
            text_emb = self.bert.bert.embeddings.word_embeddings(input_ids)
            if self.args.use_visual_personalization_token:
                if user_ids is None:
                    user_ids = ['__UNK__'] * batch_size
                loo_current_img_ids = self._get_loo_current_img_ids(
                    img_ids=img_ids, test=test)
                user_token = self._build_user_token(
                    user_ids=user_ids, current_img_ids=loo_current_img_ids, device=device)
                img_emb = self.sticker_token_proj(img_emb).unsqueeze(1)
                neg_img_emb = self.sticker_token_proj(neg_img_emb).unsqueeze(1)
            else:
                img_emb = self.img_ff(img_emb).unsqueeze(1)
                neg_img_emb = self.img_ff(neg_img_emb).unsqueeze(1)
            if self.args.add_ocr_info:
                # neg_mask_predict_inputs, neg_mask_predict_outputs, neg_cls_inputs = self.get_input_output_imglabel_by_imgid(neg_img_id)
                # mask_predict_inputs_ts = []
                # mask_predict_outputs_ts = []
                cls_inputs_ts = []
                neg_cls_inputs_ts = []

                for img_id in img_ids:
                    mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                        img_id)
                    # mask_predict_inputs_ts.append(mask_predict_inputs)
                    # mask_predict_outputs_ts.append(mask_predict_outputs)
                    cls_inputs_ts.append(cls_inputs)

                for img_id in neg_img_ids:
                    mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                        img_id)
                    neg_cls_inputs_ts.append(cls_inputs)

                # mask_predict_inputs_ts = torch.tensor(
                #     mask_predict_inputs_ts, device=device, dtype=torch.long)
                # mask_predict_outputs_ts = torch.tensor(
                #     mask_predict_outputs_ts, device=device, dtype=torch.long)
                cls_inputs_ts = torch.tensor(
                    cls_inputs_ts, device=device, dtype=torch.long)
                neg_cls_inputs_ts = torch.tensor(
                    neg_cls_inputs_ts, device=device, dtype=torch.long)

                # mask_predict_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                #     mask_predict_inputs_ts)
                cls_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                    cls_inputs_ts)
                neg_cls_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                    neg_cls_inputs_ts)

                if self.args.use_visual_personalization_token:
                    input_emb, addocr_attention_mask, token_type_ids = self._compose_personalized_inputs(
                        text_emb=text_emb,
                        attention_mask=attention_mask,
                        sticker_token=img_emb,
                        user_token=user_token,
                        aux_emb=cls_inputs_emb,
                    )
                    neg_input_emb, _, _ = self._compose_personalized_inputs(
                        text_emb=text_emb,
                        attention_mask=attention_mask,
                        sticker_token=neg_img_emb,
                        user_token=user_token,
                        aux_emb=neg_cls_inputs_emb,
                    )
                else:
                    input_emb = torch.cat(
                        [text_emb, cls_inputs_emb, img_emb, sep_emb], dim=1)  # format
                    extra_len = input_emb.size(1) - text_emb.size(1)
                    ones_mask2 = torch.ones(batch_size, extra_len, device=device)
                    addocr_attention_mask = torch.cat(
                        [attention_mask, ones_mask2], dim=1)
                    token_type_ids = torch.zeros_like(
                        addocr_attention_mask, dtype=torch.long)
                    token_type_ids[:, -extra_len:] = 1
                    neg_input_emb = torch.cat(
                        [text_emb, neg_cls_inputs_emb, neg_img_emb, sep_emb], dim=1)
                labels = torch.ones(
                    batch_size, dtype=torch.long, device=device)
                neg_labels = torch.zeros(
                    batch_size, dtype=torch.long, device=device)
                # 主任务保持不变：
                # CE(正例=1) + CE(负例=0) 再取平均，不引入额外 ranking/residual loss。
                res = self.bert(inputs_embeds=input_emb,
                                attention_mask=addocr_attention_mask, labels=labels, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
                loss = res.loss
                neg_res = self.bert(inputs_embeds=neg_input_emb,
                                    attention_mask=addocr_attention_mask, labels=neg_labels, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
                neg_loss = neg_res.loss
                cls_loss = (loss + neg_loss) / 2
                # if self.args.add_predict_img_label_task:
                #     mask_input_emb = torch.cat(
                #         [text_emb, mask_predict_inputs_emb, img_emb, sep_emb], dim=1)

                #     mask_res = self.bert(inputs_embeds=mask_input_emb,
                #                         attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
                #     mask_hidden_states = mask_res.hidden_states
                #     start = text_emb.size(1)
                #     end = text_emb.size(1) + mask_predict_inputs_emb.size(1)

                #     mask_hidden_states = mask_hidden_states[-1][:, start:end, :]
                #     logits = self.mlm_head(mask_hidden_states)
                #     # shifted_prediction_scores = logits[:, :-1, :].contiguous()
                #     # labels = mask_predict_outputs_ts[:, 1:].contiguous()
                #     labels = mask_predict_outputs_ts
                #     loss_fct = torch.nn.CrossEntropyLoss()
                #     mask_loss = loss_fct(
                #         logits.view(-1, len(self.bert_tokenizer)), labels.view(-1))
                #     final_loss = cls_loss + self.args.label_weight * mask_loss
                # else:
                final_loss = cls_loss

            else:
                if self.args.use_visual_personalization_token:
                    input_emb, noaddocr_attention_mask, token_type_ids = self._compose_personalized_inputs(
                        text_emb=text_emb,
                        attention_mask=attention_mask,
                        sticker_token=img_emb,
                        user_token=user_token,
                        aux_emb=None,
                    )
                    neg_input_emb, _, _ = self._compose_personalized_inputs(
                        text_emb=text_emb,
                        attention_mask=attention_mask,
                        sticker_token=neg_img_emb,
                        user_token=user_token,
                        aux_emb=None,
                    )
                else:
                    input_emb = torch.cat(
                        [text_emb, img_emb, sep_emb], dim=1)
                    ones_mask2 = torch.ones(batch_size, 2, device=device)
                    noaddocr_attention_mask = torch.cat(
                        [attention_mask, ones_mask2], dim=1)
                    token_type_ids = torch.zeros_like(
                        noaddocr_attention_mask, dtype=torch.long)
                    token_type_ids[:, -2:] = 1
                    neg_input_emb = torch.cat(
                        [text_emb, neg_img_emb, sep_emb], dim=1)
                labels = torch.ones(
                    batch_size, dtype=torch.long, device=device)
                neg_labels = torch.zeros(
                    batch_size, dtype=torch.long, device=device)
                # logger.debug(token_type_ids.dtype)
                res = self.bert(inputs_embeds=input_emb,
                                attention_mask=noaddocr_attention_mask, labels=labels, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
                loss = res.loss
                neg_res = self.bert(inputs_embeds=neg_input_emb,
                                    attention_mask=noaddocr_attention_mask, labels=neg_labels, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
                neg_loss = neg_res.loss
                cls_loss = (loss + neg_loss) / 2
                final_loss = cls_loss

            if self.args.add_predict_img_label_task:
                mask_predict_inputs_ts = []
                mask_predict_outputs_ts = []

                for img_id in img_ids:
                    mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                        img_id)
                    mask_predict_inputs_ts.append(mask_predict_inputs)
                    mask_predict_outputs_ts.append(mask_predict_outputs)

                mask_predict_inputs_ts = torch.tensor(
                    mask_predict_inputs_ts, device=device, dtype=torch.long)
                mask_predict_outputs_ts = torch.tensor(
                    mask_predict_outputs_ts, device=device, dtype=torch.long)

                mask_predict_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                    mask_predict_inputs_ts)
                mask_input_emb = torch.cat(
                    [text_emb, mask_predict_inputs_emb, img_emb, sep_emb], dim=1)

                predlabel_extra_len = mask_input_emb.size(1) - text_emb.size(1)
                predlabel_ones_mask2 = torch.ones(
                    batch_size, predlabel_extra_len, device=device)
                predlabel_attention_mask = torch.cat(
                    [attention_mask, predlabel_ones_mask2], dim=1)

                token_type_ids = torch.zeros_like(
                    predlabel_attention_mask, dtype=torch.long)
                token_type_ids[:, -predlabel_extra_len:] = 1

                mask_res = self.bert(inputs_embeds=mask_input_emb,
                                     attention_mask=predlabel_attention_mask, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
                mask_hidden_states = mask_res.hidden_states
                start = text_emb.size(1)
                end = text_emb.size(1) + mask_predict_inputs_emb.size(1)

                mask_hidden_states = mask_hidden_states[-1][:, start:end, :]
                logits = self.mlm_head(mask_hidden_states)
                # shifted_prediction_scores = logits[:, :-1, :].contiguous()
                # labels = mask_predict_outputs_ts[:, 1:].contiguous()
                labels = mask_predict_outputs_ts
                loss_fct = torch.nn.CrossEntropyLoss()
                mask_loss = loss_fct(
                    logits.view(-1, len(self.bert_tokenizer)), labels.view(-1))
                final_loss = final_loss + self.args.label_weight * mask_loss

            if self.args.add_emotion_task:
                hidden_states = res.hidden_states  # len=13, embedding + 12 layer
                hidden_states = hidden_states[-1][:, -2, :]
                logits = self.emotion_head(hidden_states)
                emotion_labels = torch.tensor(
                    emotion_ids, dtype=torch.long, device=device)
                loss_fct = torch.nn.CrossEntropyLoss()
                emotion_loss = loss_fct(logits, emotion_labels)
                if batch_idx % 1000 == 0:
                    logger.debug(f'{cls_loss}, {emotion_loss}')
                # return cls_loss + 0.1 * emotion_loss,
                final_loss = final_loss + self.args.emotion_weight * emotion_loss

            if self.args.add_predict_context_task:
                mask_context_text_emb = self.bert.bert.embeddings.word_embeddings(
                    mask_context_input_ids)
                if self.args.add_ocr_info:
                    mask_context_input_emb = torch.cat(
                        [mask_context_text_emb, cls_inputs_emb, img_emb, sep_emb], dim=1)
                    ctx_extra_len = mask_context_input_emb.size(1) - mask_context_text_emb.size(1)
                    ones_mask2 = torch.ones(batch_size, ctx_extra_len, device=device)
                    mask_context_attention_mask = torch.cat(
                        [mask_context_attention_mask, ones_mask2], dim=1)
                    mask_context_token_type_ids = torch.zeros_like(
                        mask_context_attention_mask, dtype=torch.long)
                    mask_context_token_type_ids[:, -ctx_extra_len:] = 1
                else:
                    mask_context_input_emb = torch.cat(
                        [mask_context_text_emb, img_emb, sep_emb], dim=1)
                    ctx_extra_len = mask_context_input_emb.size(1) - mask_context_text_emb.size(1)
                    ones_mask2 = torch.ones(batch_size, ctx_extra_len, device=device)
                    mask_context_attention_mask = torch.cat(
                        [mask_context_attention_mask, ones_mask2], dim=1)
                    mask_context_token_type_ids = torch.zeros_like(
                        mask_context_attention_mask, dtype=torch.long)
                    mask_context_token_type_ids[:, -ctx_extra_len:] = 1

                mask_context_res = self.bert(inputs_embeds=mask_context_input_emb,
                                             attention_mask=mask_context_attention_mask, token_type_ids=mask_context_token_type_ids, return_dict=True, output_hidden_states=True)
                # len=13, embedding + 12 layer
                mask_context_hidden_states = mask_context_res.hidden_states
                mask_context_hidden_states = mask_context_hidden_states[-1][:, :mask_context_output_ids.size(
                    1), :]
                logits = self.mlm_head(mask_context_hidden_states)
                # shifted_prediction_scores = logits[:, :-1, :].contiguous()
                # labels = mask_context_output_ids[:, 1:].contiguous()
                labels = mask_context_output_ids
                loss_fct = torch.nn.CrossEntropyLoss()
                mask_context_loss = loss_fct(
                    logits.view(-1, len(self.bert_tokenizer)), labels.view(-1))
                final_loss = final_loss + self.args.ctx_weight * mask_context_loss
            if batch_idx % 1000 == 0:
                mask_loss = mask_loss if 'mask_loss' in dir() else None
                emotion_loss = emotion_loss if 'emotion_loss' in dir() else None
                mask_context_loss = mask_context_loss if 'mask_context_loss' in dir() else None
                logger.debug(
                    f"cls_loss:{cls_loss}, mask_loss:{mask_loss}, emotion_loss:{emotion_loss}, mask_context_loss:{mask_context_loss}")
            return final_loss,
        else:
            device = attention_mask.device
            batch_size = input_ids.size(0)
            assert batch_size == 1
            img_num = img_emb.size(0)
            sep_emb = self.bert.bert.embeddings.word_embeddings(
                torch.tensor(self.bert_tokenizer.sep_token_id, device=device, dtype=torch.long)).unsqueeze(0).unsqueeze(0).repeat(img_num, 1, 1)
            # logger.debug(text_emb.size())
            text_emb = self.bert.bert.embeddings.word_embeddings(input_ids)
            text_emb = text_emb.repeat(img_num, 1, 1)
            if self.args.use_visual_personalization_token:
                if user_ids is None:
                    user_ids = ['__UNK__']
                loo_current_img_ids = self._get_loo_current_img_ids(
                    img_ids=img_ids, test=test)
                user_token = self._build_user_token(
                    user_ids=user_ids, current_img_ids=loo_current_img_ids, device=device)
                user_token = user_token.repeat(img_num, 1, 1)
                img_emb = self.sticker_token_proj(img_emb).unsqueeze(1)
                if self.args.add_ocr_info:
                    input_emb, _, token_type_ids = self._compose_personalized_inputs(
                        text_emb=text_emb,
                        attention_mask=attention_mask.repeat(img_num, 1),
                        sticker_token=img_emb,
                        user_token=user_token,
                        aux_emb=cls_inputs_emb,
                    )
                else:
                    input_emb, _, token_type_ids = self._compose_personalized_inputs(
                        text_emb=text_emb,
                        attention_mask=attention_mask.repeat(img_num, 1),
                        sticker_token=img_emb,
                        user_token=user_token,
                        aux_emb=None,
                    )
            else:
                img_emb = self.img_ff(img_emb).unsqueeze(1)
                if self.args.add_ocr_info:
                    input_emb = torch.cat(
                        [text_emb, cls_inputs_emb, img_emb, sep_emb], dim=1)
                else:
                    input_emb = torch.cat(
                        [text_emb, img_emb, sep_emb], dim=1)
                extra_len = input_emb.size(1) - text_emb.size(1)
                token_type_ids = torch.zeros(size=(input_emb.size(
                    0), input_emb.size(1)), device=device, dtype=torch.long)
                token_type_ids[:, -extra_len:] = 1
            res = self.bert(inputs_embeds=input_emb,
                            token_type_ids=token_type_ids, return_dict=True)
            # logger.debug(res.logits.size())
            logits = res.logits[:, 1].unsqueeze(0)
            labels = torch.tensor(img_ids, dtype=torch.long, device=device)
            return logits, labels, cands_ret

    def prepare_imgs(self, args):
        """
        准备视觉侧基础资源：图像路径映射、图像预处理器、图像缓存容器。

        输入：
        - args (Arguments): 配置对象，至少包含 `id2img_path` 与 `img_dir`。

        输出：
        - 无显式返回值。

        副作用：
        - 更新 `self.id2imgpath`、`self.id2img`、`self.img_process`。

        说明：
        - `use_img_id` 模式无需真实图像读取，会直接返回；
        - 其余模式会加载 id->文件名映射，供后续按 ID 取图。
        """
        self.args = args
        if self.args.model_choice == 'use_img_id':
            return
        img_dir = args.img_dir
        self.id2img = {}
        if self.args.model_choice == 'use_clip_repo':
            self.img_process = self.clip_process
        else:
            if hasattr(self.img_clip, 'preprocess'):
                self.img_process = self.img_clip.preprocess
            else:
                self.img_process = self.img_clip._first_module().preprocess
        logger.info("prepare img objs")
        self.id2imgpath = {}

        with open(args.id2img_path, encoding='utf-8') as f:
            id2img = json.load(f)
            for id, img in tqdm(id2img.items()):
                id = int(id)
                self.id2imgpath[id] = img

        # if args.fix_clip:
        #     logger.info('fix clip!')
        #     for p in self.img_clip.parameters():
        #         p.requires_grad = False

        # if self.args.mode != 'gen':
        #     assert self.text_clip.training == True
        #     assert self.img_clip.training == True

    def get_input_embeddings_with_segment(self, input_ids, segment_ids):
        """
        根据 `segment_ids` 把文本 embedding 与图像 embedding 融合成统一输入。

        输入：
        - input_ids (Tensor[B, L]): token 序列（图像位上存放 img_id）。
        - segment_ids (Tensor[B, L]): 段标记（约定 2 表示图像位）。

        输出：
        - Tensor[B, L, H]: 融合后的 embedding。

        流程：
        - 文本位直接取词向量；
        - 图像位取 `img_id -> 图像embedding -> img_ff`；
        - 最后叠加 segment embedding。
        """
        # position_ids will be computed by the model itself even if the inputs_embeds is given
        # position_ids = torch.arange(input_ids.size(-1), device=input_ids.device).unsqueeze(0)
        # logger.debug(f"{self.transformer.wte(input_ids).size()}, {self.transformer.wpe(position_ids).size()}, {self.segment_embedding(segment_ids).size()}")
        img_mask = (segment_ids == 2)
        text_mask = (segment_ids != 2)
        text_embedding = self.bert.embeddings.word_embeddings(
            input_ids) * (text_mask.unsqueeze(-1))
        img_pos = img_mask.nonzero(as_tuple=True)
        img_ids = input_ids[img_pos]
        img_objs = []
        for i, id in enumerate(img_ids):
            img_obj = self.id2img[id.item()]
            img_objs.append(img_obj)

        if img_objs:

            img_tokens = self.img_clip.tokenize(img_objs)
            img_tokens = batch_to_device(
                img_tokens, self.img_clip._target_device)
            # img_emb = self.img_clip.encode(img_objs, convert_to_tensor=True)
            # img_emb.requires_grad = True
            # print(img_emb.size())
            # img_emb = img_emb.to(input_ids.device)
            img_emb = self.img_clip.forward(img_tokens)['sentence_embedding']
            if self.args.mode == 'gen':
                assert self.img_clip.training == False
            emb = self.img_ff(img_emb)
            # text_embedding[img_pos[0][i], img_pos[1][i]] = emb
            # logger.debug(f"emb: {emb.requires_grad}")
            zeros = torch.zeros_like(text_embedding, device=input_ids.device)
            zeros[img_pos] = emb
            # text_embedding[img_pos] = emb
            # logger.debug(f"zeros:{zeros.requires_grad}")
            text_embedding = text_embedding + zeros

        # img_emb = self.img_clip.encode(img_objs, convert_to_tensor=True)
        # # print(img_emb.size())
        # img_emb = img_emb.to(input_ids.device)
        # emb = self.img_ff(img_emb)
        # # text_embedding[img_pos[0][i], img_pos[1][i]] = emb
        # text_embedding[img_pos] = emb

        return text_embedding + self.segment_embedding(
            segment_ids)


def _expected_cand_size_from_path(path: Optional[str]) -> Optional[int]:
    """Infer R10 vs R20 from filename (StickerChat *_with_cand_r10.json etc.)."""
    if not path:
        return None
    p = str(path)
    if "cand_r20" in p or ("_r20" in p and "cand" in p):
        return 20
    if "cand_r10" in p or ("_r10" in p and "cand" in p):
        return 10
    return None


class PLDataset(Dataset):
    def __init__(self, path, mode, args, tokenizer, expected_cand_size: Optional[int] = None):
        """
        构造数据集对象，加载 JSON 并保存运行上下文。

        输入：
        - path (str): 数据文件路径（`*_pair.json`）。
        - mode (str): 运行模式（train/test/gen/...）。
        - args (Arguments): 配置对象。
        - tokenizer: 文本分词器。
        - expected_cand_size (Optional[int]): R10/R20 长度；on-the-fly cand 时用（默认从 path 推断）。

        输出：
        - 无显式返回值（构造函数）。
        """
        with open(path, encoding='utf-8') as f:
            self.data = json.load(f)
        # self.tokenizer = tokenizer
        self.mode = mode
        self.args = args
        self.tokenizer = tokenizer
        self._expected_cand_size = (
            expected_cand_size
            if expected_cand_size is not None
            else _expected_cand_size_from_path(str(path))
        )

    def __len__(self):
        """
        返回数据集样本数量。

        输入：
        - 无。

        输出：
        - int: 样本总数。
        """
        return len(self.data)

    def random_mask(self, text_ids, return_ts):
        """
        对 token 序列做 MLM 风格随机掩码，生成“输入序列 + 监督序列”。

        输入：
        - text_ids (list[int]): 原始 token id 序列。
        - return_ts (bool): 是否返回 tensor；False 时返回 python list。

        输出：
        - tuple:
          - input_ids: mask 后输入；
          - output_ids: 监督标签（未监督位置为 -100）。

        策略（BERT风格）：
        - 15% token 进入“预测集合”；
        - 其中 80% 替换 `[MASK]`、10% 保持原词、10% 随机词。
        """
        # random mask for MLM
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        mask_id = self.tokenizer.mask_token_id
        l = len(self.tokenizer)
        for r, i in zip(rands, text_ids):
            # if i == helper.sep_id or i == helper.cls_id:
            #     input_ids.append(i)
            #     output_ids.append(-100)
            #     continue
            if r < 0.15 * 0.8:
                input_ids.append(mask_id)
                output_ids.append(i)
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)
            elif r < 0.15:
                input_ids.append(np.random.choice(l))
                output_ids.append(i)
            else:
                input_ids.append(i)
                output_ids.append(-100)
        if return_ts:
            # logger.debug(f"input_ids: {input_ids}, output_ids: {output_ids}")
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
                output_ids, dtype=torch.long)
        else:
            return input_ids, output_ids

    def __getitem__(self, index):
        """
        读取并解析单条样本，构造模型前向所需字段字典。

        输入：
        - index (int): 样本索引。

        输出：
        - dict，常见键包括：
          - `sent`: 拼接后的对话文本；
          - `img_id` / `neg_img_id`: 正负贴纸ID；
          - `user_id`: 用户键（真实或 fallback）；
          - `cand`: 候选集（R10）；
          - `emotion_id` 与上下文预测辅助字段。

        说明：
        - 对话默认取最后 `sent_num` 轮；
        - `user_id` 通过 `_resolve_user_key` 统一构造，兼容缺字段旧数据。
        """
        sample = self.data[index]
        dialog = sample['dialog']
        sent_num = self.args.sent_num
        if sent_num == 0:
            selected_dialog = dialog
        else:
            selected_dialog = dialog[-sent_num:]
        sent = ''
        img_id = None

        def pad_none(x, l=5):
            """
            将列表补齐到固定长度（内部辅助函数，当前保留兼容用途）。

            输入：
            - x (list): 原列表。
            - l (int): 目标长度。

            输出：
            - list: 补 None 后的新列表。
            """
            return x + [None] * (l - len(x))
        for i, d in enumerate(selected_dialog):
            if i == len(selected_dialog) - 1:
                # s, img_id, img_word, neg_img_id, neg_img_word = pad_none(d, 5)
                s = d.get('text', None)
                img_id = d.get('img_id', None)
                img_word = d.get('img_label', None)
                neg_img_id = d.get('neg_img_id', None)
                neg_img_word = d.get('neg_img_label', None)
                emotion_id = d.get('emotion_id', None)
            else:
                # s, _, _ = d
                s = d['text']
            sent += s
        # sent, img_id, img_word, neg_img_id, neg_img_word = dialog[-1]
        # img_id = int(img_id)
        # sent = sent.replace('[speaker1]', '').replace('[speaker2]', '')

        if self.args.add_predict_context_task:
            sent_ids = self.tokenizer.encode(sent)
            # logger.debug(sent_ids)
            mask_context_input_ids, mask_context_output_ids = self.random_mask(
                sent_ids, return_ts=True)
            # logger.debug(mask_context_input_ids)
#
            mask_context_attention_mask = torch.ones(
                len(mask_context_input_ids), dtype=torch.float)
        else:
            mask_context_input_ids, mask_context_output_ids = None, None
            mask_context_attention_mask = None

        if self.args.test_with_cand:
            raw_cand = sample.get('cand')
            # Empty JSON list [] must behave like missing cand (else collate passes [[]]).
            if isinstance(raw_cand, list) and len(raw_cand) > 0:
                cand = [int(id) for id in raw_cand]
            elif img_id is not None:
                # Global distractors (no reliance on same-pack neg_img_id from release).
                pos = int(img_id)
                size = int(self._expected_cand_size or 10)
                n_other = size - 1
                pool = [i for i in range(self.args.max_image_id) if i != pos]
                if len(pool) < n_other:
                    raise ValueError(
                        f"PLDataset index={index}: need {n_other} distractors, pool={len(pool)}"
                    )
                others = random.sample(pool, n_other)
                cand = [pos] + others
                if neg_img_id is None:
                    neg_img_id = others[0]
            else:
                cand = None
            if cand is None and (
                getattr(self.args, 'candidate_eval_only', False)
                or self.args.test_with_cand
            ):
                raise ValueError(
                    f"PLDataset index={index}: missing or empty cand and no img_id for on-the-fly build."
                )
        else:
            cand = None
        user_id = _resolve_user_key(sample, dialog, index)

        res = {
            'sent': sent,
            'img_word': img_word,
            'img_id': img_id,
            'neg_img_word': neg_img_word,
            'neg_img_id': neg_img_id,
            'user_id': user_id,
            'emotion_id': emotion_id,
            'cand': cand,
            'mask_context_output_ids': mask_context_output_ids,
            'mask_context_input_ids': mask_context_input_ids,
            'mask_context_attention_mask': mask_context_attention_mask
        }
        return res


class PLDataLoader(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        """
        初始化 LightningDataModule，保存数据路径与批处理配置。

        输入：
        - args (Arguments): 配置对象。
        - tokenizer: 文本分词器。

        输出：
        - 无显式返回值（构造函数）。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.train_data_path = args.train_data_path
        self.val_data_path = args.val_data_path
        self.test_data_path = args.test_data_path
        self.train_batch_size = args.train_batch_size
        self.valtest_batch_size = args.valtest_batch_size
        self.gen_data_path = args.gen_data_path
        self.gen_batch_size = args.gen_batch_size
        self.args = args

    def setup(self, stage: Optional[str] = None):
        """
        根据 stage 构建训练/验证/测试数据集对象。

        输入：
        - stage (Optional[str]): `fit` / `test` / `None`。

        输出：
        - 无显式返回值；副作用是写入 `self.train_dataset` 等成员。

        规则：
        - `fit`：加载 train + 可选 val；
        - `test`：加载 test（或 gen 模式对应数据）；
        - `None`：两类都准备。
        """
        if stage == 'fit' or stage is None:
            logger.info('begin get data')

            self.train_dataset = PLDataset(
                self.train_data_path,
                self.args.mode,
                self.args,
                self.tokenizer,
                expected_cand_size=_expected_cand_size_from_path(self.train_data_path),
            )
            n_train = len(self.train_dataset)
            logger.info(f"train samples={n_train}, train_batch_size={self.train_batch_size}, "
                        f"steps_per_epoch={max(1, (n_train + self.train_batch_size - 1) // self.train_batch_size)}")
            pe10 = (getattr(self.args, 'per_epoch_eval_test_r10_path', None) or '').strip()
            pe20 = (getattr(self.args, 'per_epoch_eval_test_r20_path', None) or '').strip()
            if pe10 and pe20:
                self.val_dataset_r10 = PLDataset(
                    pe10, self.args.mode, self.args, self.tokenizer, expected_cand_size=10)
                self.val_dataset_r20 = PLDataset(
                    pe20, self.args.mode, self.args, self.tokenizer, expected_cand_size=20)
                self._per_epoch_dual_val_loaders = True
                logger.info(
                    'per-epoch eval: test R10 (%d) + test R20 (%d) — no val split',
                    len(self.val_dataset_r10),
                    len(self.val_dataset_r20),
                )
            elif self.args.val_data_path:
                self._per_epoch_dual_val_loaders = False
                self.val_dataset = PLDataset(
                    self.val_data_path,
                    self.args.mode,
                    self.args,
                    self.tokenizer,
                    expected_cand_size=_expected_cand_size_from_path(self.val_data_path),
                )
                logger.info('has val!')
            else:
                self._per_epoch_dual_val_loaders = False
                logger.info('no val!')
            logger.info('finish get data')

        if stage == 'test' or stage is None:
            logger.info('begin get data')
            if self.args.mode != 'gen':
                self.test_dataset = PLDataset(
                    self.test_data_path,
                    self.args.mode,
                    self.args,
                    self.tokenizer,
                    expected_cand_size=_expected_cand_size_from_path(self.test_data_path),
                )
            else:
                self.test_dataset = PLDataset(
                    self.gen_data_path, self.args.mode, self.args, self.tokenizer)
            logger.info('finish get data')

    def collate_fn(self, batch):
        """
        将 `list[样本dict]` 整理为模型可直接消费的 batch 字典。

        输入：
        - batch (list[dict]): `PLDataset.__getitem__` 输出的样本列表。

        输出：
        - dict，核心键包括：
          - `input_ids`, `attention_mask`（文本）
          - `img_ids`, `neg_img_ids`（主任务正负贴纸）
          - `user_ids`（个性化）
          - `cands`（R10 候选）
          - `emotion_ids`、`mask_context_*`（辅助任务）

        关键点：
        - 文本统一分词并 padding；
        - 上下文预测序列使用 `pad_sequence` 对齐；
        - 保留 `user_ids` 原始字符串列表，后续在模型内映射索引。
        """
        # logger.debug(batch)
        sents = [d['sent'] for d in batch]
        # img_words = [d['img_word'] for d in batch]
        img_ids = [d['img_id'] for d in batch]
        # neg_img_words = [d['neg_img_word'] for d in batch]
        neg_img_ids = [d['neg_img_id'] for d in batch]
        user_ids = [d['user_id'] for d in batch]
        chunk = int(getattr(self.args, 'max_dialogue_length', None) or 490)
        res = self.tokenizer(sents, return_tensors='pt',
                             padding=True, truncation=True, max_length=chunk)
        input_ids = res['input_ids']
        attention_mask = res['attention_mask']
        # R10/R20：仅当 cand 非空才传 cands（避免 [[]] 触发全库回退或 OOM）。
        c0 = batch[0].get('cand')
        if isinstance(c0, list) and len(c0) > 0:
            cands = [d['cand'] for d in batch]
        else:
            cands = None

        if batch[0]['emotion_id'] is not None:
            emotion_ids = [d['emotion_id'] for d in batch]
        else:
            emotion_ids = None

        if batch[0]['mask_context_input_ids'] is not None:
            # chunk = 500
            mask_context_input_ids = pad_sequence(
                [d['mask_context_input_ids'][-chunk:] for d in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            mask_context_output_ids = pad_sequence(
                [d['mask_context_output_ids'][-chunk:] for d in batch], batch_first=True, padding_value=-100)
            mask_context_attention_mask = pad_sequence(
                [d['mask_context_attention_mask'][-chunk:] for d in batch], batch_first=True, padding_value=0)
        else:
            mask_context_input_ids = None
            mask_context_output_ids = None
            mask_context_attention_mask = None

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'img_ids': img_ids,
            'neg_img_ids': neg_img_ids,
            'user_ids': user_ids,
            'cands': cands,
            'emotion_ids': emotion_ids,
            'mask_context_input_ids': mask_context_input_ids,
            'mask_context_output_ids': mask_context_output_ids,
            'mask_context_attention_mask': mask_context_attention_mask,
        }

    def train_dataloader(self, ):
        """
        构建训练 DataLoader。

        输入：
        - 无。

        输出：
        - DataLoader: 训练迭代器（`shuffle=True`）。
        """
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          num_workers=self.args.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        """
        构建验证 DataLoader（若存在验证集路径）。

        输入：
        - 无。

        输出：
        - Optional[DataLoader]: 有验证集则返回迭代器，否则返回 None。
        """
        # return None
        if getattr(self, '_per_epoch_dual_val_loaders', False):
            return [
                DataLoader(
                    self.val_dataset_r10,
                    batch_size=self.valtest_batch_size,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                    shuffle=False,
                    collate_fn=self.collate_fn,
                ),
                DataLoader(
                    self.val_dataset_r20,
                    batch_size=self.valtest_batch_size,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                    shuffle=False,
                    collate_fn=self.collate_fn,
                ),
            ]
        if self.args.val_data_path:
            return DataLoader(self.val_dataset,
                              batch_size=self.valtest_batch_size,
                              num_workers=self.args.num_workers,
                              pin_memory=True,
                              shuffle=False,
                              collate_fn=self.collate_fn)
        else:
            return None

    def test_dataloader(self):
        """
        构建测试或生成模式 DataLoader。

        输入：
        - 无。

        输出：
        - DataLoader: 测试/生成迭代器（不打乱）。
        """
        batch_size = self.valtest_batch_size if self.args.mode != 'gen' else self.gen_batch_size
        return DataLoader(self.test_dataset,
                          batch_size=batch_size,
                          num_workers=self.args.num_workers,
                          pin_memory=True,
                          shuffle=False,
                          collate_fn=self.collate_fn)


def _candidate_eval_mode_label(test_with_cand: bool, max_cand_len: int) -> str:
    if not test_with_cand:
        return "Rall(all-stickers)"
    if max_cand_len > 0:
        return f"R{max_cand_len}(candidates)"
    return "Rcand(candidates)"


def _pl_new_eval_metric_bundle():
    return {
        'acc5': MyAccuracy(),
        'acc30': MyAccuracy(),
        'acc90': MyAccuracy(),
        'acc_r10': MyAccuracy(),
        'acc_r20': MyAccuracy(),
        'map': MyAccuracy(),
    }


def _pl_attach_per_epoch_dual_test_eval(pl_model, args):
    p10 = (getattr(args, 'per_epoch_eval_test_r10_path', None) or '').strip()
    p20 = (getattr(args, 'per_epoch_eval_test_r20_path', None) or '').strip()
    pl_model._per_epoch_dual_test_eval = bool(p10 and p20)
    if pl_model._per_epoch_dual_test_eval:
        pl_model._eval_bundles = (
            _pl_new_eval_metric_bundle(),
            _pl_new_eval_metric_bundle(),
        )
        pl_model._eval_max_cand_pair = [0, 0]
    else:
        pl_model._eval_bundles = None


class PLModel(pl.LightningModule):
    def __init__(self, args):
        """
        Lightning 训练封装层初始化。

        输入：
        - args (Arguments): 全量配置对象。

        输出：
        - 无显式返回值（构造函数）。

        初始化内容：
        - `self.model`（核心网络）；
        - train/val/test 指标对象（R@k / MRR）；
        - 图像资源准备与 `id2name` 字典。
        """
        super().__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path)
        # logger.info(f"tokenizer type:{type(self.tokenizer)}")(
        self.model = Model(args)
        # special_tokens_dict = {
        #     # 'pad_token': '[PAD]',
        #     'additional_special_tokens': ['[speaker1]', '[speaker2]', '[img]']
        # }
        # self.tokenizer.add_special_tokens(special_tokens_dict)
        # logger.info(
        #     f"tokenizer add special tokens dict: {special_tokens_dict}")
        # self.model.resize_token_embeddings(len(self.tokenizer))
        # self.lr = args.lr
        # self.acc = MyAccuracy(device=torch.device("cuda"))
        # self.train_acc5 = MyAccuracy()
        # self.train_acc30 = MyAccuracy()
        # self.train_acc90 = MyAccuracy()
        self.train_acc = MyAccuracy()
        self.valtest_acc5 = MyAccuracy()
        self.valtest_acc30 = MyAccuracy()
        self.valtest_acc90 = MyAccuracy()
        self.valtest_acc_r10 = MyAccuracy()
        self.valtest_acc_r20 = MyAccuracy()
        self.valtest_map = MyAccuracy()
        self._eval_max_cand_len = 0
        _pl_attach_per_epoch_dual_test_eval(self, args)
        self.args = args
        self.model.prepare_imgs(args)
        self.id2name = {}
        id2name_path = getattr(args, 'id2name_path', './data/id2name.json')
        with open(id2name_path, encoding='utf-8') as f:
            a = json.load(f)
            for k, v in a.items():
                self.id2name[int(k)] = v

    def _attach_runtime_version_log(self):
        """
        在 Trainer 生命周期已启动后，重新绑定到真实 version 目录日志文件。

        输入：
        - 无（依赖 self.args 与 self.trainer）。

        输出：
        - 无显式返回值。

        说明：
        - 解决 DDP 下“过早访问 trainer.log_dir 可能报错”的问题；
        - 若此时 log_dir 可用，会把日志从回退路径切换到 version_x 下。
        """
        try:
            attach_version_log_from_trainer(self.args, self.trainer)
        except Exception as e:
            logger.warning(f"attach runtime version log failed: {e}")

    def on_fit_start(self) -> None:
        """训练正式开始时绑定到真实 version 日志文件。"""
        self._attach_runtime_version_log()
        return super().on_fit_start()

    def on_test_start(self) -> None:
        """测试正式开始时绑定到真实 version 日志文件。"""
        self._attach_runtime_version_log()
        return super().on_test_start()

    def run_model_from_batch(self, batch, batch_idx, test=False):
        """
        从 batch 字典中解包字段，并统一调用 `Model.forward`。

        输入：
        - batch (dict): `collate_fn` 产出的批数据。
        - batch_idx (int): 当前批次编号。
        - test (bool): 是否走测试路径。

        输出：
        - 与 `self.model.forward(...)` 完全一致。

        作用：
        - 把 Lightning 层与底层模型输入协议解耦；
        - 统一处理可选字段（`user_ids/cands/emotion/mask_context`）。
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # img_tokens = batch['img_tokens']
        img_ids = batch['img_ids']
        # neg_img_tokens = batch['neg_img_tokens']
        neg_img_ids = batch['neg_img_ids']
        user_ids = batch.get('user_ids', None)
        cands = batch.get('cands', None)
        emotion_ids = batch.get('emotion_ids', None)
        mask_context_input_ids = batch.get('mask_context_input_ids', None)
        mask_context_output_ids = batch.get('mask_context_output_ids', None)
        mask_context_attention_mask = batch.get(
            'mask_context_attention_mask', None)

        return self.model(batch_idx=batch_idx, input_ids=input_ids, attention_mask=attention_mask, img_ids=img_ids, neg_img_ids=neg_img_ids, user_ids=user_ids,
                          test=test, cands=cands, emotion_ids=emotion_ids, mask_context_input_ids=mask_context_input_ids, mask_context_output_ids=mask_context_output_ids,
                          mask_context_attention_mask=mask_context_attention_mask)

    def compute_acc(self, sorted_idx, labels, k=5):
        """
        计算 top-k 命中统计。

        输入：
        - sorted_idx (Tensor[B, N]): 每个样本按分数降序排序后的候选索引。
        - labels (Tensor[B]): 真实标签索引（在当前索引空间中）。
        - k (int): 统计前 k 个。

        输出：
        - tuple:
          - cor (Tensor标量): 命中数；
          - total (int/Tensor): 样本总数。
        """
        idx = sorted_idx[:, :k]
        labels = labels.unsqueeze(-1).expand_as(idx)
        cor_ts = torch.eq(idx, labels).any(dim=-1)
        cor = cor_ts.sum()
        total = cor_ts.numel()
        return cor, total

    def compute_map(self, sorted_idx, labels):
        """
        计算单标签检索场景下的 MRR 累加值与样本数。

        输入：
        - sorted_idx (Tensor[B, N]): 排序后索引。
        - labels (Tensor[B]): 真实标签索引。

        输出：
        - tuple:
          - mrr_sum (Tensor标量): `sum(1/rank_i)`；
          - count (int): 样本数。
        """
        l = labels.unsqueeze(-1)
        w = sorted_idx == l
        _, idx = w.nonzero(as_tuple=True)
        s = 1. / (idx + 1)
        return torch.sum(s), labels.size(0)

    def training_step(self, batch, batch_idx):
        """
        Lightning 训练单步。

        输入：
        - batch (dict): 当前训练批次数据。
        - batch_idx (int): 批次编号。

        输出：
        - Tensor: 当前批次 loss（供 Lightning 自动反向与优化）。
        """
        # logger.debug(batch)
        outputs = self.run_model_from_batch(batch, batch_idx)

        loss = outputs[0]

        self.log('train_loss', loss)
        # logger.info(f"loss:{loss}")
        return loss

    def training_step_end(self, *args, **kwargs):
        """
        训练步后处理：裁剪 `temperature` 参数范围。

        输入：
        - *args, **kwargs: Lightning 透传参数。

        输出：
        - 父类 `training_step_end` 返回值。

        说明：
        - 把温度参数限制在 `[0, log(100)]`，防止出现过大尺度导致不稳定。
        """
        self.model.temperature.data = torch.clamp(
            self.model.temperature.data, 0, np.log(100))
        return super().training_step_end(*args, **kwargs)

    def on_validation_epoch_start(self) -> None:
        """
        验证轮开始回调：预先构建测试候选 embedding 缓存。

        输入：
        - 无。

        输出：
        - 父类回调返回值。
        """
        # if self.args.val_data_path:
        self.model.prepare_for_test()
        if getattr(self, '_per_epoch_dual_test_eval', False):
            self._eval_max_cand_pair = [0, 0]
        else:
            self._eval_max_cand_len = 0
        return super().on_validation_epoch_start()

    def log_res(self, batch, sorted_idx, batch_idx, k=5):
        """
        样本级调试日志函数（当前默认直接 return，不启用）。

        输入：
        - batch (dict): 原始 batch。
        - sorted_idx (Tensor): 排序索引。
        - batch_idx (int): 批次编号。
        - k (int): 展示 top-k。

        输出：
        - 无显式返回值。
        """
        # valtest batch size == 1
        return
        label = batch['img_ids'][0]
        input_ids = batch['input_ids']
        input_text = self.model.bert_tokenizer.decode(
            input_ids.squeeze(0).tolist())
        topk_idx = sorted_idx[0, :k].tolist()
        label_text = self.id2name[label]
        cands = batch.get('cands', None)
        if cands:
            # logger.debug(topk_idx, cands)
            topk_idx = [cands[0][idx] for idx in topk_idx]
            # assert label in topk_idx

        topk_text = [self.id2name[idx] for idx in topk_idx]
        logger.info(
            f"input_text:{input_text}, label:{label_text}, topk_text:{topk_text}, batch_idx:{batch_idx}")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        验证/测试单步：完成候选打分、排序与指标更新。

        输入：
        - batch (dict): 单批评测数据。
        - batch_idx (int): 批次编号。

        输出：
        - tuple:
          - metrics_list (list[float]): 当前样本的 `[r@1, r@2, r@5, mrr]`；
          - return_preds (list[int]): 预测排序后的贴纸ID列表；
          - return_labels (int): 真实贴纸ID。

        关键处理：
        - 若 R10（有 `cands`），先将真实贴纸ID映射到 cand 内部索引空间；
        - 再基于统一索引空间计算命中率和 MRR。
        """
        outputs = self.run_model_from_batch(batch, batch_idx, test=True)
        # labels = batch['labels']
        if self.args.mode == 'predict_img_label':
            return
        logits = outputs[0]
        labels = outputs[1]
        cands = outputs[2]  # when args.test_with_cand==True
        if cands is not None:
            cands = [int(cand) for cand in cands]
        _, idx = torch.sort(logits, dim=-1, descending=True)
        assert len(labels) == 1
        # print(cands, idx)
        if cands is not None:
            return_preds = torch.tensor(cands)[idx.squeeze(0)].tolist()
        else:
            return_preds = torch.arange(idx.size(1))[idx.squeeze(0)].tolist()
        # print(return_preds)
        return_labels = labels.item()

        self.log_res(batch, idx, batch_idx=batch_idx, k=11)

        if cands:
            # logger.debug(cands, labels)
            # R10 下 logits 的索引空间是“cand 内部索引”，
            # 所以需要把真实贴纸 id 映射到 cand 中对应位置后再算命中率/MRR。
            for i, cand in enumerate(cands):
                if int(cand) == labels.item():
                    labels[0] = i
                    # logger.debug(f'label after mapping:{i}')
                    break
            if not getattr(self, '_per_epoch_dual_test_eval', False):
                self._eval_max_cand_len = max(self._eval_max_cand_len, len(cands))

        topks = [1, 2, 5] if not self.args.test_with_cand else [1, 2, 5]
        # self.valtest_acc5.update(*self.compute_acc(idx, labels, topks[0]))
        # self.valtest_acc30.update(*self.compute_acc(idx, labels, topks[1]))
        # self.valtest_acc90.update(*self.compute_acc(idx, labels, topks[2]))
        # self.valtest_map.update(*self.compute_map(idx, labels))
        cor1, tot1 = self.compute_acc(idx, labels, topks[0])
        cor2, tot2 = self.compute_acc(idx, labels, topks[1])
        cor5, tot5 = self.compute_acc(idx, labels, topks[2])
        cor, tot = self.compute_map(idx, labels)
        if getattr(self, '_per_epoch_dual_test_eval', False):
            di = int(dataloader_idx)
            b = self._eval_bundles[di]
            b['acc5'].update(cor1, tot1)
            b['acc30'].update(cor2, tot2)
            b['acc90'].update(cor5, tot5)
            b['map'].update(cor, tot)
            if cands:
                nc = len(cands)
                self._eval_max_cand_pair[di] = max(self._eval_max_cand_pair[di], nc)
                if nc >= 10:
                    c10, t10 = self.compute_acc(idx, labels, 10)
                    b['acc_r10'].update(c10, t10)
                if nc >= 20:
                    c20, t20 = self.compute_acc(idx, labels, 20)
                    b['acc_r20'].update(c20, t20)
        else:
            self.valtest_acc5.update(cor1, tot1)
            self.valtest_acc30.update(cor2, tot2)
            self.valtest_acc90.update(cor5, tot5)
            self.valtest_map.update(cor, tot)
            if cands:
                nc = len(cands)
                if nc >= 10:
                    c10, t10 = self.compute_acc(idx, labels, 10)
                    self.valtest_acc_r10.update(c10, t10)
                if nc >= 20:
                    c20, t20 = self.compute_acc(idx, labels, 20)
                    self.valtest_acc_r20.update(c20, t20)
        return [(cor1 / tot1).item(), (cor2 / tot2).item(), (cor5 / tot5).item(), (cor / tot).item()], return_preds, return_labels

    def _pl_eval_epoch_log_and_reset_metrics(self) -> None:
        if getattr(self, '_per_epoch_dual_test_eval', False):
            for di, tag in enumerate(['test-R10', 'test-R20']):
                b = self._eval_bundles[di]
                if int(b['acc5'].total) == 0:
                    for _k, m in b.items():
                        m.reset()
                    continue
                nc = int(self._eval_max_cand_pair[di])
                mode = _candidate_eval_mode_label(bool(self.args.test_with_cand), nc)
                extra_r = ''
                if int(b['acc_r10'].total) > 0:
                    extra_r += f" r@10={b['acc_r10'].compute():.4f}"
                if int(b['acc_r20'].total) > 0:
                    extra_r += f" r@20={b['acc_r20'].compute():.4f}"
                logger.info(
                    f"\n[EvalSummary] epoch={self.current_epoch} split={tag} mode={mode} "
                    f"total={b['acc5'].total} "
                    f"r@1={b['acc5'].compute():.4f} r@2={b['acc30'].compute():.4f} "
                    f"r@5={b['acc90'].compute():.4f}{extra_r} mrr={b['map'].compute():.4f}"
                )
                for _k, m in b.items():
                    m.reset()
        else:
            eval_mode = _candidate_eval_mode_label(
                bool(self.args.test_with_cand), int(self._eval_max_cand_len)
            )
            extra_r = ''
            if int(self.valtest_acc_r10.total) > 0:
                extra_r += f" r@10={self.valtest_acc_r10.compute():.4f}"
            if int(self.valtest_acc_r20.total) > 0:
                extra_r += f" r@20={self.valtest_acc_r20.compute():.4f}"
            logger.info(
                f"\n[EvalSummary] epoch={self.current_epoch} mode={eval_mode} total={self.valtest_acc5.total} "
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
        """
        验证轮结束回调：打印摘要并重置累积指标。

        输入：
        - 无。

        输出：
        - 父类回调返回值（或在某些模式下提前 return）。
        """
        if self.args.mode == 'predict_img_label':
            logger.debug(self.model.predict_id2ocr)
            logger.debug(len(self.model.predict_id2ocr))
            with open('./data/ocr_predict.json', 'w', encoding='utf-8') as f:
                json.dump(self.model.predict_id2ocr, f,
                          indent=2, ensure_ascii=False)
            return
        self._pl_eval_epoch_log_and_reset_metrics()
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        """
        测试轮开始回调：准备候选 embedding 缓存。

        输入：
        - 无。

        输出：
        - 父类回调返回值。
        """
        self.model.prepare_for_test()
        if getattr(self, '_per_epoch_dual_test_eval', False):
            self._eval_max_cand_pair = [0, 0]
        else:
            self._eval_max_cand_len = 0
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        """
        测试单步，直接复用 `validation_step`。

        输入：
        - batch (dict): 测试批次。
        - batch_idx (int): 批次编号。

        输出：
        - 与 `validation_step` 相同。
        """
        return self.validation_step(batch, batch_idx, 0)

    def test_epoch_end(self, outputs):
        """
        测试结束回调：汇总样本指标并写入结果文件。

        输入：
        - outputs (list): 每个 `test_step` 返回项组成的列表。

        输出：
        - 无显式返回值。

        副作用：
        - 写入 `result/<test_file>_<with_cand>.json`（指标明细）；
        - 写入 `result/<test_file>_<with_cand>_pred.json`（预测与答案）；
        - 终端打印摘要路径与一条预测预览。
        """
        self.on_validation_epoch_end()
        if len(outputs) == 0:
            logger.warning("[TestSummary] empty outputs in test_epoch_end")
            return

        cols = ['r@1', 'r@2', 'r@5', 'mrr']
        res = {col: [] for col in cols}
        pred_res = []
        for item in outputs:
            metrics, pred, answer = item
            pred_res.append({'pred': pred, 'answer': answer})
            for i, val in enumerate(metrics):
                res[cols[i]].append(val)

        def get_filename_withoutext(path):
            """
            提取路径对应的“文件名主干”（去扩展名）。

            输入：
            - path (str): 文件路径。

            输出：
            - str: 不含扩展名的文件名。
            """
            filename_withext = os.path.basename(path)
            name, ext = os.path.splitext(filename_withext)
            return name

        outpath = f'result/{get_filename_withoutext(self.args.test_data_path)}_{self.args.test_with_cand}.json'
        with open(outpath, 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        pred_outpath = f'result/{get_filename_withoutext(self.args.test_data_path)}_{self.args.test_with_cand}_pred.json'
        with open(pred_outpath, 'w') as f:
            json.dump(pred_res, f, ensure_ascii=False, indent=2)

        # 测试结束时只打印摘要信息，避免把大数组直接打满终端。
        sample = pred_res[0] if pred_res else {}
        sample_pred = sample.get('pred', [])[:10]
        sample_answer = sample.get('answer', None)
        logger.info(
            f"[TestSummary] samples={len(outputs)} metrics_file={outpath} pred_file={pred_outpath}"
        )
        logger.info(
            f"[TestPreview] answer={sample_answer} top10_pred={sample_pred}"
        )

    @property
    def num_training_steps(self) -> int:
        """
        估计训练总 optimizer step 数，供 step-based scheduler 使用。

        输入：
        - 无（读取 trainer 状态）。

        输出：
        - int: 总 step 数。

        计算优先级：
        1) `trainer.max_steps`（若显式设置）；
        2) `trainer.estimated_stepping_batches`（Lightning 内建估计）；
        3) 手工估算（batch 数、设备数、梯度累积、epoch）。
        """
        # Lightning 默认 max_steps=-1（表示不限制）；不能用真值判断，
        # 否则 -1 会被当成 True，进而把 total_steps 错算成 1。
        if (self.trainer.max_steps is not None) and (self.trainer.max_steps > 0):
            return self.trainer.max_steps

        # 优先使用 Lightning 内置估计，避免 DDP / grad accumulation 下手算偏差。
        estimated_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if estimated_steps is not None and estimated_steps > 0:
            return int(estimated_steps)

        # 兜底路径：不要调用 LightningModule.train_dataloader()（本类未实现该 hook）。
        # 优先从 trainer/datamodule 读取训练 batch 数。
        batches = None
        trainer_batches = getattr(self.trainer, "num_training_batches", None)
        if isinstance(trainer_batches, int) and trainer_batches > 0:
            batches = trainer_batches

        if batches is None:
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None:
                try:
                    batches = len(dm.train_dataloader())
                except Exception:
                    batches = None

        if batches is None or batches <= 0:
            # 最后的安全回退，保证 scheduler 不会因为异常值崩溃。
            return max(1, int(self.trainer.max_epochs))

        limit_batches = self.trainer.limit_train_batches
        batches = min(batches, limit_batches) if isinstance(
            limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = max(1, self.trainer.accumulate_grad_batches * num_devices)
        return max(1, (batches // effective_accum) * self.trainer.max_epochs)

    def configure_optimizers(self):
        """
        配置优化器与学习率调度器。

        输入：
        - 无（读取模型参数与 `args`）。

        输出：
        - dict:
          - `optimizer`: AdamW
          - `lr_scheduler`: 包含 scheduler 与 `interval='step'`

        关键点：
        - 视觉编码器参数与其余参数分组，使用不同学习率；
        - cosine scheduler 必须按 step 更新，否则会出现学习率节奏错位。
        """
        logger.info(
            f"img lr:{self.args.img_lr}, fix text:{self.args.fix_text}, fix img:{self.args.fix_img}")
        assert self.args.model_choice == 'use_img_clip'
        img_params = [var for name, var in self.model.named_parameters(
        ) if name.startswith('img_clip') and var.requires_grad]
        other_params = [var for name, var in self.model.named_parameters() if
                        (not name.startswith('text_clip')) and (not name.startswith('img_clip')) and var.requires_grad]
        logger.info(
            f"len img params:{len(img_params)}, len total params:{len(list(self.model.parameters()))}")
        optimizer = AdamW([{'params': img_params, 'lr': self.args.img_lr}, {
                          'params': other_params}], lr=self.args.other_lr, betas=(0.9, 0.98), weight_decay=0.2)
        # 这个 cosine scheduler 是“按 step 设计”的，必须每个 optimizer step 更新一次；
        # 如果按 epoch 更新，会出现学习率节奏错位，甚至导致相邻 ckpt 权重重复。
        total_steps = max(1, self.num_training_steps)
        num_warmup_steps = 0
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, total_steps)
        logger.info(f"use scheduler! total_steps={total_steps}, warmup_steps={num_warmup_steps}")
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
        # return AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        # return torch.optim.SGD(self.model.parameters(), lr=0.5, momentum=0.9)
        # return AdamW(self.model.parameters(), lr=self.lr)


def test_emotion_prediction(args):
    """
    离线调试函数：遍历测试集并统计情绪预测准确率。

    输入：
    - args (Arguments): 需包含可用 `ckpt_path` 且 `mode=predict_emotion`。

    输出：
    - 无显式返回值；在终端打印有效样本数与平均准确率。

    说明：
    - 该函数不走 Lightning 的标准 test loop，用于快速人工诊断。
    """
    print('test emotion prediction!')
    model = PLModel(args)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    # print(ckpt.keys())
    model.load_state_dict(ckpt['state_dict'])
    device = torch.device('cuda')
    model.eval().to(device)
    pld = PLDataLoader(args, model.model.bert_tokenizer)
    pld.setup('test')
    dataset = pld.test_dataset
    batch_idx = 0
    scores = []

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        # print(data)
        if data['emotion_id'] == -100:
            continue

        batch = pld.collate_fn([data])
        # print(batch)
        user_ids = batch.get('user_ids', None)
        batch = batch_to_device(batch, device)
        if user_ids is not None:
            batch['user_ids'] = user_ids
        outputs = model.run_model_from_batch(batch, batch_idx, )
        # print(data['emotion_id'], outputs)
        scores.append(float(data['emotion_id'] == outputs.item()))
        batch_idx += 1
        # break

    print(f'valid samples:{len(scores)}, mean accuracy for predicting emotion:{np.mean(scores)}')


def test_sentence_gradient(args):
    """
    离线调试函数：提取 token 级梯度并导出可解释性日志。

    输入：
    - args (Arguments): 需包含可用 `ckpt_path` 且 `mode=test_gradient`。

    输出：
    - 无显式返回值。

    副作用：
    - 写入 `./log_gradient_test_easy.json`，包含 token 与对应梯度强度。
    """
    print('test sentence gradient!')
    model = PLModel(args)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    # print(ckpt.keys())
    model.load_state_dict(ckpt['state_dict'])
    device = torch.device('cuda')
    model.eval().to(device)
    pld = PLDataLoader(args, model.model.bert_tokenizer)
    pld.setup('test')
    dataset = pld.test_dataset
    batch_idx = 0
    scores = []

    tokenizer = model.model.bert_tokenizer

    res = []

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        # print(data)
        # if data['emotion_id'] == -100:
        #     continue

        batch = pld.collate_fn([data])
        # print(batch)
        user_ids = batch.get('user_ids', None)
        batch = batch_to_device(batch, device)
        if user_ids is not None:
            batch['user_ids'] = user_ids
        outputs = model.run_model_from_batch(batch, batch_idx, ) # grad
        grad, success = outputs
        print(f'success:{success}')
        # if not success:
        #     continue
        # print(f'grad:{outputs.size()}')
        input_ids = batch['input_ids'].squeeze(0).tolist()
        # print(input_ids)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(tokens)
        d = {}
        d['tokens'] = ''.join(tokens)
        grad_norm = torch.norm(grad.squeeze(0), 2, dim=-1)
        d['token_scores'] = []
        # print(grad_norm.size())
        for j, token in enumerate(tokens):
            print(token, grad_norm[j].item())
            d['token_scores'].append([token, grad_norm[j].item()])
        res.append(d)
        # print(data['emotion_id'], outputs)
        # scores.append(float(data['emotion_id'] == outputs.item()))
        batch_idx += 1
    
        # break
    with open('./log_gradient_test_easy.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f'valid samples:{len(scores)}, mean accuracy for predicting emotion:{np.mean(scores)}')

@dataclass
class Arguments:
    train_data_path: Optional[str] = field(default='./data/train_pair.json')
    val_data_path: Optional[str] = field(default='./data/validation_pair.json')
    test_data_path: Optional[str] = field(
        default='./data/validation_pair.json')
    gen_data_path: Optional[str] = field(default='./data/validation_gen.json')
    gen_out_data_path: Optional[str] = field(
        default='./data/validation_gen_out.json')
    img_lr: Optional[float] = field(default=5e-7)
    other_lr: Optional[float] = field(default=1e-5)
    gpus: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=2021)
    epochs: Optional[int] = field(default=10)
    train_batch_size: Optional[int] = field(default=8)
    valtest_batch_size: Optional[int] = field(default=1)
    gen_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    mode: Optional[str] = field(default='train')
    ckpt_path: Optional[str] = field(
        default='')
    ckpt_epoch: Optional[int] = field(default=-1)

    pretrain_path: Optional[str] = field(
        default='./ckpt/chinese-roberta-wwm-ext')
    bert_pretrain_path: Optional[str] = field(
        default='./ckpt/bert-base-chinese')
    img_pretrain_path: Optional[str] = field(
        default='./ckpt/clip-ViT-B-32')
    text_clip_pretrain_path: Optional[str] = field(
        default='./ckpt/clip-ViT-B-32-multilingual-v1')
    clip_download_root: Optional[str] = field(default='./ckpt')
    local_files_only: Optional[bool] = field(default=True)
    pl_root_dir: Optional[str] = field(default='logs/clip')
    id2img_path: Optional[str] = field(default='./data/id2img.json')
    img_dir: Optional[str] = field(default='./data/meme_set')
    fix_text: Optional[bool] = field(default=False)
    fix_img: Optional[bool] = field(default=False)
    model_choice: Optional[str] = field(default='use_img_clip')
    max_image_id: Optional[int] = field(default=307)
    max_emotion_id: Optional[int] = field(default=52)
    speaker_token_max_id: Optional[int] = field(default=2)
    init_temp: Optional[float] = field(default=np.log(1/0.07))
    sent_num: Optional[int] = field(default=1)
    num_workers: Optional[int] = field(default=32)
    max_dialogue_length: Optional[int] = field(
        default=490,
        metadata={
            'help': 'Tokenizer truncation max_length for dialogue in collate_fn; '
            'also used for mask_context_* tail slice when present.'
        },
    )
    test_with_cand: Optional[bool] = field(default=False)
    candidate_eval_only: Optional[bool] = field(
        default=False,
        metadata={
            'help': 'If True, force test_with_cand=True (e.g. StickerChat: fixed R10/R20, no Rall).'
        },
    )
    per_epoch_eval_test_r10_path: Optional[str] = field(
        default='',
        metadata={
            'help': 'If set with per_epoch_eval_test_r20_path, each train epoch validates on both test files (no val).'
        },
    )
    per_epoch_eval_test_r20_path: Optional[str] = field(
        default='',
        metadata={'help': 'Companion to per_epoch_eval_test_r10_path for dual test-set eval each epoch.'},
    )
    use_visual_personalization_token: Optional[bool] = field(default=False)
    use_visual_history_attention: Optional[bool] = field(default=True)
    visual_history_max_len: Optional[int] = field(default=50)
    visual_personalization_hidden_dim: Optional[int] = field(default=256)
    loo_filter_in_test: Optional[bool] = field(default=False)
    loo_filter_in_train: Optional[bool] = field(default=True)
    user_history_cache_path: Optional[str] = field(default='./data/user_history_cache.json')
    add_emotion_task: Optional[bool] = field(default=False)
    add_predict_context_task: Optional[bool] = field(default=False)
    add_predict_img_label_task: Optional[bool] = field(default=False)
    add_ocr_info: Optional[bool] = field(default=False)
    max_img_label_mask_num: Optional[int] = field(default=11, metadata={
                                                  'help': 'img label length plus one position for cls(start) token'})
    ocr_path: Optional[str] = field(default='./data/ocr_max10.json')
    id2name_path: Optional[str] = field(default='./data/id2name.json')
    emotion_weight: Optional[float] = field(default=0.1)
    ctx_weight: Optional[float] = field(default=0.05)
    label_weight: Optional[float] = field(default=0.1)
    prepare_batch_size: Optional[int] = field(
        default=512, metadata={'help': 'batch size for prepare_for_test encoding'})
    prepare_workers: Optional[int] = field(
        default=8, metadata={'help': 'parallel workers for loading images in prepare_for_test'})
    img_emb_cache_path: Optional[str] = field(
        default='', metadata={'help': 'path to precomputed sticker CLIP embeddings .pt; if exists, load instead of compute'})

    def __post_init__(self):
        """
        dataclass 参数后处理：解析 checkpoint 目录到具体文件。

        输入：
        - 无显式参数（使用实例字段）。

        输出：
        - 无显式返回值；可能修改 `self.ckpt_path`。

        行为：
        - 若 `ckpt_path` 指向目录：
          - 单文件目录直接选该文件；
          - 多文件目录需结合 `ckpt_epoch` 选择 `epoch=N-*.ckpt`。
        """
        # if self.add_predict_img_label_task is True:
        #     assert self.add_ocr_info is True
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            if os.path.isfile(self.ckpt_path):
                logger.info(f'real ckpt_path:{self.ckpt_path}')
            else:
                names = os.listdir(self.ckpt_path)
                if len(names) == 1:
                    self.ckpt_path = os.path.join(self.ckpt_path, names[0])
                elif len(names) > 1:
                    # 支持“目录 + epoch”方式选 ckpt：
                    # --ckpt_path 指向目录，--ckpt_epoch 指定 epoch=N 的文件。
                    # 这样测试命令更稳定，避免手工拼完整文件名。
                    if(self.ckpt_epoch == -1):
                        raise Exception(
                            "More than 1 checkpoint but ckpt_epoch is -1")
                    for name in names:
                        if f'epoch={self.ckpt_epoch}-' in name:
                            self.ckpt_path = os.path.join(self.ckpt_path, name)
                            break
                else:
                    raise Exception(f'No checkpoints in {self.ckpt_path}')

                logger.info(f'real ckpt_path:{self.ckpt_path}')
        if self.speaker_token_max_id is None:
            self.speaker_token_max_id = 2
        if self.speaker_token_max_id < 2:
            self.speaker_token_max_id = 2
        # Auto-enable R10 when vocab is large (avoids scoring over 30k+ candidates)
        if self.max_image_id and self.max_image_id > 1000:
            self.test_with_cand = True
        if getattr(self, 'candidate_eval_only', False):
            self.test_with_cand = True
        if self.mode in ('train', 'pretrain'):
            p10 = (getattr(self, 'per_epoch_eval_test_r10_path', None) or '').strip()
            p20 = (getattr(self, 'per_epoch_eval_test_r20_path', None) or '').strip()
            if p10 and p20:
                self.val_data_path = ''


def main(args):
    """
    程序总入口：根据 `args.mode` 分发到训练、评测或调试流程。

    输入：
    - args (Arguments): 全量运行配置。

    输出：
    - 无显式返回值。

    分支行为：
    - `train/pretrain`: 构建 Trainer 并执行 `fit`；
    - `test/gen`: 加载 ckpt 后执行 `trainer.test`；
    - `predict_img_label`: 跑辅助任务评测；
    - `predict_emotion/test_gradient`: 调用手工调试函数。

    注意：
    - 当 `local_files_only=True` 时会开启 HF 离线环境变量，禁止在线下载模型。
    """
    if args.local_files_only:
        # 离线模式：严格禁止 HF 在线下载，确保只使用本地 ckpt。
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    pl.seed_everything(args.seed)
    if args.mode == 'train' or args.mode == 'pretrain':
        model = PLModel(args)
        if args.ckpt_path:
            logger.info(f'load from {args.ckpt_path}')
            ckpt = torch.load(args.ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['state_dict'])
        if args.model_choice == 'use_clip_repo':
            pld = PLDataLoader(args)
        else:
            pld = PLDataLoader(args, model.model.bert_tokenizer)
        checkpoint_callback = ModelCheckpoint(save_top_k=-1,
                                              verbose=True)
        # earlystop_callback = EarlyStopping(monitor='val_loss',
        #                                    verbose=True,
        #                                    mode='min')
        trainer = pl.Trainer(
            gpus=args.gpus,
            callbacks=[checkpoint_callback],
            accelerator='ddp',
            # num_sanity_val_steps=0,
            # limit_train_batches=16,
            # limit_val_batches=0.01,
            # gradient_clip_val=1.0,
            max_epochs=args.epochs,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            default_root_dir=args.pl_root_dir)
        attach_version_log_from_trainer(args, trainer)
        trainer.fit(model, pld)
    elif args.mode == 'test' or args.mode == 'gen':
        model = PLModel(args)
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        # print(ckpt.keys())
        model.load_state_dict(ckpt['state_dict'])
        pld = PLDataLoader(args, model.model.bert_tokenizer)
        trainer = pl.Trainer(gpus=args.gpus,
                             max_epochs=args.epochs,
                             default_root_dir=args.pl_root_dir)
        attach_test_log_to_ckpt_version(args)
        trainer.test(model, datamodule=pld)

    elif args.mode == 'predict_img_label':
        model = PLModel(args)
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        # print(ckpt.keys())
        model.load_state_dict(ckpt['state_dict'], strict=False)
        pld = PLDataLoader(args, model.model.bert_tokenizer)

        trainer = pl.Trainer(gpus=args.gpus,
                             max_epochs=args.epochs,
                             default_root_dir=args.pl_root_dir)
        attach_test_log_to_ckpt_version(args)
        trainer.test(model, datamodule=pld)

    elif args.mode == 'predict_emotion':
        attach_manual_log_file(args)
        test_emotion_prediction(args)

    elif args.mode == 'test_gradient':
        attach_manual_log_file(args)
        test_sentence_gradient(args)


if __name__ == '__main__':
    try_create_dir('./logs')
    parser = HfArgumentParser(Arguments)
    args, = parser.parse_args_into_dataclasses()
    main(args)
