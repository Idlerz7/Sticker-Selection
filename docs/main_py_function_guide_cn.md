# `main.py` 逐函数输入输出说明（中文详解）

> 目标：帮助“第一次看本项目代码”的同学快速理解每个函数。
> 结构：**函数作用** / **输入** / **输出** / **核心逻辑** / **关键注意点**。

---

## 一、顶层辅助函数

### `attach_run_log_file(log_file, args=None)`
- **作用**：给当前运行绑定一个独立日志文件句柄。
- **输入**：
  - `log_file: str` 日志文件路径。
  - `args: Optional[Arguments]` 运行参数对象（可空）。
- **输出**：无显式返回；副作用是更新全局 logger 的 file handler。
- **核心逻辑**：
  - 创建日志目录；
  - 关闭旧的 `_run_file_handler`；
  - 新建 `FileHandler` 并沿用已有 formatter；
  - 写入“run log file / run args attached”提示。
- **关键注意点**：避免多次运行时日志串写到同一个 handler。

### `attach_version_log_from_trainer(args, trainer)`
- **作用**：把日志文件落到 Lightning 对应 `version_x` 目录。
- **输入**：
  - `args`: 参数对象。
  - `trainer`: `pl.Trainer` 实例。
- **输出**：无显式返回。
- **核心逻辑**：优先使用 `trainer.log_dir`；否则 fallback 到 `args.pl_root_dir/lightning_logs`。

### `attach_manual_log_file(args)`
- **作用**：为非 Trainer 模式（如手工调试）创建带时间戳日志。
- **输入**：`args` 参数对象。
- **输出**：无显式返回。
- **核心逻辑**：文件名模式 `manual_logs/<mode>-<timestamp>.log`。

### `_extract_speaker_from_text(text)`
- **作用**：从文本前缀提取说话人标记。
- **输入**：`text: str`
- **输出**：`str`，如 `[speaker1]` / `[speaker2]` / `unknown_speaker`。

### `_resolve_user_key(sample, dialog, fallback_index)`
- **作用**：解析样本所属用户键（个性化分支的主键）。
- **输入**：
  - `sample: dict` 样本；
  - `dialog: list` 对话片段；
  - `fallback_index: int` 回退索引。
- **输出**：`str` user_key。
- **核心逻辑**（优先级）：
  1. 直接用 `sample['user_id']`；
  2. 用 `dialogue_id + speaker` 拼接；
  3. 老数据用 pseudo id（哈希首句文本 + speaker）。
- **关键注意点**：pseudo id 仅兼容旧数据，不保证完全等价真实用户。

### `main(args)`
- **作用**：程序主入口，按 `mode` 分发运行流程。
- **输入**：`args: Arguments`
- **输出**：无显式返回。
- **核心逻辑**：
  - `train/pretrain` -> `trainer.fit`
  - `test/gen/predict_img_label` -> `trainer.test`
  - `predict_emotion/test_gradient` -> 手工函数

---

## 二、模型包装类与基础模块

### `BertModel.__init__(config)`
- **作用**：把 BERT 分类头固定为二分类。
- **输入**：`config`
- **输出**：构造函数无返回。
- **关键注意点**：`config.num_labels = 2`，与主任务正负二分类一致。

### `VisualHistoryAttention.__init__(input_dim, hidden_dim)`
- **作用**：初始化视觉历史注意力池化参数 `W_v1/W_v2`。
- **输入**：视觉维度、隐藏维度。
- **输出**：构造函数无返回。

### `VisualHistoryAttention.forward(history_feats, history_mask=None)`
- **作用**：对 `[B,K,D]` 的历史视觉特征做注意力聚合。
- **输入**：
  - `history_feats: Tensor[B,K,D]`
  - `history_mask: Optional[Tensor[B,K]]`
- **输出**：`Tensor[B,D]` 用户视觉偏好向量。
- **关键注意点**：无历史时返回零向量；mask 分母做 `clamp_min` 防 NaN。

### `UserVisualFusion.__init__(num_users, feature_dim)`
- **作用**：初始化用户 embedding 与门控融合层。
- **输入**：用户数、特征维度。
- **输出**：构造函数无返回。

### `UserVisualFusion.forward(visual_pref, user_indices)`
- **作用**：融合 `visual_pref` 与 `user_id embedding`。
- **输入**：
  - `visual_pref: Tensor[B,D]`
  - `user_indices: Tensor[B]`
- **输出**：`Tensor[B,D]` 融合用户向量。

### `HFClipSentenceEncoder.__init__(model_path, local_files_only=True)`
- **作用**：本地 HF-CLIP 回退编码器初始化。
- **输入**：模型路径、是否严格本地。
- **输出**：构造函数无返回。
- **关键注意点**：兼容 `0_CLIPModel` 子目录结构。

### `HFClipSentenceEncoder.tokenize(items)`
- **作用**：把图像/文本混合输入转换成 CLIP 特征字典。
- **输入**：`items: list[str|PIL.Image|Tensor]`
- **输出**：`dict`（包含 `pixel_values` 或 `input_ids`/`attention_mask`，以及 `image_text_info`）。

### `HFClipSentenceEncoder.forward(features)`
- **作用**：分别算图像与文本 embedding，并按原输入顺序拼回。
- **输入**：`features: dict`
- **输出**：`{'sentence_embedding': Tensor[N,D]}`。

---

## 三、核心主模型 `Model`（最重要）

### `Model.__init__(args)`
- **作用**：初始化整个多模态主模型与个性化组件。
- **输入**：`args: Arguments`
- **输出**：构造函数无返回。
- **核心逻辑**：
  - 加载 BERT / tokenizer；
  - 加载视觉编码器（ST / HF-CLIP / img_id embedding）；
  - 初始化主任务头与辅助任务头；
  - 读取 OCR、id2name；
  - 初始化个性化缓存与模块。

### `Model._build_user_visual_history_cache()`
- **作用**：从训练集构建每个用户的历史贴纸 id 序列。
- **输入**：无（内部用 `self.args.train_data_path`）。
- **输出**：无显式返回；更新：
  - `self.user_to_index`
  - `self.user_history_imgids`
- **关键注意点**：这里只构建静态历史，不做 LOO 去泄漏。

### `Model._save_user_history_cache()`
- **作用**：将用户映射与历史缓存写盘。
- **输入**：无（使用 `self.args.user_history_cache_path`）。
- **输出**：无显式返回。

### `Model._load_user_history_cache()`
- **作用**：从缓存文件恢复用户映射与历史。
- **输入**：无。
- **输出**：`bool`，加载成功返回 True。

### `Model._init_visual_personalization()`
- **作用**：初始化 visual-only 个性化流程。
- **输入**：无。
- **输出**：无显式返回。
- **核心逻辑**：
  - 先尝试加载缓存；
  - 训练模式缺缓存则重建并保存；
  - 测试/生成缺缓存直接报错；
  - 创建 attention/fusion/token projection 层。

### `Model._get_user_indices(user_ids, device)`
- **作用**：把字符串 `user_ids` 批量转为 embedding 行号。
- **输入**：`user_ids: list[str]`, `device`
- **输出**：`Tensor[B]`（long）。

### `Model._get_history_img_emb_cache(device)`
- **作用**：懒加载全部贴纸视觉 embedding（一次构建，多次复用）。
- **输入**：`device`
- **输出**：`Tensor[max_image_id, D]`

### `Model._get_user_visual_preference(user_indices, current_img_ids, device)`
- **作用**：计算每个样本的用户视觉偏好向量。
- **输入**：
  - `user_indices: Tensor[B]`
  - `current_img_ids: Optional[list[int]]`（用于 LOO）
  - `device`
- **输出**：`Tensor[B,D]`
- **核心逻辑**：
  - 取用户历史；
  - 如提供 `current_img_ids` 则移除同 id（LOO）；
  - 截断到 `visual_history_max_len`；
  - 查 embedding 后 attention/mean 池化。

### `Model._build_user_token(user_ids, current_img_ids, device)`
- **作用**：构建 `[USER]` token embedding。
- **输入**：用户 id 列表、当前样本 img_id（可空）、device。
- **输出**：`Tensor[B,1,H]`

### `Model._get_loo_current_img_ids(img_ids, test)`
- **作用**：统一决定当前阶段是否应用 LOO 过滤。
- **输入**：`img_ids`, `test: bool`
- **输出**：`None` 或 `img_ids` 原值。

### `Model._compose_personalized_inputs(text_emb, attention_mask, sticker_token, user_token, aux_emb=None)`
- **作用**：按个性化模板拼接 `inputs_embeds`。
- **输入**：
  - `text_emb: Tensor[B,L,H]`
  - `attention_mask: Tensor[B,L]`
  - `sticker_token: Tensor[B,1,H]`
  - `user_token: Tensor[B,1,H]`
  - `aux_emb: Optional[Tensor[B,A,H]]`
- **输出**：
  - `input_emb: Tensor[B,L',H]`
  - `full_attention_mask: Tensor[B,L']`
  - `token_type_ids: Tensor[B,L']`

### `Model.get_image_obj(id)`
- **作用**：按贴纸 id 读取并预处理图像对象（带缓存）。
- **输入**：`id: int`
- **输出**：预处理后的图像张量或对象（取决于编码器预处理）。

### `Model.prepare_for_test()`
- **作用**：测试前预编码全部候选贴纸 embedding 到 `self.all_img_embs`。
- **输入**：无。
- **输出**：无显式返回（更新内部缓存）。

### `Model.get_emb_by_imgids(img_ids)`
- **作用**：批量获取指定贴纸 id 的视觉 embedding。
- **输入**：`img_ids: list[int]`
- **输出**：`Tensor[B,D]`

### `Model.get_input_output_imglabel_by_imgid(img_id)`
- **作用**：为“图像标签预测辅助任务”构造 mask 输入与监督。
- **输入**：`img_id: int`
- **输出**：`(mask_predict_inputs, mask_predict_outputs, cls_inputs)`

### `Model.check_has_ocr(img_id)`
- **作用**：判断该贴纸是否有 OCR 文本。
- **输入**：`img_id: int`
- **输出**：`bool`

### `Model.update_predict_ocr(img_id, img_label, logits=None)`
- **作用**：更新 OCR 预测统计并记录调试信息。
- **输入**：贴纸 id、预测标签、可选 logits。
- **输出**：无显式返回。

### `Model.predict_img_label(input_ids, attention_mask, batch_idx=1, img_ids=None)`
- **作用**：执行图像标签预测模式。
- **输入**：文本输入、mask、batch 索引、目标图像 id。
- **输出**：按模式返回预测结果（内部依赖 logits 与字典更新）。

### `Model.test_img_emotion(input_ids, attention_mask, img_ids)`
- **作用**：执行情绪预测测试模式。
- **输入**：文本输入、mask、图像 id。
- **输出**：情绪类别预测（argmax）。

### `Model.test_gradient(input_ids, attention_mask, img_ids)`
- **作用**：执行梯度可解释性测试。
- **输入**：文本输入、mask、图像 id。
- **输出**：`(embeddings_grad, success_flag)`

### `Model.forward(...)`
- **作用**：主前向函数（训练/测试/调试模式统一入口）。
- **输入（关键）**：
  - `input_ids`, `attention_mask`
  - `img_ids`, `neg_img_ids`
  - `user_ids`（个性化）
  - `test`（是否测试路径）
  - `cands`（R10 候选）
  - 各辅助任务张量
- **输出**：
  - 训练：`(final_loss,)`
  - 测试：`(logits, labels, cands)`
  - 特殊模式：对应预测/梯度输出
- **核心逻辑**：
  - 训练：正负样本双分支 CE 平均；
  - 测试：对候选（R10）或全库（Rall）排序打分；
  - 个性化开关只改变输入构造，不改主任务定义。

### `Model.prepare_imgs(args)`
- **作用**：准备 id->图像路径映射与视觉预处理器。
- **输入**：`args`
- **输出**：无显式返回。

### `Model.get_input_embeddings_with_segment(input_ids, segment_ids)`
- **作用**：按 segment 将文本 embedding 与图像 embedding 融合。
- **输入**：`input_ids`, `segment_ids`
- **输出**：融合后的 embedding 张量。

---

## 四、数据集与 DataModule

### `PLDataset.__init__(path, mode, args, tokenizer)`
- **作用**：加载 JSON 样本并保存上下文配置。
- **输入**：数据路径、模式、参数、tokenizer。
- **输出**：构造函数无返回。

### `PLDataset.__len__()`
- **作用**：返回样本数量。
- **输入**：无。
- **输出**：`int`

### `PLDataset.random_mask(text_ids, return_ts)`
- **作用**：执行 MLM 风格随机 mask。
- **输入**：token id 序列、是否返回 tensor。
- **输出**：`(input_ids, output_ids)`（list 或 tensor）。

### `PLDataset.__getitem__(index)`
- **作用**：构建单条训练/测试样本字典。
- **输入**：样本索引。
- **输出**：`dict`，包含：
  - `sent`, `img_id`, `neg_img_id`
  - `user_id`, `cand`
  - 可选辅助任务字段

### `PLDataset.__getitem__.pad_none(x, l=5)`
- **作用**：局部辅助函数，补齐列表长度。
- **输入**：列表、目标长度。
- **输出**：补齐后的列表。

### `PLDataLoader.__init__(args, tokenizer)`
- **作用**：保存 DataModule 的配置与路径。

### `PLDataLoader.setup(stage=None)`
- **作用**：按 stage 构建 `train_dataset/val_dataset/test_dataset`。
- **输入**：`stage`（`fit` / `test` / `None`）。
- **输出**：无显式返回。

### `PLDataLoader.collate_fn(batch)`
- **作用**：把样本列表拼成模型可用 batch。
- **输入**：`list[dict]`
- **输出**：`dict`，核心字段：
  - `input_ids`, `attention_mask`
  - `img_ids`, `neg_img_ids`
  - `user_ids`, `cands`
  - 辅助任务张量

### `PLDataLoader.train_dataloader()`
- **作用**：返回训练 DataLoader（shuffle=True）。

### `PLDataLoader.val_dataloader()`
- **作用**：返回验证 DataLoader（如配置 val 数据）。

### `PLDataLoader.test_dataloader()`
- **作用**：返回测试/生成 DataLoader。

---

## 五、Lightning 训练封装 `PLModel`

### `PLModel.__init__(args)`
- **作用**：初始化 LightningModule 包装层。
- **输入**：参数对象。
- **输出**：构造函数无返回。

### `PLModel.run_model_from_batch(batch, batch_idx, test=False)`
- **作用**：解包 batch 并调用 `Model.forward`。
- **输入**：batch 字典、batch 索引、是否测试。
- **输出**：与 `Model.forward` 一致。

### `PLModel.compute_acc(sorted_idx, labels, k=5)`
- **作用**：计算 top-k 命中统计。
- **输入**：排序索引、标签、k。
- **输出**：`(correct, total)` 张量。

### `PLModel.compute_map(sorted_idx, labels)`
- **作用**：计算单标签场景 MRR 累加量。
- **输入**：排序索引、标签。
- **输出**：`(mrr_sum, total)`

### `PLModel.training_step(batch, batch_idx)`
- **作用**：训练单步。
- **输入**：batch、batch_idx。
- **输出**：`loss`。

### `PLModel.training_step_end(*args, **kwargs)`
- **作用**：训练步后处理（裁剪 temperature）。
- **输出**：父类返回值。

### `PLModel.on_validation_epoch_start()`
- **作用**：验证轮开始前预编码候选贴纸。

### `PLModel.log_res(batch, sorted_idx, batch_idx, k=5)`
- **作用**：样本级日志（当前默认关闭）。

### `PLModel.validation_step(batch, batch_idx)`
- **作用**：验证/测试单步评估。
- **输入**：batch、batch_idx。
- **输出**：`(metrics_list, return_preds, return_labels)`。

### `PLModel.on_validation_epoch_end()`
- **作用**：输出本轮评估摘要并重置指标统计器。

### `PLModel.on_test_epoch_start()`
- **作用**：测试开始前准备候选 embedding。

### `PLModel.test_step(batch, batch_idx)`
- **作用**：复用 `validation_step`。

### `PLModel.test_epoch_end(outputs)`
- **作用**：测试结束汇总并写结果文件。
- **输入**：`outputs`（每个 batch 的返回列表）。
- **输出**：无显式返回；写文件：
  - `result/<test_file>_<with_cand>.json`
  - `result/<test_file>_<with_cand>_pred.json`

### `PLModel.test_epoch_end.get_filename_withoutext(path)`
- **作用**：局部辅助函数，提取文件名主干。

### `PLModel.num_training_steps`（property）
- **作用**：估计训练总优化步数。
- **输出**：`int`
- **关键注意点**：优先使用 Lightning `estimated_stepping_batches`。

### `PLModel.configure_optimizers()`
- **作用**：配置 AdamW 与 cosine scheduler。
- **输出**：优化器与 scheduler 字典。
- **关键注意点**：scheduler 明确 `interval='step'`。

---

## 六、调试工具函数

### `test_emotion_prediction(args)`
- **作用**：离线遍历测试集做情绪预测评估。
- **输入**：参数对象（需 `mode=predict_emotion`）。
- **输出**：无显式返回；打印均值指标。

### `test_sentence_gradient(args)`
- **作用**：离线导出 token 级梯度分数。
- **输入**：参数对象（需 `mode=test_gradient`）。
- **输出**：无显式返回；写 `log_gradient_test_easy.json`。

---

## 七、参数类后处理

### `Arguments.__post_init__(self)`
- **作用**：参数后处理，尤其是 ckpt 路径解析。
- **输入**：实例自身。
- **输出**：无显式返回；可能更新 `self.ckpt_path`。
- **关键注意点**：
  - 当 `ckpt_path` 是目录且有多个文件时，必须提供 `ckpt_epoch`。
  - 会在日志打印最终解析的 `real ckpt_path`。

---

## 八、阅读顺序建议（给“完全看不懂”的你）

1. `main`（看模式分发）  
2. `PLDataLoader.collate_fn`（看 batch 长什么样）  
3. `PLModel.run_model_from_batch`（看怎么把 batch 喂给模型）  
4. `Model.forward`（训练/测试主路径）  
5. `_build_user_token` -> `_get_user_visual_preference` -> `_compose_personalized_inputs`（个性化核心链路）  
6. `validation_step` / `test_epoch_end`（指标与结果落盘）  
7. `configure_optimizers` / `num_training_steps`（学习率与训练稳定性）

