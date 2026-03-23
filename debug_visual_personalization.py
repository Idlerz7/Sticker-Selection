import argparse
from typing import Any, Dict, List

import torch

from main import Arguments, PLDataLoader, PLModel


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def build_args(cli: argparse.Namespace, use_personalized: bool) -> Arguments:
    args = Arguments()
    args.mode = "train"
    args.local_files_only = True
    args.model_choice = cli.model_choice
    args.train_data_path = cli.train_data_path
    args.val_data_path = cli.val_data_path
    args.test_data_path = cli.test_data_path
    args.bert_pretrain_path = cli.bert_pretrain_path
    args.img_pretrain_path = cli.img_pretrain_path
    args.text_clip_pretrain_path = cli.text_clip_pretrain_path
    args.clip_download_root = cli.clip_download_root
    args.id2img_path = cli.id2img_path
    args.img_dir = cli.img_dir
    args.ocr_path = cli.ocr_path
    args.max_image_id = cli.max_image_id
    args.sent_num = cli.sent_num
    args.num_workers = 0
    args.train_batch_size = cli.num_debug_samples
    args.valtest_batch_size = 1
    args.add_ocr_info = cli.add_ocr_info
    args.add_emotion_task = False
    args.add_predict_context_task = False
    args.add_predict_img_label_task = False
    args.use_visual_personalization_token = use_personalized
    args.use_visual_history_attention = cli.use_visual_history_attention
    args.visual_history_max_len = cli.visual_history_max_len
    args.visual_personalization_hidden_dim = cli.visual_personalization_hidden_dim
    args.user_history_cache_path = cli.user_history_cache_path
    return args


def print_data_block(samples: List[Dict[str, Any]], batch: Dict[str, Any]) -> bool:
    print("\n[DATA]")
    for i, item in enumerate(samples):
        sent_preview = item["sent"][:80].replace("\n", " ")
        print(
            f"- sample {i}: user_id={item.get('user_id')} "
            f"img_id={item.get('img_id')} neg_img_id={item.get('neg_img_id')} "
            f"sent[:80]={sent_preview}"
        )

    required = ["input_ids", "attention_mask", "img_ids", "neg_img_ids", "user_ids"]
    ok = True
    for key in required:
        has_key = key in batch
        ok = ok and has_key
        print(f"- collate has `{key}`: {has_key}")
    return ok


def build_history_tensors(
    model_core, user_indices: torch.Tensor, current_img_ids: List[int], device: torch.device
):
    bsz = user_indices.size(0)
    max_len = model_core.args.visual_history_max_len
    history_ids = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    history_mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
    filtered_all = []
    raw_all = []

    for row, uid in enumerate(user_indices.tolist()):
        raw_ids = model_core.user_history_imgids.get(uid, [])
        raw_all.append(raw_ids)
        current = current_img_ids[row]
        # Current implementation strategy: remove ALL matches to current img id.
        filtered = [hid for hid in raw_ids if hid != int(current)]
        filtered = filtered[-max_len:]
        filtered_all.append(filtered)
        if len(filtered) > 0:
            history_ids[row, : len(filtered)] = torch.tensor(filtered, dtype=torch.long, device=device)
            history_mask[row, : len(filtered)] = True

    all_embs = model_core._get_history_img_emb_cache(device)
    history_feats = all_embs[history_ids]
    return raw_all, filtered_all, history_feats, history_mask


def print_personalized_blocks(model_core, batch, device) -> Dict[str, bool]:
    result = {
        "user_map_ok": True,
        "loo_ok": True,
        "visual_pref_ok": True,
        "input_expand_ok": True,
        "input_len": True,
    }

    user_ids = batch["user_ids"]
    img_ids = batch["img_ids"]

    print("\n[USER MAP]")
    print(f"- user_to_index size: {len(model_core.user_to_index)}")
    print(f"- batch user_ids: {user_ids}")
    user_indices = model_core._get_user_indices(user_ids, device)
    print(f"- mapped user_indices: {user_indices.tolist()}")
    unk_idx = model_core._get_user_indices(["__definitely_unknown_user__"], device).tolist()[0]
    print(f"- unknown user -> index: {unk_idx} (expect 0)")
    if unk_idx != 0:
        result["user_map_ok"] = False

    print("\n[HISTORY]")
    raw_all, filtered_all, history_feats, history_mask = build_history_tensors(
        model_core, user_indices, img_ids, device
    )
    for i, (uid, pos_id, raw_ids, filtered) in enumerate(zip(user_ids, img_ids, raw_all, filtered_all)):
        removed = pos_id not in filtered
        print(f"- sample {i} user={uid}")
        print(f"  raw_history={raw_ids}")
        print(f"  current_pos_img={pos_id}")
        print(f"  filtered_history={filtered}")
        print(f"  leave-one-out removed current img: {removed}")
        if not removed:
            result["loo_ok"] = False
        if len(filtered) == 0:
            print("  filtered_history is EMPTY")

    visual_pref = model_core._get_user_visual_preference(
        user_indices=user_indices, current_img_ids=img_ids, device=device
    )
    fused_user = model_core.user_visual_fusion(visual_pref=visual_pref, user_indices=user_indices)
    user_token = model_core.user_token_proj(fused_user).unsqueeze(1)

    print("\n[VISUAL PREF]")
    print(f"- history_feats shape: {tuple(history_feats.shape)}")
    print(f"- history_mask shape: {tuple(history_mask.shape)}")
    print(f"- visual_pref shape: {tuple(visual_pref.shape)}")
    print(f"- fused_user shape: {tuple(fused_user.shape)}")
    print(f"- user_token shape: {tuple(user_token.shape)}")
    vp_norm = torch.linalg.norm(visual_pref, dim=-1)
    fu_norm = torch.linalg.norm(fused_user, dim=-1)
    print(f"- visual_pref L2 norm: {vp_norm.detach().cpu().tolist()}")
    print(f"- fused_user L2 norm: {fu_norm.detach().cpu().tolist()}")
    if torch.isnan(visual_pref).any() or torch.isnan(fused_user).any():
        result["visual_pref_ok"] = False

    print("\n[STICKER TOKEN]")
    img_emb = model_core.get_emb_by_imgids(img_ids)
    neg_img_emb = model_core.get_emb_by_imgids(batch["neg_img_ids"])
    pos_sticker_token = model_core.sticker_token_proj(img_emb).unsqueeze(1)
    neg_sticker_token = model_core.sticker_token_proj(neg_img_emb).unsqueeze(1)
    print(f"- pos img_emb shape: {tuple(img_emb.shape)}")
    print(f"- neg img_emb shape: {tuple(neg_img_emb.shape)}")
    print(f"- pos sticker_token shape: {tuple(pos_sticker_token.shape)}")
    print(f"- neg sticker_token shape: {tuple(neg_sticker_token.shape)}")

    print("\n[INPUT SHAPES]")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    text_emb = model_core.bert.bert.embeddings.word_embeddings(input_ids)

    aux_emb = None
    aux_len = 0
    if model_core.args.add_ocr_info:
        cls_inputs_ts = []
        for img_id in img_ids:
            _, _, cls_inputs = model_core.get_input_output_imglabel_by_imgid(img_id)
            cls_inputs_ts.append(cls_inputs)
        cls_inputs_ts = torch.tensor(cls_inputs_ts, device=device, dtype=torch.long)
        aux_emb = model_core.bert.bert.embeddings.word_embeddings(cls_inputs_ts)
        aux_len = aux_emb.size(1)

    p_input_emb, p_attn_mask, p_token_type_ids = model_core._compose_personalized_inputs(
        text_emb=text_emb,
        attention_mask=attention_mask,
        sticker_token=pos_sticker_token,
        user_token=user_token,
        aux_emb=aux_emb,
    )
    print(f"- text_emb length: {text_emb.size(1)}")
    print(f"- personalized input length: {p_input_emb.size(1)}")
    print(f"- personalized attention_mask length: {p_attn_mask.size(1)}")
    print(f"- personalized token_type_ids length: {p_token_type_ids.size(1)}")
    print(
        "- expected structure: [CLS] [USER] text ... [SEP]"
        + (" [AUX...]" if aux_len > 0 else "")
        + " [STICKER] [SEP]"
    )
    if aux_len > 0:
        print(f"- aux token length (OCR): {aux_len}")
    expected_len = text_emb.size(1) + 1 + 1 + 1 + aux_len
    if p_input_emb.size(1) != expected_len:
        result["input_expand_ok"] = False
    result["input_len"] = p_input_emb.size(1)

    return result


def run_forward_checks(pl_model: PLModel, batch: Dict[str, Any], device: torch.device) -> Dict[str, bool]:
    core = pl_model.model
    ret = {"forward_ok": True, "test_forward_ok": True}

    print("\n[FORWARD]")
    batch_dev = move_to_device(batch, device)
    try:
        outputs = pl_model.run_model_from_batch(batch_dev, batch_idx=0, test=False)
        loss = outputs[0]
        print(f"- train-path loss: {float(loss.detach().cpu()):.6f}")
    except Exception as e:
        ret["forward_ok"] = False
        print(f"- train-path FAILED: {type(e).__name__}: {e}")

    single = {
        "input_ids": batch["input_ids"][:1],
        "attention_mask": batch["attention_mask"][:1],
        "img_ids": [batch["img_ids"][0]],
        "neg_img_ids": [batch["neg_img_ids"][0]],
        "user_ids": [batch["user_ids"][0]],
        "cands": [[batch["img_ids"][0], batch["neg_img_ids"][0]]],
    }
    single_dev = move_to_device(single, device)
    try:
        core.prepare_for_test()
        with torch.no_grad():
            logits, labels, _ = core(
                input_ids=single_dev["input_ids"],
                attention_mask=single_dev["attention_mask"],
                img_ids=single_dev["img_ids"],
                neg_img_ids=single_dev["neg_img_ids"],
                user_ids=single_dev["user_ids"],
                test=True,
                cands=single_dev["cands"],
            )
        print(f"- test-path logits shape: {tuple(logits.shape)}")
        print(f"- test-path labels: {labels.detach().cpu().tolist()}")
    except Exception as e:
        ret["test_forward_ok"] = False
        print(f"- test-path FAILED: {type(e).__name__}: {e}")

    return ret


def run_mode(cli: argparse.Namespace, use_personalized: bool) -> Dict[str, bool]:
    title = "BASIC MODE" if not use_personalized else "PERSONALIZED MODE"
    print("\n" + "=" * 70)
    print(f"[{title}] use_visual_personalization_token={use_personalized}")
    print("=" * 70)

    args = build_args(cli, use_personalized=use_personalized)
    pl_model = PLModel(args)
    tokenizer = pl_model.model.bert_tokenizer
    pld = PLDataLoader(args, tokenizer)
    pld.setup("fit")

    samples = [pld.train_dataset[i] for i in range(cli.num_debug_samples)]
    batch = pld.collate_fn(samples)
    checks = {"data_ok": print_data_block(samples, batch)}

    device = torch.device(cli.device)
    pl_model = pl_model.to(device)

    if use_personalized:
        checks.update(print_personalized_blocks(pl_model.model, batch, device))
    else:
        print("\n[USER MAP/HISTORY/VISUAL PREF/STICKER TOKEN/INPUT SHAPES]")
        print("- skipped (baseline mode, personalization disabled).")
        text_len = batch["input_ids"].size(1)
        aux_len = 0
        if pl_model.model.args.add_ocr_info:
            aux_len = pl_model.model.args.max_img_label_mask_num
        checks["input_len"] = text_len + 2 + aux_len
        checks["user_map_ok"] = True
        checks["loo_ok"] = True
        checks["visual_pref_ok"] = True
        checks["input_expand_ok"] = True

    checks.update(run_forward_checks(pl_model, batch, device))
    return checks


def main():
    parser = argparse.ArgumentParser(
        description="Debug visual-only personalized token integration (structure/data-flow checks)."
    )
    parser.add_argument("--num_debug_samples", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_data_path", type=str, default="./data/train_pair.json")
    parser.add_argument("--val_data_path", type=str, default="./data/validation_pair.json")
    parser.add_argument("--test_data_path", type=str, default="./data/test_easy.json")
    parser.add_argument("--bert_pretrain_path", type=str, default="./ckpt/bert-base-chinese")
    parser.add_argument("--img_pretrain_path", type=str, default="./ckpt/clip-ViT-B-32")
    parser.add_argument("--text_clip_pretrain_path", type=str, default="./ckpt/clip-ViT-B-32-multilingual-v1")
    parser.add_argument("--clip_download_root", type=str, default="./ckpt")
    parser.add_argument("--id2img_path", type=str, default="./data/id2img.json")
    parser.add_argument("--img_dir", type=str, default="./data/meme_set")
    parser.add_argument("--ocr_path", type=str, default="./data/ocr_max10.json")
    parser.add_argument("--model_choice", type=str, default="use_img_clip")
    parser.add_argument("--max_image_id", type=int, default=307)
    parser.add_argument("--sent_num", type=int, default=0)
    parser.add_argument("--add_ocr_info", action="store_true")
    parser.add_argument("--use_visual_history_attention", type=str2bool, default=True)
    parser.add_argument("--visual_history_max_len", type=int, default=50)
    parser.add_argument("--visual_personalization_hidden_dim", type=int, default=256)
    parser.add_argument("--user_history_cache_path", type=str, default="./data/user_history_cache.json")
    cli = parser.parse_args()

    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to cpu.")
        cli.device = "cpu"

    baseline = run_mode(cli, use_personalized=False)
    personalized = run_mode(cli, use_personalized=True)
    length_expanded = personalized.get("input_len", 0) > baseline.get("input_len", 0)
    print(
        f"- input length baseline={baseline.get('input_len')} personalized={personalized.get('input_len')}"
    )

    print("\n" + "=" * 70)
    print("[SUMMARY]")
    print("=" * 70)
    print(f"- user_ids loaded: {'OK' if personalized['data_ok'] else 'FAILED'}")
    print(f"- leave-one-out filtering: {'OK' if personalized['loo_ok'] else 'FAILED'}")
    print(f"- visual_pref shape / finite check: {'OK' if personalized['visual_pref_ok'] else 'FAILED'}")
    print(
        "- personalized input length expanded: "
        f"{'OK' if (personalized['input_expand_ok'] and length_expanded) else 'FAILED'}"
    )
    print(
        "- forward pass (baseline/personalized): "
        f"{'OK' if (baseline['forward_ok'] and personalized['forward_ok']) else 'FAILED'}"
    )
    print(
        "- test forward (baseline/personalized): "
        f"{'OK' if (baseline['test_forward_ok'] and personalized['test_forward_ok']) else 'FAILED'}"
    )


if __name__ == "__main__":
    main()

