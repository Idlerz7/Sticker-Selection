"""Strict, read-only access to the frozen legacy MM-BERT backbone."""

from __future__ import annotations

import dataclasses

import torch

from .io import load_yaml


def load_dstc_plmodel(config: dict, device: str = "cpu"):
    import main as legacy

    values = load_yaml(config["mmbbert_config"])
    args = legacy.Arguments()
    valid = {field.name for field in dataclasses.fields(legacy.Arguments)}
    for key, value in values.items():
        if key in valid:
            setattr(args, key, value)
    args.mode = "test"
    args.gpus = 0
    args.test_with_cand = True
    args.candidate_eval_only = True
    args.num_workers = 0
    model = legacy.PLModel(args)
    checkpoint = torch.load(config["checkpoint"], map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(torch.device(device)).eval()
    cache = torch.load(config["final_clip_cache"], map_location="cpu").float().to(device)
    if cache.shape != (args.max_image_id, 512):
        raise ValueError("legacy image cache shape mismatch")
    model.model.all_img_embs = cache
    return model, args


def dialogue_text(row: dict, sent_num: int = 0) -> str:
    dialog = row["dialog"] if sent_num == 0 else row["dialog"][-sent_num:]
    return "".join(str(turn.get("text", "")) for turn in dialog)
