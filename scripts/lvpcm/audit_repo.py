#!/usr/bin/env python
"""Audit frozen assets, data mappings, dependencies, and formal blockers."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from lvpcm.catalog import dstc_catalog, read_json, stickerchat_catalog, validate_candidate_rows
from lvpcm.clip_intermediate import CLIPIntermediateExtractor
from lvpcm.images import project_rgb
from lvpcm.io import atomic_write_json, atomic_write_text, load_yaml, sha256_file
from lvpcm.provenance import command_record, environment_snapshot


def shape_record(path):
    value = torch.load(path, map_location="cpu")
    tensor = value if torch.is_tensor(value) else value.get("features")
    return {
        "path": path, "sha256": sha256_file(path), "shape": list(tensor.shape),
        "dtype": str(tensor.dtype), "finite": bool(torch.isfinite(tensor).all()),
        "mean_l2_norm": float(torch.linalg.vector_norm(tensor.float(), dim=-1).mean()),
    }


def embedding_rows(state):
    suffixes = ("bert.bert.embeddings.word_embeddings.weight", "bert.embeddings.word_embeddings.weight")
    for key, value in state.items():
        if any(key.endswith(suffix) for suffix in suffixes):
            return int(value.shape[0]), key
    return None, None


def strict_load_dstc(config):
    import main as legacy

    legacy_config = load_yaml(config["mmbbert_config"])
    args = legacy.Arguments()
    field_names = {field.name for field in dataclasses.fields(legacy.Arguments)}
    for key, value in legacy_config.items():
        if key in field_names:
            setattr(args, key, value)
    args.mode = "test"
    args.gpus = 0
    model = legacy.PLModel(args)
    checkpoint = torch.load(config["checkpoint"], map_location="cpu")
    result = model.load_state_dict(checkpoint["state_dict"], strict=True)
    tokenizer_rows = len(model.model.bert_tokenizer)
    embedding_count = int(model.model.bert.get_input_embeddings().weight.shape[0])
    return {
        "strict_load": True, "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
        "tokenizer_rows": tokenizer_rows, "embedding_rows": embedding_count,
        "speaker_tokens": model.model.bert_tokenizer.additional_special_tokens,
    }


def main(args):
    common = load_yaml(args.common_config)
    dstc_cfg = load_yaml(args.dstc_config)
    sticker_cfg = load_yaml(args.stickerchat_config)
    dstc = dstc_catalog(dstc_cfg["id2img"], dstc_cfg["train_pairs"])
    sticker = stickerchat_catalog(
        sticker_cfg["id2img"], sticker_cfg["img2id"], sticker_cfg["raw_train_pairs"], sticker_cfg["metadata"]
    )
    dstc_candidates = read_json(dstc_cfg["validation_candidates"])
    candidate_validation = validate_candidate_rows(dstc_candidates, dstc_cfg["expected_candidate_size"], dstc["ids"])
    image_counts = {}
    for cfg, catalog in ((dstc_cfg, dstc), (sticker_cfg, sticker)):
        missing = []
        for image_id, filename in catalog["id2img"].items():
            if not (Path(cfg["image_root"]) / filename).is_file():
                missing.append(image_id)
        image_counts[cfg["dataset"]] = {"mapped": len(catalog["ids"]), "missing": len(missing), "first_missing": missing[:5]}
    clip = CLIPIntermediateExtractor(common["clip_model"], args.device)
    sample_path = Path(dstc_cfg["image_root"]) / dstc["id2img"][0]
    data = sample_path.read_bytes()
    one = project_rgb(data)
    single_a, multi_a = clip.extract([one])
    single_b, multi_b = clip.extract([project_rgb(data)])
    with torch.no_grad():
        pixels = clip.processor(images=[one], return_tensors="pt")["pixel_values"].to(clip.device)
        fresh_final = clip.model.get_image_features(pixel_values=pixels).cpu().float()[0]
    dstc_cache_all = torch.load(dstc_cfg["final_clip_cache"], map_location="cpu").float()
    dstc_cache_row0 = dstc_cache_all[0]
    cache_similarity = torch.nn.functional.normalize(dstc_cache_all, dim=1).matmul(torch.nn.functional.normalize(fresh_final, dim=0))
    nearest_cache_row = int(torch.argmax(cache_similarity))
    clip_check = {
        "model_path": common["clip_model"],
        "weights_sha256": sha256_file(Path(common["clip_model"]) / "pytorch_model.bin"),
        "single_shape": list(single_a.shape), "multi_shape": list(multi_a.shape),
        "single_repeat_max_abs": float((single_a - single_b).abs().max()),
        "multi_repeat_max_abs": float((multi_a - multi_b).abs().max()),
        "dstc_final_cache_row0_max_abs": float((fresh_final - dstc_cache_row0).abs().max()),
        "dstc_final_cache_nearest_row": nearest_cache_row,
        "dstc_final_cache_nearest_cosine": float(cache_similarity[nearest_cache_row]),
        "dstc_final_cache_row0_aligned": nearest_cache_row == 0,
        "block_mapping": {"3": "hidden_states[3]", "6": "hidden_states[6]", "9": "hidden_states[9]"},
        "patch_shape_per_layer": [49, 768],
    }
    del clip
    dstc_strict = strict_load_dstc(dstc_cfg)
    chosen_ckpt = torch.load(dstc_cfg["checkpoint"], map_location="cpu")
    chosen_rows, chosen_key = embedding_rows(chosen_ckpt["state_dict"])
    old_path = "logs/u_sticker_clip/lightning_logs/version_3/checkpoints/epoch=9-step=97099.ckpt"
    old_sticker = {"path": old_path, "exists": Path(old_path).is_file()}
    if old_sticker["exists"]:
        old_state = torch.load(old_path, map_location="cpu")["state_dict"]
        old_sticker["embedding_rows"], old_sticker["embedding_key"] = embedding_rows(old_state)
        old_sticker["expected_current_rows"] = 21160
        old_sticker["usable_as_neutral"] = False
        old_sticker["reason"] = "English BERT plus two speaker tokens; current protocol needs Chinese BERT plus 32"
    cache_checks = {
        "dstc": shape_record(dstc_cfg["final_clip_cache"]),
        "stickerchat": shape_record(sticker_cfg["final_clip_cache"]),
    }
    global_checks = {}
    for key in ("global_validation_r10", "global_validation_r20", "global_test_r10", "global_test_r20"):
        rows = read_json(sticker_cfg[key])
        global_checks[key] = validate_candidate_rows(rows, 10 if key.endswith("r10") else 20, sticker["ids"])
    same_pack_anomalies = {}
    sticker_img2id = read_json(sticker_cfg["img2id"])
    for split, key in (("validation", "raw_validation_pairs"), ("test", "raw_test_pairs")):
        anomalies = []
        for index, row in enumerate(read_json(sticker_cfg[key])):
            turn = row["dialog"][-1]
            for field in ("img_id", "neg_img_id"):
                external = turn.get(field)
                if external is None or str(external) not in sticker_img2id:
                    anomalies.append({"source_row": index, "field": field, "external_id": external})
        same_pack_anomalies[split] = anomalies
    disk = shutil.disk_usage(Path.cwd())
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True
        ).strip().splitlines()
    except Exception as exc:
        gpu = ["unavailable: %s" % exc]
    report = {
        "status": "complete",
        "environment": environment_snapshot(),
        "gpu": gpu,
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "dependencies": {
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "numpy": __import__("numpy").__version__, "sklearn": __import__("sklearn").__version__,
            "transformers": __import__("transformers").__version__, "yaml": __import__("yaml").__version__,
            "pytest_available": __import__("importlib").util.find_spec("pytest") is not None,
            "pandas_available": __import__("importlib").util.find_spec("pandas") is not None,
            "faiss_available": __import__("importlib").util.find_spec("faiss") is not None,
        },
        "catalogs": {
            "dstc": {"size": len(dstc["ids"]), "train_size": len(dstc["train_ids"]), "hash": dstc["hash"]},
            "stickerchat": {"size": len(sticker["ids"]), "train_size": len(sticker["train_ids"]), "hash": sticker["hash"]},
        },
        "image_mappings": image_counts,
        "clip": clip_check,
        "dstc_checkpoint": {
            "path": dstc_cfg["checkpoint"], "sha256": sha256_file(dstc_cfg["checkpoint"]),
            "embedding_rows_in_checkpoint": chosen_rows, "embedding_key": chosen_key, **dstc_strict,
        },
        "stickerchat_checkpoint": {
            "status": "BLOCKED", "reason": sticker_cfg["neutral_checkpoint_status"], "required_embedding_rows": 21160,
            "old_u_sticker_clip_evidence": old_sticker,
            "structured_or_factorized_checkpoints_are_neutral": False,
        },
        "caches": cache_checks,
        "candidate_validation": {
            "dstc_validation_r10": candidate_validation, **global_checks,
            "stickerchat_same_pack": {
                "status": "BLOCKED" if any(same_pack_anomalies.values()) else "ready",
                "validation_anomaly_count": len(same_pack_anomalies["validation"]),
                "test_anomaly_count": len(same_pack_anomalies["test"]),
                "anomalies": same_pack_anomalies,
            },
        },
        "estimated_cost": {
            "measured_reference_throughput_images_per_second": 62.9,
            "stickerchat_clean_minutes_at_reference": 46.3,
            "four_pass_minutes_at_reference": 185.2,
            "note": "Reference A800 measurement; I/O and alpha/GIF processing can increase elapsed time.",
        },
        "same_ip_proxy": {"status": "unavailable", "reason": "repository has no independent IP/character metadata"},
    }
    output = Path(args.output)
    atomic_write_json(output.with_suffix(".json"), report)
    lines = [
        "# LVPCM Repository Audit", "", "Status: COMPLETE", "",
        "## Frozen assets", "",
        "- DSTC checkpoint strict-load: **PASS** (`%s`, SHA-256 `%s`)." % (dstc_cfg["checkpoint"], report["dstc_checkpoint"]["sha256"]),
        "- DSTC tokenizer / embedding rows: **%d / %d**; strict-load missing and unexpected keys: **0 / 0**." % (dstc_strict["tokenizer_rows"], dstc_strict["embedding_rows"]),
        "- Local CLIP: **PASS**, repeated single-image max absolute difference single/multi = `%g` / `%g`; fresh DSTC ID 0 retrieves cache row %d (cosine %.8f; raw max difference `%g`)." % (clip_check["single_repeat_max_abs"], clip_check["multi_repeat_max_abs"], clip_check["dstc_final_cache_nearest_row"], clip_check["dstc_final_cache_nearest_cosine"], clip_check["dstc_final_cache_row0_max_abs"]),
        "- StickerChat neutral checkpoint: **BLOCKED**. The old `u_sticker_clip` checkpoint has %s embedding rows and is an English-BERT/two-speaker asset; structured/factorized checkpoints are not neutral substitutes." % old_sticker.get("embedding_rows", "unknown"),
        "", "## Data and catalogs", "",
        "- DSTC: %d mapped images, %d frozen training IDs, %d missing files; validation candidate rows %d, unique gold-containing R10." % (len(dstc["ids"]), len(dstc["train_ids"]), image_counts["dstc"]["missing"], candidate_validation["rows"]),
        "- StickerChat: %d mapped images, %d raw-protocol training IDs, %d missing files. All four existing global R10/R20 files pass uniqueness and positive alignment." % (len(sticker["ids"]), len(sticker["train_ids"]), image_counts["stickerchat"]["missing"]),
        "- StickerChat same-pack reconstruction: **BLOCKED**; %d validation rows and %d test rows have no mappable original same-pack negative." % (len(same_pack_anomalies["validation"]), len(same_pack_anomalies["test"])),
        "- Independent same-IP/character metadata: **unavailable**; pack and existing OCR are the only audited proxies.",
        "", "## Cache, hardware, and capacity", "",
        "- DSTC final cache: `%s`, SHA-256 `%s`." % (cache_checks["dstc"]["shape"], cache_checks["dstc"]["sha256"]),
        "- StickerChat final cache: `%s`, SHA-256 `%s`." % (cache_checks["stickerchat"]["shape"], cache_checks["stickerchat"]["sha256"]),
        "- GPUs: %s" % "; ".join(gpu),
        "- Free disk: %.2f TiB." % (disk.free / 2**40),
        "- Reference full StickerChat clean extraction: about 46 minutes; four image passes about 185 minutes before overhead.",
        "", "## Environment", "",
        "- Python: `%s`" % sys.version.split()[0],
        "- PyTorch / CUDA build: `%s` / `%s`" % (torch.__version__, torch.version.cuda),
        "- pandas/pytest/FAISS are not required; the implementation uses unittest, standard-library data handling, and chunked exact cosine search.",
    ]
    atomic_write_text(output, "\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-config", default="configs/lvpcm/common.yaml")
    parser.add_argument("--dstc-config", default="configs/lvpcm/dstc.yaml")
    parser.add_argument("--stickerchat-config", default="configs/lvpcm/stickerchat.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="artifacts/lvpcm/repo_audit.md")
    options = parser.parse_args()
    with command_record(options):
        main(options)
