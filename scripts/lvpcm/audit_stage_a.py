#!/usr/bin/env python
"""Run descriptor, stability, leakage, A4, contact-sheet, and gate audits."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np
import torch

from lvpcm.catalog import dstc_catalog, read_json, stickerchat_catalog
from lvpcm.io import atomic_write_json, atomic_write_text, load_yaml
from lvpcm.metrics import jaccard_rows, paired_bootstrap_delta
from lvpcm.provenance import command_record
from lvpcm.stage_a import (
    a4_probe, direct_train_index_neighbors, edge_leakage, filter_duplicate_neighbors,
    full_neighbors, full_neighbors_and_duplicate_filtered, pack_recall, paired_pack_bootstrap,
)
from lvpcm.visualize import contact_sheet_pages


def load_dataset(common, cfg):
    base = Path(common["artifact_root"])
    desc = base / "descriptors" / cfg["dataset"]
    lpc = base / "lpc" / cfg["dataset"]
    descriptors = {name: torch.load(desc / (name + "_clean.pt"), map_location="cpu") for name in ("final_clip", "single", "multi")}
    lpcs = {perturb: torch.load(lpc / ("lpc_%s.pt" % perturb), map_location="cpu") for perturb in ("clean", "resize75", "jpeg75", "alpha_bbox")}
    perturb_multi = {perturb: torch.load(desc / ("multi_%s.pt" % perturb), map_location="cpu") for perturb in ("resize75", "jpeg75", "alpha_bbox")}
    manifest = read_json(desc / "extraction_manifest.json")
    canonical = [row["canonical_rgb_sha256"] for row in manifest["image_manifest"]]
    return descriptors, lpcs, perturb_multi, canonical


def ocr_map(path):
    return {int(key): (value.get("ocr", "") if isinstance(value, dict) else value) for key, value in read_json(path).items()}


def dataset_audit(common, cfg, catalog, args):
    descriptors, lpcs, perturb_multi, canonical = load_dataset(common, cfg)
    ids = descriptors["multi"]["ids"]
    duplicate_counts = collections.Counter(canonical)
    full = {}
    for name, bundle in descriptors.items():
        ordinary, duplicate_filtered = full_neighbors_and_duplicate_filtered(
            bundle["features"], ids, canonical, 10, args.device, args.query_batch_size, args.index_batch_size
        )
        full[name] = {
            "ordinary_neighbors": ordinary,
            "duplicate_filtered_neighbors": duplicate_filtered,
        }
    pack_results = None
    comparisons = None
    if cfg["dataset"] == "stickerchat":
        pack_results = {}
        for name in full:
            pack_results[name] = {
                "ordinary": pack_recall(full[name]["ordinary_neighbors"], ids, catalog["pack_by_id"]),
                "duplicate_filtered": pack_recall(full[name]["duplicate_filtered_neighbors"], ids, catalog["pack_by_id"]),
            }
        comparisons = {}
        for baseline in ("final_clip", "single"):
            left = pack_results["multi"]["duplicate_filtered"]
            right = pack_results[baseline]["duplicate_filtered"]
            comparisons["multi_minus_%s" % baseline] = {
                "r@1_delta": left["pack_macro"]["r@1"] - right["pack_macro"]["r@1"],
                "r@5_delta": left["pack_macro"]["r@5"] - right["pack_macro"]["r@5"],
                "r@10": paired_pack_bootstrap(left, right, 10, common["bootstrap_replicates"], common["seed"]),
            }
    clean_direct = direct_train_index_neighbors(
        descriptors["multi"]["features"], ids, descriptors["multi"]["train_ids"], common["lpc_k"], args.device,
        args.query_batch_size, args.index_batch_size,
    )
    vpd_stability = []
    lpc_stability = []
    per_perturbation = {}
    for perturbation, bundle in perturb_multi.items():
        direct = direct_train_index_neighbors(
            bundle["features"], ids, bundle["train_ids"], common["lpc_k"], args.device,
            args.query_batch_size, args.index_batch_size,
        )
        eligible = bundle["eligible"].bool() if perturbation == "alpha_bbox" else torch.ones(len(ids), dtype=torch.bool)
        vpd_values = jaccard_rows(clean_direct[eligible], direct[eligible])
        lpc_values = jaccard_rows(lpcs["clean"]["neighbors"][eligible], lpcs[perturbation]["neighbors"][eligible])
        if int(eligible.sum()) == 0:
            per_perturbation[perturbation] = {
                "eligible": 0, "vpd_jaccard_at_10": None, "lpc_jaccard_at_10": None, "delta": None,
                "status": "not_applicable_no_eligible_images",
            }
        else:
            per_perturbation[perturbation] = {
                "eligible": int(eligible.sum()), "vpd_jaccard_at_10": float(vpd_values.mean()),
                "lpc_jaccard_at_10": float(lpc_values.mean()), "delta": float(lpc_values.mean() - vpd_values.mean()),
            }
        vpd_full = np.full(len(ids), np.nan); lpc_full = np.full(len(ids), np.nan)
        vpd_full[eligible.numpy()] = vpd_values; lpc_full[eligible.numpy()] = lpc_values
        vpd_stability.append(vpd_full); lpc_stability.append(lpc_full)
    vpd_mean = np.nanmean(np.stack(vpd_stability), axis=0)
    lpc_mean = np.nanmean(np.stack(lpc_stability), axis=0)
    stability_bootstrap = paired_bootstrap_delta(lpc_mean, vpd_mean, common["bootstrap_replicates"], common["seed"])
    ocr = ocr_map(cfg["ocr"])
    vpd_leak = edge_leakage(clean_direct, ids, canonical, ocr, catalog.get("pack_by_id"))
    lpc_leak = edge_leakage(lpcs["clean"]["neighbors"], ids, canonical, ocr, catalog.get("pack_by_id"))
    clean_lpc = lpcs["clean"]
    locality = {
        "coverage": float((clean_lpc["degree"] > 0).float().mean()),
        "fallback_rate": float(clean_lpc["fallback"].float().mean()),
        "mean_degree": float(clean_lpc["degree"].float().mean()),
        "degree_histogram": {str(value): int((clean_lpc["degree"] == value).sum()) for value in range(common["lpc_k"] + 1)},
    }
    id2path = {int(image_id): Path(cfg["image_root"]) / filename for image_id, filename in catalog["id2img"].items()}
    if cfg["dataset"] == "stickerchat":
        rng = random.Random(common["seed"]); anchors = sorted(rng.sample(ids.tolist(), 50))
    else:
        anchors = ids.tolist()
    sheets = []
    for name in ("final_clip", "single", "multi"):
        neighbor_map = {int(image_id): full[name]["ordinary_neighbors"][row].tolist() for row, image_id in enumerate(ids.tolist())}
        sheets.extend(contact_sheet_pages(anchors, neighbor_map, id2path, Path(common["artifact_root"]) / "contact_sheets" / cfg["dataset"], name))
    return {
        "dataset": cfg["dataset"], "catalog_size": len(ids), "train_catalog_size": len(descriptors["multi"]["train_ids"]),
        "canonical_duplicate_groups": sum(count > 1 for count in duplicate_counts.values()),
        "canonical_duplicate_images": sum(count for count in duplicate_counts.values() if count > 1),
        "maximum_duplicate_group": max(duplicate_counts.values()),
        "pack_retrieval": pack_results, "vpd_core_comparisons": comparisons,
        "stability": {"per_perturbation": per_perturbation, "mean_delta_bootstrap": stability_bootstrap},
        "locality": locality, "leakage": {"multi_vpd": vpd_leak, "lpc": lpc_leak},
        "contact_sheets": sheets, "same_ip_proxy": "unavailable",
    }, descriptors, lpcs


def main(args):
    common = load_yaml(args.common_config)
    dstc_cfg = load_yaml(args.dstc_config); sticker_cfg = load_yaml(args.stickerchat_config)
    dstc_catalog_value = dstc_catalog(dstc_cfg["id2img"], dstc_cfg["train_pairs"])
    sticker_catalog_value = stickerchat_catalog(sticker_cfg["id2img"], sticker_cfg["img2id"], sticker_cfg["raw_train_pairs"], sticker_cfg["metadata"])
    dstc_result, dstc_desc, dstc_lpc = dataset_audit(common, dstc_cfg, dstc_catalog_value, args)
    sticker_result, _sticker_desc, _sticker_lpc = dataset_audit(common, sticker_cfg, sticker_catalog_value, args)
    a4 = {"dstc": {"status": "NOT_RUN"}, "stickerchat": {"status": "BLOCKED", "reason": sticker_cfg["neutral_checkpoint_status"]}}
    pair_path = Path(args.dstc_pair_cache)
    candidate_path = Path(args.dstc_candidate_manifest)
    query_records = []
    if pair_path.is_file() and candidate_path.is_file():
        pair = torch.load(pair_path, map_location="cpu")
        candidate_rows = read_json(candidate_path)["rows"]
        source_rows = read_json(dstc_cfg["validation_candidates"])
        query_records, probe = a4_probe(pair, candidate_rows, dstc_desc, dstc_lpc["clean"], source_rows, common["bootstrap_replicates"], common["seed"])
        a4["dstc"] = {"status": "COMPLETE", **probe}
        atomic_write_json(Path(common["artifact_root"]) / "stage_a" / "dstc_a4_queries.json", {"rows": query_records})
    sticker_cmp = sticker_result["vpd_core_comparisons"]
    vpd_pass = bool(sticker_cmp and all(
        value["r@10"]["estimate"] > 0 and value["r@10"]["ci95"][0] > 0
        and value["r@1_delta"] >= 0 and value["r@5_delta"] >= 0
        for value in sticker_cmp.values()
    ))
    locality_checks = []
    leakage_checks = []
    for result in (dstc_result, sticker_result):
        stability = result["stability"]["mean_delta_bootstrap"]
        locality_checks.append(
            stability["estimate"] >= 0.02 and stability["ci95"][0] > 0
            and result["locality"]["coverage"] >= 0.75 and result["locality"]["fallback_rate"] <= 0.25
        )
        before = result["leakage"]["multi_vpd"]; after = result["leakage"]["lpc"]
        check = after["exact_duplicate_share"] - before["exact_duplicate_share"] <= 0.05 and after["exact_ocr_share"] - before["exact_ocr_share"] <= 0.05
        if after["cross_pack_share"] is not None:
            check = check and after["cross_pack_share"] >= 0.25
        leakage_checks.append(check)
    local_pass = all(locality_checks); leakage_pass = all(leakage_checks)
    if not vpd_pass or not local_pass or not leakage_pass:
        verdict = "STOP"
    elif a4["dstc"].get("incremental_support") and a4["stickerchat"]["status"] == "BLOCKED":
        verdict = "CONDITIONAL GO"
    elif a4["stickerchat"]["status"] == "BLOCKED":
        verdict = "CONDITIONAL GO"
    else:
        verdict = "GO" if a4["dstc"].get("incremental_support") and a4["stickerchat"].get("incremental_support") else "STOP"
    report = {
        "status": "complete", "stage_a_verdict": verdict,
        "gate": {"vpd_core_pass": vpd_pass, "locality_pass": local_pass, "leakage_pass": leakage_pass},
        "datasets": {"dstc": dstc_result, "stickerchat": sticker_result}, "a4": a4,
        "stage_b_authorized": verdict in ("GO", "CONDITIONAL GO"),
        "known_blockers": [
            "StickerChat neutral Chinese MM-BERT checkpoint with 32 speaker tokens is unavailable",
            "StickerChat same-pack reconstruction has 16 validation and 5 test rows without an original negative",
        ],
    }
    output = Path(args.output)
    atomic_write_json(output.with_suffix(".json"), report)
    lines = [
        "# LVPCM Stage-A Report", "", "Verdict: **%s**" % verdict, "",
        "## Gate", "",
        "- Multi-layer VPD core comparison: **%s**" % ("PASS" if vpd_pass else "FAIL"),
        "- Reciprocal locality and perturbation stability: **%s**" % ("PASS" if local_pass else "FAIL"),
        "- Duplicate/OCR/cross-pack leakage: **%s**" % ("PASS" if leakage_pass else "FAIL"),
        "- DSTC A4: **%s**; StickerChat A4: **BLOCKED** (missing strict-loadable neutral checkpoint)." % a4["dstc"]["status"],
        "- StickerChat same-pack Stage-B protocol: **BLOCKED** (16 validation and 5 test rows lack an original same-pack negative).",
        "", "## Key measurements", "",
    ]
    for name, result in (("DSTC", dstc_result), ("StickerChat", sticker_result)):
        stab = result["stability"]["mean_delta_bootstrap"]
        lines.append("- %s LPC coverage/fallback: %.4f / %.4f; stability delta %.4f, 95%% CI [%.4f, %.4f]." % (
            name, result["locality"]["coverage"], result["locality"]["fallback_rate"], stab["estimate"], stab["ci95"][0], stab["ci95"][1]
        ))
    if sticker_cmp:
        for name, value in sticker_cmp.items():
            lines.append("- StickerChat %s duplicate-filtered pack-macro R@10 delta %.4f, 95%% CI [%.4f, %.4f]; R@1/R@5 deltas %.4f/%.4f." % (
                name, value["r@10"]["estimate"], value["r@10"]["ci95"][0], value["r@10"]["ci95"][1], value["r@1_delta"], value["r@5_delta"]
            ))
    lines.extend(("", "Full numeric results and all contact-sheet paths are in `%s`." % output.with_suffix(".json")))
    atomic_write_text(output, "\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-config", default="configs/lvpcm/common.yaml")
    parser.add_argument("--dstc-config", default="configs/lvpcm/dstc.yaml")
    parser.add_argument("--stickerchat-config", default="configs/lvpcm/stickerchat.yaml")
    parser.add_argument("--dstc-pair-cache", default="artifacts/lvpcm/pair_cache/dstc_validation_fixed_r10.pt")
    parser.add_argument("--dstc-candidate-manifest", default="artifacts/lvpcm/candidates/dstc_validation_fixed_r10.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--index-batch-size", type=int, default=8192)
    parser.add_argument("--output", default="artifacts/lvpcm/stage_a_report.md")
    options = parser.parse_args()
    with command_record(options):
        main(options)
