#!/usr/bin/env python
"""Build and validate all frozen LVPCM candidate protocols."""

from __future__ import annotations

import argparse
from pathlib import Path

from lvpcm.catalog import (
    dstc_catalog, dstc_training_candidates, normalized_candidate_manifest, read_json,
    stickerchat_catalog, stickerchat_same_pack_candidates, validate_candidate_rows,
)
from lvpcm.io import atomic_write_json, hash_mapping, load_yaml
from lvpcm.provenance import command_record


def write_manifest(path, rows, inputs):
    input_hash = hash_mapping(inputs)
    if Path(path).is_file():
        existing = read_json(path)
        if existing.get("input_config_hash") != input_hash:
            raise RuntimeError("refusing to overwrite incompatible candidate manifest: %s" % path)
        if existing.get("status") == "complete":
            return
    payload = {
        "status": "complete", "input_config_hash": input_hash, "schema_version": 1,
        "rows": rows, "row_count": len(rows), "candidate_sizes": sorted(set(len(row["candidate_ids"]) for row in rows)),
    }
    atomic_write_json(path, payload)


def main(args):
    dstc_cfg = load_yaml(args.dstc_config)
    sticker_cfg = load_yaml(args.stickerchat_config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dstc = dstc_catalog(dstc_cfg["id2img"], dstc_cfg["train_pairs"])
    dstc_train_rows = read_json(dstc_cfg["train_pairs"])
    dstc_train = dstc_training_candidates(dstc_train_rows, dstc["train_ids"], 10, args.seed)
    write_manifest(output / "dstc_train_uniform_r10.json", dstc_train, {"catalog": dstc["hash"], "seed": args.seed, "source": dstc_cfg["train_pairs"]})
    eval_source = read_json(dstc_cfg["validation_candidates"])
    validate_candidate_rows(eval_source, 10, dstc["ids"])
    dstc_eval = normalized_candidate_manifest(eval_source, "validation", "fixed_r10", dstc_cfg["validation_candidates"])
    write_manifest(output / "dstc_validation_fixed_r10.json", dstc_eval, {"catalog": dstc["hash"], "source": dstc_cfg["validation_candidates"]})

    sticker = stickerchat_catalog(sticker_cfg["id2img"], sticker_cfg["img2id"], sticker_cfg["raw_train_pairs"], sticker_cfg["metadata"])
    protocol_status = {}
    blocked_evidence = {}
    for split in ("validation", "test"):
        for size in (10, 20):
            key = "global_%s_r%d" % (split, size)
            source = read_json(sticker_cfg[key])
            validate_candidate_rows(source, size, sticker["ids"])
            normalized = normalized_candidate_manifest(source, split, "global_r%d" % size, sticker_cfg[key])
            write_manifest(output / ("stickerchat_%s_global_r%d.json" % (split, size)), normalized, {"catalog": sticker["hash"], "source": sticker_cfg[key]})
        raw_key = "raw_%s_pairs" % split
        raw = read_json(sticker_cfg[raw_key])
        img2id = {str(key): int(value) for key, value in read_json(sticker_cfg["img2id"]).items()}
        anomalies = []
        for index, source_row in enumerate(raw):
            turn = source_row["dialog"][-1]
            for field in ("img_id", "neg_img_id"):
                external = turn.get(field)
                if external is None or str(external) not in img2id:
                    anomalies.append({"source_row": index, "field": field, "external_id": external})
        if anomalies:
            blocked_evidence[split] = {
                "status": "BLOCKED", "reason": "raw positive/same-pack-negative mapping is incomplete",
                "anomaly_count": len(anomalies), "anomalies": anomalies,
                "source": sticker_cfg[raw_key],
            }
            atomic_write_json(output / ("stickerchat_%s_same_pack_BLOCKED.json" % split), blocked_evidence[split])
            protocol_status["stickerchat_%s_same_pack" % split] = "BLOCKED"
        else:
            for size in (10, 20):
                rebuilt = stickerchat_same_pack_candidates(raw, img2id, sticker["ids"], size, args.seed)
                for row in rebuilt:
                    row["source"] = sticker_cfg[raw_key]
                    row["split"] = split
                write_manifest(output / ("stickerchat_%s_same_pack_r%d.json" % (split, size)), rebuilt, {
                    "catalog": sticker["hash"], "source": sticker_cfg[raw_key], "seed": args.seed, "size": size,
                })
            protocol_status["stickerchat_%s_same_pack" % split] = "ready"
    status = {
        "status": "complete", "dstc": "ready", "stickerchat_global_protocols": "ready",
        **protocol_status, "same_pack_blocked_evidence": blocked_evidence,
        "stickerchat_pair_cache_and_training": "blocked_missing_neutral_checkpoint",
        "files": sorted(str(path) for path in output.glob("*.json") if path.name != "status.json"),
    }
    atomic_write_json(output / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dstc-config", default="configs/lvpcm/dstc.yaml")
    parser.add_argument("--stickerchat-config", default="configs/lvpcm/stickerchat.yaml")
    parser.add_argument("--output-dir", default="artifacts/lvpcm/candidates")
    parser.add_argument("--seed", type=int, default=2021)
    options = parser.parse_args()
    with command_record(options):
        main(options)
