#!/usr/bin/env python3
"""Strict real-asset equivalence gate for legacy and compact bank semantics."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from factorized_style_bank import FactorizedStyleBank
from scripts.audit_stickerchat_384_h1 import extract_named_json_value
from structured_retrieval_factorized import StructuredFactorizedStickerModel
from style_shapes.group_bank import GroupBank
from style_shapes.io import atomic_write_json, atomic_write_text, command_record, sha256_file


def _mapping(bank):
    return {int(row.sticker_id): int(row.proto_id) for row in bank.records}


def _members(bank):
    return {int(row.proto_id): [int(value) for value in row.member_ids] for row in bank.prototypes}


def _sampling_signature(bank, seed=2021):
    output = []
    sticker_ids = sorted(bank.sticker_to_record)
    if len(sticker_ids) > 128:
        probe_positions = sorted(set(round(i * (len(sticker_ids) - 1) / 127.0) for i in range(128)))
        sticker_ids = [sticker_ids[position] for position in probe_positions]
    for sticker_id in sticker_ids:
        same_rng = random.Random(seed + sticker_id)
        cross_rng = random.Random(seed + sticker_id)
        output.append(
            [
                sticker_id,
                bank.sample_same_proto_negative(sticker_id, same_rng),
                bank.sample_cross_proto_negative(sticker_id, cross_rng),
            ]
        )
    return output


def _prototype_tensor_signature(bank, num_stickers, seed=2021):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    c_i = torch.randn(num_stickers, 21, generator=generator)
    model = object.__new__(StructuredFactorizedStickerModel)
    model.style_bank = bank
    model.args = SimpleNamespace(factorized_proto_consistency_weight=0.0)
    model._proto_reduce_cache_by_device = {}
    prototypes, density = StructuredFactorizedStickerModel._compute_proto_vectors(model, c_i)
    q = torch.randn(7, 21, generator=generator)
    logits = torch.matmul(q, prototypes.t())
    targets = torch.tensor(
        [bank.proto_id_of(index) for index in range(7)], dtype=torch.long
    )
    loss = F.cross_entropy(logits, targets)
    return prototypes, density, logits, loss, targets


def _compare_factorized_banks(left, right, label):
    left_members = _members(left)
    right_members = _members(right)
    left_by_partition = {tuple(values): proto_id for proto_id, values in left_members.items()}
    right_by_partition = {tuple(values): proto_id for proto_id, values in right_members.items()}
    same_partition = set(left_by_partition) == set(right_by_partition)
    alignment = [left_by_partition[tuple(right_members[proto_id])] for proto_id in range(len(right_members))] if same_partition else []
    checks = {
        "membership_partition": same_partition,
        "members_up_to_proto_id_permutation": same_partition,
        "sampling": _sampling_signature(left) == _sampling_signature(right),
    }
    left_signature = _prototype_tensor_signature(left, len(left.records))
    right_signature = _prototype_tensor_signature(right, len(right.records))
    index = torch.tensor(alignment, dtype=torch.long)
    checks["prototype_vectors_elementwise"] = same_partition and torch.equal(
        left_signature[0].index_select(0, index), right_signature[0]
    )
    checks["prototype_density_elementwise"] = same_partition and torch.equal(
        left_signature[1].index_select(0, index), right_signature[1]
    )
    checks["group_scores_elementwise"] = same_partition and torch.equal(
        left_signature[2].index_select(1, index), right_signature[2]
    )
    checks["group_loss_elementwise"] = same_partition and torch.equal(
        F.cross_entropy(left_signature[2].index_select(1, index), right_signature[4]),
        right_signature[3],
    )
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("%s equivalence failed: %s" % (label, failures))
    return checks


def check_dstc():
    with open("factorized_style_bank.json", "r", encoding="utf-8") as handle:
        legacy_value = json.load(handle)
    legacy = FactorizedStyleBank(legacy_value)
    compact_path = "artifacts/style_shapes/groups/dstc/llm_original/group_bank.json"
    compact = FactorizedStyleBank.from_json(compact_path)
    checks = _compare_factorized_banks(legacy, compact, "DSTC legacy/compact")
    checks["record_neighbors_elementwise"] = all(
        legacy.sticker_to_record[index].neighbor_ids
        == compact.sticker_to_record[index].neighbor_ids
        for index in legacy.sticker_to_record
    )
    if not checks["record_neighbors_elementwise"]:
        raise RuntimeError("DSTC neighbor import is not exact")
    return {
        "status": "PASS",
        "checks": checks,
        "legacy_sha256": sha256_file("factorized_style_bank.json"),
        "compact_sha256": sha256_file(compact_path),
        "membership_hash": GroupBank.load(compact_path).membership_hash,
        "scope": "all bank fields used by minimal-core forward, losses, and negative sampling",
    }


def _stickerchat_projected_legacy():
    legacy_path = Path(
        "stickerchat/processed_style_kmeans_k384/factorized_style_bank.json"
    )
    prototypes = extract_named_json_value(legacy_path, "prototypes")
    metadata = json.load(
        open(
            "stickerchat/processed_style_kmeans_k384/sticker_metadata.json",
            "r",
            encoding="utf-8",
        )
    )["stickers"]
    neighbors_value = json.load(
        open(
            "stickerchat/processed_style_kmeans_k384/style_neighbors.json",
            "r",
            encoding="utf-8",
        )
    )
    neighbor_map = {
        int(row["id"]): [int(item["id"]) for item in row["neighbors"]]
        for row in neighbors_value["neighbors"]
    }
    group_to_proto = {str(proto["proto_key"]).split("::", 1)[1]: int(proto["proto_id"]) for proto in prototypes}
    records = []
    for row in sorted(metadata, key=lambda item: int(item["internal_img_id"])):
        sticker_id = int(row["internal_img_id"])
        proto_id = group_to_proto[str(row["img_set"])]
        records.append(
            {
                "sticker_id": sticker_id,
                "proto_id": proto_id,
                "proto_key": "img_set::%s" % row["img_set"],
                "proto_source": "img_set",
                "main_subject": "",
                "subject_category": "",
                "visual_style": "",
                "identity_summary": "",
                # Compact projection: legacy records repeat this list, but minimal-core
                # membership and sampling read the single prototypes[].member_ids copy.
                "member_ids": [],
                "neighbor_ids": neighbor_map[sticker_id],
            }
        )
    return FactorizedStyleBank(
        {
            "meta": extract_named_json_value(legacy_path, "meta"),
            "prototypes": prototypes,
            "records": records,
        }
    )


def check_stickerchat():
    legacy_path = (
        "stickerchat/processed_style_kmeans_k384/factorized_style_bank.json"
    )
    projected = _stickerchat_projected_legacy()
    compact_path = (
        "artifacts/style_shapes/groups/stickerchat/"
        "final_clip_pack_original/group_bank.json"
    )
    compact = FactorizedStyleBank.from_json(compact_path)
    checks = _compare_factorized_banks(projected, compact, "StickerChat legacy/compact")
    checks["record_neighbors_elementwise"] = all(
        projected.sticker_to_record[index].neighbor_ids
        == compact.sticker_to_record[index].neighbor_ids
        for index in projected.sticker_to_record
    )
    if not checks["record_neighbors_elementwise"]:
        raise RuntimeError("StickerChat neighbor import is not exact")
    return {
        "status": "PASS",
        "checks": checks,
        "legacy_sha256": sha256_file(legacy_path),
        "compact_sha256": sha256_file(compact_path),
        "membership_hash": GroupBank.load(compact_path).membership_hash,
        "legacy_records_section_loaded": False,
        "legacy_projection": (
            "prototypes and meta are streamed from the 3.12GB legacy bank; record order, "
            "membership, and neighbors are reconstructed from its exact sibling assets. "
            "The omitted per-record repeated member_ids are not read by minimal-core forward/loss."
        ),
        "scope": "all bank fields used by minimal-core forward, losses, and negative sampling",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dstc", "stickerchat", "all"), default="all")
    parser.add_argument("--output-dir", default="artifacts/style_shapes/equivalence")
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        result = {"status": "PASS"}
        if args.dataset in {"dstc", "all"}:
            result["dstc"] = check_dstc()
        if args.dataset in {"stickerchat", "all"}:
            result["stickerchat"] = check_stickerchat()
        output = Path(args.output_dir)
        atomic_write_json(output / "bank_equivalence.json", result)
        lines = [
            "# Style Shapes Bank Equivalence Gate",
            "",
            "Verdict: **PASS**",
            "",
        ]
        for dataset in ("dstc", "stickerchat"):
            if dataset in result:
                lines.extend(
                    [
                        "## %s" % dataset,
                        "",
                        "- Membership, prototype members, same/cross sampling: elementwise equal.",
                        "- Prototype vectors, group scores, and group CE loss: elementwise equal.",
                        "- Imported reference neighbors: elementwise equal.",
                        "- Membership hash: `%s`." % result[dataset]["membership_hash"],
                        "",
                    ]
                )
        atomic_write_text(output / "bank_equivalence.md", "\n".join(lines))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

