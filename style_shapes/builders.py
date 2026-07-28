"""Real-asset group-source builders and construction reports."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .clustering import (
    effective_group_count,
    legacy_pack_kmeans,
    random_matched_assignments,
    random_pack_matched_assignments,
    shared_initial_indices,
    spherical_kmeans,
)
from .group_bank import GroupBank
from .io import hash_value, sha256_file


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_descriptor(path: str) -> Tuple[List[int], torch.Tensor, dict]:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or "ids" not in value or "features" not in value:
        raise ValueError("descriptor bundle missing ids/features: %s" % path)
    ids = [int(item) for item in value["ids"].tolist()]
    features = value["features"].detach().cpu().float()
    if features.ndim != 2 or features.size(0) != len(ids) or not torch.isfinite(features).all():
        raise ValueError("invalid descriptor shape/values: %s" % path)
    if len(set(ids)) != len(ids):
        raise ValueError("descriptor IDs are not unique")
    features = F.normalize(features, dim=1)
    return ids, features, {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(features.shape),
        "catalog_hash": value.get("catalog_hash"),
        "input_config_hash": value.get("input_config_hash"),
    }


def _neighbor_asset(path: str) -> Dict[int, List[int]]:
    value = read_json(path)
    rows = value.get("neighbors", value)
    return {
        int(row["id"]): [
            int(item["id"] if isinstance(item, dict) else item) for item in row["neighbors"]
        ]
        for row in rows
    }


def within_group_neighbors(
    sticker_ids: Sequence[int],
    assignments: Sequence[int],
    final_clip: torch.Tensor,
    topk: int,
    query_chunk: int = 256,
) -> Dict[int, List[int]]:
    """Exact in-group cosine neighbors; stable order makes integer ID the tie-break."""
    if len(sticker_ids) != len(assignments) or final_clip.size(0) != len(sticker_ids):
        raise ValueError("neighbor inputs are not aligned")
    ids = [int(value) for value in sticker_ids]
    id_to_row = {value: index for index, value in enumerate(ids)}
    by_group: Dict[int, List[int]] = defaultdict(list)
    for sticker_id, group_id in zip(ids, assignments):
        by_group[int(group_id)].append(sticker_id)
    output: Dict[int, List[int]] = {}
    for group_id in sorted(by_group):
        members = sorted(by_group[group_id])
        rows = torch.tensor([id_to_row[value] for value in members], dtype=torch.long)
        x = F.normalize(final_clip.index_select(0, rows).double(), dim=1)
        for start in range(0, len(members), int(query_chunk)):
            end = min(start + int(query_chunk), len(members))
            similarities = torch.matmul(x[start:end], x.t())
            for local in range(end - start):
                similarities[local, start + local] = -float("inf")
            limit = min(int(topk), max(len(members) - 1, 0))
            try:
                order = torch.argsort(
                    similarities, dim=1, descending=True, stable=True
                )[:, :limit]
            except TypeError:
                import numpy as np

                order = torch.from_numpy(
                    np.argsort(-similarities.numpy(), axis=1, kind="mergesort")[:, :limit].copy()
                )
            for local, indices in enumerate(order.tolist()):
                output[members[start + local]] = [members[index] for index in indices]
    return output


def group_stats(bank: GroupBank) -> dict:
    sizes = [len(row) for row in bank.group_to_members]
    total = float(sum(sizes))
    neighbor_eligible = sum(len(row) for row in bank.group_to_members if len(row) > 1)
    return {
        "dataset": bank.dataset,
        "group_source": bank.group_source,
        "membership_hash": bank.membership_hash,
        "num_stickers": int(total),
        "num_groups": len(sizes),
        "min_group_size": min(sizes),
        "max_group_size": max(sizes),
        "mean_group_size": total / len(sizes),
        "max_group_fraction": max(sizes) / total,
        "singleton_groups": sum(value == 1 for value in sizes),
        "effective_group_count": effective_group_count(sizes),
        "same_group_negative_coverage": neighbor_eligible / total,
    }


def _provenance(
    config: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    final_clip: Mapping[str, Any],
    clustering: Mapping[str, Any],
    catalog_hash: str,
) -> dict:
    frozen = {
        "config": dict(config),
        "descriptor": dict(descriptor),
        "final_clip": dict(final_clip),
        "clustering": dict(clustering),
        "catalog_hash": catalog_hash,
    }
    return {**frozen, "input_config_hash": hash_value(frozen)}


def build_dstc(config: Mapping[str, Any]) -> Dict[str, GroupBank]:
    ids, vpd, vpd_meta = load_descriptor(config["vpd_bundle"])
    final_ids, final_clip, final_meta = load_descriptor(config["final_clip_bundle"])
    if ids != final_ids or ids != list(range(int(config["expected_stickers"]))):
        raise ValueError("DSTC descriptors do not align to the complete ID catalog")
    legacy = read_json(config["legacy_bank"])
    llm = GroupBank.from_legacy_dict(
        legacy,
        dataset="dstc",
        group_source="llm_original",
        provenance={
            "legacy_bank_path": config["legacy_bank"],
            "legacy_bank_sha256": sha256_file(config["legacy_bank"]),
            "catalog_hash": vpd_meta["catalog_hash"],
            "conversion": "lossless_membership_and_neighbors",
        },
    )
    k = int(config["num_groups"])
    starts = shared_initial_indices(len(ids), k, int(config["n_init"]), int(config["seed"]))
    final_assign, final_info = spherical_kmeans(
        final_clip,
        k,
        seed=int(config["seed"]),
        n_init=int(config["n_init"]),
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        initial_index_sets=starts,
    )
    vpd_assign, vpd_info = spherical_kmeans(
        vpd,
        k,
        seed=int(config["seed"]),
        n_init=int(config["n_init"]),
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        initial_index_sets=starts,
    )
    final_neighbors = within_group_neighbors(
        ids, final_assign.tolist(), final_clip, int(config["neighbor_topk"])
    )
    vpd_neighbors = within_group_neighbors(
        ids, vpd_assign.tolist(), final_clip, int(config["neighbor_topk"])
    )
    final_bank = GroupBank.create(
        "dstc",
        "final_clip",
        ids,
        final_assign.tolist(),
        final_neighbors,
        _provenance(config, final_meta, final_meta, final_info, str(vpd_meta["catalog_hash"])),
    )
    vpd_bank = GroupBank.create(
        "dstc",
        "vpd_multi",
        ids,
        vpd_assign.tolist(),
        vpd_neighbors,
        _provenance(config, vpd_meta, final_meta, vpd_info, str(vpd_meta["catalog_hash"])),
    )
    random_assign = random_matched_assignments(
        ids, [len(row) for row in vpd_bank.group_to_members], int(config["random_seed"])
    )
    random_neighbors = within_group_neighbors(
        ids, random_assign, final_clip, int(config["neighbor_topk"])
    )
    random_bank = GroupBank.create(
        "dstc",
        "random_matched",
        ids,
        random_assign,
        random_neighbors,
        _provenance(
            config,
            vpd_meta,
            final_meta,
            {
                "algorithm": "seeded_random_permutation_exact_vpd_size_multiset",
                "seed": int(config["random_seed"]),
                "target_membership_hash": vpd_bank.membership_hash,
            },
            str(vpd_meta["catalog_hash"]),
        ),
    )
    return {
        "llm_original": llm,
        "final_clip": final_bank,
        "vpd_multi": vpd_bank,
        "random_matched": random_bank,
    }


def _pack_members(
    metadata_path: str, expected_ids: Sequence[int]
) -> Tuple[List[str], Dict[str, List[int]]]:
    value = read_json(metadata_path)
    by_pack: Dict[str, List[int]] = defaultdict(list)
    found = []
    for row in value.get("stickers", value):
        sticker_id = int(row["internal_img_id"])
        found.append(sticker_id)
        by_pack[str(row["img_set"])].append(sticker_id)
    if sorted(found) != sorted(int(value) for value in expected_ids):
        raise ValueError("StickerChat metadata does not align to descriptor catalog")
    for pack in by_pack:
        by_pack[pack].sort()
    names = sorted(by_pack)
    return names, dict(by_pack)


def _pack_centroids(
    ids: Sequence[int],
    features: torch.Tensor,
    pack_names: Sequence[str],
    pack_to_ids: Mapping[str, Sequence[int]],
) -> torch.Tensor:
    id_to_row = {int(value): index for index, value in enumerate(ids)}
    rows = []
    for pack in pack_names:
        index = torch.tensor(
            [id_to_row[int(value)] for value in pack_to_ids[pack]], dtype=torch.long
        )
        rows.append(F.normalize(features.index_select(0, index).mean(dim=0), dim=0))
    return torch.stack(rows)


def _sticker_assignments(
    ids: Sequence[int],
    pack_names: Sequence[str],
    pack_to_ids: Mapping[str, Sequence[int]],
    pack_assignments: Sequence[int],
) -> List[int]:
    by_id = {}
    for pack, group_id in zip(pack_names, pack_assignments):
        for sticker_id in pack_to_ids[pack]:
            by_id[int(sticker_id)] = int(group_id)
    return [by_id[int(sticker_id)] for sticker_id in ids]


def _reference_k384_membership(path: str, ids: Sequence[int]) -> List[int]:
    value = read_json(path)
    by_id = {}
    for row in value.get("stickers", value):
        name = str(row["img_set"])
        if not name.startswith("kc_"):
            raise ValueError("unexpected reference group name: %s" % name)
        by_id[int(row["internal_img_id"])] = int(name.split("_", 1)[1])
    return [by_id[int(sticker_id)] for sticker_id in ids]


def build_stickerchat(config: Mapping[str, Any]) -> Tuple[Dict[str, GroupBank], dict]:
    ids, vpd, vpd_meta = load_descriptor(config["vpd_bundle"])
    final_ids, final_clip, final_meta = load_descriptor(config["final_clip_bundle"])
    if ids != final_ids or ids != list(range(int(config["expected_stickers"]))):
        raise ValueError("StickerChat descriptors do not align to the complete ID catalog")
    pack_names, pack_to_ids = _pack_members(config["metadata"], ids)
    if len(pack_names) != int(config["expected_packs"]):
        raise ValueError("StickerChat pack count mismatch")
    final_pack = _pack_centroids(ids, final_clip, pack_names, pack_to_ids)
    vpd_pack = _pack_centroids(ids, vpd, pack_names, pack_to_ids)
    k = int(config["num_groups"])
    ref_pack_assign, ref_info = legacy_pack_kmeans(
        final_pack, k, int(config["iterations"]), int(config["seed"])
    )
    expected = _reference_k384_membership(config["reference_metadata"], ids)
    rebuilt = _sticker_assignments(ids, pack_names, pack_to_ids, ref_pack_assign.tolist())
    reference_exact = rebuilt == expected
    if not reference_exact:
        mismatches = sum(left != right for left, right in zip(rebuilt, expected))
        raise RuntimeError("K384 rebuild equivalence failed for %d stickers" % mismatches)
    reference_neighbors = _neighbor_asset(config["reference_neighbors"])
    reference = GroupBank.create(
        "stickerchat",
        "final_clip_pack_original",
        ids,
        rebuilt,
        reference_neighbors,
        _provenance(config, final_meta, final_meta, ref_info, str(vpd_meta["catalog_hash"])),
    )
    vpd_pack_assign, vpd_info = legacy_pack_kmeans(
        vpd_pack,
        k,
        int(config["iterations"]),
        int(config["seed"]),
        initial_indices=ref_info["initial_indices"],
    )
    sticker_vpd_assign = _sticker_assignments(
        ids, pack_names, pack_to_ids, vpd_pack_assign.tolist()
    )
    if len(set(sticker_vpd_assign)) != k:
        raise RuntimeError("VPD pack K-means produced an empty final group")
    vpd_neighbors = within_group_neighbors(
        ids, sticker_vpd_assign, final_clip, int(config["neighbor_topk"])
    )
    vpd_bank = GroupBank.create(
        "stickerchat",
        "vpd_pack",
        ids,
        sticker_vpd_assign,
        vpd_neighbors,
        _provenance(config, vpd_meta, final_meta, vpd_info, str(vpd_meta["catalog_hash"])),
    )
    pack_sizes = [len(pack_to_ids[name]) for name in pack_names]
    pack_random, random_info = random_pack_matched_assignments(
        pack_sizes,
        [len(row) for row in vpd_bank.group_to_members],
        int(config["random_seed"]),
    )
    sticker_random = _sticker_assignments(ids, pack_names, pack_to_ids, pack_random)
    random_neighbors = within_group_neighbors(
        ids, sticker_random, final_clip, int(config["neighbor_topk"])
    )
    random_bank = GroupBank.create(
        "stickerchat",
        "random_pack_matched",
        ids,
        sticker_random,
        random_neighbors,
        _provenance(config, vpd_meta, final_meta, random_info, str(vpd_meta["catalog_hash"])),
    )
    return {
        "final_clip_pack_original": reference,
        "vpd_pack": vpd_bank,
        "random_pack_matched": random_bank,
    }, {"reference_rebuild_exact": reference_exact, "random_size_match": random_info}

