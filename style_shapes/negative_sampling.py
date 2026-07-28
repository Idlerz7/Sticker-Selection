"""Optional StickerChat dual-local negative sampling for Style Shapes."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .group_bank import GroupBank
from .io import hash_value, sha256_file


DUAL_LOCAL_VPD_POLICY = "stickerchat_same_pack_plus_vpd_top32"
DUAL_LOCAL_SEMSP_POLICY = "stickerchat_same_pack_plus_semsp_top32"
# Backward-compatible name used by the original VPD-only implementation.
DUAL_LOCAL_POLICY = DUAL_LOCAL_VPD_POLICY
DUAL_LOCAL_POLICY_BY_GROUP_SOURCE = {
    "vpd_pack": DUAL_LOCAL_VPD_POLICY,
    "final_clip_pack_original": DUAL_LOCAL_SEMSP_POLICY,
}
DUAL_LOCAL_GROUP_SOURCE_BY_POLICY = {
    policy: source for source, policy in DUAL_LOCAL_POLICY_BY_GROUP_SOURCE.items()
}
DUAL_LOCAL_POLICIES = frozenset(DUAL_LOCAL_GROUP_SOURCE_BY_POLICY)
DUAL_LOCAL_SCHEMA = "style_shapes.dual_local_eligibility.v1"


def terminal_positive(row: Mapping[str, Any]) -> int:
    dialog = row.get("dialog")
    if not isinstance(dialog, list) or not dialog:
        raise ValueError("training row has no dialogue")
    return int(dialog[-1]["img_id"])


def load_id_to_pack(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    rows = value.get("stickers", value) if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("StickerChat metadata must contain a stickers list")
    output: Dict[int, str] = {}
    for row in rows:
        sticker_id = int(row["internal_img_id"])
        pack = str(row["img_set"])
        if sticker_id in output:
            raise ValueError("duplicate sticker metadata ID %d" % sticker_id)
        output[sticker_id] = pack
    if not output:
        raise ValueError("StickerChat metadata is empty")
    return output


def pack_members_from_mapping(id_to_pack: Mapping[int, str]) -> Dict[str, List[int]]:
    output: Dict[str, List[int]] = defaultdict(list)
    for sticker_id, pack in id_to_pack.items():
        output[str(pack)].append(int(sticker_id))
    for pack in output:
        output[pack].sort()
    return dict(output)


def _bank_neighbors(bank: GroupBank) -> Dict[int, List[int]]:
    return {
        int(key): [int(value) for value in values]
        for key, values in bank.value.get("same_group_neighbors", {}).items()
    }


def validate_group_top32_bank(
    bank: GroupBank,
    id_to_pack: Mapping[int, str],
    expected_policy: Optional[str] = None,
) -> dict:
    if bank.dataset != "stickerchat":
        raise ValueError("dual-local negatives require a StickerChat Group Bank")
    policy = DUAL_LOCAL_POLICY_BY_GROUP_SOURCE.get(bank.group_source)
    if policy is None:
        raise ValueError(
            "dual-local negatives do not support Group Bank source %s"
            % bank.group_source
        )
    if expected_policy is not None and str(expected_policy) != policy:
        raise ValueError(
            "dual-local policy %s requires Group Bank source %s"
            % (
                expected_policy,
                DUAL_LOCAL_GROUP_SOURCE_BY_POLICY.get(str(expected_policy), "unknown"),
            )
        )
    ids = bank.sticker_ids
    if set(ids) != set(int(value) for value in id_to_pack):
        raise ValueError("pack metadata and Group Bank sticker IDs do not align")
    id_to_group = dict(zip(ids, bank.sticker_to_group))
    neighbors = _bank_neighbors(bank)
    missing = [sticker_id for sticker_id in ids if sticker_id not in neighbors]
    if missing:
        raise ValueError("Group Bank misses neighbor rows for %d stickers" % len(missing))
    edge_count = 0
    for sticker_id in ids:
        values = neighbors[sticker_id]
        if len(values) > 32:
            raise ValueError("sticker %d has more than 32 group neighbors" % sticker_id)
        if len(set(values)) != len(values) or sticker_id in values:
            raise ValueError("invalid group neighbor list for sticker %d" % sticker_id)
        for neighbor in values:
            if neighbor not in id_to_group:
                raise ValueError("group neighbor %d is outside the catalog" % neighbor)
            if id_to_group[neighbor] != id_to_group[sticker_id]:
                raise ValueError("neighbor relation crosses groups")
        edge_count += len(values)
    return {
        "group_source": bank.group_source,
        "negative_policy": policy,
        "stickers": len(ids),
        "neighbor_edges": edge_count,
        "neighbor_content_hash": hash_value(
            [[sticker_id, neighbors[sticker_id]] for sticker_id in sorted(ids)]
        ),
    }


def validate_vpd_top32_bank(bank: GroupBank, id_to_pack: Mapping[int, str]) -> dict:
    """Backward-compatible VPD-specific validation entrypoint."""
    return validate_group_top32_bank(
        bank, id_to_pack, expected_policy=DUAL_LOCAL_VPD_POLICY
    )


class StickerChatDualLocalNegativeSampler:
    """Sample one raw-pack negative and one distinct group top-32 neighbor."""

    def __init__(
        self,
        bank: GroupBank,
        id_to_pack: Mapping[int, str],
        seed: int = 2021,
        policy: Optional[str] = None,
    ):
        audit = validate_group_top32_bank(bank, id_to_pack, expected_policy=policy)
        self.bank = bank
        self.policy = str(audit["negative_policy"])
        self.group_source = str(bank.group_source)
        self.neighbor_trace_field = (
            "vpd_top32"
            if self.policy == DUAL_LOCAL_VPD_POLICY
            else "semsp_top32"
        )
        self.id_to_pack = {int(key): str(value) for key, value in id_to_pack.items()}
        self.pack_to_members = pack_members_from_mapping(self.id_to_pack)
        self.neighbors = _bank_neighbors(bank)
        self.id_to_group = dict(zip(bank.sticker_ids, bank.sticker_to_group))
        self.rng = random.Random(int(seed))
        self.seed = int(seed)

    def valid_pack_choices(self, positive: int) -> List[int]:
        positive = int(positive)
        pack = self.id_to_pack[positive]
        pack_pool = [
            value for value in self.pack_to_members[pack] if value != positive
        ]
        group_pool = self.neighbors[positive]
        return [
            pack_negative
            for pack_negative in pack_pool
            if any(value != pack_negative for value in group_pool)
        ]

    def sample_one(self, positive: int) -> Tuple[int, int]:
        positive = int(positive)
        pack_choices = self.valid_pack_choices(positive)
        if not pack_choices:
            raise RuntimeError(
                "positive %d has no distinct same-pack plus group-top32 pair"
                % positive
            )
        same_pack = int(self.rng.choice(pack_choices))
        group_choices = [
            value
            for value in self.neighbors[positive]
            if value != positive and value != same_pack
        ]
        if not group_choices:
            raise RuntimeError("positive %d exhausted its group top-32 pool" % positive)
        group_top32 = int(self.rng.choice(group_choices))
        if len({positive, same_pack, group_top32}) != 3:
            raise RuntimeError("dual-local negative IDs are not unique")
        return same_pack, group_top32

    def resolve(
        self,
        *,
        pos_ids: Sequence[int],
    ) -> Tuple[List[int], List[int], Dict[str, Any]]:
        same_pack_ids: List[int] = []
        group_top32_ids: List[int] = []
        for positive in pos_ids:
            same_pack, group_top32 = self.sample_one(int(positive))
            same_pack_ids.append(same_pack)
            group_top32_ids.append(group_top32)
        debug = {
            "negative_policy": self.policy,
            "seed": self.seed,
            "group_source": self.group_source,
            "group_top32_ids_preview": group_top32_ids[:8],
            "%s_ids_preview" % self.neighbor_trace_field: group_top32_ids[:8],
            "same_pack_ids_preview": same_pack_ids[:8],
        }
        # The factorized core applies expr_rank_loss to the second ("same") slot.
        # Therefore group top-32 occupies cross, and raw-pack occupies same.
        return group_top32_ids, same_pack_ids, debug

    def install(self, factorized_model: Any) -> None:
        def resolver(
            q_expr,
            pos_ids,
            fallback_neg_ids,
            style_bank_a,
        ):
            del q_expr, fallback_neg_ids, style_bank_a
            return self.resolve(pos_ids=pos_ids)

        factorized_model._resolve_prototype_aware_negatives = resolver


def build_eligibility_manifest(
    train_rows: Sequence[Mapping[str, Any]],
    sampler: StickerChatDualLocalNegativeSampler,
    *,
    inputs: Optional[Mapping[str, Any]] = None,
) -> dict:
    eligible: List[int] = []
    excluded: List[dict] = []
    for source_row, row in enumerate(train_rows):
        positive = terminal_positive(row)
        pack = sampler.id_to_pack.get(positive)
        if pack is None:
            raise ValueError("training positive %d is absent from pack metadata" % positive)
        pack_pool = [
            value
            for value in sampler.pack_to_members[pack]
            if value != positive
        ]
        if not pack_pool:
            excluded.append(
                {
                    "source_row": int(source_row),
                    "positive": positive,
                    "reason": "singleton_original_pack",
                }
            )
            continue
        if not sampler.valid_pack_choices(positive):
            reason = (
                "no_distinct_vpd_top32_pair"
                if sampler.policy == DUAL_LOCAL_VPD_POLICY
                else "no_distinct_semsp_top32_pair"
            )
            excluded.append(
                {
                    "source_row": int(source_row),
                    "positive": positive,
                    "reason": reason,
                }
            )
            continue
        eligible.append(int(source_row))
    if sampler.policy == DUAL_LOCAL_VPD_POLICY:
        group_constraint_name = "vpd_negative"
        group_constraint = "uniform_vpd_group_final_clip_top32"
        group_order_name = "vpd_top32"
    else:
        group_constraint_name = "semsp_negative"
        group_constraint = "uniform_semsp_group_final_clip_top32"
        group_order_name = "semsp_top32"
    core = {
        "schema_version": DUAL_LOCAL_SCHEMA,
        "negative_policy": sampler.policy,
        "seed": sampler.seed,
        "constraints": {
            "same_pack_negative": "uniform_valid_original_pack_member",
            group_constraint_name: group_constraint,
            "three_ids_unique": True,
            "fallback": "none",
            "expr_rank_negative": "same_pack",
            "train_logit_order": ["positive", group_order_name, "same_pack"],
        },
        "total_rows": len(train_rows),
        "eligible_rows": eligible,
        "eligible_count": len(eligible),
        "excluded_rows": excluded,
        "excluded_count": len(excluded),
        "excluded_reason_counts": {
            reason: sum(row["reason"] == reason for row in excluded)
            for reason in sorted({row["reason"] for row in excluded})
        },
        "inputs": dict(inputs or {}),
    }
    return {**core, "manifest_hash": hash_value(core)}


def validate_eligibility_manifest(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != DUAL_LOCAL_SCHEMA:
        raise ValueError("unsupported dual-local eligibility schema")
    if value.get("negative_policy") not in DUAL_LOCAL_POLICIES:
        raise ValueError("dual-local eligibility policy mismatch")
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != hash_value(core):
        raise ValueError("dual-local eligibility manifest hash mismatch")
    eligible = [int(item) for item in value.get("eligible_rows", [])]
    excluded = list(value.get("excluded_rows", []))
    if eligible != sorted(eligible) or len(set(eligible)) != len(eligible):
        raise ValueError("eligible source rows must be sorted and unique")
    total = int(value.get("total_rows", -1))
    if int(value.get("eligible_count", -1)) != len(eligible):
        raise ValueError("eligible row count mismatch")
    if int(value.get("excluded_count", -1)) != len(excluded):
        raise ValueError("excluded row count mismatch")
    excluded_ids = [int(row["source_row"]) for row in excluded]
    if len(set(excluded_ids)) != len(excluded_ids):
        raise ValueError("duplicate excluded source rows")
    if sorted(eligible + excluded_ids) != list(range(total)):
        raise ValueError("eligible/excluded rows do not partition the training data")


def load_eligibility_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    validate_eligibility_manifest(value)
    return value


def load_dual_local_runtime(
    policy_config: Mapping[str, Any],
    *,
    train_data_path: str,
    group_bank_path: str,
    bank: GroupBank,
) -> Tuple[StickerChatDualLocalNegativeSampler, dict]:
    requested_policy = str(policy_config.get("mode", ""))
    if requested_policy not in DUAL_LOCAL_POLICIES:
        raise ValueError("unsupported Style Shapes negative policy")
    metadata_path = str(policy_config["pack_metadata_path"])
    eligibility_path = str(policy_config["eligibility_manifest"])
    manifest = load_eligibility_manifest(eligibility_path)
    if manifest.get("negative_policy") != requested_policy:
        raise RuntimeError("dual-local eligibility policy does not match configuration")
    inputs = manifest.get("inputs", {})
    expected = {
        "train_data": (str(train_data_path), sha256_file(train_data_path)),
        "pack_metadata": (metadata_path, sha256_file(metadata_path)),
        "group_bank": (str(group_bank_path), sha256_file(group_bank_path)),
    }
    for name, (path, digest) in expected.items():
        record = inputs.get(name, {})
        if record.get("path") != path or record.get("sha256") != digest:
            raise RuntimeError("dual-local %s provenance mismatch" % name)
    if inputs.get("membership_hash") != bank.membership_hash:
        raise RuntimeError("dual-local membership hash mismatch")
    id_to_pack = load_id_to_pack(metadata_path)
    sampler = StickerChatDualLocalNegativeSampler(
        bank,
        id_to_pack,
        seed=int(policy_config.get("seed", 2021)),
        policy=requested_policy,
    )
    bank_audit = validate_group_top32_bank(
        bank, id_to_pack, expected_policy=requested_policy
    )
    if inputs.get("neighbor_content_hash") != bank_audit["neighbor_content_hash"]:
        raise RuntimeError("dual-local group top-32 content hash mismatch")
    return sampler, manifest


def validate_dual_local_trace_record(
    record: Mapping[str, Any],
    sampler: StickerChatDualLocalNegativeSampler,
) -> None:
    if record.get("negative_policy") != sampler.policy:
        raise ValueError("dual-local trace policy mismatch")
    positive = int(record["positive"])
    same_pack = int(record["same_pack"])
    source_specific = record.get(sampler.neighbor_trace_field)
    generic = record.get("group_top32")
    if source_specific is None and generic is None:
        raise ValueError("dual-local trace misses the group top-32 negative")
    group_top32 = int(
        source_specific if source_specific is not None else generic
    )
    if generic is not None and int(generic) != group_top32:
        raise ValueError("dual-local generic/source-specific aliases disagree")
    if int(record["same"]) != same_pack or int(record["cross"]) != group_top32:
        raise ValueError("dual-local trace semantic aliases are inconsistent")
    if bool(record.get("fallback_used", True)):
        raise ValueError("dual-local trace unexpectedly used fallback")
    if len({positive, same_pack, group_top32}) != 3:
        raise ValueError("dual-local trace IDs are not unique")
    if sampler.id_to_pack[positive] != sampler.id_to_pack[same_pack]:
        raise ValueError("same-pack trace negative is outside the original pack")
    group = sampler.id_to_group[positive]
    if (
        sampler.id_to_group[same_pack] != group
        or sampler.id_to_group[group_top32] != group
    ):
        raise ValueError("dual-local trace negative is outside the positive group")
    if group_top32 not in sampler.neighbors[positive]:
        raise ValueError("trace negative is outside the positive group top-32 list")
