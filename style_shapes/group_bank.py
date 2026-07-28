"""Compact, validated Group Bank v1 schema and legacy adapter."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .io import atomic_write_json, hash_value

GROUP_BANK_SCHEMA = "style_shapes.group_bank.v1"


def membership_pairs(
    sticker_ids: Sequence[int], sticker_to_group: Sequence[int]
) -> List[List[int]]:
    if len(sticker_ids) != len(sticker_to_group):
        raise ValueError("sticker_ids and sticker_to_group length mismatch")
    pairs = sorted(
        ([int(sticker_id), int(group_id)] for sticker_id, group_id in zip(sticker_ids, sticker_to_group)),
        key=lambda row: row[0],
    )
    if len({row[0] for row in pairs}) != len(pairs):
        raise ValueError("duplicate sticker IDs")
    return pairs


def membership_sha256(
    sticker_ids: Sequence[int], sticker_to_group: Sequence[int]
) -> str:
    return hash_value(membership_pairs(sticker_ids, sticker_to_group))


class GroupBank:
    """Validated compact partition with an in-memory legacy representation."""

    def __init__(self, value: Mapping[str, Any]):
        self.value = dict(value)
        self.validate()

    @property
    def dataset(self) -> str:
        return str(self.value["dataset"])

    @property
    def group_source(self) -> str:
        return str(self.value["group_source"])

    @property
    def num_groups(self) -> int:
        return int(self.value["num_groups"])

    @property
    def sticker_ids(self) -> List[int]:
        return [int(value) for value in self.value["sticker_ids"]]

    @property
    def sticker_to_group(self) -> List[int]:
        return [int(value) for value in self.value["sticker_to_group"]]

    @property
    def group_to_members(self) -> List[List[int]]:
        return [[int(value) for value in row] for row in self.value["group_to_members"]]

    @property
    def membership_hash(self) -> str:
        return str(self.value["provenance"]["membership_hash"])

    def validate(self) -> None:
        if self.value.get("schema_version") != GROUP_BANK_SCHEMA:
            raise ValueError("unsupported group bank schema: %r" % self.value.get("schema_version"))
        ids = [int(value) for value in self.value.get("sticker_ids", [])]
        groups = [int(value) for value in self.value.get("sticker_to_group", [])]
        k = int(self.value.get("num_groups", 0))
        members = [
            [int(value) for value in row] for row in self.value.get("group_to_members", [])
        ]
        sizes = [int(value) for value in self.value.get("group_sizes", [])]
        if not self.value.get("dataset") or not self.value.get("group_source"):
            raise ValueError("dataset and group_source are required")
        if k <= 0 or len(members) != k or len(sizes) != k:
            raise ValueError("num_groups/group_to_members/group_sizes mismatch")
        if len(ids) != len(groups) or len(set(ids)) != len(ids):
            raise ValueError("sticker IDs are not a unique aligned sequence")
        if sorted(set(groups)) != list(range(k)):
            raise ValueError("group IDs must be dense and every group non-empty")
        flat = [item for row in members for item in row]
        if len(flat) != len(ids) or len(set(flat)) != len(ids) or set(flat) != set(ids):
            raise ValueError("group_to_members is not a complete partition")
        id_to_group = dict(zip(ids, groups))
        for group_id, row in enumerate(members):
            if not row or len(row) != sizes[group_id]:
                raise ValueError("empty group or incorrect group size at %d" % group_id)
            if any(id_to_group[item] != group_id for item in row):
                raise ValueError("bidirectional membership mismatch at %d" % group_id)
        neighbors = self.value.get("same_group_neighbors", {})
        for raw_sticker_id, raw_neighbors in neighbors.items():
            sticker_id = int(raw_sticker_id)
            if sticker_id not in id_to_group:
                raise ValueError("neighbor record has unknown sticker %d" % sticker_id)
            seen = set()
            for raw_neighbor in raw_neighbors:
                neighbor = int(raw_neighbor)
                if neighbor == sticker_id or neighbor in seen:
                    raise ValueError("invalid or duplicate same-group neighbor")
                if neighbor not in id_to_group:
                    raise ValueError("neighbor record references an unknown sticker")
                seen.add(neighbor)
        if self.value.get("cross_group_pool") != {"type": "catalog_minus_group"}:
            raise ValueError("cross_group_pool must be declarative catalog_minus_group")
        expected = membership_sha256(ids, groups)
        actual = str(self.value.get("provenance", {}).get("membership_hash", ""))
        if actual != expected:
            raise ValueError("membership hash mismatch: %s != %s" % (actual, expected))

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.value)

    def save(self, path: str) -> None:
        atomic_write_json(path, self.value)

    def to_legacy_dict(self) -> Dict[str, Any]:
        neighbors = {
            int(key): [int(value) for value in values]
            for key, values in self.value.get("same_group_neighbors", {}).items()
        }
        members = self.group_to_members
        max_size = max(len(row) for row in members)
        prototypes = []
        for group_id, row in enumerate(members):
            key = "style_shapes::%s::%04d" % (self.group_source, group_id)
            prototypes.append(
                {
                    "proto_id": group_id,
                    "proto_key": key,
                    "proto_source": self.group_source,
                    "member_ids": row,
                    "member_count": len(row),
                    "main_subject": "",
                    "subject_category": self.dataset,
                    "visual_style": self.group_source,
                    "identity_summary": "%d stickers" % len(row),
                    "proto_density": math.sqrt(float(len(row))) / math.sqrt(float(max_size)),
                    "neighbor_coverage": 0.0,
                    "style_consistency": 1.0,
                }
            )
        id_to_group = dict(zip(self.sticker_ids, self.sticker_to_group))
        records = []
        for sticker_id in self.sticker_ids:
            group_id = id_to_group[sticker_id]
            key = "style_shapes::%s::%04d" % (self.group_source, group_id)
            records.append(
                {
                    "sticker_id": sticker_id,
                    "proto_id": group_id,
                    "proto_key": key,
                    "proto_source": self.group_source,
                    "main_subject": "",
                    "subject_category": self.dataset,
                    "visual_style": self.group_source,
                    "identity_summary": str(sticker_id),
                    "member_ids": [],
                    "neighbor_ids": neighbors.get(sticker_id, []),
                }
            )
        return {
            "meta": {
                "schema_version": GROUP_BANK_SCHEMA,
                "dataset": self.dataset,
                "group_source": self.group_source,
                "num_prototypes": self.num_groups,
                "max_image_id": max(self.sticker_ids) + 1,
                "membership_hash": self.membership_hash,
                "compact_records": True,
                "provenance": self.value.get("provenance", {}),
            },
            "prototypes": prototypes,
            "records": records,
        }

    @classmethod
    def load(cls, path: str) -> "GroupBank":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    @classmethod
    def create(
        cls,
        dataset: str,
        group_source: str,
        sticker_ids: Sequence[int],
        sticker_to_group: Sequence[int],
        same_group_neighbors: Optional[Mapping[int, Sequence[int]]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "GroupBank":
        pairs = membership_pairs(sticker_ids, sticker_to_group)
        ordered_ids = [row[0] for row in pairs]
        ordered_groups = [row[1] for row in pairs]
        unique_groups = sorted(set(ordered_groups))
        if unique_groups != list(range(len(unique_groups))):
            raise ValueError("group IDs must be dense")
        group_to_members: List[List[int]] = [[] for _ in unique_groups]
        for sticker_id, group_id in pairs:
            group_to_members[group_id].append(sticker_id)
        provenance_value = dict(provenance or {})
        provenance_value.setdefault("created_utc", dt.datetime.now(dt.timezone.utc).isoformat())
        provenance_value["membership_hash"] = membership_sha256(ordered_ids, ordered_groups)
        value = {
            "schema_version": GROUP_BANK_SCHEMA,
            "dataset": str(dataset),
            "group_source": str(group_source),
            "num_groups": len(unique_groups),
            "sticker_ids": ordered_ids,
            "sticker_to_group": ordered_groups,
            "group_to_members": group_to_members,
            "group_sizes": [len(row) for row in group_to_members],
            "same_group_neighbors": {
                str(int(key)): [int(value) for value in values]
                for key, values in (same_group_neighbors or {}).items()
            },
            "cross_group_pool": {"type": "catalog_minus_group"},
            "provenance": provenance_value,
        }
        return cls(value)

    @classmethod
    def from_legacy_dict(
        cls,
        value: Mapping[str, Any],
        dataset: str,
        group_source: str,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "GroupBank":
        records = value.get("records", [])
        prototypes = value.get("prototypes", [])
        raw_group_ids = sorted(int(row["proto_id"]) for row in prototypes)
        remap = {old: new for new, old in enumerate(raw_group_ids)}
        ids = [int(row["sticker_id"]) for row in records]
        groups = [remap[int(row["proto_id"])] for row in records]
        neighbors = {
            int(row["sticker_id"]): [int(item) for item in row.get("neighbor_ids", [])]
            for row in records
        }
        prov = dict(provenance or {})
        prov["legacy_meta"] = dict(value.get("meta", {}))
        prov["neighbor_scope"] = "legacy_import_exact"
        return cls.create(dataset, group_source, ids, groups, neighbors, prov)

