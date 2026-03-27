import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def normalize_sticker_id(raw_id: Any) -> Optional[int]:
    if raw_id is None:
        return None
    text = str(raw_id).strip()
    if text == "":
        return None
    if text.isdigit():
        return int(text)
    return None


def normalize_proto_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text == "":
        return ""
    return "_".join(text.replace("-", "_").split())


@dataclass(frozen=True)
class FactorizedStyleRecord:
    sticker_id: int
    proto_id: int
    proto_key: str
    proto_source: str
    main_subject: str
    subject_category: str
    visual_style: str
    identity_summary: str
    member_ids: List[int]
    neighbor_ids: List[int]


@dataclass(frozen=True)
class FactorizedPrototype:
    proto_id: int
    proto_key: str
    proto_source: str
    member_ids: List[int]
    main_subject: str
    subject_category: str
    visual_style: str
    identity_summary: str
    proto_density: float


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pseudo_labels(path: str) -> Dict[int, Dict[str, str]]:
    records: Dict[int, Dict[str, str]] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pseudo label file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sticker_id = normalize_sticker_id(item.get("id"))
            if sticker_id is None:
                continue
            label = item.get("label") or {}
            records[sticker_id] = {
                "main_subject": normalize_proto_text(label.get("main_subject", "")),
                "subject_category": normalize_proto_text(label.get("subject_category", "")),
                "visual_style": normalize_proto_text(label.get("visual_style", "")),
                "identity_summary": str(label.get("identity_summary", "")).strip(),
            }
    return records


def _load_style_neighbors(path: str) -> Dict[int, List[int]]:
    if not path or not os.path.exists(path):
        return {}
    raw = _read_json(path)
    rows = raw.get("neighbors", []) if isinstance(raw, dict) else raw
    neighbors: Dict[int, List[int]] = {}
    for row in rows:
        sticker_id = normalize_sticker_id(row.get("id"))
        if sticker_id is None:
            continue
        ids: List[int] = []
        for neighbor in row.get("neighbors", []):
            neighbor_id = normalize_sticker_id(neighbor.get("id"))
            if neighbor_id is None or neighbor_id == sticker_id:
                continue
            ids.append(neighbor_id)
        neighbors[sticker_id] = ids
    return neighbors


def _load_style_metadata(path: str) -> Dict[int, Dict[str, str]]:
    if not path:
        raise ValueError("style metadata path is required.")
    raw = _read_json(path)
    rows = raw.get("stickers", []) if isinstance(raw, dict) else raw
    metadata: Dict[int, Dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sticker_id = normalize_sticker_id(
            row.get("internal_img_id", row.get("sticker_id", row.get("id")))
        )
        if sticker_id is None:
            continue
        img_set = str(row.get("img_set", "")).strip()
        if not img_set:
            img_set = f"missing_{sticker_id}"
        metadata[sticker_id] = {
            "img_set": img_set,
            "external_img_id": str(
                row.get("external_img_id", row.get("filename", ""))
            ).strip(),
            "filename": str(row.get("filename", "")).strip(),
        }
    return metadata


def _pick_majority_text(values: Iterable[str], fallback: str = "") -> str:
    cleaned = [x for x in values if x]
    if not cleaned:
        return fallback
    return Counter(cleaned).most_common(1)[0][0]


def _build_fine_key(label: Mapping[str, str]) -> str:
    main_subject = label.get("main_subject", "")
    subject_category = label.get("subject_category", "")
    visual_style = label.get("visual_style", "")
    if main_subject and subject_category and visual_style:
        return f"{main_subject}|{subject_category}|{visual_style}"
    return ""


def _build_coarse_key(label: Mapping[str, str]) -> str:
    subject_category = label.get("subject_category", "")
    visual_style = label.get("visual_style", "")
    if subject_category and visual_style:
        return f"{subject_category}|{visual_style}"
    return ""


def build_factorized_style_bank_dict(
    pseudo_label_path: str,
    style_neighbors_path: str,
    max_image_id: int,
    min_fine_proto_size: int = 2,
    min_coarse_proto_size: int = 2,
) -> Dict[str, Any]:
    labels = _load_pseudo_labels(pseudo_label_path)
    neighbors = _load_style_neighbors(style_neighbors_path)

    fine_groups: Dict[str, List[int]] = defaultdict(list)
    coarse_groups: Dict[str, List[int]] = defaultdict(list)
    sticker_meta: Dict[int, Dict[str, str]] = {}
    for sticker_id in range(max_image_id):
        label = labels.get(
            sticker_id,
            {
                "main_subject": "",
                "subject_category": "",
                "visual_style": "",
                "identity_summary": "",
            },
        )
        sticker_meta[sticker_id] = dict(label)
        fine_key = _build_fine_key(label)
        coarse_key = _build_coarse_key(label)
        if fine_key:
            fine_groups[fine_key].append(sticker_id)
        if coarse_key:
            coarse_groups[coarse_key].append(sticker_id)

    proto_key_by_sticker: Dict[int, str] = {}
    proto_source_by_sticker: Dict[int, str] = {}
    for sticker_id in range(max_image_id):
        label = sticker_meta[sticker_id]
        fine_key = _build_fine_key(label)
        coarse_key = _build_coarse_key(label)
        if fine_key and len(fine_groups[fine_key]) >= min_fine_proto_size:
            proto_key_by_sticker[sticker_id] = f"fine::{fine_key}"
            proto_source_by_sticker[sticker_id] = "fine"
            continue
        if coarse_key and len(coarse_groups[coarse_key]) >= min_coarse_proto_size:
            proto_key_by_sticker[sticker_id] = f"coarse::{coarse_key}"
            proto_source_by_sticker[sticker_id] = "coarse"
            continue
        proto_key_by_sticker[sticker_id] = f"singleton::{sticker_id}"
        proto_source_by_sticker[sticker_id] = "singleton"

    proto_members: Dict[str, List[int]] = defaultdict(list)
    for sticker_id, proto_key in proto_key_by_sticker.items():
        proto_members[proto_key].append(sticker_id)

    sorted_proto_keys = sorted(
        proto_members.keys(),
        key=lambda key: (-len(proto_members[key]), key),
    )
    proto_id_map = {key: idx for idx, key in enumerate(sorted_proto_keys)}
    max_proto_size = max((len(v) for v in proto_members.values()), default=1)

    prototypes: List[Dict[str, Any]] = []
    for proto_key in sorted_proto_keys:
        member_ids = sorted(proto_members[proto_key])
        proto_labels = [sticker_meta[idx] for idx in member_ids]
        proto_id = proto_id_map[proto_key]
        proto_density = math.sqrt(float(len(member_ids))) / math.sqrt(float(max_proto_size))
        prototype = {
            "proto_id": proto_id,
            "proto_key": proto_key,
            "proto_source": proto_source_by_sticker[member_ids[0]],
            "member_ids": member_ids,
            "main_subject": _pick_majority_text(x.get("main_subject", "") for x in proto_labels),
            "subject_category": _pick_majority_text(
                x.get("subject_category", "") for x in proto_labels
            ),
            "visual_style": _pick_majority_text(x.get("visual_style", "") for x in proto_labels),
            "identity_summary": _pick_majority_text(
                x.get("identity_summary", "") for x in proto_labels
            ),
            "proto_density": proto_density,
        }
        prototypes.append(prototype)

    records: List[Dict[str, Any]] = []
    for sticker_id in range(max_image_id):
        proto_key = proto_key_by_sticker[sticker_id]
        proto_id = proto_id_map[proto_key]
        label = sticker_meta[sticker_id]
        record = {
            "sticker_id": sticker_id,
            "proto_id": proto_id,
            "proto_key": proto_key,
            "proto_source": proto_source_by_sticker[sticker_id],
            "main_subject": label.get("main_subject", ""),
            "subject_category": label.get("subject_category", ""),
            "visual_style": label.get("visual_style", ""),
            "identity_summary": label.get("identity_summary", ""),
            "member_ids": sorted(proto_members[proto_key]),
            "neighbor_ids": [x for x in neighbors.get(sticker_id, []) if x < max_image_id],
        }
        records.append(record)

    return {
        "meta": {
            "pseudo_label_path": pseudo_label_path,
            "style_neighbors_path": style_neighbors_path,
            "max_image_id": int(max_image_id),
            "min_fine_proto_size": int(min_fine_proto_size),
            "min_coarse_proto_size": int(min_coarse_proto_size),
            "num_prototypes": len(prototypes),
        },
        "prototypes": prototypes,
        "records": records,
    }


def build_factorized_style_bank_from_style_metadata(
    style_metadata_path: str,
    style_neighbors_path: str,
    max_image_id: int,
) -> Dict[str, Any]:
    style_metadata = _load_style_metadata(style_metadata_path)
    neighbors = _load_style_neighbors(style_neighbors_path)

    style_groups: Dict[str, List[int]] = defaultdict(list)
    sticker_meta: Dict[int, Dict[str, str]] = {}
    missing_ids: List[int] = []
    for sticker_id in range(max_image_id):
        row = style_metadata.get(sticker_id)
        if row is None:
            missing_ids.append(sticker_id)
            row = {
                "img_set": f"missing_{sticker_id}",
                "external_img_id": str(sticker_id),
                "filename": "",
            }
        sticker_meta[sticker_id] = dict(row)
        style_groups[row["img_set"]].append(sticker_id)

    sorted_style_keys = sorted(
        style_groups.keys(),
        key=lambda key: (-len(style_groups[key]), key),
    )
    proto_id_map = {key: idx for idx, key in enumerate(sorted_style_keys)}
    max_proto_size = max((len(v) for v in style_groups.values()), default=1)

    prototypes: List[Dict[str, Any]] = []
    for img_set in sorted_style_keys:
        member_ids = sorted(style_groups[img_set])
        proto_id = proto_id_map[img_set]
        proto_density = math.sqrt(float(len(member_ids))) / math.sqrt(float(max_proto_size))
        prototypes.append(
            {
                "proto_id": proto_id,
                "proto_key": f"img_set::{img_set}",
                "proto_source": "img_set",
                "member_ids": member_ids,
                "main_subject": f"set_{img_set}",
                "subject_category": "stickerchat_pack",
                "visual_style": f"pack_{img_set}",
                "identity_summary": f"{len(member_ids)} stickers from set {img_set}",
                "proto_density": proto_density,
            }
        )

    records: List[Dict[str, Any]] = []
    for sticker_id in range(max_image_id):
        row = sticker_meta[sticker_id]
        img_set = row["img_set"]
        records.append(
            {
                "sticker_id": sticker_id,
                "proto_id": proto_id_map[img_set],
                "proto_key": f"img_set::{img_set}",
                "proto_source": "img_set",
                "main_subject": f"set_{img_set}",
                "subject_category": "stickerchat_pack",
                "visual_style": f"pack_{img_set}",
                "identity_summary": row.get("external_img_id", str(sticker_id)),
                "member_ids": sorted(style_groups[img_set]),
                "neighbor_ids": [x for x in neighbors.get(sticker_id, []) if x < max_image_id],
            }
        )

    return {
        "meta": {
            "style_metadata_path": style_metadata_path,
            "style_neighbors_path": style_neighbors_path,
            "max_image_id": int(max_image_id),
            "num_prototypes": len(prototypes),
            "num_missing_style_metadata_ids": len(missing_ids),
        },
        "prototypes": prototypes,
        "records": records,
    }


class FactorizedStyleBank:
    def __init__(self, bank_dict: Dict[str, Any]):
        meta = bank_dict.get("meta", {})
        self.meta = dict(meta)
        self.records: List[FactorizedStyleRecord] = [
            FactorizedStyleRecord(**row) for row in bank_dict.get("records", [])
        ]
        self.prototypes: List[FactorizedPrototype] = [
            FactorizedPrototype(**row) for row in bank_dict.get("prototypes", [])
        ]
        self.sticker_to_record: Dict[int, FactorizedStyleRecord] = {
            row.sticker_id: row for row in self.records
        }
        self.proto_to_record: Dict[int, FactorizedPrototype] = {
            row.proto_id: row for row in self.prototypes
        }
        self.sticker_to_proto: Dict[int, int] = {
            row.sticker_id: row.proto_id for row in self.records
        }
        self.proto_to_members: Dict[int, List[int]] = {
            row.proto_id: list(row.member_ids) for row in self.prototypes
        }

    @classmethod
    def from_assets(
        cls,
        pseudo_label_path: str,
        style_neighbors_path: str,
        max_image_id: int,
        min_fine_proto_size: int = 2,
        min_coarse_proto_size: int = 2,
    ) -> "FactorizedStyleBank":
        bank_dict = build_factorized_style_bank_dict(
            pseudo_label_path=pseudo_label_path,
            style_neighbors_path=style_neighbors_path,
            max_image_id=max_image_id,
            min_fine_proto_size=min_fine_proto_size,
            min_coarse_proto_size=min_coarse_proto_size,
        )
        return cls(bank_dict)

    @classmethod
    def from_style_metadata(
        cls,
        style_metadata_path: str,
        style_neighbors_path: str,
        max_image_id: int,
    ) -> "FactorizedStyleBank":
        bank_dict = build_factorized_style_bank_from_style_metadata(
            style_metadata_path=style_metadata_path,
            style_neighbors_path=style_neighbors_path,
            max_image_id=max_image_id,
        )
        return cls(bank_dict)

    @classmethod
    def from_json(cls, path: str) -> "FactorizedStyleBank":
        return cls(_read_json(path))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": dict(self.meta),
            "prototypes": [row.__dict__ for row in self.prototypes],
            "records": [row.__dict__ for row in self.records],
        }

    def save_json(self, path: str) -> None:
        p = Path(path)
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def proto_id_of(self, sticker_id: int) -> int:
        return self.sticker_to_proto[int(sticker_id)]

    def proto_density(self, proto_id: int) -> float:
        return float(self.proto_to_record[int(proto_id)].proto_density)

    def members_of_proto(self, proto_id: int) -> List[int]:
        return list(self.proto_to_members.get(int(proto_id), []))

    def expand_top_prototypes(
        self,
        proto_ids: Sequence[int],
        allowed_ids: Optional[Iterable[int]] = None,
        per_proto_limit: Optional[int] = None,
    ) -> List[int]:
        allowed = set(int(x) for x in allowed_ids) if allowed_ids is not None else None
        expanded: List[int] = []
        for proto_id in proto_ids:
            members = self.members_of_proto(int(proto_id))
            if allowed is not None:
                members = [x for x in members if x in allowed]
            if per_proto_limit is not None and per_proto_limit > 0:
                members = members[:per_proto_limit]
            expanded.extend(members)
        return expanded

    def sample_same_proto_negative(
        self,
        sticker_id: int,
        rng: random.Random,
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> Optional[int]:
        exclude = {int(sticker_id)}
        if exclude_ids is not None:
            exclude.update(int(x) for x in exclude_ids)
        proto_id = self.proto_id_of(int(sticker_id))
        candidates = [x for x in self.members_of_proto(proto_id) if x not in exclude]
        if not candidates:
            return None
        return int(rng.choice(candidates))

    def sample_cross_proto_negative(
        self,
        sticker_id: int,
        rng: random.Random,
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> Optional[int]:
        exclude = {int(sticker_id)}
        if exclude_ids is not None:
            exclude.update(int(x) for x in exclude_ids)
        proto_id = self.proto_id_of(int(sticker_id))
        candidates = [
            row.sticker_id
            for row in self.records
            if row.proto_id != proto_id and row.sticker_id not in exclude
        ]
        if not candidates:
            return None
        return int(rng.choice(candidates))
