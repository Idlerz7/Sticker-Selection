#!/usr/bin/env python3
"""Read-only H0/H1 audit for StickerChat candidate sets under a prototype bank.

The script reads the prototype mapping from ``prototypes[].proto_id`` and
``prototypes[].member_ids`` in a factorized bank.  It deliberately stops before
the usually very large ``records`` section, so a multi-GB bank can be audited
without loading it in full.

Candidate schema is detected from the JSON itself.  Supported gold locations
are ``dialog[-1].img_id`` and the top-level ``gold_id``/``img_id`` fields;
supported candidate fields are ``cand``, ``candidate_ids`` and ``candidates``.
An unrecognised or ambiguous schema is an error, never a silent fallback.

Example (audit both candidate generations with the same K=384 bank):

  python scripts/audit_stickerchat_384_h1.py \
    --bank-path stickerchat/processed_style_kmeans_k384/factorized_style_bank.json \
    --val-r10-path stickerchat/processed_style_kmeans_k384/release_val_u_sticker_format_int_with_cand_r10.json \
    --val-r20-path stickerchat/processed_style_kmeans_k384/release_val_u_sticker_format_int_with_cand_r20.json \
    --test-r10-path stickerchat/processed_style_kmeans_k384/release_test_u_sticker_format_int_with_cand_r10.json \
    --test-r20-path stickerchat/processed_style_kmeans_k384/release_test_u_sticker_format_int_with_cand_r20.json \
    --primary-label rebuilt_candidates \
    --comparison-val-r10-path stickerchat/processed/release_val_u_sticker_format_int_with_cand_r10.json \
    --comparison-val-r20-path stickerchat/processed/release_val_u_sticker_format_int_with_cand_r20.json \
    --comparison-test-r10-path stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json \
    --comparison-test-r20-path stickerchat/processed/release_test_u_sticker_format_int_with_cand_r20.json \
    --comparison-label original_candidates \
    --output-dir outputs/stickerchat_384_h1_audit

Only files below ``--output-dir`` are opened for writing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class SchemaError(ValueError):
    """Raised when an input JSON does not have a supported, unambiguous schema."""


PROTOCOLS: Tuple[Tuple[str, str, int], ...] = (
    ("validation_r10", "val_r10", 10),
    ("validation_r20", "val_r20", 20),
    ("test_r10", "test_r10", 10),
    ("test_r20", "test_r20", 20),
)

OUTPUT_FILENAMES = {
    "stats.json",
    "h0_ids.jsonl",
    "h1_ids.jsonl",
    "multiplicity_slices.json",
    "StickerChat_384_H1_audit.md",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-path", type=Path, required=True)
    p.add_argument("--val-r10-path", type=Path, required=True)
    p.add_argument("--val-r20-path", type=Path, required=True)
    p.add_argument("--test-r10-path", type=Path, required=True)
    p.add_argument("--test-r20-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--primary-label", default="primary_candidates")

    p.add_argument("--comparison-val-r10-path", type=Path)
    p.add_argument("--comparison-val-r20-path", type=Path)
    p.add_argument("--comparison-test-r10-path", type=Path)
    p.add_argument("--comparison-test-r20-path", type=Path)
    p.add_argument("--comparison-label", default="comparison_candidates")

    p.add_argument(
        "--primary-config",
        type=Path,
        help="Optional main-experiment YAML; its effective bank/candidate/log paths are traced.",
    )
    p.add_argument(
        "--comparison-config",
        type=Path,
        help="Optional comparison/legacy-evaluation YAML to trace.",
    )
    return p.parse_args()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def display_path(path: Path, base: Optional[Path] = None) -> str:
    p = resolved(path)
    root = resolved(base or Path.cwd())
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return str(p)


def require_input_file(path: Path, label: str) -> Path:
    p = resolved(path)
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found or not a file: {p}")
    return p


def validate_output_scope(output_dir: Path, input_paths: Iterable[Path]) -> Path:
    out = resolved(output_dir)
    inputs = [resolved(p) for p in input_paths]
    for src in inputs:
        if out == src or out == src.parent:
            raise ValueError(
                f"--output-dir must be a separate audit directory, not an input path/parent: {out}"
            )
        for name in OUTPUT_FILENAMES:
            if out / name == src:
                raise ValueError(f"Audit output would overwrite input file: {src}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"Invalid JSON in {path}: {exc}") from exc


def extract_named_json_value(path: Path, key: str, chunk_size: int = 1 << 20) -> Any:
    """Extract the first JSON value belonging to *key* without reading the rest.

    This scanner respects JSON string escaping while balancing objects/arrays.
    It is suitable for extracting the small ``meta`` and ``prototypes`` values
    near the beginning of the multi-GB factorized bank.
    """

    needle = json.dumps(key).encode("utf-8")
    buffer = bytearray()
    found_at: Optional[int] = None
    start: Optional[int] = None
    scan_pos = 0
    stack: List[int] = []
    in_string = False
    escaped = False

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if chunk:
                buffer.extend(chunk)

            if found_at is None:
                search_from = max(0, len(buffer) - len(chunk) - len(needle) - 8)
                pos = bytes(buffer).find(needle, search_from)
                while pos >= 0:
                    j = pos + len(needle)
                    while j < len(buffer) and chr(buffer[j]).isspace():
                        j += 1
                    if j < len(buffer) and buffer[j] == ord(":"):
                        found_at = pos
                        scan_pos = j + 1
                        break
                    pos = bytes(buffer).find(needle, pos + 1)

            if found_at is not None and start is None:
                while scan_pos < len(buffer) and chr(buffer[scan_pos]).isspace():
                    scan_pos += 1
                if scan_pos < len(buffer):
                    if buffer[scan_pos] not in (ord("{"), ord("[")):
                        raise SchemaError(
                            f"Bank field {key!r} in {path} must be an object or array"
                        )
                    start = scan_pos
                    stack = [buffer[scan_pos]]
                    scan_pos += 1

            if start is not None:
                while scan_pos < len(buffer):
                    ch = buffer[scan_pos]
                    if in_string:
                        if escaped:
                            escaped = False
                        elif ch == ord("\\"):
                            escaped = True
                        elif ch == ord('"'):
                            in_string = False
                    else:
                        if ch == ord('"'):
                            in_string = True
                        elif ch in (ord("{"), ord("[")):
                            stack.append(ch)
                        elif ch in (ord("}"), ord("]")):
                            if not stack:
                                raise SchemaError(f"Unbalanced JSON while reading {key!r} from {path}")
                            opener = stack.pop()
                            if (opener, ch) not in ((ord("{"), ord("}")), (ord("["), ord("]"))):
                                raise SchemaError(f"Mismatched JSON delimiters in {key!r} from {path}")
                            if not stack:
                                raw = bytes(buffer[start : scan_pos + 1])
                                try:
                                    return json.loads(raw)
                                except json.JSONDecodeError as exc:
                                    raise SchemaError(
                                        f"Could not decode bank field {key!r} in {path}: {exc}"
                                    ) from exc
                    scan_pos += 1

            if not chunk:
                break
            if len(buffer) > (512 << 20):
                raise SchemaError(
                    f"Bank field {key!r} exceeded 512 MiB before closing; refusing unsafe parse"
                )

    raise SchemaError(f"Could not find a complete top-level-like field {key!r} in bank {path}")


def detect_common_field(
    rows: Sequence[Mapping[str, Any]], candidates: Sequence[str], description: str
) -> str:
    matches = [name for name in candidates if all(name in row for row in rows)]
    if len(matches) != 1:
        raise SchemaError(
            f"Expected exactly one supported {description} field among {list(candidates)}, "
            f"found {matches}. Example keys: {sorted(rows[0].keys()) if rows else []}"
        )
    return matches[0]


def parse_sticker_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a sticker ID")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        result = int(value.strip())
    else:
        raise ValueError(f"not a non-negative integer ID: {value!r}")
    if result < 0:
        raise ValueError(f"negative sticker ID: {result}")
    return result


def numeric_summary(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def histogram(values: Sequence[int]) -> Dict[str, int]:
    counts = Counter(values)
    return {str(k): counts[k] for k in sorted(counts)}


def file_info(path: Path) -> Dict[str, Any]:
    p = resolved(path)
    if not p.exists():
        return {"path": display_path(p), "exists": False}
    st = p.stat()
    return {
        "path": display_path(p),
        "exists": True,
        "size_bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
    }


def analyze_bank(bank_path: Path) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str], Dict[str, Any]]:
    bank_meta = extract_named_json_value(bank_path, "meta")
    prototypes = extract_named_json_value(bank_path, "prototypes")
    if not isinstance(bank_meta, dict):
        raise SchemaError(f"bank.meta must be an object: {bank_path}")
    if not isinstance(prototypes, list) or not prototypes:
        raise SchemaError(f"bank.prototypes must be a non-empty array: {bank_path}")
    if not all(isinstance(row, dict) for row in prototypes):
        raise SchemaError("Every bank.prototypes element must be an object")

    proto_id_field = detect_common_field(
        prototypes, ("proto_id", "prototype_id"), "prototype ID"
    )
    member_field = detect_common_field(
        prototypes, ("member_ids", "sticker_ids"), "prototype member IDs"
    )

    sticker_to_proto: Dict[int, int] = {}
    proto_sizes: Dict[int, int] = {}
    proto_keys: Dict[int, str] = {}
    duplicate_members: List[int] = []
    duplicate_proto_ids: List[int] = []
    for index, proto in enumerate(prototypes):
        try:
            proto_id = parse_sticker_id(proto[proto_id_field])
        except ValueError as exc:
            raise SchemaError(f"Invalid prototype ID at prototypes[{index}]: {exc}") from exc
        if proto_id in proto_sizes:
            duplicate_proto_ids.append(proto_id)
            continue
        members = proto[member_field]
        if not isinstance(members, list):
            raise SchemaError(f"prototypes[{index}].{member_field} must be an array")
        parsed_members: List[int] = []
        for member_index, raw in enumerate(members):
            try:
                sticker_id = parse_sticker_id(raw)
            except ValueError as exc:
                raise SchemaError(
                    f"Invalid member ID at prototypes[{index}].{member_field}[{member_index}]: {exc}"
                ) from exc
            if sticker_id in sticker_to_proto:
                duplicate_members.append(sticker_id)
            else:
                sticker_to_proto[sticker_id] = proto_id
            parsed_members.append(sticker_id)
        if "member_count" in proto and int(proto["member_count"]) != len(parsed_members):
            raise SchemaError(
                f"prototypes[{index}].member_count={proto['member_count']} "
                f"but {member_field} has {len(parsed_members)} entries"
            )
        proto_sizes[proto_id] = len(parsed_members)
        if "proto_key" in proto:
            proto_keys[proto_id] = str(proto["proto_key"])

    if duplicate_proto_ids:
        raise SchemaError(f"Duplicate prototype IDs in bank: {duplicate_proto_ids[:20]}")
    if duplicate_members:
        raise SchemaError(
            f"Sticker IDs assigned to multiple prototypes: {sorted(set(duplicate_members))[:20]}"
        )
    declared = bank_meta.get("num_prototypes")
    if declared is not None and int(declared) != len(proto_sizes):
        raise SchemaError(
            f"bank.meta.num_prototypes={declared}, parsed non-empty prototypes={len(proto_sizes)}"
        )

    asset_dir = bank_path.parent
    sibling_paths = {
        name: asset_dir / name
        for name in (
            "sticker_metadata.json",
            "img_set_to_ids.json",
            "style_regroup_analysis.json",
            "summary.json",
        )
    }
    for name, path in sibling_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required K384 sibling asset missing: {name}: {path}")

    analysis = load_json(sibling_paths["style_regroup_analysis.json"])
    summary = load_json(sibling_paths["summary.json"])
    metadata = load_json(sibling_paths["sticker_metadata.json"])
    group_members = load_json(sibling_paths["img_set_to_ids.json"])
    if not isinstance(analysis, dict) or not isinstance(summary, dict):
        raise SchemaError("style_regroup_analysis.json and summary.json must be objects")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("stickers"), list):
        raise SchemaError("sticker_metadata.json must contain a stickers array")
    if not isinstance(group_members, dict):
        raise SchemaError("img_set_to_ids.json must be an object")

    metadata_rows = metadata["stickers"]
    if not metadata_rows or not all(isinstance(row, dict) for row in metadata_rows):
        raise SchemaError("sticker_metadata.stickers must contain objects")
    metadata_id_field = detect_common_field(
        metadata_rows, ("internal_img_id", "sticker_id", "id"), "metadata sticker ID"
    )
    metadata_group_field = detect_common_field(
        metadata_rows, ("img_set", "prototype_key", "proto_key"), "metadata group"
    )
    original_pack_field = (
        "original_img_set"
        if all("original_img_set" in row for row in metadata_rows)
        else None
    )

    group_to_proto: Dict[str, int] = {}
    for pid, key in proto_keys.items():
        group = key.split("::", 1)[1] if "::" in key else key
        if group in group_to_proto and group_to_proto[group] != pid:
            raise SchemaError(f"Multiple prototypes resolve to group key {group!r}")
        group_to_proto[group] = pid

    original_pack_by_sticker: Dict[int, str] = {}
    metadata_mapping_mismatches = 0
    metadata_missing_groups = 0
    for row in metadata_rows:
        sid = parse_sticker_id(row[metadata_id_field])
        group = str(row[metadata_group_field])
        pid = group_to_proto.get(group)
        if pid is None:
            metadata_missing_groups += 1
        elif sticker_to_proto.get(sid) != pid:
            metadata_mapping_mismatches += 1
        if original_pack_field is not None:
            original_pack_by_sticker[sid] = str(row[original_pack_field])

    img_set_mismatches = 0
    img_set_unknown_groups = 0
    for group, raw_ids in group_members.items():
        if not isinstance(raw_ids, list):
            raise SchemaError(f"img_set_to_ids[{group!r}] must be an array")
        pid = group_to_proto.get(str(group))
        if pid is None:
            img_set_unknown_groups += 1
            continue
        actual = sorted(parse_sticker_id(x) for x in raw_ids)
        expected = sorted(sid for sid, mapped_pid in sticker_to_proto.items() if mapped_pid == pid)
        if actual != expected:
            img_set_mismatches += 1

    if metadata_mapping_mismatches or metadata_missing_groups or img_set_mismatches or img_set_unknown_groups:
        raise SchemaError(
            "K384 sibling assets disagree with bank mapping: "
            f"metadata_mapping_mismatches={metadata_mapping_mismatches}, "
            f"metadata_missing_groups={metadata_missing_groups}, "
            f"img_set_mismatches={img_set_mismatches}, "
            f"img_set_unknown_groups={img_set_unknown_groups}"
        )

    sizes = list(proto_sizes.values())
    requested_k = analysis.get("num_clusters_requested", metadata.get("meta", {}).get("num_clusters_requested"))
    audit = {
        "path": display_path(bank_path),
        "schema": {
            "prototype_array": "prototypes",
            "prototype_id_field": f"prototypes[].{proto_id_field}",
            "member_id_field": f"prototypes[].{member_field}",
            "sticker_to_prototype_mapping": f"prototypes[].{member_field} -> prototypes[].{proto_id_field}",
            "records_section_loaded": False,
        },
        "bank_meta": bank_meta,
        "requested_k": int(requested_k) if requested_k is not None else None,
        "actual_nonempty_prototypes": len(proto_sizes),
        "total_mapped_stickers": len(sticker_to_proto),
        "original_pack_count": analysis.get("num_original_packs", metadata.get("meta", {}).get("num_original_packs")),
        "kmeans_seed": analysis.get("seed", metadata.get("meta", {}).get("seed")),
        "kmeans_iters": analysis.get("kmeans_iters", metadata.get("meta", {}).get("kmeans_iters")),
        "input_clip_embedding_cache": analysis.get("img_emb_cache_path"),
        "prototype_size": numeric_summary(sizes),
        "prototype_size_histogram": histogram(sizes),
        "mapped_sticker_id_range": {
            "min": min(sticker_to_proto) if sticker_to_proto else None,
            "max": max(sticker_to_proto) if sticker_to_proto else None,
        },
        "sibling_assets": {name: file_info(path) for name, path in sibling_paths.items()},
        "sibling_consistency": {
            "metadata_rows": len(metadata_rows),
            "metadata_id_field": metadata_id_field,
            "metadata_group_field": metadata_group_field,
            "original_pack_field": original_pack_field,
            "metadata_mapping_mismatches": metadata_mapping_mismatches,
            "img_set_group_count": len(group_members),
            "img_set_membership_mismatches": img_set_mismatches,
            "summary_num_final_groups": summary.get("num_final_groups"),
            "analysis_method": analysis.get("style_regroup_method"),
        },
    }
    return sticker_to_proto, proto_sizes, original_pack_by_sticker, audit


GoldGetter = Tuple[str, ...]


def get_path(row: Mapping[str, Any], path: GoldGetter) -> Any:
    value: Any = row
    for part in path:
        if part == "[-1]":
            if not isinstance(value, list) or not value:
                raise KeyError("[-1]")
            value = value[-1]
        else:
            if not isinstance(value, dict) or part not in value:
                raise KeyError(part)
            value = value[part]
    return value


def detect_candidate_schema(rows: Sequence[Mapping[str, Any]], path: Path) -> Dict[str, Any]:
    if not rows:
        raise SchemaError(f"Candidate JSON array is empty: {path}")
    sample = rows[: min(100, len(rows))]
    candidate_fields = [
        name
        for name in ("cand", "candidate_ids", "candidates")
        if all(name in row and isinstance(row[name], list) for row in sample)
    ]
    if len(candidate_fields) != 1:
        raise SchemaError(
            f"{path}: expected exactly one candidate list field among "
            f"['cand', 'candidate_ids', 'candidates']; found {candidate_fields}. "
            f"Example keys: {sorted(sample[0].keys())}"
        )

    gold_locations: List[GoldGetter] = []
    for location in (("dialog", "[-1]", "img_id"), ("gold_id",), ("img_id",)):
        try:
            values = [get_path(row, location) for row in sample]
        except KeyError:
            continue
        if all(value is not None for value in values):
            gold_locations.append(location)
    if len(gold_locations) != 1:
        rendered = [".".join(x).replace(".[-1]", "[-1]") for x in gold_locations]
        raise SchemaError(
            f"{path}: expected one supported gold location; found {rendered}. "
            "Supported: dialog[-1].img_id, gold_id, img_id"
        )

    query_field: Optional[str] = None
    query_field_unique = False
    for field in ("query_id", "dialogue_id", "user_id", "id"):
        if all(field in row and isinstance(row[field], (str, int)) for row in rows):
            values = [str(row[field]) for row in rows]
            if len(set(values)) == len(values):
                query_field = field
                query_field_unique = True
                break
            if query_field is None:
                query_field = field
    return {
        "gold_location_tuple": gold_locations[0],
        "gold_location": ".".join(gold_locations[0]).replace(".[-1]", "[-1]"),
        "candidate_field": candidate_fields[0],
        "query_id_field": query_field or "row_index",
        "query_id_unique": query_field_unique if query_field else True,
    }


def bucket_multiplicity(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    return ">=3"


def bucket_distractors(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    return ">=3"


def audit_candidate_file(
    path: Path,
    dataset_label: str,
    protocol: str,
    expected_count: int,
    sticker_to_proto: Mapping[int, int],
    proto_sizes: Mapping[int, int],
    original_pack_by_sticker: Mapping[int, str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise SchemaError(f"Candidate file must be a top-level JSON array: {path}")
    if not all(isinstance(row, dict) for row in raw):
        bad = next(i for i, row in enumerate(raw) if not isinstance(row, dict))
        raise SchemaError(f"Candidate row {bad} in {path} is not an object")
    rows: List[Mapping[str, Any]] = raw
    schema = detect_candidate_schema(rows, path)
    gold_path: GoldGetter = schema.pop("gold_location_tuple")
    candidate_field = schema["candidate_field"]
    query_field = schema["query_id_field"]

    counts = Counter()
    h0: List[Dict[str, Any]] = []
    h1: List[Dict[str, Any]] = []
    slices: Dict[str, List[Dict[str, Any]]] = {"1": [], "2": [], ">=3": []}
    unique_proto_counts: List[int] = []
    max_proto_multiplicities: List[int] = []
    gold_proto_sizes: List[int] = []
    positive_multiplicities: List[int] = []
    original_pack_multiplicities: List[int] = []
    status_by_query: Dict[str, Dict[str, Any]] = {}
    examples: Dict[str, List[Any]] = {
        "duplicate_candidates": [],
        "gold_not_in_candidates": [],
        "missing_prototype_mapping": [],
        "wrong_candidate_count": [],
        "illegal_sticker_id": [],
    }
    seen_query_ids: Counter[str] = Counter()

    for index, row in enumerate(rows):
        counts["queries"] += 1
        if query_field == "row_index":
            raw_query_id = str(index)
        else:
            if query_field not in row:
                raise SchemaError(f"{path}: row {index} lacks detected query field {query_field!r}")
            raw_query_id = str(row[query_field])
        seen_query_ids[raw_query_id] += 1
        query_id = raw_query_id
        if seen_query_ids[raw_query_id] > 1:
            query_id = f"{raw_query_id}#row={index}"

        try:
            raw_gold = get_path(row, gold_path)
        except KeyError as exc:
            raise SchemaError(
                f"{path}: row {index} lacks detected gold location {schema['gold_location']}"
            ) from exc
        raw_candidates = row.get(candidate_field)
        if not isinstance(raw_candidates, list):
            raise SchemaError(
                f"{path}: row {index}.{candidate_field} is not an array; schema changed within file"
            )

        illegal_occurrences = 0
        try:
            gold_id: Optional[int] = parse_sticker_id(raw_gold)
        except ValueError as exc:
            gold_id = None
            illegal_occurrences += 1
            if len(examples["illegal_sticker_id"]) < 20:
                examples["illegal_sticker_id"].append(
                    {"query_id": query_id, "location": schema["gold_location"], "value": raw_gold, "error": str(exc)}
                )

        candidate_ids: List[Optional[int]] = []
        for cand_index, value in enumerate(raw_candidates):
            try:
                candidate_ids.append(parse_sticker_id(value))
            except ValueError as exc:
                candidate_ids.append(None)
                illegal_occurrences += 1
                if len(examples["illegal_sticker_id"]) < 20:
                    examples["illegal_sticker_id"].append(
                        {"query_id": query_id, "location": f"{candidate_field}[{cand_index}]", "value": value, "error": str(exc)}
                    )
        if illegal_occurrences:
            counts["illegal_id_queries"] += 1
            counts["illegal_id_occurrences"] += illegal_occurrences

        parsed_candidates = [x for x in candidate_ids if x is not None]
        duplicate_excess = len(parsed_candidates) - len(set(parsed_candidates))
        has_duplicates = duplicate_excess > 0
        if has_duplicates:
            counts["duplicate_candidate_queries"] += 1
            counts["duplicate_candidate_excess"] += duplicate_excess
            if len(examples["duplicate_candidates"]) < 20:
                examples["duplicate_candidates"].append(
                    {"query_id": query_id, "duplicate_excess": duplicate_excess}
                )

        wrong_count = len(raw_candidates) != expected_count
        if wrong_count:
            counts["wrong_candidate_count_queries"] += 1
            if len(examples["wrong_candidate_count"]) < 20:
                examples["wrong_candidate_count"].append(
                    {"query_id": query_id, "actual": len(raw_candidates), "expected": expected_count}
                )

        gold_absent = gold_id is not None and gold_id not in parsed_candidates
        if gold_absent:
            counts["gold_not_in_candidates_queries"] += 1
            if len(examples["gold_not_in_candidates"]) < 20:
                examples["gold_not_in_candidates"].append({"query_id": query_id, "gold_id": gold_id})

        ids_to_map = ([gold_id] if gold_id is not None else []) + parsed_candidates
        missing_ids = [sid for sid in ids_to_map if sid not in sticker_to_proto]
        if missing_ids:
            counts["missing_mapping_queries"] += 1
            counts["missing_mapping_occurrences"] += len(missing_ids)
            if len(examples["missing_prototype_mapping"]) < 20:
                examples["missing_prototype_mapping"].append(
                    {"query_id": query_id, "sticker_ids": sorted(set(missing_ids))[:20]}
                )

        valid = not (
            illegal_occurrences
            or has_duplicates
            or wrong_count
            or gold_absent
            or missing_ids
            or gold_id is None
        )
        candidate_signature = tuple(parsed_candidates) if len(parsed_candidates) == len(raw_candidates) else None
        if not valid:
            status_by_query[query_id] = {
                "status": "invalid",
                "gold_id": gold_id,
                "candidate_signature": candidate_signature,
            }
            continue

        counts["valid_queries"] += 1
        assert gold_id is not None
        gold_proto = sticker_to_proto[gold_id]
        candidate_protos = [sticker_to_proto[sid] for sid in parsed_candidates]
        proto_counts = Counter(candidate_protos)
        positive_multiplicity = proto_counts[gold_proto]
        same_proto_distractors = sum(
            1
            for sid, pid in zip(parsed_candidates, candidate_protos)
            if sid != gold_id and pid == gold_proto
        )
        unique_proto_count = len(proto_counts)
        max_proto_multiplicity = max(proto_counts.values())
        gold_proto_size = proto_sizes[gold_proto]

        original_pack_multiplicity: Optional[int] = None
        if original_pack_by_sticker and gold_id in original_pack_by_sticker:
            gold_pack = original_pack_by_sticker[gold_id]
            if all(sid in original_pack_by_sticker for sid in parsed_candidates):
                original_pack_multiplicity = sum(
                    1 for sid in parsed_candidates if original_pack_by_sticker[sid] == gold_pack
                )
                original_pack_multiplicities.append(original_pack_multiplicity)
                if original_pack_multiplicity >= 2:
                    counts["original_pack_h1_queries"] += 1

        multiplicity_bucket = bucket_multiplicity(positive_multiplicity)
        counts[f"multiplicity_{multiplicity_bucket}"] += 1
        if positive_multiplicity == 1:
            hypothesis = "H0"
            counts["h0"] += 1
        else:
            hypothesis = "H1"
            counts["h1"] += 1
            distractor_bucket = bucket_distractors(same_proto_distractors)
            counts[f"h1_distractors_{distractor_bucket}"] += 1
            if original_pack_multiplicity == 1:
                counts["h1_created_by_pack_merge"] += 1

        manifest = {
            "dataset": dataset_label,
            "protocol": protocol,
            "query_index": index,
            "query_id": query_id,
            "gold_id": gold_id,
            "gold_prototype_id": gold_proto,
            "positive_prototype_multiplicity": positive_multiplicity,
            "same_prototype_distractors": same_proto_distractors,
            "unique_prototype_count": unique_proto_count,
            "max_candidate_prototype_multiplicity": max_proto_multiplicity,
            "gold_prototype_size": gold_proto_size,
            "original_pack_positive_multiplicity": original_pack_multiplicity,
        }
        if hypothesis == "H0":
            h0.append(manifest)
        else:
            h1.append(manifest)
        slices[multiplicity_bucket].append(manifest)
        unique_proto_counts.append(unique_proto_count)
        max_proto_multiplicities.append(max_proto_multiplicity)
        gold_proto_sizes.append(gold_proto_size)
        positive_multiplicities.append(positive_multiplicity)
        status_by_query[query_id] = {
            "status": hypothesis,
            "gold_id": gold_id,
            "positive_multiplicity": positive_multiplicity,
            "candidate_signature": candidate_signature,
        }

    valid_count = counts["valid_queries"]
    h1_count = counts["h1"]
    stats = {
        "path": display_path(path),
        "expected_candidate_count": expected_count,
        "schema": schema,
        "query_count": counts["queries"],
        "valid_query_count": valid_count,
        "h0_count": counts["h0"],
        "h1_count": h1_count,
        "h1_coverage": h1_count / valid_count if valid_count else None,
        "positive_prototype_multiplicity": {
            "1": counts["multiplicity_1"],
            "2": counts["multiplicity_2"],
            ">=3": counts["multiplicity_>=3"],
            "histogram": histogram(positive_multiplicities),
        },
        "h1_same_prototype_distractors": {
            "1": counts["h1_distractors_1"],
            "2": counts["h1_distractors_2"],
            ">=3": counts["h1_distractors_>=3"],
        },
        "unique_prototype_count_per_query": {
            "summary": numeric_summary(unique_proto_counts),
            "histogram": histogram(unique_proto_counts),
        },
        "max_candidate_prototype_multiplicity_per_query": {
            "summary": numeric_summary(max_proto_multiplicities),
            "histogram": histogram(max_proto_multiplicities),
        },
        "gold_full_prototype_size_per_query": {
            "summary": numeric_summary(gold_proto_sizes),
            "histogram": histogram(gold_proto_sizes),
        },
        "original_pack_positive_multiplicity": {
            "available": bool(original_pack_by_sticker),
            "h1_count": counts["original_pack_h1_queries"],
            "histogram": histogram(original_pack_multiplicities),
        },
        "h1_created_by_original_pack_merge_count": counts["h1_created_by_pack_merge"],
        "anomalies": {
            "duplicate_candidates": {
                "query_count": counts["duplicate_candidate_queries"],
                "duplicate_excess": counts["duplicate_candidate_excess"],
            },
            "gold_not_in_candidates": {"query_count": counts["gold_not_in_candidates_queries"]},
            "missing_prototype_mapping": {
                "query_count": counts["missing_mapping_queries"],
                "id_occurrences": counts["missing_mapping_occurrences"],
            },
            "wrong_candidate_count": {"query_count": counts["wrong_candidate_count_queries"]},
            "illegal_sticker_id": {
                "query_count": counts["illegal_id_queries"],
                "id_occurrences": counts["illegal_id_occurrences"],
            },
            "examples": examples,
        },
    }
    return stats, h0, h1, slices, status_by_query


def parse_yaml_scalar(text: str) -> Any:
    value = text.strip()
    if not value:
        return ""
    if value.startswith(("'", '"')):
        try:
            return json.loads(value) if value.startswith('"') else value.strip("'")
        except json.JSONDecodeError:
            return value.strip("\"'")
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("null", "none", "~"):
        return None
    if re.fullmatch(r"[-+]?[0-9]+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+)", value):
        return float(value)
    return value


def parse_simple_yaml(path: Path) -> Dict[str, Any]:
    """Parse this repository's top-level scalar config keys without PyYAML."""
    values: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line or line[0].isspace() or line.lstrip().startswith("#"):
                continue
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):(?:\s*(.*?))?\s*$", line.rstrip("\n"))
            if not match:
                continue
            key, raw = match.group(1), (match.group(2) or "")
            if raw in (">", "|"):
                continue
            if " #" in raw:
                raw = raw.split(" #", 1)[0].rstrip()
            values[key] = parse_yaml_scalar(raw)
    return values


def resolve_config(path: Path, seen: Optional[List[Path]] = None) -> Tuple[Dict[str, Any], List[Path]]:
    path = require_input_file(path, "config")
    chain = list(seen or [])
    if path in chain:
        raise ValueError(f"Config inheritance cycle: {' -> '.join(map(str, chain + [path]))}")
    chain.append(path)
    local = parse_simple_yaml(path)
    parent_name = local.get("extends")
    effective: Dict[str, Any] = {}
    resolved_chain: List[Path] = []
    if parent_name:
        parent = Path(str(parent_name))
        if not parent.suffix:
            parent = parent.with_suffix(".yaml")
        if not parent.is_absolute():
            parent = path.parent / parent
        effective, resolved_chain = resolve_config(parent, chain)
    effective.update(local)
    resolved_chain.append(path)
    return effective, resolved_chain


def resolve_project_value(value: Any, config_path: Path) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(value)
    if p.is_absolute():
        return resolved(p)
    # Project configs are relative to the working tree, not to configs/.
    cwd_candidate = resolved(Path.cwd() / p)
    if cwd_candidate.exists() or str(value).startswith("."):
        return cwd_candidate
    return resolved(config_path.parent / p)


def trace_config(config_path: Path, expected: Mapping[str, Path]) -> Dict[str, Any]:
    effective, chain = resolve_config(config_path)
    fields = (
        "name",
        "pl_root_dir",
        "train_data_path",
        "test_data_path",
        "per_epoch_eval_test_r10_path",
        "per_epoch_eval_test_r20_path",
        "factorized_bank_path",
        "style_neighbors_path",
        "factorized_style_metadata_path",
    )
    selected = {key: effective.get(key) for key in fields}
    resolved_fields: Dict[str, Optional[str]] = {}
    path_matches: Dict[str, Optional[bool]] = {}
    for key in fields[1:]:
        p = resolve_project_value(effective.get(key), config_path)
        resolved_fields[key] = display_path(p) if p is not None else None
        if key in expected and p is not None:
            path_matches[key] = p == resolved(expected[key])

    log_path = resolve_project_value(effective.get("pl_root_dir"), config_path)
    checkpoints: List[Dict[str, Any]] = []
    hparams: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    if log_path and log_path.is_dir():
        checkpoints = [file_info(p) for p in sorted(log_path.rglob("*.ckpt"))]
        hparams = [file_info(p) for p in sorted(log_path.rglob("hparams.yaml"))]
        events = [file_info(p) for p in sorted(log_path.rglob("events.out*"))]
    final_checkpoints = [row for row in checkpoints if Path(row["path"]).name == "final.ckpt"]
    latest_final = max(final_checkpoints, key=lambda row: row["mtime_utc"]) if final_checkpoints else None
    return {
        "config_path": display_path(config_path),
        "inheritance_chain": [display_path(p) for p in chain],
        "effective_fields": selected,
        "resolved_paths": resolved_fields,
        "expected_path_matches": path_matches,
        "log_dir_exists": bool(log_path and log_path.is_dir()),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "latest_final_checkpoint": latest_final,
        "hparams_files": hparams,
        "event_files": events,
    }


def compare_candidate_sets(
    primary: Mapping[str, Mapping[str, Dict[str, Any]]],
    comparison: Mapping[str, Mapping[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for protocol, _, _ in PROTOCOLS:
        left = primary[protocol]
        right = comparison[protocol]
        shared = sorted(set(left) & set(right))
        transitions = Counter()
        changed_candidates = 0
        gold_mismatches = 0
        for query_id in shared:
            lrow, rrow = left[query_id], right[query_id]
            transitions[f"{lrow['status']}->{rrow['status']}"] += 1
            if lrow.get("candidate_signature") != rrow.get("candidate_signature"):
                changed_candidates += 1
            if lrow.get("gold_id") != rrow.get("gold_id"):
                gold_mismatches += 1
        output[protocol] = {
            "primary_query_count": len(left),
            "comparison_query_count": len(right),
            "aligned_query_count": len(shared),
            "only_primary_query_count": len(set(left) - set(right)),
            "only_comparison_query_count": len(set(right) - set(left)),
            "gold_id_mismatch_count": gold_mismatches,
            "candidate_list_changed_count": changed_candidates,
            "candidate_list_changed_rate": changed_candidates / len(shared) if shared else None,
            "h0_h1_transitions": dict(sorted(transitions.items())),
        }
    return output


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            f.write("\n")


def pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def feasibility_label(stats: Mapping[str, Any]) -> str:
    coverage = stats.get("h1_coverage")
    h1_count = int(stats.get("h1_count", 0))
    if coverage is not None and coverage >= 0.15 and h1_count >= 200:
        return "meets threshold"
    if coverage is not None and coverage < 0.15:
        return "coverage below 15%"
    return "fewer than 200 H1 queries"


def render_report(stats: Mapping[str, Any]) -> str:
    lines = [
        "# StickerChat 384 H1 audit",
        "",
        "| Split/Protocol | Queries | H0 | H1 | H1 coverage | m=1 | m=2 | m>=3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_label, dataset in stats["datasets"].items():
        for protocol, protocol_stats in dataset["protocols"].items():
            mult = protocol_stats["positive_prototype_multiplicity"]
            lines.append(
                f"| {dataset_label}/{protocol} | {protocol_stats['query_count']} | "
                f"{protocol_stats['h0_count']} | {protocol_stats['h1_count']} | "
                f"{pct(protocol_stats['h1_coverage'])} | {mult['1']} | {mult['2']} | {mult['>=3']} |"
            )

    bank = stats["bank"]
    psize = bank["prototype_size"]
    lines.extend(
        [
            "",
            "## Audited prototype assets",
            "",
            f"- Bank: `{bank['path']}`",
            f"- Mapping schema: `{bank['schema']['member_id_field']} -> {bank['schema']['prototype_id_field']}`",
            f"- Requested K / actual non-empty: {bank['requested_k']} / {bank['actual_nonempty_prototypes']}",
            f"- Stickers / original packs: {bank['total_mapped_stickers']} / {bank['original_pack_count']}",
            f"- K-means seed / iterations: {bank['kmeans_seed']} / {bank['kmeans_iters']}",
            f"- CLIP embedding cache: `{bank['input_clip_embedding_cache']}`",
            "- Prototype size min / median / mean / max: "
            f"{psize['min']} / {psize['median']} / {psize['mean']:.4f} / {psize['max']}",
            "",
            "The adjacent `sticker_metadata.json`, `img_set_to_ids.json`, "
            "`style_regroup_analysis.json`, and `summary.json` were loaded and cross-checked "
            "against the bank mapping; any mismatch would have stopped this audit.",
        ]
    )

    if stats.get("config_traces"):
        lines.extend(["", "## Config → bank/candidates → log/checkpoint trace", ""])
        for label, trace in stats["config_traces"].items():
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"- Config: `{trace['config_path']}`")
            lines.append(f"- Effective name: `{trace['effective_fields'].get('name')}`")
            lines.append(f"- Effective bank: `{trace['resolved_paths'].get('factorized_bank_path')}`")
            lines.append(
                f"- Effective R10/R20 candidates: `{trace['resolved_paths'].get('per_epoch_eval_test_r10_path')}` / "
                f"`{trace['resolved_paths'].get('per_epoch_eval_test_r20_path')}`"
            )
            lines.append(f"- Log directory: `{trace['resolved_paths'].get('pl_root_dir')}`")
            latest = trace.get("latest_final_checkpoint")
            lines.append(
                f"- Latest final checkpoint: `{latest['path']}`" if latest else "- Latest final checkpoint: not found in this log directory"
            )
            matches = trace.get("expected_path_matches", {})
            if matches:
                lines.append(f"- Explicit input-path matches: `{json.dumps(matches, sort_keys=True)}`")
            lines.append("")

    lines.extend(["## H1 interpretation", ""])
    for dataset_label, dataset in stats["datasets"].items():
        lines.append(f"### {dataset_label}")
        lines.append("")
        for protocol, protocol_stats in dataset["protocols"].items():
            lines.append(
                f"- {protocol}: H1={protocol_stats['h1_count']} "
                f"({pct(protocol_stats['h1_coverage'])}); {feasibility_label(protocol_stats)}."
            )
        lines.append("")

    primary_label = stats["primary_dataset_label"]
    primary_protocols = stats["datasets"][primary_label]["protocols"]
    primary_pass = [
        s["h1_coverage"] is not None and s["h1_coverage"] >= 0.15 and s["h1_count"] >= 200
        for s in primary_protocols.values()
    ]
    coverages = [s["h1_coverage"] for s in primary_protocols.values() if s["h1_coverage"] is not None]
    h1_total = sum(s["h1_count"] for s in primary_protocols.values())
    m2_total = sum(s["positive_prototype_multiplicity"]["2"] for s in primary_protocols.values())
    m3_total = sum(s["positive_prototype_multiplicity"][">=3"] for s in primary_protocols.values())

    lines.extend(["## Answers", ""])
    if all(primary_pass):
        lines.append("1. **H1 sufficiency:** yes for every primary protocol under the stated 15% / 200-query rule.")
    else:
        lines.append(
            "1. **H1 sufficiency:** no for the primary candidate set; at least one protocol fails "
            "the stated 15% / 200-query rule."
        )
    if coverages and max(coverages) - min(coverages) <= 0.05 and len(set(primary_pass)) == 1:
        lines.append("2. **Validation/test and R10/R20 consistency:** qualitatively consistent under the feasibility rule.")
    else:
        lines.append("2. **Validation/test and R10/R20 consistency:** not fully consistent; inspect the protocol rows above.")
    if h1_total == 0:
        lines.append("3. **Multiplicity balance:** no primary H1 population exists, so an H1 balance analysis is not meaningful.")
    else:
        dominant = max(m2_total, m3_total) / h1_total
        severity = "severely imbalanced" if dominant >= 0.8 else "not severely imbalanced by the 80% dominance rule"
        lines.append(f"3. **Multiplicity balance:** {severity} (m=2: {m2_total}; m>=3: {m3_total}).")
    lines.append(
        f"4. **Main-experiment assets:** the primary rows above use `{primary_label}` with the audited K384 bank; "
        "the config trace states the exact effective paths."
    )
    if all(primary_pass):
        lines.append("5. **Flat/Real/Permuted:** the primary audit supports proceeding to the three-arm evaluation.")
    else:
        lines.append(
            "5. **Flat/Real/Permuted:** not recommended on the primary fixed candidates because the H1 feasibility threshold is not met."
        )

    if stats.get("candidate_comparison"):
        lines.extend(
            [
                "",
                "## Candidate-generation confounding",
                "",
                "Both candidate generations were audited with the identical K384 sticker→prototype mapping. "
                "`candidate_list_changed_rate` and the per-query H0/H1 transition counts in `stats.json` "
                "therefore isolate candidate resampling effects. `h1_created_by_original_pack_merge_count` "
                "counts H1 queries whose candidates contain only one sticker from the gold original pack, "
                "so their H1 status is attributable to merging distinct original packs into one K384 prototype.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    primary_paths = {
        "val_r10": require_input_file(args.val_r10_path, "--val-r10-path"),
        "val_r20": require_input_file(args.val_r20_path, "--val-r20-path"),
        "test_r10": require_input_file(args.test_r10_path, "--test-r10-path"),
        "test_r20": require_input_file(args.test_r20_path, "--test-r20-path"),
    }
    bank_path = require_input_file(args.bank_path, "--bank-path")

    comparison_args = (
        args.comparison_val_r10_path,
        args.comparison_val_r20_path,
        args.comparison_test_r10_path,
        args.comparison_test_r20_path,
    )
    if any(x is not None for x in comparison_args) and not all(x is not None for x in comparison_args):
        raise ValueError("All four --comparison-*-path arguments must be supplied together")
    comparison_paths: Optional[Dict[str, Path]] = None
    if all(x is not None for x in comparison_args):
        comparison_paths = {
            "val_r10": require_input_file(args.comparison_val_r10_path, "--comparison-val-r10-path"),
            "val_r20": require_input_file(args.comparison_val_r20_path, "--comparison-val-r20-path"),
            "test_r10": require_input_file(args.comparison_test_r10_path, "--comparison-test-r10-path"),
            "test_r20": require_input_file(args.comparison_test_r20_path, "--comparison-test-r20-path"),
        }

    all_inputs: List[Path] = [bank_path, *primary_paths.values()]
    if comparison_paths:
        all_inputs.extend(comparison_paths.values())
    if args.primary_config:
        all_inputs.append(require_input_file(args.primary_config, "--primary-config"))
    if args.comparison_config:
        all_inputs.append(require_input_file(args.comparison_config, "--comparison-config"))
    output_dir = validate_output_scope(args.output_dir, all_inputs)

    sticker_to_proto, proto_sizes, original_pack_by_sticker, bank_stats = analyze_bank(bank_path)
    datasets: Dict[str, Any] = {}
    all_h0: List[Dict[str, Any]] = []
    all_h1: List[Dict[str, Any]] = []
    all_slices: Dict[str, Any] = {"datasets": {}}
    status_maps: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

    bundles: List[Tuple[str, Dict[str, Path], str]] = [
        (args.primary_label, primary_paths, "primary/main-config candidates")
    ]
    if comparison_paths:
        bundles.append((args.comparison_label, comparison_paths, "comparison candidates"))

    for dataset_label, paths, role in bundles:
        protocol_stats: Dict[str, Any] = {}
        dataset_slices: Dict[str, Any] = {}
        dataset_status: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for protocol, path_key, expected_count in PROTOCOLS:
            result, h0, h1, slices, status = audit_candidate_file(
                paths[path_key],
                dataset_label,
                protocol,
                expected_count,
                sticker_to_proto,
                proto_sizes,
                original_pack_by_sticker,
            )
            protocol_stats[protocol] = result
            dataset_slices[protocol] = slices
            dataset_status[protocol] = status
            all_h0.extend(h0)
            all_h1.extend(h1)
        datasets[dataset_label] = {
            "role": role,
            "paths": {key: display_path(value) for key, value in paths.items()},
            "protocols": protocol_stats,
        }
        all_slices["datasets"][dataset_label] = dataset_slices
        status_maps[dataset_label] = dataset_status

    config_traces: Dict[str, Any] = {}
    if args.primary_config:
        config_traces["primary_main_experiment"] = trace_config(
            resolved(args.primary_config),
            {
                "factorized_bank_path": bank_path,
                "per_epoch_eval_test_r10_path": primary_paths["test_r10"],
                "per_epoch_eval_test_r20_path": primary_paths["test_r20"],
                "test_data_path": primary_paths["test_r10"],
            },
        )
    if args.comparison_config and comparison_paths:
        config_traces["comparison_legacy_evaluation"] = trace_config(
            resolved(args.comparison_config),
            {
                "factorized_bank_path": bank_path,
                "per_epoch_eval_test_r10_path": comparison_paths["test_r10"],
                "per_epoch_eval_test_r20_path": comparison_paths["test_r20"],
                "test_data_path": comparison_paths["test_r10"],
            },
        )

    comparison_stats = None
    if comparison_paths:
        comparison_stats = compare_candidate_sets(
            status_maps[args.primary_label], status_maps[args.comparison_label]
        )

    stats: Dict[str, Any] = {
        "audit_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_contract": {
            "input_files_opened_for_writing": False,
            "output_dir": display_path(output_dir),
            "outputs": sorted(OUTPUT_FILENAMES),
        },
        "bank": bank_stats,
        "primary_dataset_label": args.primary_label,
        "datasets": datasets,
        "candidate_comparison": comparison_stats,
        "config_traces": config_traces,
    }

    write_json(output_dir / "stats.json", stats)
    write_jsonl(output_dir / "h0_ids.jsonl", all_h0)
    write_jsonl(output_dir / "h1_ids.jsonl", all_h1)
    write_json(output_dir / "multiplicity_slices.json", all_slices)
    with (output_dir / "StickerChat_384_H1_audit.md").open("w", encoding="utf-8") as f:
        f.write(render_report(stats))
    print(json.dumps({
        "output_dir": display_path(output_dir),
        "bank_prototypes": bank_stats["actual_nonempty_prototypes"],
        "mapped_stickers": bank_stats["total_mapped_stickers"],
        "datasets": {
            label: {
                protocol: {
                    "queries": item["query_count"],
                    "valid": item["valid_query_count"],
                    "h0": item["h0_count"],
                    "h1": item["h1_count"],
                    "h1_coverage": item["h1_coverage"],
                }
                for protocol, item in data["protocols"].items()
            }
            for label, data in datasets.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (SchemaError, ValueError, FileNotFoundError) as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
