"""Code-compatible StickerChat evaluation assets for the released PBR bug.

This module is intentionally audit-only. It reproduces the released PBR
candidate construction bug without changing any training or formal evaluation
protocol:

* candidate slot 0 is the gold sticker;
* the first nine ``emoji_mapping.txt`` IDs are appended without removing gold,
  because the released code checks ``"<gold>.npy"`` against extension-free IDs;
* short packs are padded with the same RGB=127 gray sentinel used by the clean
  fixed-candidate implementation.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np

from .fixed_same_pack import CANDIDATE_COUNT, GRAY_SENTINEL_ID, normalize_external_id
from .io import atomic_write_json, canonical_json, hash_value, sha256_file


PBR_BUG_POLICY = "pbr_released_code_bug_compatible_r10"
PBR_BUG_SCHEMA = "style_shapes.stickerchat_pbr_bug_compatible_r10.v1"


def parse_pbr_mapping_order(text: str) -> List[str]:
    """Match ``emojis[key] = value; list(emojis.keys())`` in PBR exactly."""
    ordered: Dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            raise ValueError(
                "invalid PBR emoji_mapping.txt line %d: %r"
                % (line_number, raw_line)
            )
        sticker = normalize_external_id(fields[0])
        # Updating an existing Python dict key preserves its first insertion
        # position, which is exactly what the released PBR loader does.
        ordered[sticker] = fields[1]
    return list(ordered.keys())


def pbr_bug_candidates_for_gold(
    pack_id: Any,
    gold_external_id: Any,
    ordered_pack_stickers: Sequence[Any],
    img2id: Mapping[str, Any],
    negative_count: int = 9,
) -> Tuple[List[int], List[bool]]:
    """Reproduce the released suffix mismatch and gray padding."""
    pack = normalize_external_id(pack_id)
    gold = normalize_external_id(gold_external_id)
    gold_key = "%s-%s" % (pack, gold)
    if gold_key not in img2id:
        raise KeyError("missing internal mapping for gold %s" % gold_key)
    gold_internal = int(img2id[gold_key])

    # Released PBR code effectively does:
    #   negative_ids = mapping_keys[:]
    #   if gold + ".npy" in negative_ids: remove(...)
    # The mapping keys have no suffix, so the condition is false.
    negatives: List[int] = []
    for value in list(ordered_pack_stickers)[: int(negative_count)]:
        external = normalize_external_id(value)
        key = "%s-%s" % (pack, external)
        if key not in img2id:
            raise KeyError("missing internal mapping for PBR candidate %s" % key)
        negatives.append(int(img2id[key]))

    candidates = [gold_internal] + negatives
    candidates.extend(
        [GRAY_SENTINEL_ID]
        * (1 + int(negative_count) - len(candidates))
    )
    if len(candidates) != 1 + int(negative_count):
        raise AssertionError("PBR bug-compatible candidate width changed")
    gray_mask = [value == GRAY_SENTINEL_ID for value in candidates]
    if candidates[0] != gold_internal or gray_mask[0]:
        raise AssertionError("PBR gold must remain in candidate slot 0")
    return candidates, gray_mask


def _release_member(archive: zipfile.ZipFile, split: str) -> str:
    suffix = "/release_%s.json" % split
    exact = "stickerchat/release_%s.json" % split
    if exact in archive.namelist():
        return exact
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError("cannot uniquely resolve raw %s release file" % split)
    return matches[0]


def _release_rows(
    archive: zipfile.ZipFile, member: str
) -> Iterator[Tuple[int, Mapping[str, Any]]]:
    with archive.open(member, "r") as handle:
        source_row = 0
        for raw in handle:
            if not raw.strip():
                continue
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict) or not isinstance(
                value.get("current"), dict
            ):
                raise ValueError(
                    "%s row %d has invalid release schema"
                    % (member, source_row)
                )
            yield source_row, value
            source_row += 1


def load_pbr_pack_orders(
    archive: zipfile.ZipFile,
) -> Tuple[Dict[str, List[str]], str, List[str]]:
    """Load mapping keys exactly as PBR does, including empty mappings."""
    orders: Dict[str, List[str]] = {}
    hash_rows = []
    empty = []
    for member in sorted(
        name
        for name in archive.namelist()
        if name.endswith("/emoji_mapping.txt")
    ):
        parts = member.rstrip("/").split("/")
        if len(parts) < 2:
            raise ValueError("cannot infer pack ID from %s" % member)
        pack = normalize_external_id(parts[-2])
        if pack in orders:
            raise ValueError("duplicate emoji mapping for pack %s" % pack)
        order = parse_pbr_mapping_order(
            archive.read(member).decode("utf-8")
        )
        orders[pack] = order
        if not order:
            empty.append(pack)
        hash_rows.append((pack, order))
    if not orders:
        raise ValueError("archive contains no emoji_mapping.txt files")
    return orders, hash_value(hash_rows), sorted(empty, key=int)


def build_pbr_bug_split(
    archive: zipfile.ZipFile,
    split: str,
    pack_orders: Mapping[str, Sequence[str]],
    img2id: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    member = _release_member(archive, split)
    rows: List[Dict[str, Any]] = []
    duplicate_gold_rows = 0
    duplicate_gold_slots = 0
    gray_rows = 0
    gray_slots = 0
    gold_absent_from_mapping = 0
    for source_row, raw_row in _release_rows(archive, member):
        current = raw_row["current"]
        pack = normalize_external_id(current["sticker_set_id"])
        gold_external = normalize_external_id(current["sticker_id"])
        if pack not in pack_orders:
            raise KeyError("raw row %d references missing pack %s" % (source_row, pack))
        order = pack_orders[pack]
        candidates, gray_mask = pbr_bug_candidates_for_gold(
            pack, gold_external, order, img2id
        )
        gold_internal = int(candidates[0])
        duplicate_slots = sum(
            int(value == gold_internal) for value in candidates[1:]
        )
        duplicate_gold_rows += int(duplicate_slots > 0)
        duplicate_gold_slots += duplicate_slots
        row_gray = sum(bool(value) for value in gray_mask)
        gray_rows += int(row_gray > 0)
        gray_slots += row_gray
        gold_absent_from_mapping += int(gold_external not in set(order))
        rows.append(
            {
                "source_row": int(source_row),
                "pack_id": pack,
                "gold_external_id": gold_external,
                "gold_internal_id": gold_internal,
                "candidate_ids": [int(value) for value in candidates],
                "gray_mask": [bool(value) for value in gray_mask],
                "positive_index": 0,
                "duplicate_gold_negative_indices": [
                    index
                    for index, value in enumerate(candidates)
                    if index > 0 and int(value) == gold_internal
                ],
            }
        )
    stats = {
        "rows": len(rows),
        "candidate_size": CANDIDATE_COUNT,
        "duplicate_gold_rows": int(duplicate_gold_rows),
        "duplicate_gold_slots": int(duplicate_gold_slots),
        "gray_rows": int(gray_rows),
        "gray_slots": int(gray_slots),
        "gold_absent_from_mapping_rows": int(gold_absent_from_mapping),
        "raw_release_member": member,
    }
    return rows, stats


def _atomic_write_jsonl(path: os.PathLike, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".%s." % target.name, dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_pbr_bug_manifest(
    manifest: Mapping[str, Any], verify_files: bool = True
) -> None:
    if manifest.get("schema_version") != PBR_BUG_SCHEMA:
        raise ValueError("unsupported PBR bug-compatible manifest schema")
    if manifest.get("protocol") != PBR_BUG_POLICY:
        raise ValueError("PBR bug-compatible policy mismatch")
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("manifest_hash") != hash_value(core):
        raise ValueError("PBR bug-compatible manifest hash mismatch")
    if verify_files:
        for metadata in manifest["splits"].values():
            for name in ("rows", "evaluation_data"):
                entry = metadata[name]
                if not Path(entry["path"]).exists():
                    raise FileNotFoundError(entry["path"])
                if sha256_file(entry["path"]) != entry["sha256"]:
                    raise ValueError("PBR bug-compatible %s hash mismatch" % name)
        gray = manifest["gray_embedding"]
        if not Path(gray["path"]).exists():
            raise FileNotFoundError(gray["path"])
        if sha256_file(gray["path"]) != gray["sha256"]:
            raise ValueError("PBR bug-compatible gray embedding hash mismatch")


def write_pbr_bug_eval_assets(
    zip_path: str,
    img2id_path: str,
    processed_validation_path: str,
    processed_test_path: str,
    gray_embedding_path: str,
    output_dir: str,
    validation_output_path: str,
    test_output_path: str,
) -> Dict[str, Any]:
    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        validate_pbr_bug_manifest(existing, verify_files=True)
        return dict(existing)

    with open(img2id_path, "r", encoding="utf-8") as handle:
        img2id = json.load(handle)
    with zipfile.ZipFile(zip_path, "r") as archive:
        pack_orders, mapping_hash, empty_packs = load_pbr_pack_orders(archive)
        built = {
            "validation": build_pbr_bug_split(
                archive, "val", pack_orders, img2id
            ),
            "test": build_pbr_bug_split(
                archive, "test", pack_orders, img2id
            ),
        }

    processed_paths = {
        "validation": processed_validation_path,
        "test": processed_test_path,
    }
    output_paths = {
        "validation": validation_output_path,
        "test": test_output_path,
    }
    split_manifest = {}
    for split in ("validation", "test"):
        rows, stats = built[split]
        if stats["rows"] != 10000:
            raise RuntimeError(
                "%s PBR bug-compatible row count changed: %d"
                % (split, stats["rows"])
            )
        with open(processed_paths[split], "r", encoding="utf-8") as handle:
            processed = json.load(handle)
        if len(processed) != len(rows):
            raise ValueError("%s processed/raw row count mismatch" % split)
        for index, (processed_row, bug_row) in enumerate(zip(processed, rows)):
            dialogue = processed_row.get("dialog")
            if not isinstance(dialogue, list) or not dialogue:
                raise ValueError("%s processed row %d has no dialogue" % (split, index))
            if int(dialogue[-1]["img_id"]) != int(bug_row["gold_internal_id"]):
                raise ValueError("%s row %d gold alignment mismatch" % (split, index))
            processed_row["cand"] = list(bug_row["candidate_ids"])
            processed_row["gray_mask"] = list(bug_row["gray_mask"])
            processed_row["pbr_bug_compatible_protocol"] = PBR_BUG_POLICY
            processed_row["positive_index"] = 0
        atomic_write_json(output_paths[split], processed)
        rows_path = output / ("%s.jsonl" % split)
        _atomic_write_jsonl(rows_path, rows)
        split_manifest[split] = {
            "stats": stats,
            "rows": {
                "path": str(rows_path),
                "sha256": sha256_file(rows_path),
            },
            "evaluation_data": {
                "path": str(output_paths[split]),
                "sha256": sha256_file(output_paths[split]),
                "source_path": processed_paths[split],
                "source_sha256": sha256_file(processed_paths[split]),
            },
        }

    core = {
        "schema_version": PBR_BUG_SCHEMA,
        "protocol": PBR_BUG_POLICY,
        "audit_only": True,
        "candidate_count": CANDIDATE_COUNT,
        "positive_index": 0,
        "gray_sentinel_id": GRAY_SENTINEL_ID,
        "selection_policy": (
            "candidate 0 is gold; emulate released PBR suffix mismatch by "
            "checking <gold>.npy against extension-free mapping keys, therefore "
            "do not remove gold; take the first nine mapping keys; RGB=127 "
            "gray-pad the shortfall"
        ),
        "inputs": {
            "raw_zip": {"path": zip_path, "sha256": sha256_file(zip_path)},
            "img2id": {"path": img2id_path, "sha256": sha256_file(img2id_path)},
            "emoji_mapping_content_hash": mapping_hash,
            "pack_count": len(pack_orders),
            "empty_mapping_packs": empty_packs,
        },
        "gray_embedding": {
            "path": gray_embedding_path,
            "sha256": sha256_file(gray_embedding_path),
            "rgb": [127, 127, 127],
        },
        "splits": split_manifest,
    }
    manifest = {**core, "manifest_hash": hash_value(core)}
    atomic_write_json(manifest_path, manifest)
    validate_pbr_bug_manifest(manifest, verify_files=True)
    return manifest


def pbr_released_metrics(score_rows: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Compute the released PBR single-positive metrics, including tie behavior."""
    scores = np.asarray(score_rows, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != CANDIDATE_COUNT:
        raise ValueError("PBR metric scores must have shape [Q,10]")
    if scores.shape[0] == 0 or not np.isfinite(scores).all():
        raise ValueError("PBR metric scores must be non-empty and finite")
    indices = np.argsort(-scores, axis=1)
    locations = np.argwhere(indices == 0)
    if locations.shape[0] != scores.shape[0]:
        raise AssertionError("each query must rank positive slot 0 exactly once")
    ranks = np.empty(scores.shape[0], dtype=np.int64)
    ranks[locations[:, 0]] = locations[:, 1] + 1
    return {
        "queries": int(scores.shape[0]),
        "r_at_1": float(np.mean(ranks <= 1)),
        "r_at_2": float(np.mean(ranks <= 2)),
        "r_at_5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "map": float(np.mean(1.0 / ranks)),
        # This odd metric in the released code compares the positive only with
        # candidate slot 1, not with the hardest negative.
        "released_r2_at_1": float(np.mean(scores[:, 0] > scores[:, 1])),
        "rank_distribution": {
            str(rank): int(np.sum(ranks == rank))
            for rank in range(1, CANDIDATE_COUNT + 1)
        },
        "positive_slot": 0,
        "tie_order": "numpy.argsort(-scores, axis=1), matching released metrics.py",
    }
