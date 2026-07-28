"""Frozen StickerChat same-pack R10 candidates and listwise-loss helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    hash_value,
    sha256_file,
)


FIXED_SAME_PACK_POLICY = "fixed_same_pack_listwise"
FIXED_SAME_PACK_SCHEMA = "style_shapes.stickerchat_fixed_same_pack_r10.v1"
GRAY_SENTINEL_ID = -1
CANDIDATE_COUNT = 10
EXPECTED_SPLIT_STATS = {
    "train": {"rows": 320168, "gray_rows": 5803, "gray_slots": 27828},
    "validation": {"rows": 10000, "gray_rows": 163, "gray_slots": 826},
    "test": {"rows": 10000, "gray_rows": 164, "gray_slots": 776},
}


def normalize_external_id(value: Any) -> str:
    """Normalize raw StickerChat IDs without changing their integer identity."""
    text = str(value).strip()
    if text.endswith(".npy"):
        text = text[:-4]
    if not text or not text.lstrip("-").isdigit():
        raise ValueError("invalid StickerChat external ID: %r" % value)
    return str(int(text))


def parse_emoji_mapping(text: str, allow_empty: bool = False) -> List[str]:
    """Read sticker IDs in the exact order recorded by emoji_mapping.txt."""
    result: List[str] = []
    seen = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        sticker = normalize_external_id(line.split("\t", 1)[0])
        if sticker in seen:
            raise ValueError(
                "duplicate sticker %s in emoji_mapping.txt line %d"
                % (sticker, line_number)
            )
        seen.add(sticker)
        result.append(sticker)
    if not result and not allow_empty:
        raise ValueError("emoji_mapping.txt has no stickers")
    return result


def fixed_candidates_for_gold(
    pack_id: Any,
    gold_external_id: Any,
    ordered_pack_stickers: Sequence[Any],
    img2id: Mapping[str, Any],
    negative_count: int = 9,
    gray_sentinel: int = GRAY_SENTINEL_ID,
) -> Tuple[List[int], List[bool]]:
    """Build gold + first N mapping-ordered negatives, then gray-pad."""
    pack = normalize_external_id(pack_id)
    gold = normalize_external_id(gold_external_id)
    ordered = [normalize_external_id(item) for item in ordered_pack_stickers]
    if gold not in set(ordered):
        raise ValueError("gold %s is absent from pack %s emoji mapping" % (gold, pack))
    gold_key = "%s-%s" % (pack, gold)
    if gold_key not in img2id:
        raise KeyError("missing internal mapping for gold %s" % gold_key)
    gold_internal = int(img2id[gold_key])

    negatives: List[int] = []
    for external in ordered:
        if external == gold:
            continue
        key = "%s-%s" % (pack, external)
        if key not in img2id:
            raise KeyError("missing internal mapping for candidate %s" % key)
        negatives.append(int(img2id[key]))
        if len(negatives) == int(negative_count):
            break
    candidates = [gold_internal] + negatives
    candidates.extend([int(gray_sentinel)] * (1 + int(negative_count) - len(candidates)))
    gray_mask = [item == int(gray_sentinel) for item in candidates]
    validate_candidate_row(candidates, gray_mask, gold_internal, 1 + int(negative_count))
    return candidates, gray_mask


def validate_candidate_row(
    candidates: Sequence[int],
    gray_mask: Sequence[bool],
    gold_internal_id: int,
    candidate_count: int = CANDIDATE_COUNT,
) -> None:
    if len(candidates) != int(candidate_count) or len(gray_mask) != int(candidate_count):
        raise ValueError("fixed candidate row must contain exactly %d slots" % candidate_count)
    if int(candidates[0]) != int(gold_internal_id):
        raise ValueError("gold must be candidate 0")
    if bool(gray_mask[0]):
        raise ValueError("gold candidate cannot be gray")
    if [int(item) == GRAY_SENTINEL_ID for item in candidates] != [
        bool(item) for item in gray_mask
    ]:
        raise ValueError("gray mask does not match sentinel IDs")
    real = [int(item) for item in candidates if int(item) != GRAY_SENTINEL_ID]
    if len(real) != len(set(real)):
        raise ValueError("real fixed candidates must be unique")
    if GRAY_SENTINEL_ID in real:
        raise ValueError("gray sentinel entered the real candidate set")


def flatten_query_major(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[B,L] + [B,N] -> query-major [B*N,L] and [B*N]."""
    if candidate_ids.ndim != 2:
        raise ValueError("candidate_ids must have shape [B,N]")
    batch, count = candidate_ids.shape
    if input_ids.size(0) != batch or attention_mask.size(0) != batch:
        raise ValueError("dialogue and candidate batch dimensions are not aligned")
    return (
        input_ids.repeat_interleave(count, dim=0),
        attention_mask.repeat_interleave(count, dim=0),
        candidate_ids.reshape(-1),
    )


def listwise_match_loss(scores: torch.Tensor) -> torch.Tensor:
    """Ten-way (or general N-way) CE with the gold fixed at index 0."""
    if scores.ndim != 2 or scores.size(1) < 2:
        raise ValueError("listwise scores must have shape [B,N] with N >= 2")
    labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, labels)


def hardest_expression_rank_loss(
    expression_scores: torch.Tensor, margin: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Margin loss against only the highest-scoring expression negative."""
    if expression_scores.ndim != 2 or expression_scores.size(1) < 2:
        raise ValueError("expression scores must have shape [B,N] with N >= 2")
    hardest_value, hardest_offset = expression_scores[:, 1:].max(dim=1)
    loss = F.relu(float(margin) - expression_scores[:, 0] + hardest_value).mean()
    return loss, hardest_offset + 1


def _atomic_torch_save(path: os.PathLike, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".%s." % target.name, dir=str(target.parent))
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_jsonl(path: os.PathLike, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".%s." % target.name, dir=str(target.parent))
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


def _zip_member_sha256(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member, "r") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_release_rows(
    archive: zipfile.ZipFile, member: str
) -> Iterator[Tuple[int, Mapping[str, Any]]]:
    with archive.open(member, "r") as handle:
        for source_row, raw in enumerate(handle):
            if not raw.strip():
                continue
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict) or not isinstance(value.get("current"), dict):
                raise ValueError("%s row %d has invalid release schema" % (member, source_row))
            yield source_row, value


def load_pack_orders_from_zip(
    archive: zipfile.ZipFile,
) -> Tuple[Dict[str, List[str]], str, List[str]]:
    pack_orders: Dict[str, List[str]] = {}
    hash_rows: List[Tuple[str, str, List[str]]] = []
    empty_mapping_fallback_packs: List[str] = []
    archive_names = archive.namelist()
    for member in sorted(
        name for name in archive_names if name.endswith("/emoji_mapping.txt")
    ):
        parts = member.rstrip("/").split("/")
        if len(parts) < 2:
            raise ValueError("cannot infer pack ID from %s" % member)
        pack = normalize_external_id(parts[-2])
        if pack in pack_orders:
            raise ValueError("duplicate emoji mapping for pack %s" % pack)
        ordered = parse_emoji_mapping(
            archive.read(member).decode("utf-8"), allow_empty=True
        )
        source = "emoji_mapping"
        if not ordered:
            # Two upstream packs have empty mapping files. Preserve the original
            # ZIP member order of their .npy files; this is the only fallback that
            # reproduces the pre-registered gray-slot counts.
            prefix = member.rsplit("/", 1)[0] + "/"
            ordered = [
                normalize_external_id(Path(name).stem)
                for name in archive_names
                if name.startswith(prefix) and name.endswith(".npy")
            ]
            if not ordered:
                raise ValueError(
                    "empty emoji mapping pack %s also has no .npy stickers" % pack
                )
            source = "zip_member_order_empty_mapping_fallback"
            empty_mapping_fallback_packs.append(pack)
        pack_orders[pack] = ordered
        hash_rows.append((pack, source, ordered))
    if not pack_orders:
        raise ValueError("archive contains no emoji_mapping.txt files")
    return (
        pack_orders,
        hash_value(hash_rows),
        sorted(empty_mapping_fallback_packs, key=int),
    )


def _release_member(archive: zipfile.ZipFile, split: str) -> str:
    name = "stickerchat/release_%s.json" % split
    if name not in archive.namelist():
        matches = [item for item in archive.namelist() if item.endswith("/release_%s.json" % split)]
        if len(matches) != 1:
            raise ValueError("cannot uniquely resolve raw %s release file" % split)
        name = matches[0]
    return name


def build_split_candidates(
    archive: zipfile.ZipFile,
    split: str,
    pack_orders: Mapping[str, Sequence[str]],
    img2id: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    member = _release_member(archive, split)
    rows: List[Dict[str, Any]] = []
    all_candidates: List[List[int]] = []
    all_gray: List[List[bool]] = []
    gray_rows = 0
    gray_slots = 0
    exposure: Dict[int, int] = {}
    for source_row, raw_row in _zip_release_rows(archive, member):
        current = raw_row["current"]
        pack = normalize_external_id(current["sticker_set_id"])
        gold_external = normalize_external_id(current["sticker_id"])
        if pack not in pack_orders:
            raise KeyError("raw row %d references missing pack %s" % (source_row, pack))
        candidates, gray_mask = fixed_candidates_for_gold(
            pack, gold_external, pack_orders[pack], img2id
        )
        real_negatives = [item for item in candidates[1:] if item != GRAY_SENTINEL_ID]
        for item in real_negatives:
            exposure[int(item)] = exposure.get(int(item), 0) + 1
        row_gray = sum(bool(item) for item in gray_mask)
        gray_rows += int(row_gray > 0)
        gray_slots += row_gray
        rows.append(
            {
                "source_row": int(source_row),
                "pack_id": pack,
                "gold_external_id": gold_external,
                "gold_internal_id": int(candidates[0]),
                "candidate_ids": [int(item) for item in candidates],
                "gray_mask": [bool(item) for item in gray_mask],
                "positive_index": 0,
            }
        )
        all_candidates.append(candidates)
        all_gray.append(gray_mask)
    candidates_tensor = torch.tensor(all_candidates, dtype=torch.int32)
    gray_tensor = torch.tensor(all_gray, dtype=torch.bool)
    counts = sorted(exposure.values())
    stats = {
        "rows": len(rows),
        "candidate_size": CANDIDATE_COUNT,
        "gray_rows": int(gray_rows),
        "gray_slots": int(gray_slots),
        "real_negative_slots": int(len(rows) * 9 - gray_slots),
        "negative_exposure": {
            "unique_stickers": len(exposure),
            "minimum": counts[0] if counts else 0,
            "maximum": counts[-1] if counts else 0,
            "mean": (sum(counts) / float(len(counts))) if counts else 0.0,
        },
        "raw_release_member": member,
        "raw_release_member_sha256": _zip_member_sha256(archive, member),
    }
    return rows, candidates_tensor, gray_tensor, stats


def _validate_processed_alignment(
    processed_rows: Sequence[Mapping[str, Any]],
    fixed_rows: Sequence[Mapping[str, Any]],
    split: str,
) -> None:
    if len(processed_rows) != len(fixed_rows):
        raise ValueError(
            "%s processed/raw row-count mismatch: %d != %d"
            % (split, len(processed_rows), len(fixed_rows))
        )
    for index, (processed, fixed) in enumerate(zip(processed_rows, fixed_rows)):
        dialogue = processed.get("dialog")
        if not isinstance(dialogue, list) or not dialogue:
            raise ValueError("%s processed row %d has no dialogue" % (split, index))
        internal_gold = int(dialogue[-1]["img_id"])
        if internal_gold != int(fixed["gold_internal_id"]):
            raise ValueError(
                "%s row %d gold alignment mismatch: %d != %d"
                % (split, index, internal_gold, int(fixed["gold_internal_id"]))
            )


def validate_fixed_manifest(value: Mapping[str, Any], verify_files: bool = True) -> None:
    if value.get("schema_version") != FIXED_SAME_PACK_SCHEMA:
        raise ValueError("unsupported fixed same-pack manifest schema")
    if value.get("negative_policy") != FIXED_SAME_PACK_POLICY:
        raise ValueError("fixed same-pack negative policy mismatch")
    if int(value.get("gray_sentinel_id", 0)) != GRAY_SENTINEL_ID:
        raise ValueError("fixed same-pack gray sentinel mismatch")
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != hash_value(core):
        raise ValueError("fixed same-pack manifest hash mismatch")
    if verify_files:
        for split, metadata in value["splits"].items():
            for key in ("rows", "runtime_tensor"):
                entry = metadata[key]
                if not Path(entry["path"]).exists():
                    raise FileNotFoundError(entry["path"])
                if sha256_file(entry["path"]) != entry["sha256"]:
                    raise ValueError("%s %s file hash mismatch" % (split, key))
            eval_entry = metadata.get("evaluation_data")
            if eval_entry is not None:
                if sha256_file(eval_entry["path"]) != eval_entry["sha256"]:
                    raise ValueError("%s evaluation data hash mismatch" % split)
        gray = value["gray_embedding"]
        if not Path(gray["path"]).exists() or sha256_file(gray["path"]) != gray["sha256"]:
            raise ValueError("gray embedding file hash mismatch")


@dataclass
class FixedSamePackRuntime:
    manifest: Dict[str, Any]
    candidate_ids: torch.Tensor
    gray_mask: torch.Tensor
    policy: str = FIXED_SAME_PACK_POLICY

    def for_source_rows(
        self, source_rows: Sequence[int]
    ) -> Tuple[List[List[int]], List[List[bool]]]:
        index = torch.tensor([int(item) for item in source_rows], dtype=torch.long)
        if index.numel() and (
            int(index.min()) < 0 or int(index.max()) >= int(self.candidate_ids.size(0))
        ):
            raise IndexError("fixed candidate source row is outside the manifest")
        candidates = self.candidate_ids.index_select(0, index)
        gray = self.gray_mask.index_select(0, index)
        return candidates.tolist(), gray.tolist()


def validate_fixed_trace_record(
    record: Mapping[str, Any],
    runtime: FixedSamePackRuntime,
) -> None:
    if str(record.get("negative_policy", "")) != FIXED_SAME_PACK_POLICY:
        raise ValueError("fixed trace negative policy mismatch")
    source_row = int(record["source_row"])
    expected_candidates, expected_gray = runtime.for_source_rows([source_row])
    candidates = [int(item) for item in record["candidate_ids"]]
    gray_mask = [bool(item) for item in record["gray_mask"]]
    if candidates != expected_candidates[0] or gray_mask != expected_gray[0]:
        raise ValueError("fixed trace candidate manifest mismatch")
    if int(record["positive"]) != candidates[0]:
        raise ValueError("fixed trace positive is not candidate 0")
    score_fields = (
        "base_scores",
        "expression_scores",
        "group_scores",
        "final_scores",
    )
    for name in score_fields:
        scores = [float(item) for item in record[name]]
        if len(scores) != CANDIDATE_COUNT or not all(
            math.isfinite(item) for item in scores
        ):
            raise ValueError("fixed trace %s is invalid" % name)
    group_scores = [float(item) for item in record["group_scores"]]
    if any(group_scores[index] != 0.0 for index, gray in enumerate(gray_mask) if gray):
        raise ValueError("gray fixed candidates must have exactly zero group score")
    hardest_index = int(record["hardest_expression_index"])
    if hardest_index < 1 or hardest_index >= CANDIDATE_COUNT:
        raise ValueError("fixed trace hardest expression index is outside negatives")
    if int(record["hardest_expression_id"]) != candidates[hardest_index]:
        raise ValueError("fixed trace hardest expression ID/index mismatch")
    expression_scores = [float(item) for item in record["expression_scores"]]
    if expression_scores[hardest_index] != max(expression_scores[1:]):
        raise ValueError("fixed trace did not select the hardest expression negative")


def load_fixed_same_pack_runtime(
    config: Mapping[str, Any],
    train_data_path: Optional[str] = None,
) -> FixedSamePackRuntime:
    if str(config.get("mode", "")) != FIXED_SAME_PACK_POLICY:
        raise ValueError("fixed-candidate config mode mismatch")
    manifest_path = str(config["manifest_path"])
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_fixed_manifest(manifest, verify_files=True)
    if train_data_path:
        expected = manifest["inputs"]["processed_train"]["sha256"]
        if sha256_file(train_data_path) != expected:
            raise ValueError("processed training data hash does not match fixed candidates")
    runtime_path = manifest["splits"]["train"]["runtime_tensor"]["path"]
    payload = torch.load(runtime_path, map_location="cpu")
    candidate_ids = payload["candidate_ids"].to(dtype=torch.long)
    gray_mask = payload["gray_mask"].to(dtype=torch.bool)
    expected_rows = int(manifest["splits"]["train"]["stats"]["rows"])
    if tuple(candidate_ids.shape) != (expected_rows, CANDIDATE_COUNT):
        raise ValueError("fixed train candidate tensor shape mismatch")
    if gray_mask.shape != candidate_ids.shape:
        raise ValueError("fixed train gray mask shape mismatch")
    if not torch.equal(gray_mask, candidate_ids.eq(GRAY_SENTINEL_ID)):
        raise ValueError("fixed train gray mask content mismatch")
    return FixedSamePackRuntime(dict(manifest), candidate_ids, gray_mask)


def write_fixed_same_pack_assets(
    zip_path: str,
    img2id_path: str,
    processed_train_path: str,
    processed_val_path: str,
    processed_test_path: str,
    output_dir: str,
    val_output_path: str,
    test_output_path: str,
    gray_embedding: torch.Tensor,
    enforce_expected_stats: bool = True,
) -> Dict[str, Any]:
    """Build all frozen assets. The caller supplies the CLIP gray embedding."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = output / "manifest.json"
    if existing_manifest_path.exists():
        with existing_manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        validate_fixed_manifest(existing, verify_files=True)
        requested_inputs = {
            "raw_zip": (zip_path, sha256_file(zip_path)),
            "img2id": (img2id_path, sha256_file(img2id_path)),
            "processed_train": (
                processed_train_path,
                sha256_file(processed_train_path),
            ),
        }
        for name, (path, digest) in requested_inputs.items():
            recorded = existing["inputs"][name]
            if recorded["path"] != path or recorded["sha256"] != digest:
                raise RuntimeError(
                    "refusing to reuse fixed candidates with changed %s" % name
                )
        recorded_eval = {
            existing["splits"][split]["evaluation_data"]["path"]
            for split in ("validation", "test")
        }
        if recorded_eval != {val_output_path, test_output_path}:
            raise RuntimeError(
                "refusing to reuse fixed candidates with changed eval outputs"
            )
        return existing
    with open(img2id_path, "r", encoding="utf-8") as handle:
        img2id = json.load(handle)
    with zipfile.ZipFile(zip_path, "r") as archive:
        (
            pack_orders,
            mapping_hash,
            empty_mapping_fallback_packs,
        ) = load_pack_orders_from_zip(archive)
        built = {}
        for split in ("train", "validation", "test"):
            raw_split = "val" if split == "validation" else split
            built[split] = build_split_candidates(
                archive, raw_split, pack_orders, img2id
            )

    split_sources = {
        "validation": processed_val_path,
        "test": processed_test_path,
    }
    split_outputs = {
        "validation": val_output_path,
        "test": test_output_path,
    }
    with open(processed_train_path, "r", encoding="utf-8") as handle:
        processed_train = json.load(handle)
    _validate_processed_alignment(
        processed_train, built["train"][0], "train"
    )
    del processed_train
    for split in ("validation", "test"):
        with open(split_sources[split], "r", encoding="utf-8") as handle:
            processed = json.load(handle)
        rows = built[split][0]
        _validate_processed_alignment(processed, rows, split)
        for processed_row, fixed_row in zip(processed, rows):
            processed_row["cand"] = list(fixed_row["candidate_ids"])
            processed_row["gray_mask"] = list(fixed_row["gray_mask"])
            processed_row["fixed_candidate_policy"] = FIXED_SAME_PACK_POLICY
        atomic_write_json(split_outputs[split], processed)

    split_manifest = {}
    for split, (rows, candidates, gray, stats) in built.items():
        if enforce_expected_stats:
            expected = EXPECTED_SPLIT_STATS[split]
            observed = {
                "rows": stats["rows"],
                "gray_rows": stats["gray_rows"],
                "gray_slots": stats["gray_slots"],
            }
            if observed != expected:
                raise RuntimeError(
                    "%s fixed-candidate statistics changed: %r != %r"
                    % (split, observed, expected)
                )
        rows_path = output / ("%s.jsonl" % split)
        tensor_path = output / ("%s_candidates.pt" % split)
        _atomic_write_jsonl(rows_path, rows)
        _atomic_torch_save(
            tensor_path,
            {
                "schema_version": FIXED_SAME_PACK_SCHEMA,
                "split": split,
                "candidate_ids": candidates,
                "gray_mask": gray,
            },
        )
        entry = {
            "stats": stats,
            "rows": {"path": str(rows_path), "sha256": sha256_file(rows_path)},
            "runtime_tensor": {
                "path": str(tensor_path),
                "sha256": sha256_file(tensor_path),
                "schema": {
                    "candidate_ids": [stats["rows"], CANDIDATE_COUNT],
                    "gray_mask": [stats["rows"], CANDIDATE_COUNT],
                },
            },
        }
        if split in split_outputs:
            entry["evaluation_data"] = {
                "path": str(split_outputs[split]),
                "sha256": sha256_file(split_outputs[split]),
                "source_path": split_sources[split],
                "source_sha256": sha256_file(split_sources[split]),
            }
        split_manifest[split] = entry

    gray_embedding = gray_embedding.detach().cpu().float().reshape(-1)
    if int(gray_embedding.numel()) != 512 or not torch.isfinite(gray_embedding).all():
        raise ValueError("gray CLIP embedding must be a finite 512-D vector")
    gray_path = output / "gray_clip_embedding.pt"
    _atomic_torch_save(
        gray_path,
        {
            "schema_version": FIXED_SAME_PACK_SCHEMA,
            "rgb": [127, 127, 127],
            "embedding": gray_embedding,
        },
    )
    core = {
        "schema_version": FIXED_SAME_PACK_SCHEMA,
        "negative_policy": FIXED_SAME_PACK_POLICY,
        "candidate_count": CANDIDATE_COUNT,
        "positive_index": 0,
        "gray_sentinel_id": GRAY_SENTINEL_ID,
        "gray_policy": "RGB=127; MM-BERT/expression score normally; group score masked to zero",
        "selection_policy": (
            "remove gold by normalized external ID, then take first nine remaining "
            "emoji_mapping.txt entries in original order; gray-pad the shortfall"
        ),
        "inputs": {
            "raw_zip": {"path": zip_path, "sha256": sha256_file(zip_path)},
            "img2id": {"path": img2id_path, "sha256": sha256_file(img2id_path)},
            "processed_train": {
                "path": processed_train_path,
                "sha256": sha256_file(processed_train_path),
            },
            "emoji_mapping_content_hash": mapping_hash,
            "pack_count": len(pack_orders),
            "empty_mapping_fallback": {
                "policy": (
                    "only for an empty emoji_mapping.txt, use .npy entries in "
                    "original ZIP member order"
                ),
                "packs": empty_mapping_fallback_packs,
                "count": len(empty_mapping_fallback_packs),
            },
        },
        "gray_embedding": {
            "path": str(gray_path),
            "sha256": sha256_file(gray_path),
            "shape": [512],
            "dtype": "float32",
        },
        "splits": split_manifest,
    }
    manifest = {**core, "manifest_hash": hash_value(core)}
    manifest_path = output / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    validate_fixed_manifest(manifest, verify_files=True)
    return manifest
