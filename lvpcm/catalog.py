"""Frozen catalog and candidate-manifest construction."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .io import hash_mapping


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def terminal_turn(row: dict) -> dict:
    dialog = row.get("dialog")
    if not isinstance(dialog, list) or not dialog:
        raise ValueError("row has no non-empty dialog")
    return dialog[-1]


def dstc_catalog(id2img_path: str, train_pairs_path: str) -> dict:
    id2img_raw = read_json(id2img_path)
    id2img = {int(key): value for key, value in id2img_raw.items()}
    if sorted(id2img) != list(range(len(id2img))):
        raise ValueError("DSTC image IDs are not contiguous")
    train_ids = set()
    for row in read_json(train_pairs_path):
        turn = terminal_turn(row)
        for field in ("img_id", "neg_img_id"):
            value = turn.get(field)
            if value is not None:
                train_ids.add(int(value))
    unknown = train_ids.difference(id2img)
    if unknown:
        raise ValueError("DSTC training catalog contains unknown IDs: %s" % sorted(unknown)[:5])
    return {
        "dataset": "dstc",
        "ids": sorted(id2img),
        "train_ids": sorted(train_ids),
        "id2img": id2img,
        "hash": hash_mapping({"ids": sorted(id2img), "train_ids": sorted(train_ids), "id2img": id2img}),
    }


def stickerchat_catalog(id2img_path: str, img2id_path: str, raw_train_path: str, metadata_path: str) -> dict:
    raw_id2img = read_json(id2img_path)
    id2img = {int(key): value for key, value in raw_id2img.items()}
    if sorted(id2img) != list(range(len(id2img))):
        raise ValueError("StickerChat image IDs are not contiguous")
    img2id = {str(key): int(value) for key, value in read_json(img2id_path).items()}
    train_ids = set()
    missing = set()
    for row in read_json(raw_train_path):
        turn = terminal_turn(row)
        for field in ("img_id", "neg_img_id"):
            external = turn.get(field)
            if external is None:
                continue
            if str(external) not in img2id:
                missing.add(str(external))
            else:
                train_ids.add(img2id[str(external)])
    if missing:
        raise ValueError("StickerChat raw training IDs missing from img2id: %s" % sorted(missing)[:5])
    metadata = read_json(metadata_path)
    stickers = metadata.get("stickers", [])
    pack_by_id = {}
    external_by_id = {}
    for item in stickers:
        internal = int(item["internal_img_id"])
        pack_by_id[internal] = str(item["img_set"])
        external_by_id[internal] = str(item["external_img_id"])
    if set(pack_by_id) != set(id2img):
        raise ValueError("StickerChat metadata and ID catalog do not align")
    frozen = {
        "ids": sorted(id2img),
        "train_ids": sorted(train_ids),
        "id2img": id2img,
        "pack_by_id": pack_by_id,
        "external_by_id": external_by_id,
    }
    return {"dataset": "stickerchat", **frozen, "hash": hash_mapping(frozen)}


def validate_candidate_rows(rows: Sequence[dict], expected_n: int = None, allowed_ids: Iterable[int] = None) -> dict:
    sizes = []
    allowed = None if allowed_ids is None else set(int(value) for value in allowed_ids)
    for index, row in enumerate(rows):
        candidates = [int(value) for value in row["cand"]]
        gold = int(terminal_turn(row)["img_id"])
        if len(candidates) != len(set(candidates)):
            raise ValueError("duplicate candidate at row %d" % index)
        if candidates.count(gold) != 1:
            raise ValueError("gold candidate does not occur exactly once at row %d" % index)
        if expected_n is not None and len(candidates) != expected_n:
            raise ValueError("candidate count mismatch at row %d" % index)
        if allowed is not None and any(value not in allowed for value in candidates):
            raise ValueError("candidate ID outside frozen catalog at row %d" % index)
        sizes.append(len(candidates))
    return {"rows": len(rows), "candidate_sizes": sorted(set(sizes)), "all_ids_in_frozen_catalog": allowed is not None}


def normalized_candidate_manifest(rows: Sequence[dict], split: str, protocol: str, source: str) -> List[dict]:
    output = []
    for index, row in enumerate(rows):
        candidates = [int(value) for value in row["cand"]]
        gold = int(terminal_turn(row)["img_id"])
        if len(candidates) != len(set(candidates)) or candidates.count(gold) != 1:
            raise ValueError("invalid candidates at source row %d" % index)
        output.append({
            "query_id": index,
            "source": source,
            "source_row": index,
            "dialogue_id": row.get("dialogue_id"),
            "user_id": row.get("user_id"),
            "split": split,
            "protocol": protocol,
            "candidate_ids": candidates,
            "positive_index": candidates.index(gold),
        })
    return output


def dstc_training_candidates(rows: Sequence[dict], train_catalog: Sequence[int], n: int = 10, seed: int = 2021) -> List[dict]:
    catalog = sorted(int(value) for value in train_catalog)
    rng = random.Random(seed)
    output = []
    for index, row in enumerate(rows):
        gold = int(terminal_turn(row)["img_id"])
        pool = [value for value in catalog if value != gold]
        negatives = rng.sample(pool, n - 1)
        candidates = [gold] + negatives
        rng.shuffle(candidates)
        output.append({
            "query_id": index,
            "source": "data/train_pair.json",
            "source_row": index,
            "dialogue_id": row.get("dialogue_id"),
            "user_id": row.get("user_id"),
            "split": "train",
            "protocol": "uniform_r%d" % n,
            "candidate_ids": candidates,
            "positive_index": candidates.index(gold),
        })
    return output


def stickerchat_same_pack_candidates(
    rows: Sequence[dict], img2id: Dict[str, int], all_ids: Sequence[int], n: int, seed: int = 2021
) -> List[dict]:
    """Rebuild the original hard-negative protocol without mutating source data."""
    rng = random.Random(seed + n)
    pool = sorted(int(value) for value in all_ids)
    output = []
    for index, row in enumerate(rows):
        turn = terminal_turn(row)
        external_gold = str(turn["img_id"])
        external_negative = str(turn["neg_img_id"])
        if external_gold not in img2id or external_negative not in img2id:
            raise ValueError("external/internal mapping failure at row %d" % index)
        gold = int(img2id[external_gold])
        hard = int(img2id[external_negative])
        if gold == hard:
            raise ValueError("hard negative equals gold at row %d" % index)
        if str(turn.get("img_set")) != str(turn.get("neg_img_set")):
            raise ValueError("raw hard negative is not same-pack at row %d" % index)
        chosen = [gold, hard]
        while len(chosen) < n:
            candidate = pool[rng.randrange(len(pool))]
            if candidate not in chosen:
                chosen.append(candidate)
        output.append({
            "query_id": index,
            "source_row": index,
            "dialogue_id": row.get("dialogue_id"),
            "user_id": row.get("user_id"),
            "split": "validation" if "val" in str(row.get("source", "")) else "unspecified",
            "protocol": "same_pack_r%d" % n,
            "candidate_ids": chosen,
            "positive_index": 0,
            "external_positive_id": external_gold,
            "external_hard_negative_id": external_negative,
            "positive_pack": str(turn.get("img_set")),
            "hard_negative_pack": str(turn.get("neg_img_set")),
        })
    return output
