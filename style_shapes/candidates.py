"""Fixed StickerChat candidate protocols for the Style Shapes pilot."""

from __future__ import annotations

import copy
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


def pack_from_image_name(value: str) -> str:
    """Return the original pack prefix from ``<pack>-<sticker>.<suffix>``."""
    stem = Path(str(value)).stem
    if "-" not in stem:
        raise ValueError("StickerChat image name has no pack separator: %s" % value)
    pack, sticker = stem.split("-", 1)
    if not pack or not sticker:
        raise ValueError("malformed StickerChat image name: %s" % value)
    return pack


def id_to_pack_from_id2img(value: Mapping) -> Dict[int, str]:
    result = {int(sticker_id): pack_from_image_name(name) for sticker_id, name in value.items()}
    if sorted(result) != list(range(len(result))):
        raise ValueError("StickerChat internal IDs must be contiguous from zero")
    return result


def _gold_id(row: Mapping) -> int:
    dialog = row.get("dialog")
    if not isinstance(dialog, list) or not dialog:
        raise ValueError("candidate row has no dialogue")
    return int(dialog[-1]["img_id"])


def build_same_pack_candidates(
    rows: Sequence[Mapping],
    id_to_pack: Mapping[int, str],
    candidate_size: int,
    seed: int,
) -> Tuple[list, dict]:
    """Build gold-first candidates, filling only same-pack shortfalls globally.

    Each query receives every possible same-pack negative up to ``candidate_size - 1``.
    When the original pack is large enough, all negatives are sampled from that pack.
    When it is too small, all available same-pack negatives are retained and the exact
    shortfall is sampled without replacement from the rest of the global catalog.
    """
    candidate_size = int(candidate_size)
    if candidate_size < 2:
        raise ValueError("candidate_size must be at least two")
    normalized = {int(key): str(value) for key, value in id_to_pack.items()}
    all_ids = sorted(normalized)
    if all_ids != list(range(len(all_ids))):
        raise ValueError("StickerChat internal IDs must be contiguous from zero")

    pack_to_ids = defaultdict(list)
    for sticker_id in all_ids:
        pack_to_ids[normalized[sticker_id]].append(sticker_id)

    output = []
    same_counts = Counter()
    global_counts = Counter()
    fallback_rows = 0
    need = candidate_size - 1
    for row_index, source in enumerate(rows):
        gold = _gold_id(source)
        if gold not in normalized:
            raise ValueError("row %d gold is outside StickerChat catalog" % row_index)
        pack = normalized[gold]
        same_pool = [value for value in pack_to_ids[pack] if value != gold]
        rng = random.Random(int(seed) + row_index)

        same_take = min(need, len(same_pool))
        same_negatives = rng.sample(same_pool, same_take)
        chosen = set(same_negatives)
        chosen.add(gold)
        global_take = need - same_take
        global_negatives = []
        if global_take:
            global_pool = [value for value in all_ids if value not in chosen]
            global_negatives = rng.sample(global_pool, global_take)
            fallback_rows += 1

        negatives = same_negatives + global_negatives
        rng.shuffle(negatives)
        candidates = [gold] + negatives
        if len(candidates) != candidate_size or len(set(candidates)) != candidate_size:
            raise RuntimeError("row %d candidate construction failed" % row_index)
        if any(normalized[value] != pack for value in same_negatives):
            raise RuntimeError("row %d same-pack construction failed" % row_index)
        if any(normalized[value] == pack for value in global_negatives):
            raise RuntimeError("row %d fallback unexpectedly remained in gold pack" % row_index)

        remapped = copy.deepcopy(source)
        remapped["cand"] = candidates
        remapped["dialog"][-1]["neg_img_id"] = int(candidates[1])
        output.append(remapped)
        same_counts[same_take] += 1
        global_counts[global_take] += 1

    stats = {
        "rows": len(output),
        "candidate_size": candidate_size,
        "seed": int(seed),
        "catalog_stickers": len(all_ids),
        "packs": len(pack_to_ids),
        "all_same_pack_rows": len(output) - fallback_rows,
        "global_fallback_rows": fallback_rows,
        "same_pack_negative_count_distribution": {
            str(key): same_counts[key] for key in sorted(same_counts)
        },
        "global_fallback_count_distribution": {
            str(key): global_counts[key] for key in sorted(global_counts)
        },
        "same_pack_negatives": sum(key * value for key, value in same_counts.items()),
        "global_fallback_negatives": sum(key * value for key, value in global_counts.items()),
    }
    return output, stats


def audit_same_pack_candidates(
    rows: Sequence[Mapping],
    id_to_pack: Mapping[int, str],
    candidate_size: int = 10,
) -> dict:
    """Verify that global negatives occur only when the gold pack is too small."""
    normalized = {int(key): str(value) for key, value in id_to_pack.items()}
    pack_sizes = Counter(normalized.values())
    same_counts = Counter()
    global_counts = Counter()
    need = int(candidate_size) - 1
    for row_index, row in enumerate(rows):
        gold = _gold_id(row)
        values = [int(value) for value in row.get("cand", [])]
        if len(values) != int(candidate_size) or values.count(gold) != 1:
            raise ValueError("row %d candidate size/gold contract failed" % row_index)
        if len(set(values)) != len(values):
            raise ValueError("row %d has duplicate candidates" % row_index)
        gold_pack = normalized[gold]
        same = sum(normalized[value] == gold_pack for value in values if value != gold)
        expected_same = min(need, pack_sizes[gold_pack] - 1)
        if same != expected_same:
            raise ValueError(
                "row %d has %d same-pack negatives, expected %d"
                % (row_index, same, expected_same)
            )
        same_counts[same] += 1
        global_counts[need - same] += 1
    return {
        "rows": len(rows),
        "candidate_size": int(candidate_size),
        "all_same_pack_rows": same_counts[need],
        "global_fallback_rows": len(rows) - same_counts[need],
        "same_pack_negative_count_distribution": {
            str(key): same_counts[key] for key in sorted(same_counts)
        },
        "global_fallback_count_distribution": {
            str(key): global_counts[key] for key in sorted(global_counts)
        },
    }
