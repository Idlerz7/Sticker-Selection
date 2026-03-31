#!/usr/bin/env python3
"""
Filter StickerChat train JSON to keep samples whose last-turn gold img_id is frequent enough.

Example:
  python scripts/filter_stickerchat_train_by_frequency.py \\
    --input ./stickerchat/processed/release_train_u_sticker_format_int.json \\
    --output ./stickerchat/processed/release_train_u_sticker_format_int_freq_ge_8.json \\
    --min-frequency 8
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _last_turn_img_id(sample: Dict[str, Any]) -> Optional[int]:
    dialog = sample.get("dialog") or []
    if not dialog:
        return None
    last = dialog[-1]
    if not isinstance(last, dict):
        return None
    raw = last.get("img_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--min-frequency",
        type=int,
        default=8,
        help="Keep sample if gold img_id appears at least this many times in the train file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    counts = Counter()
    for sample in data:
        iid = _last_turn_img_id(sample)
        if iid is not None:
            counts[iid] += 1

    kept: List[Dict[str, Any]] = []
    for sample in data:
        iid = _last_turn_img_id(sample)
        if iid is None:
            continue
        if counts[iid] >= args.min_frequency:
            kept.append(sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)

    print(
        "input_samples=",
        len(data),
        "kept=",
        len(kept),
        "min_frequency=",
        args.min_frequency,
        "unique_gold_ids=",
        len(counts),
    )


if __name__ == "__main__":
    main()
