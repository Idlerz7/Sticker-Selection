#!/usr/bin/env python3
"""
Add fixed candidate sticker ids to u-sticker validation JSON.

Default strategy ``hard_neg`` matches ``main.py`` PLDataset when
``test_with_cand`` is True and the sample has no ``cand`` field:
  cand = [pos, neg_img_id] + 8 extras sampled from
  {0 .. max_image_id-1} \\ {pos, neg}.

Strategy ``uniform`` matches ``build_validation_candidates.py`` (meme):
  cand = (cand_size - 1) random negatives + true label, then shuffled
  (does not use neg_img_id from the dialog).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build u-sticker validation JSON with fixed R10 (or custom) cand lists."
    )
    p.add_argument(
        "--input",
        default="./u-sticker/u_sticker_val_split.json",
        help="Input validation json (list of samples with dialog).",
    )
    p.add_argument(
        "--id2img",
        default="./u-sticker/u_sticker_id2img.json",
        help="id2img json (for strategy uniform and max id inference).",
    )
    p.add_argument(
        "--output",
        default="./u-sticker/u_sticker_val_split_with_cand.json",
        help="Output path with cand added.",
    )
    p.add_argument(
        "--strategy",
        choices=("hard_neg", "uniform"),
        default="hard_neg",
        help=(
            "hard_neg: pos + neg_img_id + 8 random (same as main.py on-the-fly R10). "
            "uniform: (cand_size-1) random negs + pos, shuffled (meme build_validation_candidates)."
        ),
    )
    p.add_argument(
        "--cand-size",
        type=int,
        default=10,
        help="Total candidates (including the true sticker).",
    )
    p.add_argument(
        "--max-image-id",
        type=int,
        default=None,
        help=(
            "Sticker id upper bound (ids used: 0 .. max_image_id-1). "
            "Default: inferred from id2img (max_key+1) for hard_neg; uniform uses id2img keys."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2021,
        help="RNG seed (reproducible cand).",
    )
    return p.parse_args()


def _last_turn(dialog: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not dialog:
        return None
    return dialog[-1]


def _infer_max_image_id(id2img: Dict[str, Any]) -> int:
    keys = [int(k) for k in id2img.keys()]
    if not keys:
        raise ValueError("id2img is empty")
    return max(keys) + 1


def make_cand_hard_neg(
    pos: int,
    neg: int,
    max_image_id: int,
    n_extra: int,
    rng: random.Random,
) -> List[int]:
    exclude = {pos, neg}
    pool = [i for i in range(max_image_id) if i not in exclude]
    k = min(n_extra, len(pool))
    extra = rng.sample(pool, k) if k > 0 else []
    return [pos, neg] + extra


def make_cand_uniform(
    true_id: int,
    all_ids: List[int],
    cand_size: int,
    rng: random.Random,
) -> List[int]:
    neg_size = cand_size - 1
    pool = [x for x in all_ids if x != true_id]
    if len(pool) < neg_size:
        raise ValueError(
            f"Not enough negatives: need {neg_size}, pool has {len(pool)} (true_id={true_id})"
        )
    negs = rng.sample(pool, k=neg_size)
    cand = negs + [true_id]
    rng.shuffle(cand)
    return cand


def main() -> None:
    args = parse_args()
    if args.cand_size < 2:
        raise ValueError("--cand-size must be >= 2")

    rng = random.Random(args.seed)

    with open(args.id2img, encoding="utf-8") as f:
        id2img = json.load(f)
    all_ids_sorted = sorted(int(k) for k in id2img.keys())

    max_image_id = args.max_image_id
    if max_image_id is None:
        max_image_id = _infer_max_image_id(id2img)

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples.")

    out: List[Dict[str, Any]] = []
    skipped_no_img = 0
    skipped_no_neg = 0

    for sample in data:
        dialog = sample.get("dialog") or []
        last = _last_turn(dialog)
        if last is None:
            skipped_no_img += 1
            continue
        raw_img = last.get("img_id")
        if raw_img is None:
            skipped_no_img += 1
            continue
        true_id = int(raw_img)

        sample_out = dict(sample)

        if args.strategy == "uniform":
            sample_out["cand"] = make_cand_uniform(
                true_id, all_ids_sorted, args.cand_size, rng
            )
            out.append(sample_out)
            continue

        # hard_neg
        raw_neg = last.get("neg_img_id")
        if raw_neg is None:
            skipped_no_neg += 1
            continue
        neg_id = int(raw_neg)
        n_extra = args.cand_size - 2
        if n_extra < 0:
            raise ValueError("cand-size must be >= 2 for hard_neg")
        sample_out["cand"] = make_cand_hard_neg(
            true_id, neg_id, max_image_id, n_extra, rng
        )
        out.append(sample_out)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(
        f"input={args.input} total_in={len(data)} "
        f"saved={len(out)} skipped_no_pos={skipped_no_img} skipped_no_neg={skipped_no_neg}"
    )
    print(
        f"output={args.output} strategy={args.strategy} "
        f"cand_size={args.cand_size} max_image_id={max_image_id} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
