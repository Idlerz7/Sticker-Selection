#!/usr/bin/env python3
"""Split u_sticker_train_pair.json into train + val."""
import argparse
import json
import random
import re

# 每轮 text 恰好为 [speaker1] 或 [speaker2]，无其它正文（与 u_sticker 中「空对话」样本一致）
_TEXT_PLACEHOLDER_ONLY = re.compile(r"^\[(speaker1|speaker2)\]$")


def is_text_placeholder_only_dialog(dialog) -> bool:
    if not dialog:
        return False
    for turn in dialog:
        t = turn.get("text")
        if not isinstance(t, str):
            return False
        if not _TEXT_PLACEHOLDER_ONLY.match(t.strip()):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./u-sticker/u_sticker_train_pair.json")
    ap.add_argument("--train-out", default="./u-sticker/u_sticker_train_split.json")
    ap.add_argument("--val-out", default="./u-sticker/u_sticker_val_split.json")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument(
        "--drop-text-placeholder-only",
        action="store_true",
        help="Remove samples whose every turn's text is only [speaker1] or [speaker2] (no real utterance). Applied before shuffle/split.",
    )
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    n_raw = len(data)
    if args.drop_text_placeholder_only:
        data = [x for x in data if not is_text_placeholder_only_dialog(x.get("dialog") or [])]
        print(
            f"Filtered text-placeholder-only: {n_raw} -> {len(data)} "
            f"(dropped {n_raw - len(data)})"
        )
    random.seed(args.seed)
    random.shuffle(data)
    n_val = max(1, int(len(data) * args.val_ratio))
    val_data = data[:n_val]
    train_data = data[n_val:]
    with open(args.train_out, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(args.val_out, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"Split: train={len(train_data)}, val={len(val_data)}")


if __name__ == "__main__":
    main()
