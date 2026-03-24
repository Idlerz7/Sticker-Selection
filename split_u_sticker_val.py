#!/usr/bin/env python3
"""Split u_sticker_train_pair.json into train + val."""
import json
import random
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./u-sticker/u_sticker_train_pair.json")
    ap.add_argument("--train-out", default="./u-sticker/u_sticker_train_split.json")
    ap.add_argument("--val-out", default="./u-sticker/u_sticker_val_split.json")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
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
