import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add candidate sets to validation samples."
    )
    parser.add_argument(
        "--input",
        default="data/validation_pair.json",
        help="Input validation pair json path.",
    )
    parser.add_argument(
        "--id2img",
        default="data/id2img.json",
        help="Path to id2img mapping (used to get sticker id universe).",
    )
    parser.add_argument(
        "--output",
        default="data/validation_pair_with_cand.json",
        help="Output json path with candidate set added.",
    )
    parser.add_argument(
        "--cand-size",
        type=int,
        default=10,
        help="Total candidate size (includes the true label).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2021,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cand_size < 2:
        raise ValueError("--cand-size must be >= 2")

    random.seed(args.seed)

    with open(args.id2img, encoding="utf-8") as f:
        id2img = json.load(f)
    all_ids = sorted(int(k) for k in id2img.keys())

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    skipped = 0
    out = []
    neg_size = args.cand_size - 1
    for sample in data:
        dialog = sample.get("dialog", [])
        if not dialog:
            skipped += 1
            continue
        true_id = dialog[-1].get("img_id")
        if true_id is None:
            skipped += 1
            continue
        true_id = int(true_id)

        pool = [x for x in all_ids if x != true_id]
        if len(pool) < neg_size:
            raise ValueError(
                f"Not enough negatives: requested {neg_size}, available {len(pool)}"
            )
        negs = random.sample(pool, k=neg_size)
        cand = negs + [true_id]
        random.shuffle(cand)

        sample_out = dict(sample)
        sample_out["cand"] = cand
        out.append(sample_out)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"input: {args.input}, total: {len(data)}")
    print(f"output: {args.output}, saved: {len(out)}, skipped: {skipped}")
    print(f"cand size: {args.cand_size}, seed: {args.seed}")


if __name__ == "__main__":
    main()
