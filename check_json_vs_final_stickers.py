#!/usr/bin/env python3
"""Check which JSON-referenced media files exist in final_stickers."""
import json
import os
from collections import Counter

MEDIA_ROOT = "./u-sticker/final_stickers"
INPUT_JSON = "./u-sticker/all_filtered_sticker_output.json"


def main():
    # Collect all refs from JSON
    refs = set()
    with open(INPUT_JSON, encoding="utf-8") as f:
        data = json.load(f)
    for sample in data:
        dialog = sample.get("dialog", [])
        if not isinstance(dialog, list):
            continue
        for turn in dialog:
            if not isinstance(turn, dict):
                continue
            for key in ("img_id", "neg_img_id"):
                v = turn.get(key)
                if v and isinstance(v, str):
                    name = os.path.basename(v.strip())
                    refs.add(name)

    # Check existence
    exists = []
    missing = []
    for name in sorted(refs):
        path = os.path.join(MEDIA_ROOT, name)
        if os.path.isfile(path):
            exists.append(name)
        else:
            missing.append(name)

    # Stats by extension
    ext_exists = Counter(os.path.splitext(n)[1].lower() for n in exists)
    ext_missing = Counter(os.path.splitext(n)[1].lower() for n in missing)

    print("[JSON vs final_stickers Check]")
    print(f"- Total unique refs in JSON: {len(refs)}")
    print(f"- Exist in final_stickers: {len(exists)}")
    print(f"- Missing: {len(missing)}")
    print(f"\nBy extension (exists): {dict(ext_exists)}")
    print(f"By extension (missing): {dict(ext_missing)}")
    if missing:
        print(f"\nSample missing (first 20):")
        for n in missing[:20]:
            print(f"  {n}")


if __name__ == "__main__":
    main()
