#!/usr/bin/env python3
"""Generate id2name and ocr from u_sticker id2img for English training."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id2img", default="./u-sticker/u_sticker_id2img.json")
    ap.add_argument("--id2name-out", default="./u-sticker/u_sticker_id2name.json")
    ap.add_argument("--ocr-out", default="./u-sticker/u_sticker_ocr.json")
    ap.add_argument("--img2id-out", default="./u-sticker/u_sticker_img2id.json")
    args = ap.parse_args()

    with open(args.id2img, encoding="utf-8") as f:
        id2img = json.load(f)

    id2name = {}
    ocr = {}
    img2id = {}
    for k, filename in id2img.items():
        stem, _ = os.path.splitext(filename)
        id2name[k] = stem
        ocr[k] = {"ocr": "", "name": filename}
        img2id[filename] = str(k)

    out_dir = os.path.dirname(os.path.abspath(args.id2name_out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.id2name_out, "w", encoding="utf-8") as f:
        json.dump(id2name, f, ensure_ascii=False, indent=2)
    with open(args.ocr_out, "w", encoding="utf-8") as f:
        json.dump(ocr, f, ensure_ascii=False, indent=2)
    with open(args.img2id_out, "w", encoding="utf-8") as f:
        json.dump(img2id, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.id2name_out}, {args.ocr_out}, {args.img2id_out} ({len(id2name)} entries)")


if __name__ == "__main__":
    main()
