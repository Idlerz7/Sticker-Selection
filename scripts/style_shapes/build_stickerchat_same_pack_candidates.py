#!/usr/bin/env python3
"""Freeze StickerChat same-pack R10 with global fallback only for small packs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from style_shapes.candidates import (
    audit_same_pack_candidates,
    build_same_pack_candidates,
    id_to_pack_from_id2img,
)
from style_shapes.io import atomic_write_json, command_record, sha256_file
from style_shapes.validation import candidate_file_audit


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_frozen(path: str, value) -> None:
    target = Path(path)
    if target.exists():
        existing = _read_json(str(target))
        if existing != value:
            raise RuntimeError("refusing to overwrite incompatible candidate file: %s" % target)
        return
    atomic_write_json(target, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id2img", default="stickerchat/processed/id2img.json")
    parser.add_argument(
        "--validation-source",
        default="stickerchat/processed/release_val_u_sticker_format_int_with_cand_r10.json",
    )
    parser.add_argument(
        "--test-source",
        default="stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json",
    )
    parser.add_argument(
        "--validation-output",
        default="stickerchat/processed/release_val_u_sticker_format_int_with_cand_same_pack_r10.json",
    )
    parser.add_argument(
        "--test-output",
        default="stickerchat/processed/release_test_u_sticker_format_int_with_cand_same_pack_r10.json",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/style_shapes/candidates/stickerchat_same_pack_r10_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=20260326)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    args = parser.parse_args()

    with command_record(args, args.artifact_root):
        id_to_pack = id_to_pack_from_id2img(_read_json(args.id2img))
        protocols = {}
        contracts = (
            (
                "validation",
                args.validation_source,
                args.validation_output,
                int(args.seed) + 1_000_000,
            ),
            ("test", args.test_source, args.test_output, int(args.seed) + 2_000_000),
        )
        for split, source, output, split_seed in contracts:
            rows, stats = build_same_pack_candidates(
                _read_json(source), id_to_pack, candidate_size=10, seed=split_seed
            )
            _write_frozen(output, rows)
            audit = candidate_file_audit(output, 10)
            composition = audit_same_pack_candidates(rows, id_to_pack, 10)
            if composition["global_fallback_rows"] != stats["global_fallback_rows"]:
                raise RuntimeError("%s same-pack composition audit mismatch" % split)
            protocols[split] = {
                **stats,
                "source": source,
                "source_sha256": sha256_file(source),
                "output": output,
                "output_sha256": audit["sha256"],
                "normalized_hash": audit["normalized_hash"],
            }

        manifest = {
            "schema_version": "style_shapes.stickerchat_same_pack_r10.v1",
            "policy": (
                "gold plus nine unique same-pack negatives; if the original pack has fewer "
                "than nine alternatives, retain all of them and fill only the shortfall "
                "with unique global negatives"
            ),
            "pack_source": "first hyphen-delimited component of id2img filename",
            "id2img": args.id2img,
            "id2img_sha256": sha256_file(args.id2img),
            "base_seed": int(args.seed),
            "protocols": protocols,
        }
        _write_frozen(args.manifest, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
