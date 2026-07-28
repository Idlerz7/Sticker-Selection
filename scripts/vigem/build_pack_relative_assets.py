#!/usr/bin/env python3
"""Build the content-addressed StickerChat original-pack residual bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lvpcm.io import atomic_torch_save
from style_shapes.io import atomic_write_json, command_record, sha256_file
from vigem.pack_config import read_pack_config
from vigem.pack_relative import build_pack_relative_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", default="artifacts/vigem")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        config = read_pack_config(args.config)
        output = Path(config["pack_residual_bundle"])
        manifest_path = output.with_suffix(".manifest.json")
        source_paths = {
            "sticker_metadata": config["sticker_metadata_path"],
            "group_bank": config["group_bank"],
            "final_clip": config["final_clip_descriptor"],
            "vpd": config["vpd_descriptor"],
        }
        expected_hashes = {
            key: sha256_file(path) for key, path in source_paths.items()
        }
        if output.exists() or manifest_path.exists():
            if not (output.exists() and manifest_path.exists()):
                raise RuntimeError("refusing incomplete pack-relative asset pair")
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if (
                existing.get("source_hashes") != expected_hashes
                or existing.get("bundle", {}).get("sha256")
                != sha256_file(output)
            ):
                raise RuntimeError("refusing incompatible pack-relative assets")
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return

        payload, manifest = build_pack_relative_bundle(
            metadata_path=config["sticker_metadata_path"],
            group_bank_path=config["group_bank"],
            final_clip_path=config["final_clip_descriptor"],
            vpd_path=config["vpd_descriptor"],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(output, payload)
        manifest["bundle"] = {
            "path": str(output),
            "sha256": sha256_file(output),
            "shape": list(payload["residuals"].shape),
            "dtype": str(payload["residuals"].dtype),
        }
        atomic_write_json(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
