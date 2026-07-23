#!/usr/bin/env python3
"""Audit immutable Style Shapes inputs, environment, candidates, disk, and GPUs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from style_shapes.builders import load_descriptor
from style_shapes.io import (
    atomic_write_json,
    atomic_write_text,
    command_record,
    environment_snapshot,
    sha256_file,
)
from style_shapes.validation import candidate_file_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/style_shapes")
    args = parser.parse_args()
    with command_record(args, args.output_dir):
        descriptors = {}
        for dataset, count in (("dstc", 307), ("stickerchat", 174695)):
            descriptors[dataset] = {}
            for family, dim in (("multi", 256), ("final_clip", 512)):
                path = "artifacts/lvpcm/descriptors/%s/%s_clean.pt" % (dataset, family)
                ids, features, meta = load_descriptor(path)
                if ids != list(range(count)) or list(features.shape) != [count, dim]:
                    raise RuntimeError("%s %s descriptor contract failed" % (dataset, family))
                descriptors[dataset][family] = meta
        candidates = {
            "dstc_validation_r10": candidate_file_audit(
                "data/validation_pair_with_cand.json", 10
            ),
            "stickerchat_validation_r10": candidate_file_audit(
                "stickerchat/processed/release_val_u_sticker_format_int_with_cand_r10.json",
                10,
            ),
            "stickerchat_validation_r20": candidate_file_audit(
                "stickerchat/processed/release_val_u_sticker_format_int_with_cand_r20.json",
                20,
            ),
            "stickerchat_test_r10": candidate_file_audit(
                "stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json",
                10,
            ),
            "stickerchat_test_r20": candidate_file_audit(
                "stickerchat/processed/release_test_u_sticker_format_int_with_cand_r20.json",
                20,
            ),
        }
        disk = shutil.disk_usage(".")
        audit = {
            "status": "COMPLETE",
            "repository": environment_snapshot(),
            "conda_contract": {
                "required": "stickr-select",
                "actual": os.environ.get("CONDA_DEFAULT_ENV"),
                "pass": os.environ.get("CONDA_DEFAULT_ENV") == "stickr-select",
            },
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
            },
            "descriptors": descriptors,
            "legacy_banks": {
                "dstc": {
                    "path": "factorized_style_bank.json",
                    "sha256": sha256_file("factorized_style_bank.json"),
                },
                "stickerchat": {
                    "path": "stickerchat/processed_style_kmeans_k384/factorized_style_bank.json",
                    "sha256": sha256_file(
                        "stickerchat/processed_style_kmeans_k384/factorized_style_bank.json"
                    ),
                },
            },
            "candidates": candidates,
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
            },
            "resource_policy": {
                "dstc_world_size": 4,
                "stickerchat_world_size": 8,
                "preempt_occupied_gpu": False,
                "formal_state_until_equivalence_and_resources": "RESOURCE_BLOCKED",
            },
        }
        output = Path(args.output_dir)
        atomic_write_json(output / "repo_audit.json", audit)
        lines = [
            "# Style Shapes Repository Audit",
            "",
            "Status: **COMPLETE**",
            "",
            "- Required Conda environment: `stickr-select`; active: `%s`."
            % os.environ.get("CONDA_DEFAULT_ENV"),
            "- PyTorch `%s`, CUDA build `%s`, visible GPU count `%d`."
            % (torch.__version__, torch.version.cuda, torch.cuda.device_count()),
            "- VPD bundles: DSTC `[307,256]`; StickerChat `[174695,256]`, finite and ID-aligned.",
            "- Final-CLIP bundles: DSTC `[307,512]`; StickerChat `[174695,512]`, finite and ID-aligned.",
            "- Fixed candidates validated: DSTC validation R10; StickerChat validation/test R10/R20.",
            "- Free filesystem space: %.1f GiB." % (disk.free / 2**30),
            "- Formal training remains resource-gated; occupied GPUs will not be preempted.",
            "",
            "All exact paths and SHA-256 values are in `repo_audit.json`.",
        ]
        atomic_write_text(output / "repo_audit.md", "\n".join(lines) + "\n")
        print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

