#!/usr/bin/env python3
"""Evaluate one checkpoint on fixed R10, random same-pack R10, and global R20."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from style_shapes.io import atomic_write_json, command_record, sha256_file


PROTOCOLS = {
    "fixed_same_pack_r10": (
        "stickerchat/processed/"
        "release_test_u_sticker_format_int_with_cand_fixed_same_pack_r10.json"
    ),
    "random_same_pack_r10": (
        "stickerchat/processed/"
        "release_test_u_sticker_format_int_with_cand_same_pack_r10.json"
    ),
    "global_random_r20": (
        "stickerchat/processed/"
        "release_test_u_sticker_format_int_with_cand_r20.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with command_record(args, args.artifact_root):
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        results = {}
        for protocol, data_path in PROTOCOLS.items():
            protocol_output = output / protocol
            command = [
                sys.executable,
                str(REPO / "scripts/style_shapes/run_pilot.py"),
                "--config",
                args.config,
                "--run-mode",
                "test",
                "--checkpoint-path",
                args.checkpoint,
                "--test-data-path",
                data_path,
                "--run-output-dir",
                str(protocol_output),
            ]
            subprocess.run(command, cwd=str(REPO), check=True)
            metrics_path = protocol_output / "scores" / "rank_00_metrics.json"
            if not metrics_path.exists():
                raise RuntimeError(
                    "evaluation did not produce metrics: %s" % metrics_path
                )
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            if not metrics.get("map_equals_mrr"):
                raise RuntimeError("single-positive MAP=MRR assertion failed")
            results[protocol] = {
                "test_data_path": data_path,
                "output_dir": str(protocol_output),
                "metrics_path": str(metrics_path),
                "metrics": metrics,
            }
        manifest = {
            "status": "EVAL_SUITE_COMPLETE",
            "config": args.config,
            "config_sha256": sha256_file(args.config),
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "protocols": results,
            "checkpoint_selection": "external; this suite never selects on test",
        }
        atomic_write_json(output / "evaluation_suite_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
