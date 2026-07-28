#!/usr/bin/env python3
"""Evaluate a Style Shapes/SEMSP checkpoint with the released PBR R10 bug."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from style_shapes.io import (
    atomic_write_json,
    atomic_write_text,
    command_record,
    sha256_file,
)
from style_shapes.pbr_bug_eval import (
    PBR_BUG_POLICY,
    pbr_released_metrics,
    validate_pbr_bug_manifest,
)


DEFAULT_MANIFEST = (
    "artifacts/style_shapes/candidates/"
    "stickerchat_pbr_bug_compatible_r10/manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split", choices=("validation", "test"), default="test"
    )
    parser.add_argument("--candidate-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional explicit output. Default: the configured pilot output/"
            "final_eval/pbr_bug_compatible_<split>_r10_<candidate hash>."
        ),
    )
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    return parser.parse_args()


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Style Shapes config root must be a mapping")
    return value


def main() -> None:
    args = parse_args()
    with command_record(args, args.artifact_root):
        with open(args.candidate_manifest, "r", encoding="utf-8") as handle:
            candidate_manifest = json.load(handle)
        validate_pbr_bug_manifest(candidate_manifest, verify_files=True)
        split_entry = candidate_manifest["splits"][args.split]["evaluation_data"]
        candidate_path = str(split_entry["path"])
        candidate_hash = str(split_entry["sha256"])
        checkpoint_hash = sha256_file(args.checkpoint)

        config = _load_yaml(args.config)
        if config.get("dataset") != "stickerchat":
            raise ValueError("PBR bug-compatible evaluation is StickerChat-only")
        overrides = dict(config.get("model_overrides", {}))
        overrides["factorized_gray_sentinel_id"] = -1
        overrides["factorized_gray_embedding_path"] = candidate_manifest[
            "gray_embedding"
        ]["path"]
        config["model_overrides"] = overrides

        if args.output_dir:
            output = Path(args.output_dir)
        else:
            output = (
                Path(config["output_dir"])
                / "final_eval"
                / (
                    "pbr_bug_compatible_%s_r10_%s"
                    % (
                        args.split,
                        "%s_%s" % (
                            candidate_hash[:12],
                            checkpoint_hash[:12],
                        ),
                    )
                )
            )
        output.mkdir(parents=True, exist_ok=True)
        effective_config = output / "effective_eval_config.yaml"
        atomic_write_text(
            effective_config,
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        )

        command = [
            sys.executable,
            str(REPO / "scripts/style_shapes/run_pilot.py"),
            "--config",
            str(effective_config),
            "--run-mode",
            "test",
            "--checkpoint-path",
            args.checkpoint,
            "--test-data-path",
            candidate_path,
            "--run-output-dir",
            str(output),
        ]
        subprocess.run(command, cwd=str(REPO), check=True)

        scores_path = output / "scores" / "rank_00_scores.json"
        if not scores_path.exists():
            raise RuntimeError("evaluation did not produce scores: %s" % scores_path)
        with scores_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if len(rows) != int(
            candidate_manifest["splits"][args.split]["stats"]["rows"]
        ):
            raise RuntimeError("PBR bug score row count mismatch")

        score_fields = (
            "base_scores",
            "instance_scores",
            "group_scores",
            "final_scores",
        )
        metrics = {}
        duplicate_rows = 0
        for row_index, row in enumerate(rows):
            candidates = [int(value) for value in row["candidate_ids"]]
            gold = int(row["gold"])
            if len(candidates) != 10 or candidates[0] != gold:
                raise RuntimeError(
                    "query %d does not preserve PBR positive slot 0" % row_index
                )
            duplicate_rows += int(gold in candidates[1:])
        for field in score_fields:
            metrics[field] = pbr_released_metrics(
                [row[field] for row in rows]
            )

        expected_duplicates = int(
            candidate_manifest["splits"][args.split]["stats"][
                "duplicate_gold_rows"
            ]
        )
        if duplicate_rows != expected_duplicates:
            raise RuntimeError(
                "scored duplicate-gold rows changed: %d != %d"
                % (duplicate_rows, expected_duplicates)
            )
        result = {
            "status": "PBR_BUG_COMPATIBLE_EVAL_COMPLETE",
            "audit_only": True,
            "protocol": PBR_BUG_POLICY,
            "warning": (
                "This intentionally reproduces a released-code candidate bug "
                "and must not replace the clean formal result."
            ),
            "split": args.split,
            "config": {
                "path": args.config,
                "sha256": sha256_file(args.config),
                "effective_path": str(effective_config),
                "effective_sha256": sha256_file(effective_config),
            },
            "checkpoint": {
                "path": args.checkpoint,
                "sha256": checkpoint_hash,
            },
            "candidate_manifest": {
                "path": args.candidate_manifest,
                "sha256": sha256_file(args.candidate_manifest),
                "manifest_hash": candidate_manifest["manifest_hash"],
            },
            "candidate_data": {
                "path": candidate_path,
                "sha256": candidate_hash,
                "duplicate_gold_rows": duplicate_rows,
                "gray_rows": int(
                    candidate_manifest["splits"][args.split]["stats"][
                        "gray_rows"
                    ]
                ),
            },
            "score_path": str(scores_path),
            "score_sha256": sha256_file(scores_path),
            "metrics": metrics,
            "primary_result": metrics["final_scores"],
            "metric_implementation": (
                "positive label is candidate slot 0; NumPy argsort tie behavior "
                "matches released PBR metrics.py"
            ),
        }
        result_path = output / "pbr_bug_compatible_metrics.json"
        atomic_write_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
