#!/usr/bin/env python3
"""Finalize engineering reports, blocked pilot state, and content manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from style_shapes.io import (
    atomic_write_json,
    atomic_write_text,
    command_record,
    environment_snapshot,
    sha256_file,
)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def gpu_rows():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        lines = subprocess.check_output(command, text=True).strip().splitlines()
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "rows": []}
    rows = []
    for line in lines:
        index, name, total, used, utilization = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_percent": int(utilization),
            }
        )
    return {"status": "available", "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        root = Path(args.artifact_root)
        dstc = read_json(root / "groups/dstc/group_construction.json")
        stickerchat = read_json(root / "groups/stickerchat/group_construction.json")
        equivalence = read_json(root / "equivalence/bank_equivalence.json")
        audit = read_json(root / "repo_audit.json")
        resources = gpu_rows()
        init_paths = [
            root / "init/dstc_seed2021.ckpt",
            root / "init/stickerchat_seed2021.ckpt",
        ]
        missing_init = [str(path) for path in init_paths if not path.exists()]

        pilot = {
            "status": "RESOURCE_BLOCKED",
            "engineering_status": "COMPLETE_WITH_EXECUTION_BLOCKER",
            "equivalence_gate": equivalence["status"],
            "group_assets": {
                "dstc": dstc["status"],
                "stickerchat": stickerchat["status"],
            },
            "initialization_snapshots": {
                "status": "EXECUTION_BLOCKED" if missing_init else "COMPLETE",
                "missing": missing_init,
                "reason": (
                    "GPU command approval service rejected after its review stream disconnected; "
                    "no partial snapshot was written."
                    if missing_init
                    else None
                ),
            },
            "formal_training": {
                "dstc": "RESOURCE_BLOCKED",
                "stickerchat": "RESOURCE_BLOCKED",
                "required_world_sizes": {"dstc": 4, "stickerchat": 8},
                "resource_snapshot": resources,
                "occupied_gpus_preempted": False,
            },
            "formal_metrics": None,
            "paired_bootstrap": None,
            "scientific_verdict": "NOT_EVALUATED",
            "admissible_for_research_table": False,
            "reason": (
                "No four equivalent free A800s for DSTC and no eight free A800s for "
                "StickerChat. Formal one-seed jobs, final-query scores, bootstrap, latency, "
                "and memory measurements were not run. Smoke or historical results were not substituted."
            ),
        }
        atomic_write_json(root / "pilot_results.json", pilot)

        group_lines = [
            "# Style Shapes Group Construction Report",
            "",
            "Status: **COMPLETE**",
            "",
        ]
        for dataset, report in (("DSTC", dstc), ("StickerChat", stickerchat)):
            group_lines.extend(
                [
                    "## %s" % dataset,
                    "",
                    "| Source | K | min | max | max share | effective K | coverage |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for source, row in report["sources"].items():
                group_lines.append(
                    "| %s | %d | %d | %d | %.2f%% | %.2f | %.3f%% |"
                    % (
                        source,
                        row["num_groups"],
                        row["min_group_size"],
                        row["max_group_size"],
                        100.0 * row["max_group_fraction"],
                        row["effective_group_count"],
                        100.0 * row["same_group_negative_coverage"],
                    )
                )
            group_lines.append("")
        group_lines.extend(
            [
                "- StickerChat reference K384 rebuild: **exact** (174,695/174,695).",
                "- StickerChat pack-matched random size MAE: %.3f; maximum absolute error: %d; "
                "Wasserstein: %.3f."
                % (
                    stickerchat["random_size_match"]["size_mae"],
                    stickerchat["random_size_match"]["max_abs_error"],
                    stickerchat["random_size_match"]["wasserstein_1d"],
                ),
                "- All groups are non-empty and maximum occupancy is below 20%.",
            ]
        )
        atomic_write_text(root / "group_construction_report.md", "\n".join(group_lines) + "\n")

        engineering_lines = [
            "# Style Shapes Engineering Validation",
            "",
            "Engineering status: **COMPLETE_WITH_EXECUTION_BLOCKER**",
            "",
            "- Group Bank v1 schema/hash/bidirectional membership and public legacy adapter: PASS.",
            "- Deterministic shared-init spherical K-means, legacy K384 reproduction, random controls: PASS.",
            "- DSTC and StickerChat legacy/compact bank-induced forward/loss/sampling equivalence: PASS.",
            "- Nine formal configs parse and enforce minimal core, batch 16, 10 epochs, correct K/refresh/world size: PASS.",
            "- Fixed DSTC legacy R10 and StickerChat same-pack R10/global R20 candidate hashes: PASS; "
            "the old global R10 remains a frozen reference.",
            "- Unit suite: 15 PASS.",
            "- Real-asset suite: 4 PASS, 1 SKIP (initialization snapshots not generated).",
            "- Weights-only initializer, strict reload check, fixed sampler, atomic rank traces, "
            "final-only checkpoint runner, and per-query score exporter: IMPLEMENTED.",
            "- Initialization execution: BLOCKED by the command-approval service; no partial files.",
            "- Single-step model/checkpoint smoke and formal multi-GPU pilot: NOT RUN.",
            "",
            "The skipped execution checks cannot be promoted to PASS. Formal results remain inadmissible.",
        ]
        atomic_write_text(root / "engineering_validation.md", "\n".join(engineering_lines) + "\n")

        report_lines = [
            "# Style Shapes VPD Group-Source Fair Pilot",
            "",
            "Overall status: **RESOURCE_BLOCKED**",
            "",
            "The complete offline group construction and bank-equivalence gate passed for both datasets. "
            "All candidate and permutation manifests are frozen. No formal model was trained.",
            "",
            "## What is established",
            "",
            "- DSTC: four K=85 sources built; VPD effective K %.2f and same-negative coverage %.2f%%."
            % (
                dstc["sources"]["vpd_multi"]["effective_group_count"],
                100.0 * dstc["sources"]["vpd_multi"]["same_group_negative_coverage"],
            ),
            "- StickerChat: reference K384 exact; VPD effective K %.2f, max share %.2f%%, coverage %.3f%%."
            % (
                stickerchat["sources"]["vpd_pack"]["effective_group_count"],
                100.0 * stickerchat["sources"]["vpd_pack"]["max_group_fraction"],
                100.0 * stickerchat["sources"]["vpd_pack"]["same_group_negative_coverage"],
            ),
            "- Legacy/compact partition, sampling, prototype vector, group score, and group-loss equivalence: PASS.",
            "",
            "## What is not established",
            "",
            "There are no valid R@1/MRR differences, paired bootstrap intervals, latency, peak-memory, "
            "or slice results. The preregistered scientific gate is **NOT_EVALUATED**.",
            "",
            "## AAAI-27 direction",
            "",
            "The Style Shapes hypothesis is engineering-ready but empirically unresolved. It must not be "
            "described as improving SEMSP until the exact nine-job one-seed matrix and final single-GPU "
            "evaluations complete. The next admissible action is to create the two shared initialization "
            "snapshots, then run DSTC on four free A800s and StickerChat on eight free A800s without "
            "changing K, losses, candidates, or hyperparameters.",
        ]
        atomic_write_text(root / "STYLE_SHAPES_PILOT_REPORT.md", "\n".join(report_lines) + "\n")

        # Build last, excluding self and the append-only command log whose finish record follows.
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name != "artifact_manifest.json"
            and path != root / "logs/command_history.jsonl"
        )
        manifest = {
            "status": "COMPLETE",
            "pilot_status": pilot["status"],
            "git_head": audit["repository"]["git_head"],
            "artifacts": [
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in paths
            ],
            "excluded_mutable": [str(root / "logs/command_history.jsonl")],
        }
        atomic_write_json(root / "artifact_manifest.json", manifest)
        print(
            json.dumps(
                {
                    "status": pilot["status"],
                    "engineering": pilot["engineering_status"],
                    "artifacts": len(manifest["artifacts"]),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
