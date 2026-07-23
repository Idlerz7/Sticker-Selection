#!/usr/bin/env python3
"""Build and validate all compact Style Shapes group-source banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from style_shapes.builders import build_dstc, build_stickerchat, group_stats
from style_shapes.io import atomic_write_json, atomic_write_text, command_record, hash_value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def main():
    args = parse_args()
    with command_record(args, args.artifact_root):
        config = load_config(args.config)
        if config["dataset"] == "dstc":
            banks = build_dstc(config)
            extra = {}
        elif config["dataset"] == "stickerchat":
            banks, extra = build_stickerchat(config)
        else:
            raise ValueError("unsupported dataset")
        output = Path(config["output_dir"])
        stats = {}
        for source, bank in banks.items():
            bank_path = output / source / "group_bank.json"
            if bank_path.exists():
                existing = type(bank).load(str(bank_path))
                def stable_provenance(value):
                    return {key: item for key, item in value.items() if key not in {"created_utc", "membership_hash"}}
                compatible = (
                    existing.membership_hash == bank.membership_hash
                    and hash_value(stable_provenance(existing.value["provenance"]))
                    == hash_value(stable_provenance(bank.value["provenance"]))
                )
                if not compatible:
                    raise RuntimeError("refusing to overwrite incompatible group bank: %s" % bank_path)
            else:
                bank.save(str(bank_path))
            atomic_write_text(output / source / "membership.sha256", bank.membership_hash + "\n")
            stats[source] = group_stats(bank)
        report = {
            "status": "COMPLETE",
            "dataset": config["dataset"],
            "sources": stats,
            **extra,
        }
        atomic_write_json(output / "group_construction.json", report)
        lines = [
            "# %s Group Construction" % config["dataset"],
            "",
            "Status: **COMPLETE**",
            "",
            "| Source | K | min | max | effective K | same-negative coverage | membership |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for source, row in stats.items():
            lines.append(
                "| %s | %d | %d | %d | %.3f | %.2f%% | `%s` |"
                % (
                    source,
                    row["num_groups"],
                    row["min_group_size"],
                    row["max_group_size"],
                    row["effective_group_count"],
                    100.0 * row["same_group_negative_coverage"],
                    row["membership_hash"][:16],
                )
            )
        atomic_write_text(output / "group_construction.md", "\n".join(lines) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

