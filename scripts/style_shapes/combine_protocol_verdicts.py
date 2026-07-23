#!/usr/bin/env python3
"""Combine preregistered protocol results; a missing required protocol is STOP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from style_shapes.io import atomic_write_json, atomic_write_text, command_record, sha256_file


REQUIRED = {
    "dstc": {"fixed_validation_r10"},
    "stickerchat": {"global_test_r10", "global_test_r20"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(REQUIRED), required=True)
    parser.add_argument("--result", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-root", default="artifacts/style_shapes")
    args = parser.parse_args()
    with command_record(args, args.artifact_root):
        results = {}
        files = {}
        for path in args.result:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            if value.get("dataset") != args.dataset or value.get("status") != "COMPLETE":
                raise ValueError("incompatible or incomplete protocol result: %s" % path)
            protocol = str(value["protocol"])
            if protocol in results:
                raise ValueError("duplicate protocol result: %s" % protocol)
            results[protocol] = value
            files[protocol] = {"path": path, "sha256": sha256_file(path)}
        missing = sorted(REQUIRED[args.dataset].difference(results))
        failed = sorted(
            protocol for protocol, value in results.items() if value.get("verdict") != "GO"
        )
        verdict = "GO" if not missing and not failed else "STOP"
        output = {
            "status": "COMPLETE",
            "dataset": args.dataset,
            "required_protocols": sorted(REQUIRED[args.dataset]),
            "missing_protocols": missing,
            "failed_protocols": failed,
            "protocol_results": files,
            "verdict": verdict,
        }
        atomic_write_json(args.output, output)
        atomic_write_text(
            str(Path(args.output).with_suffix(".md")),
            "# %s Style Shapes Final Verdict\n\nVerdict: **%s**\n\n"
            "Missing protocols: %s\n\nFailed protocols: %s\n"
            % (args.dataset, verdict, missing, failed),
        )
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

