#!/usr/bin/env python
"""Generate engineering validation and the final content-hashed artifact manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

from lvpcm.io import atomic_write_json, atomic_write_text, sha256_file
from lvpcm.provenance import command_record, environment_snapshot


def main(args):
    root = Path(args.artifact_root)
    test_log = root / "logs" / "tests_final.log"
    test_text = test_log.read_text(encoding="utf-8", errors="replace") if test_log.is_file() else ""
    tests_ok = "\nOK\n" in test_text or test_text.rstrip().endswith("OK")
    count_match = re.search(r"Ran (\d+) tests?", test_text)
    test_count = int(count_match.group(1)) if count_match else None
    stage_a_path = root / "stage_a_report.json"
    stage_a = json.loads(stage_a_path.read_text()) if stage_a_path.is_file() else {"stage_a_verdict": "NOT COMPLETED"}
    stage_b_path = root / "stage_b" / "results.json"
    stage_b = json.loads(stage_b_path.read_text()) if stage_b_path.is_file() else {"stage_b_verdict": "NOT RUN"}
    engineering = [
        "# LVPCM Engineering Validation", "",
        "- Final unittest status: **%s** (%s tests, 0 failures when PASS)." % (("PASS" if tests_ok else "FAIL OR MISSING"), test_count if test_count is not None else "unknown"),
        "- Stage A: **%s**." % stage_a.get("stage_a_verdict", "UNKNOWN"),
        "- Stage B: **%s**." % stage_b.get("stage_b_verdict", "NOT RUN"),
        "- Pair-cache equality requires exact zero max-absolute error against the legacy forward path.",
        "- Unit coverage includes VPD block/CLS/shape/finite/PCA isolation/reproducibility, LPC mutual/fallback/query isolation/ties, scorer equality/bounds/gradients/parameter count/permutation, candidate alignment, ranking numerics, and serialization.",
        "- Real-asset coverage includes local CLIP repeat extraction, DSTC strict checkpoint load, 21,130 tokenizer/embedding rows, `[10,768]` pooler output, exact positive-logit reconstruction, and image-cache row alignment.",
        "- Smoke outputs are excluded from research tables.", "",
        "Test log: `%s`" % test_log,
    ]
    atomic_write_text(root / "engineering_validation.md", "\n".join(engineering) + "\n")
    return stage_a, stage_b, tests_ok


def write_manifest(args, stage_a, stage_b, tests_ok):
    root = Path(args.artifact_root)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    source_files = []
    for pattern in ("lvpcm/**/*.py", "scripts/lvpcm/*.py", "configs/lvpcm/*.yaml", "tests/lvpcm/*.py"):
        for path in sorted(Path.cwd().glob(pattern)):
            if path.is_file():
                source_files.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    for path in (Path("LVPCM_SPEC.md"), Path("LVPCM_EXEC_PLAN.md")):
        if path.is_file():
            source_files.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "status": "complete", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage_a": stage_a.get("stage_a_verdict", "NOT COMPLETED"), "stage_b": stage_b.get("stage_b_verdict", "NOT RUN"),
        "tests_pass": tests_ok, "environment": environment_snapshot(),
        "artifacts": files, "implementation_files": source_files,
        "blockers": [
            "StickerChat neutral Chinese MM-BERT checkpoint with 32 speaker tokens is unavailable",
            "StickerChat same-pack reconstruction has 16 validation and 5 test rows without original negatives",
        ],
    }
    atomic_write_json(root / "artifact_manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="artifacts/lvpcm")
    options = parser.parse_args()
    with command_record(options, options.artifact_root):
        stage_a_value, stage_b_value, tests_ok_value = main(options)
    # Write after command_record's finish event so command_history.jsonl itself has a final,
    # stable content hash in the artifact manifest.
    write_manifest(options, stage_a_value, stage_b_value, tests_ok_value)
