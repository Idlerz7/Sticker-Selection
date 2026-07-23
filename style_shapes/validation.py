"""Engineering gates for candidates, negative traces, and artifact manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .io import hash_value, sha256_file


def terminal_gold(row: Mapping) -> int:
    dialog = row.get("dialog")
    if not isinstance(dialog, list) or not dialog:
        raise ValueError("candidate row has no dialogue")
    return int(dialog[-1]["img_id"])


def candidate_file_audit(path: str, expected_candidates: int) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    for index, row in enumerate(rows):
        candidates = [int(value) for value in row.get("cand", [])]
        gold = terminal_gold(row)
        if len(candidates) != int(expected_candidates):
            raise ValueError("candidate size mismatch at row %d" % index)
        if len(set(candidates)) != len(candidates) or candidates.count(gold) != 1:
            raise ValueError("candidate uniqueness/positive mismatch at row %d" % index)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "candidate_count": int(expected_candidates),
        "normalized_hash": hash_value(
            [[terminal_gold(row), [int(value) for value in row["cand"]]] for row in rows]
        ),
    }


def merge_and_validate_traces(
    paths: Sequence[str],
    num_rows: int,
    epochs: int,
    membership_hash: str,
    output_path: str = "",
) -> dict:
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    counts = Counter()
    for record in records:
        if record["membership_hash"] != membership_hash:
            raise ValueError("trace membership hash mismatch")
        key = (int(record["epoch"]), int(record["source_row"]))
        counts[key] += 1
        for name in ("positive", "fallback", "cross", "same"):
            int(record[name])
    missing = [
        [epoch, source_row]
        for epoch in range(int(epochs))
        for source_row in range(int(num_rows))
        if counts[(epoch, source_row)] == 0
    ]
    if missing:
        raise ValueError("negative trace misses %d epoch/source rows" % len(missing))
    duplicates = sum(value - 1 for value in counts.values() if value > 1)
    result = {
        "status": "COMPLETE",
        "rank_trace_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "records": len(records),
        "expected_unique_records": int(num_rows) * int(epochs),
        "known_ddp_padding_records": duplicates,
        "complete_coverage": True,
        "membership_hash": membership_hash,
    }
    if output_path:
        from .io import atomic_write_json

        atomic_write_json(output_path, result)
    return result


def artifact_records(paths: Iterable[str]) -> list:
    output = []
    for raw in paths:
        path = Path(raw)
        output.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return output

