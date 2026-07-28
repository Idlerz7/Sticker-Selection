"""Completion contracts that prevent partial formal jobs from being accepted."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping


PILOT_SCHEMA_VERSION = "style_shapes.pilot.v2"
SCHEDULER_CONTRACT = "rank_sharded_loader_no_world_redivision.v1"


def completed_manifest_uses_current_contract(
    value: Mapping, expected_world_size: int
) -> bool:
    return (
        value.get("status") == "TRAIN_COMPLETE"
        and value.get("schema_version") == PILOT_SCHEMA_VERSION
        and value.get("scheduler_contract") == SCHEDULER_CONTRACT
        and int(value.get("world_size", -1)) == int(expected_world_size)
        and int(value.get("expected_optimizer_steps", -1))
        == int(value.get("completed_optimizer_steps", -2))
    )


def training_completion_errors(
    *,
    interrupted: bool,
    completed_optimizer_steps: int,
    expected_optimizer_steps: int,
    current_epoch: int,
    expected_epochs: int,
    trace_dir: str,
    world_size: int,
) -> List[str]:
    errors: List[str] = []
    if interrupted:
        errors.append("trainer reported interruption")
    if int(completed_optimizer_steps) != int(expected_optimizer_steps):
        errors.append(
            "optimizer steps %d != %d"
            % (int(completed_optimizer_steps), int(expected_optimizer_steps))
        )
    if int(current_epoch) < int(expected_epochs) - 1:
        errors.append(
            "current epoch %d did not reach %d"
            % (int(current_epoch), int(expected_epochs) - 1)
        )
    root = Path(trace_dir)
    for rank in range(int(world_size)):
        final_path = root / ("rank_%02d.jsonl" % rank)
        partial_path = root / ("rank_%02d.jsonl.partial" % rank)
        if not final_path.exists():
            errors.append("missing finalized trace %s" % final_path)
        if partial_path.exists():
            errors.append("partial trace remains %s" % partial_path)
    return errors
