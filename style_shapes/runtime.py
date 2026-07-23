"""Runtime resource resolution for Style Shapes formal jobs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Tuple

from .io import hash_value
from .permutations import (
    load_permutation_manifest,
    save_permutation_manifest,
    validate_permutation_manifest,
)


def derive_permutation_world_size(value: Mapping, world_size: int) -> dict:
    """Preserve every global epoch order while changing only DDP sharding."""
    validate_permutation_manifest(value)
    if int(world_size) <= 0:
        raise ValueError("world_size must be positive")
    core = {
        key: item for key, item in value.items() if key != "manifest_hash"
    }
    core["world_size"] = int(world_size)
    return {**core, "manifest_hash": hash_value(core)}


def _world_size_path(path: str, world_size: int) -> Path:
    source = Path(path)
    match = re.search(r"_ws\d+$", source.stem)
    if match:
        stem = source.stem[: match.start()] + "_ws%d" % int(world_size)
    else:
        stem = source.stem + "_ws%d" % int(world_size)
    return source.with_name(stem + source.suffix)


def resolve_permutation_world_size(path: str, world_size: int) -> Tuple[str, dict]:
    """Return an idempotent manifest matching the runtime GPU world size."""
    source = load_permutation_manifest(path)
    if int(source["world_size"]) == int(world_size):
        return str(path), source

    expected = derive_permutation_world_size(source, world_size)
    target = _world_size_path(path, world_size)
    if target.exists():
        existing = load_permutation_manifest(str(target), int(source["num_rows"]))
        if existing != expected:
            raise RuntimeError(
                "existing runtime permutation is incompatible: %s" % target
            )
        return str(target), existing

    save_permutation_manifest(str(target), expected)
    return str(target), expected
