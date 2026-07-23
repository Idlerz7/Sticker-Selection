"""Fixed global epoch permutations and a DDP sampler that replays them."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterator, List, Mapping

import torch
from torch.utils.data import Dataset, Sampler

from .io import atomic_write_json, hash_value


def create_permutation_manifest(
    num_rows: int, epochs: int, world_size: int, seed: int = 2021
) -> dict:
    if num_rows <= 0 or epochs <= 0 or world_size <= 0:
        raise ValueError("num_rows, epochs and world_size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutations = [
        torch.randperm(num_rows, generator=generator).tolist() for _ in range(epochs)
    ]
    core = {
        "schema_version": "style_shapes.epoch_permutations.v1",
        "num_rows": int(num_rows),
        "epochs": int(epochs),
        "world_size": int(world_size),
        "seed": int(seed),
        "permutations": permutations,
        "sharding": "pad_from_epoch_prefix_then_rank_strided",
    }
    return {**core, "manifest_hash": hash_value(core)}


def validate_permutation_manifest(value: Mapping, expected_rows: int = None) -> None:
    if value.get("schema_version") != "style_shapes.epoch_permutations.v1":
        raise ValueError("unsupported permutation schema")
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != hash_value(core):
        raise ValueError("permutation manifest hash mismatch")
    num_rows = int(value["num_rows"])
    if expected_rows is not None and num_rows != int(expected_rows):
        raise ValueError("permutation dataset-size mismatch")
    expected = list(range(num_rows))
    if len(value["permutations"]) != int(value["epochs"]):
        raise ValueError("permutation epoch count mismatch")
    for epoch, row in enumerate(value["permutations"]):
        if sorted(int(item) for item in row) != expected:
            raise ValueError("epoch %d is not a complete permutation" % epoch)


def save_permutation_manifest(path: str, value: Mapping) -> None:
    validate_permutation_manifest(value)
    atomic_write_json(path, value)


def load_permutation_manifest(path: str, expected_rows: int = None) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    validate_permutation_manifest(value, expected_rows)
    return value


def sharded_epoch_indices(value: Mapping, epoch: int, rank: int) -> List[int]:
    world_size = int(value["world_size"])
    if rank < 0 or rank >= world_size:
        raise ValueError("rank outside manifest world size")
    base = [int(item) for item in value["permutations"][int(epoch) % int(value["epochs"])]]
    total_size = int(math.ceil(len(base) / float(world_size))) * world_size
    padded = base + base[: total_size - len(base)]
    return padded[rank:total_size:world_size]


class FixedEpochDistributedSampler(Sampler):
    def __init__(self, dataset: Dataset, manifest: Mapping, rank: int):
        validate_permutation_manifest(manifest, len(dataset))
        self.dataset = dataset
        self.manifest = dict(manifest)
        self.rank = int(rank)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        return iter(sharded_epoch_indices(self.manifest, self.epoch, self.rank))

    def __len__(self) -> int:
        return int(math.ceil(len(self.dataset) / float(int(self.manifest["world_size"]))))


class ExactDistributedEvalSampler(Sampler):
    """Rank-strided evaluation shard without padding or duplicated examples."""

    def __init__(self, dataset: Dataset, world_size: int, rank: int):
        if int(world_size) <= 0:
            raise ValueError("world_size must be positive")
        if int(rank) < 0 or int(rank) >= int(world_size):
            raise ValueError("rank outside evaluation world size")
        self.dataset = dataset
        self.world_size = int(world_size)
        self.rank = int(rank)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return int(math.ceil(remaining / float(self.world_size)))


class IndexedDataset(Dataset):
    """Add immutable source-row identity without changing the legacy dataset."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        value = dict(self.dataset[int(index)])
        value["_style_shapes_source_row"] = int(index)
        return value


def process_rank(world_size: int) -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
    else:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if rank < 0 or rank >= int(world_size):
        raise RuntimeError("runtime rank %d is incompatible with world size %d" % (rank, world_size))
    return rank
