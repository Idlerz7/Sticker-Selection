"""Deterministic spherical clustering and matched random controls."""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def l2_normalize(features: torch.Tensor) -> torch.Tensor:
    value = features.detach().cpu().float()
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError("features must be a finite rank-2 tensor")
    if (value.norm(dim=1) <= 0).any():
        raise ValueError("zero-norm feature row")
    return F.normalize(value, dim=1)


def shared_initial_indices(num_items: int, k: int, n_init: int, seed: int) -> List[List[int]]:
    if not 0 < k <= num_items or n_init <= 0:
        raise ValueError("invalid K or n_init")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return [
        torch.randperm(num_items, generator=generator)[:k].tolist() for _ in range(n_init)
    ]


def _repair_empty(
    features: torch.Tensor, assignments: torch.Tensor, similarities: torch.Tensor, k: int
) -> torch.Tensor:
    repaired = assignments.clone()
    counts = torch.bincount(repaired, minlength=k)
    empties = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if not empties:
        return repaired
    confidence = similarities.gather(1, repaired.unsqueeze(1)).squeeze(1)
    donor_order = sorted(range(len(repaired)), key=lambda index: (float(confidence[index]), index))
    used = set()
    for empty in empties:
        donor = next(
            index
            for index in donor_order
            if index not in used and int(counts[int(repaired[index])]) > 1
        )
        old = int(repaired[donor])
        repaired[donor] = int(empty)
        counts[old] -= 1
        counts[empty] += 1
        used.add(donor)
    return repaired


def spherical_lloyd(
    features: torch.Tensor,
    initial_indices: Sequence[int],
    max_iter: int,
    tol: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
    x = l2_normalize(features)
    k = len(initial_indices)
    centers = x.index_select(0, torch.tensor(list(initial_indices), dtype=torch.long)).clone()
    previous_objective = None
    assignments = torch.zeros(x.size(0), dtype=torch.long)
    for iteration in range(max(1, int(max_iter))):
        similarities = torch.matmul(x, centers.t())
        assignments = torch.argmax(similarities, dim=1)
        assignments = _repair_empty(x, assignments, similarities, k)
        new_centers = []
        for group_id in range(k):
            center = x[assignments == group_id].mean(dim=0)
            new_centers.append(F.normalize(center, dim=0))
        centers = torch.stack(new_centers)
        objective = float(
            torch.matmul(x, centers.t()).gather(1, assignments.unsqueeze(1)).mean().item()
        )
        if previous_objective is not None and abs(objective - previous_objective) <= float(tol):
            return assignments, centers, objective, iteration + 1
        previous_objective = objective
    return assignments, centers, float(previous_objective), max(1, int(max_iter))


def spherical_kmeans(
    features: torch.Tensor,
    k: int,
    seed: int = 2021,
    n_init: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
    initial_index_sets: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[torch.Tensor, dict]:
    x = l2_normalize(features)
    starts = (
        [list(row) for row in initial_index_sets]
        if initial_index_sets is not None
        else shared_initial_indices(x.size(0), k, n_init, seed)
    )
    if not starts or any(len(row) != k for row in starts):
        raise ValueError("invalid initial_index_sets")
    candidates = []
    for run, initial in enumerate(starts):
        assignments, _centers, objective, iterations = spherical_lloyd(
            x, initial, max_iter=max_iter, tol=tol
        )
        candidates.append((objective, run, assignments, iterations))
    objective, run, assignments, iterations = max(candidates, key=lambda row: (row[0], -row[1]))
    return assignments, {
        "objective": objective,
        "selected_init": run,
        "iterations": iterations,
        "initial_index_sets": starts,
        "seed": int(seed),
        "n_init": len(starts),
        "max_iter": int(max_iter),
        "tol": float(tol),
    }


def legacy_pack_kmeans(
    features: torch.Tensor,
    k: int,
    iterations: int = 40,
    seed: int = 20260330,
    initial_indices: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Byte-for-algorithm reproduction of the repository's old pack K-means."""
    x = l2_normalize(features)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    seeded_initial = torch.randperm(x.size(0), generator=generator)[:k]
    if initial_indices is None:
        initial = seeded_initial
    else:
        initial = torch.tensor(list(initial_indices), dtype=torch.long)
    centers = x.index_select(0, initial).clone()
    assignments = torch.zeros(x.size(0), dtype=torch.long)
    for _ in range(max(1, int(iterations))):
        similarities = torch.matmul(x, centers.t())
        assignments = torch.argmax(similarities, dim=1)
        next_centers = []
        for group_id in range(k):
            mask = assignments == group_id
            if bool(mask.any()):
                next_centers.append(F.normalize(x[mask].mean(dim=0), dim=0))
            else:
                index = int(torch.randint(x.size(0), (1,), generator=generator).item())
                next_centers.append(x[index])
        centers = torch.stack(next_centers)
    return assignments, {
        "seed": int(seed),
        "iterations": int(iterations),
        "initial_indices": initial.tolist(),
    }


def random_matched_assignments(
    sticker_ids: Sequence[int], target_sizes: Sequence[int], seed: int = 2021
) -> List[int]:
    if sum(int(value) for value in target_sizes) != len(sticker_ids):
        raise ValueError("target sizes do not cover sticker catalog")
    if any(int(value) <= 0 for value in target_sizes):
        raise ValueError("target groups must be non-empty")
    shuffled = list(range(len(sticker_ids)))
    random.Random(int(seed)).shuffle(shuffled)
    by_position = [0] * len(sticker_ids)
    cursor = 0
    for group_id, size in enumerate(target_sizes):
        for position in shuffled[cursor : cursor + int(size)]:
            by_position[position] = group_id
        cursor += int(size)
    return by_position


def random_pack_matched_assignments(
    pack_sizes: Sequence[int], target_sizes: Sequence[int], seed: int = 2021
) -> Tuple[List[int], dict]:
    if len(pack_sizes) < len(target_sizes) or sum(pack_sizes) != sum(target_sizes):
        raise ValueError("pack and target sizes are incompatible")
    if any(size <= 0 for size in pack_sizes) or any(size <= 0 for size in target_sizes):
        raise ValueError("all packs and groups must be non-empty")
    rng = random.Random(int(seed))
    tie = [rng.random() for _ in pack_sizes]
    order = sorted(range(len(pack_sizes)), key=lambda index: (-pack_sizes[index], tie[index], index))
    assignment = [-1] * len(pack_sizes)
    loads = [0] * len(target_sizes)
    # Seed every group first, choosing the largest remaining pack for the largest target.
    target_order = sorted(range(len(target_sizes)), key=lambda group: (-target_sizes[group], group))
    for pack_index, group_id in zip(order[: len(target_sizes)], target_order):
        assignment[pack_index] = group_id
        loads[group_id] += int(pack_sizes[pack_index])
    for pack_index in order[len(target_sizes) :]:
        group_id = max(
            range(len(target_sizes)),
            key=lambda group: (
                int(target_sizes[group]) - loads[group],
                -abs(int(target_sizes[group]) - (loads[group] + int(pack_sizes[pack_index]))),
                -group,
            ),
        )
        assignment[pack_index] = group_id
        loads[group_id] += int(pack_sizes[pack_index])
    errors = [loads[index] - int(target_sizes[index]) for index in range(len(loads))]
    return assignment, {
        "seed": int(seed),
        "target_sizes": [int(value) for value in target_sizes],
        "actual_sizes": loads,
        "size_mae": sum(abs(value) for value in errors) / float(len(errors)),
        "max_abs_error": max(abs(value) for value in errors),
        "wasserstein_1d": sum(
            abs(left - right)
            for left, right in zip(sorted(loads), sorted(int(value) for value in target_sizes))
        )
        / float(len(loads)),
    }


def effective_group_count(sizes: Iterable[int]) -> float:
    values = [float(value) for value in sizes]
    total = sum(values)
    probabilities = [value / total for value in values if value > 0]
    return math.exp(-sum(value * math.log(value) for value in probabilities))

