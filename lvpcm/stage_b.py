"""One-seed frozen Stage-B training and evaluation."""

from __future__ import annotations

import math
import time
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from .metrics import paired_bootstrap_delta, ranking_metrics
from .scorers import ConditionalScorer, UnconditionalScorer, trainable_parameter_count


VARIANTS = ("base", "unconditional", "final_clip", "single_vpd", "local_lpc", "shuffled_lpc")


def centered_pca_condition(bundle: dict, n_components=256, seed=2021) -> dict:
    ids = bundle["ids"].long()
    features = bundle["features"].float().numpy()
    train_ids = set(int(value) for value in bundle["train_ids"].tolist())
    train_rows = [row for row, value in enumerate(ids.tolist()) if int(value) in train_ids]
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=False, random_state=seed)
    pca.fit(features[train_rows])
    transformed = pca.transform(features).astype(np.float32)
    transformed /= np.maximum(np.linalg.norm(transformed, axis=1, keepdims=True), 1e-12)
    return {
        "ids": ids, "features": torch.from_numpy(transformed), "train_ids": bundle["train_ids"].long(),
        "pca_mean": torch.from_numpy(pca.mean_.astype(np.float32)),
        "components": torch.from_numpy(pca.components_.astype(np.float32)),
        "explained_variance": torch.from_numpy(pca.explained_variance_.astype(np.float32)),
    }


def split_safe_shuffle(bundle: dict, seed=2021):
    ids = bundle["ids"].long()
    train_set = set(int(value) for value in bundle["train_ids"].tolist())
    train_rows = [row for row, value in enumerate(ids.tolist()) if int(value) in train_set]
    nontrain_rows = [row for row, value in enumerate(ids.tolist()) if int(value) not in train_set]
    generator = torch.Generator().manual_seed(seed)
    output = bundle["features"].clone()
    mapping = {}
    for rows in (train_rows, nontrain_rows):
        if not rows:
            continue
        permutation = torch.randperm(len(rows), generator=generator).tolist()
        for destination_offset, source_offset in enumerate(permutation):
            destination = rows[destination_offset]; source = rows[source_offset]
            output[destination] = bundle["features"][source]
            mapping[int(ids[destination])] = int(ids[source])
    if set(mapping) != set(ids.tolist()) or len(set(mapping.values())) != len(ids):
        raise RuntimeError("shuffled condition is not a bijection")
    return {**bundle, "features": output}, mapping


def condition_tensor(bundle: dict, candidate_ids: torch.Tensor) -> torch.Tensor:
    row_by_id = {int(value): row for row, value in enumerate(bundle["ids"].tolist())}
    rows = torch.tensor([[row_by_id[int(value)] for value in row] for row in candidate_ids.tolist()], dtype=torch.long)
    return bundle["features"][rows].float()


def epoch_permutations(query_count: int, epochs: int, seed=2021):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randperm(query_count, generator=generator) for _ in range(epochs)]


def _score(model, variant, base, h, condition=None):
    if variant == "base":
        return base
    if variant == "unconditional":
        return model(base, h)
    return model(base, h, condition)


def train_variant(variant, train_cache, condition, sigma_b, config, permutations, device):
    if variant == "base":
        return None, []
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config["seed"])
    model = UnconditionalScorer(sigma_b, rank=config["unconditional_rank"]) if variant == "unconditional" else ConditionalScorer(
        sigma_b, rank=config["condition_rank"], condition_dim=config["condition_dim"]
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], betas=tuple(config["betas"]),
        eps=config["epsilon"], weight_decay=config["weight_decay"],
    )
    steps_per_epoch = math.ceil(len(train_cache["b"]) / config["batch_size"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * config["epochs"])
    losses = []
    for epoch, permutation in enumerate(permutations):
        epoch_losses = []
        for start in range(0, len(permutation), config["batch_size"]):
            indices = permutation[start:start + config["batch_size"]]
            base = train_cache["b"][indices].to(device)
            h = train_cache["h"][indices].to(device)
            positive = train_cache["positive_index"][indices].to(device)
            z = None if condition is None else condition[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = _score(model, variant, base, h, z)
            loss = F.cross_entropy(scores, positive)
            loss.backward(); optimizer.step(); scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append({"epoch": epoch + 1, "mean_listwise_cross_entropy": float(np.mean(epoch_losses)), "final_lr": scheduler.get_last_lr()[0]})
    return model.eval(), losses


@torch.no_grad()
def evaluate_variant(model, variant, cache, condition, device, batch_size=256):
    all_scores = []
    all_delta = []
    latencies = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(cache["b"]), batch_size):
        end = min(start + batch_size, len(cache["b"]))
        base = cache["b"][start:end].to(device)
        h = cache["h"][start:end].to(device)
        z = None if condition is None else condition[start:end].to(device)
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        before = time.perf_counter()
        scores = _score(model, variant, base, h, z)
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - before) * 1000)
        all_scores.append(scores.cpu())
        all_delta.append((scores - base).cpu())
    scores = torch.cat(all_scores); delta = torch.cat(all_delta)
    metrics = ranking_metrics(scores, cache["positive_index"])
    peak = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
    return scores, delta, {
        **{key: value for key, value in metrics.items() if key != "ranks"},
        "ranks": metrics["ranks"],
        "latency_ms_mean_per_query_batch": float(np.mean(latencies)),
        "latency_ms_p95_per_query_batch": float(np.quantile(latencies, 0.95)),
        "latency_batch_size": batch_size, "peak_cuda_memory_bytes": int(peak),
        "delta": {"mean": float(delta.mean()), "std": float(delta.std(unbiased=False)), "max_abs": float(delta.abs().max())},
    }


def per_query_metrics(scores, positive):
    order = torch.argsort(scores, dim=1, descending=True)
    ranks = (order == positive.view(-1, 1)).nonzero(as_tuple=False)[:, 1] + 1
    return {
        "mrr": (1 / ranks.float()).numpy(),
        "r@1": (ranks <= 1).float().numpy(),
        "r@2": (ranks <= 2).float().numpy(),
        "r@5": (ranks <= 5).float().numpy(),
        "r@10": (ranks <= 10).float().numpy(),
    }


def paired_comparisons(score_by_variant, positive, replicates=10000, seed=2021):
    local = per_query_metrics(score_by_variant["local_lpc"], positive)
    comparisons = {}
    for other in ("unconditional", "final_clip", "single_vpd", "shuffled_lpc"):
        baseline = per_query_metrics(score_by_variant[other], positive)
        comparisons["local_minus_%s" % other] = {
            metric: paired_bootstrap_delta(local[metric], baseline[metric], replicates, seed)
            for metric in ("mrr", "r@1", "r@2", "r@5", "r@10")
        }
    return comparisons


def slice_metrics(scores, positive, local_bundle, candidate_ids):
    row_by_id = {int(value): row for row, value in enumerate(local_bundle["ids"].tolist())}
    gold_ids = candidate_ids[torch.arange(len(candidate_ids)), positive]
    degrees = torch.tensor([local_bundle["degree"][row_by_id[int(value)]] for value in gold_ids])
    fallback = torch.tensor([local_bundle["fallback"][row_by_id[int(value)]] for value in gold_ids])
    dispersions = []
    for candidates in candidate_ids.tolist():
        values = torch.stack([local_bundle["features"][row_by_id[int(value)]] for value in candidates]).float()
        distances = 1 - values.matmul(values.T)
        upper = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
        dispersions.append(float(upper.mean()))
    dispersions = torch.tensor(dispersions)
    lower, upper = torch.quantile(dispersions, torch.tensor([1 / 3, 2 / 3]))
    result = {}
    masks = {
        "fallback": fallback, "nonfallback": ~fallback,
        "degree_1_3": (degrees >= 1) & (degrees <= 3), "degree_4_7": (degrees >= 4) & (degrees <= 7),
        "degree_8_10": degrees >= 8,
        "dispersion_low": dispersions <= lower,
        "dispersion_mid": (dispersions > lower) & (dispersions <= upper),
        "dispersion_high": dispersions > upper,
    }
    for name, mask in masks.items():
        if mask.any():
            result[name] = {"queries": int(mask.sum()), **{k: v for k, v in ranking_metrics(scores[mask], positive[mask]).items() if k != "ranks"}}
    return result
