"""Visual patch-distribution feature definitions and train-only reducers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BLOCKS = (3, 6, 9)
SINGLE_BLOCK = 6


def patch_distribution(hidden_states: Sequence[torch.Tensor], blocks: Sequence[int]) -> torch.Tensor:
    parts = []
    for block in blocks:
        if block <= 0 or block >= len(hidden_states):
            raise ValueError("block %d is not available in hidden_states" % block)
        state = hidden_states[block]
        if state.ndim != 3 or state.shape[1:] != (50, 768):
            raise ValueError("expected CLIP state [B,50,768], got %s" % (tuple(state.shape),))
        patches = state[:, 1:, :]
        parts.extend((patches.mean(dim=1), patches.std(dim=1, unbiased=False)))
    value = torch.cat(parts, dim=-1)
    if not torch.isfinite(value).all():
        raise ValueError("non-finite VPD values")
    return value


def single_vpd(hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
    return patch_distribution(hidden_states, (SINGLE_BLOCK,))


def multi_vpd(hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
    return patch_distribution(hidden_states, BLOCKS)


def l2_normalize(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(value.float(), p=2, dim=-1, eps=eps)


@dataclass
class FrozenReducer:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    train_ids: np.ndarray
    random_state: int = 2021

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        scaled = (values - self.scaler_mean) / self.scaler_scale
        reduced = (scaled - self.pca_mean).dot(self.components.T)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        return (reduced / np.maximum(norms, 1e-12)).astype(np.float32)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        output = self.transform_numpy(values.detach().cpu().numpy())
        return torch.from_numpy(output)

    def state_dict(self) -> dict:
        return {
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "pca_mean": self.pca_mean,
            "components": self.components,
            "explained_variance": self.explained_variance,
            "train_ids": self.train_ids,
            "random_state": self.random_state,
            "whiten": False,
            "svd_solver": "randomized",
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "FrozenReducer":
        return cls(**{key: state[key] for key in (
            "scaler_mean", "scaler_scale", "pca_mean", "components", "explained_variance", "train_ids", "random_state"
        )})


def fit_reducer(values: torch.Tensor, ids: torch.Tensor, train_ids: Iterable[int], n_components: int = 256, seed: int = 2021) -> FrozenReducer:
    values_np = values.detach().cpu().numpy().astype(np.float32, copy=False)
    ids_np = ids.detach().cpu().numpy().astype(np.int64, copy=False)
    row_by_id = {int(value): index for index, value in enumerate(ids_np.tolist())}
    frozen_train_ids = np.asarray(sorted(int(value) for value in train_ids), dtype=np.int64)
    try:
        rows = np.asarray([row_by_id[int(value)] for value in frozen_train_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("train ID absent from feature rows: %s" % exc)
    train = values_np[rows]
    if n_components > min(train.shape):
        raise ValueError("PCA components exceed training matrix rank bound")
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled = scaler.fit_transform(train)
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=False, random_state=seed)
    pca.fit(scaled)
    return FrozenReducer(
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        train_ids=frozen_train_ids,
        random_state=seed,
    )
