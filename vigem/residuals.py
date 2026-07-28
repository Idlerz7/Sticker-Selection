"""Frozen group-centred visual residuals and their masked listwise objective."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn.functional as F

from style_shapes.group_bank import GroupBank
from style_shapes.io import hash_value, sha256_file


INSTANCE_RESIDUAL_SCHEMA = "vigem.instance_residual.v1"


def _load_descriptor(path: str, expected_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError("descriptor bundle must be a mapping: %s" % path)
    ids = value.get("ids")
    features = value.get("features")
    if not isinstance(ids, torch.Tensor) or not isinstance(features, torch.Tensor):
        raise ValueError("descriptor bundle requires tensor ids/features: %s" % path)
    ids = ids.detach().cpu().long().reshape(-1)
    features = features.detach().cpu().float()
    if tuple(features.shape) != (int(ids.numel()), int(expected_dim)):
        raise ValueError(
            "descriptor shape mismatch for %s: expected [%d,%d], got %s"
            % (path, int(ids.numel()), int(expected_dim), tuple(features.shape))
        )
    if not torch.isfinite(features).all():
        raise ValueError("descriptor contains non-finite values: %s" % path)
    if len(set(int(item) for item in ids.tolist())) != int(ids.numel()):
        raise ValueError("descriptor IDs are not unique: %s" % path)
    return ids, features


def build_instance_residual_bundle(
    dataset: str,
    group_bank_path: str,
    final_clip_path: str,
    vpd_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build ``z_i - mean_g(z)`` after independently normalizing CLIP and VPD."""
    bank = GroupBank.load(group_bank_path)
    if str(bank.dataset) != str(dataset):
        raise ValueError("group bank dataset mismatch")
    clip_ids, clip = _load_descriptor(final_clip_path, 512)
    vpd_ids, vpd = _load_descriptor(vpd_path, 256)
    bank_ids = torch.tensor(bank.sticker_ids, dtype=torch.long)
    if not torch.equal(clip_ids, vpd_ids) or not torch.equal(clip_ids, bank_ids):
        raise ValueError("CLIP, VPD, and Group Bank ID sequences must be identical")
    expected_ids = torch.arange(int(bank_ids.numel()), dtype=torch.long)
    if not torch.equal(bank_ids, expected_ids):
        raise ValueError(
            "runtime residual lookup requires dense sticker IDs ordered from 0"
        )

    clip = F.normalize(clip, dim=-1)
    vpd = F.normalize(vpd, dim=-1)
    z = torch.cat([clip, vpd], dim=-1).contiguous()
    group_ids = torch.tensor(bank.sticker_to_group, dtype=torch.long)
    group_sizes = torch.tensor(
        [int(value) for value in bank.value["group_sizes"]], dtype=torch.long
    )
    centroids = torch.zeros(
        (int(bank.num_groups), int(z.size(1))), dtype=torch.float32
    )
    centroids.index_add_(0, group_ids, z)
    centroids = centroids / group_sizes.float().unsqueeze(-1)
    residuals = (z - centroids.index_select(0, group_ids)).contiguous()
    singleton = group_sizes.index_select(0, group_ids).eq(1)
    residuals[singleton] = 0.0
    if not torch.isfinite(residuals).all():
        raise RuntimeError("computed residual bundle contains non-finite values")
    if singleton.any() and torch.count_nonzero(residuals[singleton]).item() != 0:
        raise RuntimeError("singleton residuals must be exactly zero")

    source_hashes = {
        "group_bank": sha256_file(group_bank_path),
        "final_clip": sha256_file(final_clip_path),
        "vpd": sha256_file(vpd_path),
    }
    manifest_core = {
        "schema_version": INSTANCE_RESIDUAL_SCHEMA,
        "dataset": str(dataset),
        "group_source": str(bank.group_source),
        "num_stickers": int(bank_ids.numel()),
        "num_groups": int(bank.num_groups),
        "feature_schema": {
            "final_clip_dim": 512,
            "vpd_dim": 256,
            "residual_dim": 768,
            "dtype": "float32",
            "normalization": "family-wise L2 before concatenation",
            "centering": "full Group Bank catalog mean",
        },
        "membership_hash": str(bank.membership_hash),
        "singleton_stickers": int(singleton.sum().item()),
        "inputs": {
            "group_bank": {
                "path": str(group_bank_path),
                "sha256": source_hashes["group_bank"],
            },
            "final_clip": {
                "path": str(final_clip_path),
                "sha256": source_hashes["final_clip"],
            },
            "vpd": {
                "path": str(vpd_path),
                "sha256": source_hashes["vpd"],
            },
        },
    }
    manifest_hash = hash_value(manifest_core)
    manifest = dict(manifest_core)
    manifest["manifest_hash"] = manifest_hash
    payload = {
        "schema_version": INSTANCE_RESIDUAL_SCHEMA,
        "ids": bank_ids,
        "residuals": residuals,
        "group_ids": group_ids,
        "group_sizes": group_sizes,
        "membership_hash": str(bank.membership_hash),
        "manifest_hash": manifest_hash,
    }
    return payload, manifest


@dataclass(frozen=True)
class InstanceResidualBundle:
    ids: torch.Tensor
    residuals: torch.Tensor
    group_ids: torch.Tensor
    group_sizes: torch.Tensor
    membership_hash: str
    manifest_hash: str

    @classmethod
    def load(
        cls,
        path: str,
        expected_membership_hash: str = "",
    ) -> "InstanceResidualBundle":
        value = torch.load(path, map_location="cpu")
        if not isinstance(value, Mapping):
            raise ValueError("residual bundle must be a mapping")
        if value.get("schema_version") != INSTANCE_RESIDUAL_SCHEMA:
            raise ValueError("unsupported residual bundle schema")
        bundle = cls(
            ids=value["ids"].detach().cpu().long().reshape(-1),
            residuals=value["residuals"].detach().cpu().float(),
            group_ids=value["group_ids"].detach().cpu().long().reshape(-1),
            group_sizes=value["group_sizes"].detach().cpu().long().reshape(-1),
            membership_hash=str(value["membership_hash"]),
            manifest_hash=str(value["manifest_hash"]),
        )
        bundle.validate(expected_membership_hash)
        return bundle

    def validate(self, expected_membership_hash: str = "") -> None:
        count = int(self.ids.numel())
        if not torch.equal(self.ids, torch.arange(count, dtype=torch.long)):
            raise ValueError("residual bundle IDs must be dense and ordered")
        if tuple(self.residuals.shape) != (count, 768):
            raise ValueError("residual bundle must have shape [N,768]")
        if tuple(self.group_ids.shape) != (count,):
            raise ValueError("residual group_ids must have shape [N]")
        if self.group_sizes.ndim != 1 or int(self.group_sizes.numel()) <= 0:
            raise ValueError("residual group_sizes must be a non-empty vector")
        if self.group_ids.min().item() < 0 or self.group_ids.max().item() >= int(
            self.group_sizes.numel()
        ):
            raise ValueError("residual group ID outside group_sizes")
        observed = torch.bincount(
            self.group_ids, minlength=int(self.group_sizes.numel())
        )
        if not torch.equal(observed, self.group_sizes):
            raise ValueError("residual group sizes are inconsistent")
        if not torch.isfinite(self.residuals).all():
            raise ValueError("residual bundle contains non-finite values")
        singleton = self.group_sizes.index_select(0, self.group_ids).eq(1)
        if singleton.any() and torch.count_nonzero(self.residuals[singleton]).item():
            raise ValueError("singleton residuals must be zero")
        if expected_membership_hash and self.membership_hash != str(
            expected_membership_hash
        ):
            raise ValueError("residual membership hash mismatch")


def masked_instance_listwise_loss(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CE over valid gold/local negatives; skip non-informative rows.

    A gold candidate can be intentionally invalid for the Instance objective
    when its visual group is a singleton (and therefore its group-centred
    residual is exactly zero).  Such rows still participate in the final and
    group objectives, but must be skipped by this auxiliary loss.
    """
    if scores.ndim != 2 or valid_mask.shape != scores.shape:
        raise ValueError("scores and valid_mask must have the same [B,N] shape")
    if scores.size(1) < 2:
        raise ValueError("instance listwise loss requires at least two candidates")
    valid_mask = valid_mask.bool()
    eligible = valid_mask[:, 0] & valid_mask[:, 1:].any(dim=1)
    if not bool(eligible.any()):
        return scores.sum() * 0.0, eligible
    selected_scores = scores[eligible]
    selected_mask = valid_mask[eligible]
    masked_scores = selected_scores.masked_fill(
        ~selected_mask, torch.finfo(selected_scores.dtype).min
    )
    labels = torch.zeros(
        int(masked_scores.size(0)), dtype=torch.long, device=scores.device
    )
    return F.cross_entropy(masked_scores, labels), eligible
