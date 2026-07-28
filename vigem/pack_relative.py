"""Original-pack-centred residual assets for the independent VIGEM repair."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from style_shapes.group_bank import GroupBank
from style_shapes.io import hash_value, sha256_file


PACK_RELATIVE_SCHEMA = "vigem.pack_relative_residual.v1"


def _load_descriptor(path: str, expected_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError("descriptor must be a mapping: %s" % path)
    ids = value.get("ids")
    features = value.get("features")
    if not isinstance(ids, torch.Tensor) or not isinstance(features, torch.Tensor):
        raise ValueError("descriptor must contain tensor ids/features: %s" % path)
    ids = ids.detach().cpu().long()
    features = features.detach().cpu().float()
    if tuple(features.shape) != (int(ids.numel()), int(expected_dim)):
        raise ValueError(
            "descriptor shape mismatch for %s: %s" % (path, tuple(features.shape))
        )
    if not torch.isfinite(features).all():
        raise ValueError("descriptor contains non-finite values: %s" % path)
    return ids, features


def _pack_membership(
    metadata_path: str, expected_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, Sequence[str], str]:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    rows = value.get("stickers") if isinstance(value, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("sticker metadata must contain a stickers list")
    by_id: Dict[int, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid sticker metadata row")
        sticker_id = int(row["internal_img_id"])
        pack_key = str(row["img_set"])
        if sticker_id in by_id:
            raise ValueError("duplicate internal sticker ID in metadata")
        if not pack_key:
            raise ValueError("empty original pack ID")
        by_id[sticker_id] = pack_key
    ordered_ids = [int(item) for item in expected_ids.tolist()]
    if set(by_id) != set(ordered_ids):
        missing = sorted(set(ordered_ids) - set(by_id))
        extra = sorted(set(by_id) - set(ordered_ids))
        raise ValueError(
            "metadata/descriptor ID mismatch: missing=%s extra=%s"
            % (missing[:8], extra[:8])
        )
    pack_keys = sorted(set(by_id.values()))
    pack_to_id = {key: index for index, key in enumerate(pack_keys)}
    pack_ids = torch.tensor(
        [pack_to_id[by_id[sticker_id]] for sticker_id in ordered_ids],
        dtype=torch.long,
    )
    pack_sizes = torch.bincount(pack_ids, minlength=len(pack_keys)).long()
    if int(pack_sizes.numel()) != len(pack_keys) or bool(pack_sizes.eq(0).any()):
        raise ValueError("original pack IDs must form a dense non-empty partition")
    membership_hash = hash_value(
        [[sticker_id, by_id[sticker_id]] for sticker_id in sorted(ordered_ids)]
    )
    return pack_ids, pack_sizes, pack_keys, membership_hash


def build_pack_relative_bundle(
    metadata_path: str,
    group_bank_path: str,
    final_clip_path: str,
    vpd_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build CLIP+VPD residuals centred within each original StickerChat pack."""
    bank = GroupBank.load(group_bank_path)
    if bank.dataset != "stickerchat":
        raise ValueError("pack-relative residuals are StickerChat-only")
    clip_ids, clip = _load_descriptor(final_clip_path, 512)
    vpd_ids, vpd = _load_descriptor(vpd_path, 256)
    bank_ids = torch.tensor(bank.sticker_ids, dtype=torch.long)
    if not torch.equal(clip_ids, vpd_ids) or not torch.equal(clip_ids, bank_ids):
        raise ValueError("CLIP, VPD, and Group Bank ID sequences must be identical")
    expected = torch.arange(int(clip_ids.numel()), dtype=torch.long)
    if not torch.equal(clip_ids, expected):
        raise ValueError("runtime pack-relative lookup requires dense ordered IDs")

    pack_ids, pack_sizes, pack_keys, pack_hash = _pack_membership(
        metadata_path, clip_ids
    )
    clip = F.normalize(clip, dim=-1)
    vpd = F.normalize(vpd, dim=-1)
    features = torch.cat([clip, vpd], dim=-1).contiguous()
    centroids = torch.zeros(
        int(pack_sizes.numel()), int(features.size(1)), dtype=torch.float32
    )
    centroids.index_add_(0, pack_ids, features)
    centroids = centroids / pack_sizes.float().unsqueeze(-1)
    residuals = (
        features - centroids.index_select(0, pack_ids)
    ).contiguous()
    singleton = pack_sizes.index_select(0, pack_ids).eq(1)
    residuals[singleton] = 0.0
    if not torch.isfinite(residuals).all():
        raise RuntimeError("pack-relative residual contains non-finite values")
    if singleton.any() and int(torch.count_nonzero(residuals[singleton]).item()):
        raise RuntimeError("singleton original-pack residuals must be zero")

    vpd_group_ids = torch.tensor(bank.sticker_to_group, dtype=torch.long)
    vpd_group_sizes = torch.tensor(
        bank.value["group_sizes"], dtype=torch.long
    )
    source_hashes = {
        "sticker_metadata": sha256_file(metadata_path),
        "group_bank": sha256_file(group_bank_path),
        "final_clip": sha256_file(final_clip_path),
        "vpd": sha256_file(vpd_path),
    }
    manifest_core = {
        "schema_version": PACK_RELATIVE_SCHEMA,
        "dataset": "stickerchat",
        "num_stickers": int(clip_ids.numel()),
        "num_packs": int(pack_sizes.numel()),
        "num_vpd_groups": int(vpd_group_sizes.numel()),
        "feature_schema": {
            "final_clip_dim": 512,
            "vpd_dim": 256,
            "residual_dim": 768,
            "family_normalization": "independent_l2",
            "centering": "original_pack_mean",
            "post_center_normalization": "none",
        },
        "pack_membership_hash": pack_hash,
        "vpd_membership_hash": bank.membership_hash,
        # Compatibility alias used by the common VIGEM model constructor.
        "membership_hash": bank.membership_hash,
        "source_hashes": source_hashes,
    }
    manifest_hash = hash_value(manifest_core)
    payload = {
        "schema_version": PACK_RELATIVE_SCHEMA,
        "ids": clip_ids,
        "residuals": residuals,
        "pack_ids": pack_ids,
        "pack_sizes": pack_sizes,
        "pack_keys": list(pack_keys),
        "vpd_group_ids": vpd_group_ids,
        "vpd_group_sizes": vpd_group_sizes,
        "pack_membership_hash": pack_hash,
        "vpd_membership_hash": bank.membership_hash,
        "manifest_hash": manifest_hash,
    }
    manifest = dict(manifest_core)
    manifest["manifest_hash"] = manifest_hash
    return payload, manifest


@dataclass
class PackRelativeResidualBundle:
    ids: torch.Tensor
    residuals: torch.Tensor
    pack_ids: torch.Tensor
    pack_sizes: torch.Tensor
    pack_keys: Sequence[str]
    vpd_group_ids: torch.Tensor
    vpd_group_sizes: torch.Tensor
    pack_membership_hash: str
    vpd_membership_hash: str
    manifest_hash: str

    @property
    def group_ids(self) -> torch.Tensor:
        """Compatibility alias for inherited VIGEM lookup helpers."""
        return self.vpd_group_ids

    @property
    def group_sizes(self) -> torch.Tensor:
        return self.vpd_group_sizes

    @classmethod
    def load(
        cls, path: str, expected_vpd_membership_hash: str
    ) -> "PackRelativeResidualBundle":
        value = torch.load(path, map_location="cpu")
        if not isinstance(value, Mapping):
            raise ValueError("pack-relative bundle must be a mapping")
        if value.get("schema_version") != PACK_RELATIVE_SCHEMA:
            raise ValueError("unsupported pack-relative residual schema")
        bundle = cls(
            ids=value["ids"].detach().cpu().long(),
            residuals=value["residuals"].detach().cpu().float(),
            pack_ids=value["pack_ids"].detach().cpu().long(),
            pack_sizes=value["pack_sizes"].detach().cpu().long(),
            pack_keys=[str(item) for item in value["pack_keys"]],
            vpd_group_ids=value["vpd_group_ids"].detach().cpu().long(),
            vpd_group_sizes=value["vpd_group_sizes"].detach().cpu().long(),
            pack_membership_hash=str(value["pack_membership_hash"]),
            vpd_membership_hash=str(value["vpd_membership_hash"]),
            manifest_hash=str(value["manifest_hash"]),
        )
        bundle.validate()
        if bundle.vpd_membership_hash != str(expected_vpd_membership_hash):
            raise ValueError("pack-relative VPD membership hash mismatch")
        return bundle

    def validate(self) -> None:
        count = int(self.ids.numel())
        if not torch.equal(self.ids, torch.arange(count, dtype=torch.long)):
            raise ValueError("pack-relative IDs must be dense and ordered")
        if tuple(self.residuals.shape) != (count, 768):
            raise ValueError("pack-relative residuals must have shape [N,768]")
        for name, value in (
            ("pack_ids", self.pack_ids),
            ("vpd_group_ids", self.vpd_group_ids),
        ):
            if tuple(value.shape) != (count,):
                raise ValueError("%s must have shape [N]" % name)
        if int(self.pack_sizes.numel()) != len(self.pack_keys):
            raise ValueError("pack keys/sizes length mismatch")
        if bool(self.pack_sizes.le(0).any()) or bool(
            self.vpd_group_sizes.le(0).any()
        ):
            raise ValueError("pack and VPD group sizes must be positive")
        if int(self.pack_ids.min().item()) < 0 or int(
            self.pack_ids.max().item()
        ) >= int(self.pack_sizes.numel()):
            raise ValueError("pack ID outside pack sizes")
        if int(self.vpd_group_ids.min().item()) < 0 or int(
            self.vpd_group_ids.max().item()
        ) >= int(self.vpd_group_sizes.numel()):
            raise ValueError("VPD group ID outside group sizes")
        if not torch.equal(
            torch.bincount(self.pack_ids, minlength=int(self.pack_sizes.numel())),
            self.pack_sizes,
        ):
            raise ValueError("pack sizes are inconsistent")
        if not torch.equal(
            torch.bincount(
                self.vpd_group_ids,
                minlength=int(self.vpd_group_sizes.numel()),
            ),
            self.vpd_group_sizes,
        ):
            raise ValueError("VPD group sizes are inconsistent")
        expected_hash = hash_value(
            [
                [sticker_id, self.pack_keys[int(self.pack_ids[sticker_id])]]
                for sticker_id in range(count)
            ]
        )
        if expected_hash != self.pack_membership_hash:
            raise ValueError("original pack membership hash mismatch")
        if not torch.isfinite(self.residuals).all():
            raise ValueError("pack-relative residuals contain non-finite values")
        singleton = self.pack_sizes.index_select(0, self.pack_ids).eq(1)
        if singleton.any() and int(
            torch.count_nonzero(self.residuals[singleton]).item()
        ):
            raise ValueError("singleton pack residuals must be zero")
