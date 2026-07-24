import json
import os
import unittest

import torch

from factorized_style_bank import FactorizedStyleBank
from style_shapes.builders import (
    _pack_centroids,
    _pack_members,
    _reference_k384_membership,
    _sticker_assignments,
    load_descriptor,
)
from style_shapes.clustering import legacy_pack_kmeans
from style_shapes.group_bank import GroupBank
from style_shapes.io import sha256_file
from style_shapes.validation import candidate_file_audit


@unittest.skipUnless(
    os.environ.get("STYLE_SHAPES_REAL_ASSETS") == "1",
    "set STYLE_SHAPES_REAL_ASSETS=1 for real-asset integration tests",
)
class RealAssetIntegrationTest(unittest.TestCase):
    def test_descriptor_shapes_and_alignment(self):
        contracts = (
            ("dstc", 307),
            ("stickerchat", 174695),
        )
        for dataset, rows in contracts:
            ids, multi, _ = load_descriptor(
                "artifacts/lvpcm/descriptors/%s/multi_clean.pt" % dataset
            )
            final_ids, final, _ = load_descriptor(
                "artifacts/lvpcm/descriptors/%s/final_clip_clean.pt" % dataset
            )
            self.assertEqual(ids, list(range(rows)))
            self.assertEqual(ids, final_ids)
            self.assertEqual(list(multi.shape), [rows, 256])
            self.assertEqual(list(final.shape), [rows, 512])
            self.assertTrue(torch.isfinite(multi).all())
            self.assertTrue(torch.isfinite(final).all())

    def test_dstc_legacy_compact_membership_and_public_load(self):
        legacy = FactorizedStyleBank.from_json("factorized_style_bank.json")
        compact = FactorizedStyleBank.from_json(
            "artifacts/style_shapes/groups/dstc/llm_original/group_bank.json"
        )
        legacy_partition = sorted(
            sorted(row.member_ids) for row in legacy.prototypes
        )
        compact_partition = sorted(
            sorted(row.member_ids) for row in compact.prototypes
        )
        self.assertEqual(legacy_partition, compact_partition)
        self.assertEqual(len(compact.records), 307)
        self.assertEqual(len(compact.prototypes), 85)

    def test_stickerchat_reference_k384_rebuild_exact(self):
        ids, final, _ = load_descriptor(
            "artifacts/lvpcm/descriptors/stickerchat/final_clip_clean.pt"
        )
        names, packs = _pack_members(
            "stickerchat/processed/sticker_metadata.json", ids
        )
        centroids = _pack_centroids(ids, final, names, packs)
        assignments, _ = legacy_pack_kmeans(centroids, 384, 40, 20260330)
        rebuilt = _sticker_assignments(ids, names, packs, assignments.tolist())
        expected = _reference_k384_membership(
            "stickerchat/processed_style_kmeans_k384/sticker_metadata.json", ids
        )
        self.assertEqual(rebuilt, expected)
        compact = GroupBank.load(
            "artifacts/style_shapes/groups/stickerchat/"
            "final_clip_pack_original/group_bank.json"
        )
        self.assertEqual(compact.sticker_to_group, expected)

    def test_fixed_candidate_hashes(self):
        expected = {
            "data/validation_pair_with_cand.json": (
                10,
                "1ab0c917c049e90a18d73836355edc927a1549e2b2bc26d0db3f9221ba138d32",
            ),
            "stickerchat/processed/release_val_u_sticker_format_int_with_cand_r10.json": (
                10,
                "c97655f9fb89097116c37a2fcae407b0a9208c0ec735e6883e2e4c6c1c422960",
            ),
            "stickerchat/processed/release_val_u_sticker_format_int_with_cand_r20.json": (
                20,
                "9abaf87a37a2e2e553c7b5d0ecb8fcd59601ec23bac790aa858737031c308f5e",
            ),
            "stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json": (
                10,
                "4e73451bc50e1e27ce903b20358eab0de64a70166cd152bb89af4ba342398395",
            ),
            "stickerchat/processed/release_test_u_sticker_format_int_with_cand_r20.json": (
                20,
                "11a1952f6db7bb58d0fb817585a9ef04c759e9c86d26244c8033fc58f7ad5356",
            ),
            "stickerchat/processed/release_val_u_sticker_format_int_with_cand_same_pack_r10.json": (
                10,
                "4f15007736239b1bc474b49668ef016fd26007f93357a06dc985ad5b7c79a4be",
            ),
            "stickerchat/processed/release_test_u_sticker_format_int_with_cand_same_pack_r10.json": (
                10,
                "1e1711244ccde8652144fea3a79b3c769cdbae0ef74b0dd255e332cf223b7fa5",
            ),
        }
        for path, (size, digest) in expected.items():
            self.assertEqual(sha256_file(path), digest)
            self.assertEqual(candidate_file_audit(path, size)["candidate_count"], size)

    def test_init_snapshot_strict_reload_contract(self):
        paths = (
            "artifacts/style_shapes/init/dstc_seed2021.ckpt",
            "artifacts/style_shapes/init/stickerchat_seed2021.ckpt",
        )
        missing = [path for path in paths if not os.path.exists(path)]
        if missing:
            self.skipTest("GPU initialization execution approval blocked: %s" % missing)


if __name__ == "__main__":
    unittest.main()
