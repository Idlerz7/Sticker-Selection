import json
import os
import unittest
from pathlib import Path

import torch

from style_shapes.group_bank import GroupBank
from vigem.pack_relative import PackRelativeResidualBundle


REAL = os.environ.get("VIGEM_REAL_ASSETS") == "1"


@unittest.skipUnless(REAL, "set VIGEM_REAL_ASSETS=1 for real assets")
class PackRelativeRealAssetTest(unittest.TestCase):
    def test_catalog_pack_and_candidate_contract(self):
        bundle = PackRelativeResidualBundle.load(
            "artifacts/vigem/residuals/stickerchat_pack_relative.pt",
            GroupBank.load(
                "artifacts/style_shapes/groups/stickerchat/vpd_pack/"
                "group_bank.json"
            ).membership_hash,
        )
        self.assertEqual(tuple(bundle.residuals.shape), (174695, 768))
        self.assertEqual(len(bundle.pack_keys), 3516)
        candidates = torch.load(
            "artifacts/style_shapes/candidates/"
            "stickerchat_fixed_same_pack_r10/train_candidates.pt",
            map_location="cpu",
        )["candidate_ids"].long()
        self.assertEqual(tuple(candidates.shape), (320168, 10))
        safe = candidates.clamp_min(0)
        packs = bundle.pack_ids.index_select(
            0, safe.reshape(-1)
        ).reshape(candidates.shape)
        real = candidates.ge(0)
        self.assertTrue(bool((packs.eq(packs[:, :1]) | ~real).all()))

    def test_config_contains_no_r20_path(self):
        import yaml

        config = yaml.safe_load(
            Path(
                "configs/vigem/"
                "stickerchat_vpd_pack_relative_setwise_r10.yaml"
            ).read_text(encoding="utf-8")
        )
        serialized = json.dumps(config).lower()
        self.assertNotIn("with_cand_r20", serialized)


if __name__ == "__main__":
    unittest.main()
