import os
import unittest

import torch

from style_shapes.group_bank import GroupBank
from vigem.residuals import InstanceResidualBundle


REAL = os.environ.get("VIGEM_REAL_ASSETS") == "1"


@unittest.skipUnless(REAL, "set VIGEM_REAL_ASSETS=1 for real assets")
class VigemRealAssetTest(unittest.TestCase):
    def test_residual_shapes_membership_and_singletons(self):
        cases = (
            (
                "artifacts/style_shapes/groups/dstc/vpd_multi/group_bank.json",
                "artifacts/vigem/residuals/dstc_vpd_multi.pt",
                (307, 768),
            ),
            (
                "artifacts/style_shapes/groups/stickerchat/vpd_pack/group_bank.json",
                "artifacts/vigem/residuals/stickerchat_vpd_pack.pt",
                (174695, 768),
            ),
        )
        for bank_path, bundle_path, shape in cases:
            bank = GroupBank.load(bank_path)
            bundle = InstanceResidualBundle.load(
                bundle_path, bank.membership_hash
            )
            self.assertEqual(tuple(bundle.residuals.shape), shape)
            singleton = bundle.group_sizes.index_select(
                0, bundle.group_ids
            ).eq(1)
            if singleton.any():
                self.assertEqual(
                    torch.count_nonzero(bundle.residuals[singleton]).item(), 0
                )

    def test_stickerchat_fixed_real_candidates_stay_in_vpd_group(self):
        bank = GroupBank.load(
            "artifacts/style_shapes/groups/stickerchat/vpd_pack/group_bank.json"
        )
        value = torch.load(
            "artifacts/style_shapes/candidates/"
            "stickerchat_fixed_same_pack_r10/train_candidates.pt",
            map_location="cpu",
        )
        candidates = value["candidate_ids"].long()
        groups = torch.tensor(bank.sticker_to_group, dtype=torch.long)
        safe = candidates.clamp_min(0)
        candidate_groups = groups.index_select(0, safe.reshape(-1)).reshape(
            candidates.shape
        )
        real = candidates.ne(-1)
        same = candidate_groups.eq(candidate_groups[:, :1])
        self.assertTrue(bool((same | ~real).all()))

    def test_new_snapshots_preserve_every_legacy_state_tensor(self):
        cases = (
            (
                "artifacts/style_shapes/init/dstc_seed2021.ckpt",
                "artifacts/vigem/init/dstc_vpd_multi_seed2021.ckpt",
            ),
            (
                "artifacts/style_shapes/init/stickerchat_seed2021.ckpt",
                "artifacts/vigem/init/stickerchat_vpd_pack_seed2021.ckpt",
            ),
        )
        for legacy_path, new_path in cases:
            legacy = torch.load(legacy_path, map_location="cpu")["state_dict"]
            new = torch.load(new_path, map_location="cpu")["state_dict"]
            self.assertTrue(set(legacy).issubset(set(new)))
            for key, value in legacy.items():
                self.assertTrue(
                    torch.equal(value, new[key]),
                    msg="shared initialization changed at %s" % key,
                )
            new_keys = sorted(set(new) - set(legacy))
            self.assertTrue(new_keys)
            self.assertTrue(
                all(
                    key.startswith("model.instance_")
                    for key in new_keys
                )
            )
