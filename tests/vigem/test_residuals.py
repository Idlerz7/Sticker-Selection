import tempfile
import unittest
from pathlib import Path

import torch

from style_shapes.group_bank import GroupBank
from vigem.residuals import (
    INSTANCE_RESIDUAL_SCHEMA,
    InstanceResidualBundle,
    build_instance_residual_bundle,
)


class InstanceResidualAssetTest(unittest.TestCase):
    def test_family_normalization_group_centering_and_singleton(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank_path = root / "bank.json"
            clip_path = root / "clip.pt"
            vpd_path = root / "vpd.pt"
            bank = GroupBank.create(
                dataset="toy",
                group_source="vpd",
                sticker_ids=[0, 1, 2],
                sticker_to_group=[0, 0, 1],
            )
            bank.save(str(bank_path))
            clip = torch.zeros(3, 512)
            clip[0, 0] = 2.0
            clip[1, 1] = 3.0
            clip[2, 2] = 4.0
            vpd = torch.zeros(3, 256)
            vpd[0, 0] = 5.0
            vpd[1, 1] = 6.0
            vpd[2, 2] = 7.0
            torch.save(
                {"ids": torch.arange(3), "features": clip}, clip_path
            )
            torch.save(
                {"ids": torch.arange(3), "features": vpd}, vpd_path
            )
            payload, manifest = build_instance_residual_bundle(
                "toy", str(bank_path), str(clip_path), str(vpd_path)
            )
            self.assertEqual(payload["schema_version"], INSTANCE_RESIDUAL_SCHEMA)
            self.assertEqual(tuple(payload["residuals"].shape), (3, 768))
            self.assertTrue(
                torch.allclose(
                    payload["residuals"][:2].mean(dim=0),
                    torch.zeros(768),
                )
            )
            self.assertEqual(
                torch.count_nonzero(payload["residuals"][2]).item(), 0
            )
            self.assertEqual(
                payload["membership_hash"], bank.membership_hash
            )
            self.assertEqual(
                payload["manifest_hash"], manifest["manifest_hash"]
            )

    def test_id_order_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank = GroupBank.create(
                "toy", "vpd", [0, 1], [0, 0]
            )
            bank.save(str(root / "bank.json"))
            torch.save(
                {
                    "ids": torch.tensor([0, 1]),
                    "features": torch.ones(2, 512),
                },
                root / "clip.pt",
            )
            torch.save(
                {
                    "ids": torch.tensor([1, 0]),
                    "features": torch.ones(2, 256),
                },
                root / "vpd.pt",
            )
            with self.assertRaises(ValueError):
                build_instance_residual_bundle(
                    "toy",
                    str(root / "bank.json"),
                    str(root / "clip.pt"),
                    str(root / "vpd.pt"),
                )

    def test_bundle_validation_rejects_nonzero_singleton(self):
        bundle = InstanceResidualBundle(
            ids=torch.arange(2),
            residuals=torch.stack(
                [torch.ones(768), torch.zeros(768)], dim=0
            ),
            group_ids=torch.tensor([0, 1]),
            group_sizes=torch.tensor([1, 1]),
            membership_hash="x",
            manifest_hash="y",
        )
        with self.assertRaises(ValueError):
            bundle.validate()
