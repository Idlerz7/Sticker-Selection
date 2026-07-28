import json
import tempfile
import unittest
from pathlib import Path

import torch

from style_shapes.group_bank import GroupBank
from vigem.pack_relative import (
    PACK_RELATIVE_SCHEMA,
    PackRelativeResidualBundle,
    build_pack_relative_bundle,
)


class PackRelativeAssetTest(unittest.TestCase):
    def _build(self, root: Path):
        bank = GroupBank.create(
            "stickerchat", "vpd_pack", [0, 1, 2], [0, 0, 1]
        )
        bank.save(str(root / "bank.json"))
        metadata = {
            "stickers": [
                {"internal_img_id": 0, "img_set": "pack-b"},
                {"internal_img_id": 1, "img_set": "pack-b"},
                {"internal_img_id": 2, "img_set": "pack-a"},
            ]
        }
        (root / "metadata.json").write_text(json.dumps(metadata))
        clip = torch.zeros(3, 512)
        clip[0, 0] = 2
        clip[1, 1] = 3
        clip[2, 2] = 4
        vpd = torch.zeros(3, 256)
        vpd[0, 0] = 5
        vpd[1, 1] = 6
        vpd[2, 2] = 7
        torch.save(
            {"ids": torch.arange(3), "features": clip},
            root / "clip.pt",
        )
        torch.save(
            {"ids": torch.arange(3), "features": vpd},
            root / "vpd.pt",
        )
        payload, manifest = build_pack_relative_bundle(
            str(root / "metadata.json"),
            str(root / "bank.json"),
            str(root / "clip.pt"),
            str(root / "vpd.pt"),
        )
        return bank, payload, manifest

    def test_pack_centroid_singleton_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            bank, payload, manifest = self._build(Path(directory))
            self.assertEqual(payload["schema_version"], PACK_RELATIVE_SCHEMA)
            self.assertEqual(tuple(payload["residuals"].shape), (3, 768))
            self.assertTrue(
                torch.allclose(
                    payload["residuals"][:2].mean(dim=0),
                    torch.zeros(768),
                )
            )
            self.assertEqual(
                int(torch.count_nonzero(payload["residuals"][2]).item()), 0
            )
            self.assertEqual(payload["pack_keys"], ["pack-a", "pack-b"])
            self.assertEqual(payload["pack_ids"].tolist(), [1, 1, 0])
            self.assertEqual(
                manifest["vpd_membership_hash"], bank.membership_hash
            )
            self.assertEqual(
                payload["manifest_hash"], manifest["manifest_hash"]
            )

    def test_round_trip_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank, payload, _ = self._build(root)
            torch.save(payload, root / "bundle.pt")
            bundle = PackRelativeResidualBundle.load(
                str(root / "bundle.pt"), bank.membership_hash
            )
            self.assertEqual(bundle.pack_sizes.tolist(), [1, 2])
            self.assertEqual(bundle.group_ids.tolist(), [0, 0, 1])

    def test_metadata_id_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _ = self._build(root)
            metadata = json.loads((root / "metadata.json").read_text())
            metadata["stickers"].pop()
            (root / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                build_pack_relative_bundle(
                    str(root / "metadata.json"),
                    str(root / "bank.json"),
                    str(root / "clip.pt"),
                    str(root / "vpd.pt"),
                )


if __name__ == "__main__":
    unittest.main()
