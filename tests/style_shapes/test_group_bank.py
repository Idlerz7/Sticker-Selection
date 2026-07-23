import json
import random
import tempfile
import unittest
from pathlib import Path

from factorized_style_bank import FactorizedStyleBank
from style_shapes.group_bank import GROUP_BANK_SCHEMA, GroupBank, membership_sha256


class GroupBankTest(unittest.TestCase):
    def setUp(self):
        self.bank = GroupBank.create(
            "toy",
            "vpd",
            [4, 1, 3, 2],
            [1, 0, 1, 0],
            {1: [2], 2: [1], 3: [4], 4: [3]},
            {"seed": 2021},
        )

    def test_schema_hash_and_bidirectional_membership(self):
        self.assertEqual(self.bank.value["schema_version"], GROUP_BANK_SCHEMA)
        self.assertEqual(self.bank.sticker_ids, [1, 2, 3, 4])
        self.assertEqual(self.bank.group_to_members, [[1, 2], [3, 4]])
        self.assertEqual(
            self.bank.membership_hash,
            membership_sha256([4, 1, 3, 2], [1, 0, 1, 0]),
        )

    def test_serialization_and_public_legacy_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            self.bank.save(str(path))
            loaded = GroupBank.load(str(path))
            self.assertEqual(loaded.membership_hash, self.bank.membership_hash)
            legacy = FactorizedStyleBank.from_json(str(path))
            self.assertEqual(legacy.members_of_proto(0), [1, 2])
            self.assertEqual(legacy.members_of_proto(1), [3, 4])
            self.assertEqual(legacy.sticker_to_record[1].member_ids, [])

    def test_same_and_cross_sampling(self):
        legacy = FactorizedStyleBank(self.bank.to_legacy_dict())
        self.assertEqual(legacy.sample_same_proto_negative(1, random.Random(2)), 2)
        self.assertIn(
            legacy.sample_cross_proto_negative(1, random.Random(2)),
            {3, 4},
        )

    def test_hash_corruption_rejected(self):
        value = self.bank.to_dict()
        value["provenance"] = dict(value["provenance"])
        value["provenance"]["membership_hash"] = "0" * 64
        with self.assertRaises(ValueError):
            GroupBank(value)

    def test_empty_group_rejected(self):
        value = self.bank.to_dict()
        value["num_groups"] = 3
        value["group_to_members"] = value["group_to_members"] + [[]]
        value["group_sizes"] = value["group_sizes"] + [0]
        with self.assertRaises(ValueError):
            GroupBank(value)


if __name__ == "__main__":
    unittest.main()

