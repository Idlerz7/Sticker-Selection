import tempfile
import unittest
from pathlib import Path

from torch.utils.data import Dataset

from style_shapes.group_bank import GroupBank
from style_shapes.negative_sampling import (
    DUAL_LOCAL_POLICY,
    DUAL_LOCAL_SEMSP_POLICY,
    StickerChatDualLocalNegativeSampler,
    build_eligibility_manifest,
    validate_dual_local_trace_record,
    validate_eligibility_manifest,
)
from style_shapes.permutations import IndexedSubsetDataset


def row(positive):
    return {"dialog": [{"text": "x", "img_id": int(positive)}]}


def toy_bank(group_source="vpd_pack"):
    return GroupBank.create(
        "stickerchat",
        group_source,
        [0, 1, 2, 3, 4],
        [0, 0, 0, 0, 0],
        {
            0: [1, 2, 3],
            1: [0, 2, 3],
            2: [3, 0, 1],
            3: [2, 0, 1],
            4: [0, 2, 3],
        },
    )


class TinyDataset(Dataset):
    def __init__(self):
        self.values = [{"value": value} for value in range(5)]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class DualLocalNegativeTest(unittest.TestCase):
    def setUp(self):
        self.id_to_pack = {0: "a", 1: "a", 2: "b", 3: "b", 4: "solo"}

    def test_sampling_is_deterministic_distinct_and_semantically_ordered(self):
        first = StickerChatDualLocalNegativeSampler(
            toy_bank(), self.id_to_pack, seed=2021
        )
        second = StickerChatDualLocalNegativeSampler(
            toy_bank(), self.id_to_pack, seed=2021
        )
        first_value = first.resolve(pos_ids=[0, 2])
        second_value = second.resolve(pos_ids=[0, 2])
        self.assertEqual(first_value, second_value)
        cross_vpd, same_pack, meta = first_value
        self.assertEqual(meta["negative_policy"], DUAL_LOCAL_POLICY)
        for positive, vpd_negative, pack_negative in zip(
            [0, 2], cross_vpd, same_pack
        ):
            self.assertEqual(
                self.id_to_pack[positive], self.id_to_pack[pack_negative]
            )
            self.assertIn(vpd_negative, first.neighbors[positive])
            self.assertEqual(len({positive, vpd_negative, pack_negative}), 3)

    def test_singleton_pack_has_no_silent_fallback(self):
        sampler = StickerChatDualLocalNegativeSampler(
            toy_bank(), self.id_to_pack, seed=2021
        )
        with self.assertRaisesRegex(RuntimeError, "no distinct"):
            sampler.sample_one(4)

    def test_semsp_bank_uses_an_independent_policy_with_the_same_contract(self):
        sampler = StickerChatDualLocalNegativeSampler(
            toy_bank("final_clip_pack_original"), self.id_to_pack, seed=2021
        )
        cross_semsp, same_pack, meta = sampler.resolve(pos_ids=[0, 2])
        self.assertEqual(meta["negative_policy"], DUAL_LOCAL_SEMSP_POLICY)
        self.assertEqual(sampler.neighbor_trace_field, "semsp_top32")
        for positive, semsp_negative, pack_negative in zip(
            [0, 2], cross_semsp, same_pack
        ):
            self.assertEqual(
                self.id_to_pack[positive], self.id_to_pack[pack_negative]
            )
            self.assertIn(semsp_negative, sampler.neighbors[positive])
            self.assertEqual(len({positive, semsp_negative, pack_negative}), 3)

        record = {
            "negative_policy": DUAL_LOCAL_SEMSP_POLICY,
            "positive": 0,
            "fallback": 4,
            "cross": cross_semsp[0],
            "same": same_pack[0],
            "group_top32": cross_semsp[0],
            "semsp_top32": cross_semsp[0],
            "same_pack": same_pack[0],
            "fallback_used": False,
        }
        validate_dual_local_trace_record(record, sampler)

    def test_eligibility_manifest_filters_only_singleton_pack(self):
        sampler = StickerChatDualLocalNegativeSampler(
            toy_bank(), self.id_to_pack, seed=2021
        )
        value = build_eligibility_manifest(
            [row(0), row(2), row(4)], sampler, inputs={"toy": True}
        )
        validate_eligibility_manifest(value)
        self.assertEqual(value["eligible_rows"], [0, 1])
        self.assertEqual(value["eligible_count"], 2)
        self.assertEqual(value["excluded_count"], 1)
        self.assertEqual(
            value["excluded_reason_counts"], {"singleton_original_pack": 1}
        )

    def test_trace_aliases_and_semantics(self):
        sampler = StickerChatDualLocalNegativeSampler(
            toy_bank(), self.id_to_pack, seed=2021
        )
        same_pack, vpd_top32 = sampler.sample_one(0)
        record = {
            "negative_policy": DUAL_LOCAL_POLICY,
            "positive": 0,
            "fallback": 4,
            "cross": vpd_top32,
            "same": same_pack,
            "vpd_top32": vpd_top32,
            "same_pack": same_pack,
            "fallback_used": False,
        }
        validate_dual_local_trace_record(record, sampler)
        broken = dict(record, vpd_top32=same_pack)
        with self.assertRaises(ValueError):
            validate_dual_local_trace_record(broken, sampler)

    def test_indexed_subset_preserves_original_source_rows(self):
        dataset = IndexedSubsetDataset(TinyDataset(), [1, 3, 4])
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0]["value"], 1)
        self.assertEqual(dataset[0]["_style_shapes_source_row"], 1)
        self.assertEqual(dataset[2]["_style_shapes_source_row"], 4)


if __name__ == "__main__":
    unittest.main()
