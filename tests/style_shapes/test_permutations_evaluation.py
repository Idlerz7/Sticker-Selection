import unittest

from style_shapes.evaluation import metrics_from_ranks, paired_bootstrap, scientific_gate
from style_shapes.permutations import (
    ExactDistributedEvalSampler,
    create_permutation_manifest,
    sharded_epoch_indices,
    validate_permutation_manifest,
)


class PermutationEvaluationTest(unittest.TestCase):
    def test_epoch_permutation_and_ddp_shards(self):
        value = create_permutation_manifest(11, 3, 4, 2021)
        validate_permutation_manifest(value, 11)
        self.assertNotEqual(value["permutations"][0], value["permutations"][1])
        shards = [sharded_epoch_indices(value, 0, rank) for rank in range(4)]
        self.assertTrue(all(len(row) == 3 for row in shards))
        merged = [item for row in shards for item in row]
        self.assertEqual(set(merged), set(range(11)))

    def test_manifest_corruption(self):
        value = create_permutation_manifest(5, 2, 2, 2021)
        value["permutations"][0][0] = value["permutations"][0][1]
        with self.assertRaises(ValueError):
            validate_permutation_manifest(value)

    def test_exact_eval_shards_have_no_padding_or_duplicates(self):
        dataset = list(range(11))
        shards = [
            list(ExactDistributedEvalSampler(dataset, 4, rank))
            for rank in range(4)
        ]
        self.assertEqual([len(row) for row in shards], [3, 3, 3, 2])
        merged = [item for row in shards for item in row]
        self.assertEqual(sorted(merged), dataset)
        self.assertEqual(len(merged), len(set(merged)))

    def test_metrics_and_bootstrap(self):
        self.assertEqual(metrics_from_ranks([1, 2, 5])["r@1"], 1 / 3)
        result = paired_bootstrap([1, 1, 2, 1], [2, 3, 4, 2], 1000, 2021)
        self.assertGreater(result["mrr"]["ci95"][0], 0)

    def test_gate(self):
        good = {
            name: {
                "r@1": {"difference": 0.02, "ci95": [0.01, 0.03]},
                "mrr": {"difference": 0.02, "ci95": [0.01, 0.03]},
            }
            for name in (
                "vpd_vs_random",
                "vpd_vs_final_clip",
                "vpd_vs_base_only",
                "vpd_vs_reference",
            )
        }
        self.assertEqual(scientific_gate("dstc", good)["verdict"], "GO")
        good["vpd_vs_random"]["mrr"]["ci95"][0] = -0.01
        self.assertEqual(scientific_gate("dstc", good)["verdict"], "STOP")


if __name__ == "__main__":
    unittest.main()
