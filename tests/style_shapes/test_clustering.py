import unittest

import torch

from style_shapes.clustering import (
    legacy_pack_kmeans,
    random_matched_assignments,
    random_pack_matched_assignments,
    shared_initial_indices,
    spherical_kmeans,
)


class ClusteringTest(unittest.TestCase):
    def test_shared_initialization_and_determinism(self):
        features = torch.Generator().manual_seed(7)
        x = torch.randn(30, 8, generator=features)
        starts = shared_initial_indices(30, 5, 3, 2021)
        left, left_meta = spherical_kmeans(
            x, 5, initial_index_sets=starts, max_iter=20
        )
        right, right_meta = spherical_kmeans(
            x, 5, initial_index_sets=starts, max_iter=20
        )
        self.assertTrue(torch.equal(left, right))
        self.assertEqual(left_meta, right_meta)
        self.assertEqual(sorted(set(left.tolist())), list(range(5)))

    def test_legacy_pack_kmeans_repeats(self):
        x = torch.randn(40, 7, generator=torch.Generator().manual_seed(4))
        left, meta = legacy_pack_kmeans(x, 6, iterations=5, seed=20260330)
        right, _ = legacy_pack_kmeans(
            x, 6, iterations=5, seed=20260330, initial_indices=meta["initial_indices"]
        )
        self.assertTrue(torch.equal(left, right))

    def test_random_exact_size_matching(self):
        assignment = random_matched_assignments(list(range(10)), [2, 3, 5], 2021)
        sizes = [assignment.count(group) for group in range(3)]
        self.assertEqual(sizes, [2, 3, 5])
        self.assertEqual(
            assignment,
            random_matched_assignments(list(range(10)), [2, 3, 5], 2021),
        )

    def test_pack_matching_preserves_pack_and_nonempty(self):
        pack_sizes = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        assignments, info = random_pack_matched_assignments(
            pack_sizes, [12, 11, 10, 7, 5], 2021
        )
        self.assertEqual(len(assignments), len(pack_sizes))
        self.assertEqual(sorted(set(assignments)), list(range(5)))
        self.assertEqual(sum(info["actual_sizes"]), sum(pack_sizes))


if __name__ == "__main__":
    unittest.main()

