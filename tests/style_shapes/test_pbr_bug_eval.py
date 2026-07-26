import unittest

import numpy as np

from style_shapes.pbr_bug_eval import (
    parse_pbr_mapping_order,
    pbr_bug_candidates_for_gold,
    pbr_released_metrics,
)


class PbrBugCandidateTest(unittest.TestCase):
    def test_suffix_bug_keeps_gold_in_negative_slots(self):
        img2id = {
            "10-100": 0,
            "10-101": 1,
            "10-102": 2,
        }
        candidates, gray = pbr_bug_candidates_for_gold(
            10, 101, [100, 101, 102], img2id
        )
        self.assertEqual(candidates[:4], [1, 0, 1, 2])
        self.assertEqual(candidates.count(1), 2)
        self.assertEqual(candidates[4:], [-1] * 6)
        self.assertEqual(gray, [False] * 4 + [True] * 6)

    def test_mapping_matches_python_dict_insertion_order(self):
        order = parse_pbr_mapping_order(
            "100\tfirst\n101\tsecond\n100\toverwritten\n"
        )
        self.assertEqual(order, ["100", "101"])

    def test_released_metrics_label_only_slot_zero(self):
        scores = np.zeros((2, 10), dtype=np.float32)
        scores[0, 0] = 3.0
        scores[0, 1] = 2.0
        scores[1, 0] = 2.0
        scores[1, 1:4] = [5.0, 4.0, 3.0]
        metrics = pbr_released_metrics(scores)
        self.assertEqual(metrics["queries"], 2)
        self.assertAlmostEqual(metrics["r_at_1"], 0.5)
        self.assertAlmostEqual(metrics["mrr"], 0.625)
        self.assertAlmostEqual(metrics["map"], metrics["mrr"])
        self.assertAlmostEqual(metrics["released_r2_at_1"], 0.5)


if __name__ == "__main__":
    unittest.main()
