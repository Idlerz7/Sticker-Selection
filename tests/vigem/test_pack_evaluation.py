import json
import tempfile
import unittest
from pathlib import Path

from vigem.pack_evaluation import evaluate_pack_relative_scores


def _row(index, rank, without_rank):
    candidates = list(range(index * 10, index * 10 + 10))
    final = [float(10 - value) for value in range(10)]
    without = list(final)
    gold = candidates[0]
    return {
        "query_index": index,
        "candidate_ids": candidates,
        "gold": gold,
        "rank": rank,
        "rank_without_instance": without_rank,
        "holistic_scores": list(final),
        "instance_scores": [0.1 * value for value in range(10)],
        "group_scores": [0.5] * 10,
        "final_scores": list(final),
        "final_without_instance_scores": without,
    }


class PackEvaluationTest(unittest.TestCase):
    def test_within_ablation_bootstrap_and_pending_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [_row(0, 1, 2), _row(1, 1, 1), _row(2, 2, 3)]
            (root / "scores.json").write_text(json.dumps(rows))
            result = evaluate_pack_relative_scores(
                str(root / "scores.json"),
                str(root / "report.json"),
                iterations=100,
                seed=2021,
            )
            self.assertEqual(result["verdict"], "BASELINE_PENDING")
            self.assertEqual(result["rank_changes"]["improved"], 2)
            self.assertGreater(
                result["full_vs_without_instance"]["mrr"]["difference"], 0
            )

    def test_baseline_alignment_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [_row(0, 1, 2), _row(1, 1, 2)]
            baseline = [_row(0, 2, 2), _row(1, 2, 2)]
            baseline[1]["candidate_ids"][1] = 999
            (root / "scores.json").write_text(json.dumps(rows))
            (root / "baseline.json").write_text(json.dumps(baseline))
            with self.assertRaises(ValueError):
                evaluate_pack_relative_scores(
                    str(root / "scores.json"),
                    str(root / "report.json"),
                    str(root / "baseline.json"),
                    iterations=20,
                )


if __name__ == "__main__":
    unittest.main()
