import unittest

from lvpcm.catalog import (
    dstc_training_candidates, normalized_candidate_manifest, stickerchat_same_pack_candidates,
    validate_candidate_rows,
)


def row(gold, negative=None, candidates=None, dialogue="d"):
    turn = {"text": "hello", "img_id": gold}
    if negative is not None:
        turn.update({"neg_img_id": negative, "img_set": "p", "neg_img_set": "p"})
    value = {"dialog": [turn], "dialogue_id": dialogue, "user_id": dialogue + "::u"}
    if candidates is not None:
        value["cand"] = candidates
    return value


class CandidateTests(unittest.TestCase):
    def test_positive_index_and_uniqueness(self):
        rows = [row(2, candidates=[4, 2, 1])]
        self.assertEqual(validate_candidate_rows(rows, 3)["rows"], 1)
        manifest = normalized_candidate_manifest(rows, "validation", "fixed", "source.json")
        self.assertEqual(manifest[0]["positive_index"], 1)
        with self.assertRaises(ValueError):
            validate_candidate_rows([row(2, candidates=[2, 2, 1])])
        with self.assertRaises(ValueError):
            validate_candidate_rows(rows, 3, allowed_ids=[0, 1, 3, 4])

    def test_dstc_uniform_determinism(self):
        rows = [row(0), row(1)]
        a = dstc_training_candidates(rows, range(12), n=10, seed=2021)
        b = dstc_training_candidates(rows, range(12), n=10, seed=2021)
        self.assertEqual(a, b)
        for source, manifest in zip(rows, a):
            gold = source["dialog"][-1]["img_id"]
            self.assertEqual(manifest["candidate_ids"].count(gold), 1)
            self.assertEqual(manifest["candidate_ids"][manifest["positive_index"]], gold)

    def test_stickerchat_external_internal_and_pack_alignment(self):
        rows = [row("a", "b")]
        rebuilt = stickerchat_same_pack_candidates(rows, {"a": 7, "b": 8}, range(30), 10, seed=2021)
        self.assertEqual(rebuilt[0]["candidate_ids"][:2], [7, 8])
        self.assertEqual(len(set(rebuilt[0]["candidate_ids"])), 10)
        bad = [row("a", "b")]
        bad[0]["dialog"][-1]["neg_img_set"] = "other"
        with self.assertRaises(ValueError):
            stickerchat_same_pack_candidates(bad, {"a": 7, "b": 8}, range(30), 10)


if __name__ == "__main__":
    unittest.main()
