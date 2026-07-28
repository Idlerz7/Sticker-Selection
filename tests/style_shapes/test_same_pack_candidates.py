import unittest

from style_shapes.candidates import (
    audit_same_pack_candidates,
    build_same_pack_candidates,
    id_to_pack_from_id2img,
    pack_from_image_name,
)


def _row(gold):
    return {"dialog": [{"text": "context"}, {"text": "target", "img_id": int(gold)}]}


class SamePackCandidatesTest(unittest.TestCase):
    def test_pack_is_filename_prefix(self):
        self.assertEqual(pack_from_image_name("pack-a-sticker.png"), "pack")
        self.assertEqual(
            id_to_pack_from_id2img({"0": "100-200.png", "1": "100-201.png"}),
            {0: "100", 1: "100"},
        )
        with self.assertRaises(ValueError):
            pack_from_image_name("missing_separator.png")

    def test_same_pack_and_global_fallback_are_exact_and_deterministic(self):
        id_to_pack = {
            0: "a",
            1: "a",
            2: "a",
            3: "a",
            4: "b",
            5: "b",
            6: "c",
            7: "c",
        }
        first, stats = build_same_pack_candidates(
            [_row(0), _row(4)], id_to_pack, candidate_size=4, seed=2021
        )
        second, second_stats = build_same_pack_candidates(
            [_row(0), _row(4)], id_to_pack, candidate_size=4, seed=2021
        )
        self.assertEqual(first, second)
        self.assertEqual(stats, second_stats)

        self.assertEqual(first[0]["cand"][0], 0)
        self.assertEqual(set(first[0]["cand"][1:]), {1, 2, 3})
        self.assertEqual(first[1]["cand"][0], 4)
        self.assertIn(5, first[1]["cand"][1:])
        self.assertEqual(len(set(first[1]["cand"])), 4)
        self.assertEqual(stats["all_same_pack_rows"], 1)
        self.assertEqual(stats["global_fallback_rows"], 1)
        self.assertEqual(stats["global_fallback_negatives"], 2)
        audit = audit_same_pack_candidates(first, id_to_pack, candidate_size=4)
        self.assertEqual(audit["all_same_pack_rows"], 1)
        self.assertEqual(audit["global_fallback_rows"], 1)

        broken = list(first)
        broken[0] = {"dialog": first[0]["dialog"], "cand": [0, 1, 2, 4]}
        with self.assertRaises(ValueError):
            audit_same_pack_candidates(broken, id_to_pack, candidate_size=4)


if __name__ == "__main__":
    unittest.main()
