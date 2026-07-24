import json
import tempfile
import unittest
from pathlib import Path

import torch

from factorized_style_bank import FactorizedStyleBank
from structured_retrieval_factorized import StructuredFactorizedStickerModel
from style_shapes.group_bank import GroupBank
from style_shapes.validation import merge_and_validate_traces


class PrototypeValidationTest(unittest.TestCase):
    def test_prototype_pooling_uses_current_c_i(self):
        bank = GroupBank.create("toy", "vpd", [0, 1, 2], [0, 0, 1])
        model = object.__new__(StructuredFactorizedStickerModel)
        model.style_bank = FactorizedStyleBank(bank.to_legacy_dict())
        from types import SimpleNamespace
        model.args = SimpleNamespace(factorized_proto_consistency_weight=0.0)
        model._proto_reduce_cache_by_device = {}
        first = torch.tensor([[1.0, 3.0], [3.0, 5.0], [9.0, 11.0]])
        second = first + 2.0
        first_proto, _ = StructuredFactorizedStickerModel._compute_proto_vectors(model, first)
        second_proto, _ = StructuredFactorizedStickerModel._compute_proto_vectors(model, second)
        self.assertTrue(torch.equal(first_proto, torch.tensor([[2.0, 4.0], [9.0, 11.0]])))
        self.assertTrue(torch.equal(second_proto, first_proto + 2.0))

    def test_trace_complete_coverage_and_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for rank in range(2):
                path = Path(directory) / ("rank%d.jsonl" % rank)
                rows = []
                for epoch in range(2):
                    for source_row in range(rank, 4, 2):
                        rows.append(
                            {
                                "epoch": epoch,
                                "global_step": epoch * 2,
                                "rank": rank,
                                "source_row": source_row,
                                "positive": source_row,
                                "fallback": (source_row + 1) % 4,
                                "cross": (source_row + 2) % 4,
                                "same": (source_row + 3) % 4,
                                "membership_hash": "abc",
                            }
                        )
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                paths.append(str(path))
            result = merge_and_validate_traces(paths, 4, 2, "abc")
            self.assertTrue(result["complete_coverage"])
            self.assertEqual(result["known_ddp_padding_records"], 0)

    def test_trace_complete_coverage_for_filtered_source_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0.jsonl"
            records = []
            for epoch in range(2):
                for source_row in (1, 3):
                    records.append(
                        {
                            "epoch": epoch,
                            "global_step": epoch,
                            "rank": 0,
                            "source_row": source_row,
                            "positive": source_row,
                            "fallback": 0,
                            "cross": 2,
                            "same": 4,
                            "membership_hash": "abc",
                        }
                    )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            result = merge_and_validate_traces(
                [str(path)],
                2,
                2,
                "abc",
                expected_source_rows=[1, 3],
            )
            self.assertEqual(result["expected_source_row_count"], 2)
            self.assertTrue(result["complete_coverage"])


if __name__ == "__main__":
    unittest.main()
