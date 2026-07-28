import tempfile
import unittest
from pathlib import Path

from style_shapes.contracts import (
    PILOT_SCHEMA_VERSION,
    SCHEDULER_CONTRACT,
    completed_manifest_uses_current_contract,
    training_completion_errors,
)


class CompletionContractsTest(unittest.TestCase):
    def test_legacy_interrupted_manifest_is_rejected(self):
        legacy = {
            "status": "TRAIN_COMPLETE",
            "world_size": 4,
        }
        self.assertFalse(completed_manifest_uses_current_contract(legacy, 4))

    def test_current_manifest_requires_equal_expected_and_completed_steps(self):
        value = {
            "status": "TRAIN_COMPLETE",
            "schema_version": PILOT_SCHEMA_VERSION,
            "scheduler_contract": SCHEDULER_CONTRACT,
            "world_size": 6,
            "expected_optimizer_steps": 33360,
            "completed_optimizer_steps": 33360,
        }
        self.assertTrue(completed_manifest_uses_current_contract(value, 6))
        value["completed_optimizer_steps"] = 140
        self.assertFalse(completed_manifest_uses_current_contract(value, 6))

    def test_completion_requires_finalized_trace_for_every_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in range(2):
                (root / ("rank_%02d.jsonl" % rank)).write_text("{}\n")
            self.assertEqual(
                training_completion_errors(
                    interrupted=False,
                    completed_optimizer_steps=20,
                    expected_optimizer_steps=20,
                    current_epoch=1,
                    expected_epochs=2,
                    trace_dir=str(root),
                    world_size=2,
                ),
                [],
            )
            (root / "rank_01.jsonl").unlink()
            (root / "rank_01.jsonl.partial").write_text("{}\n")
            errors = training_completion_errors(
                interrupted=True,
                completed_optimizer_steps=3,
                expected_optimizer_steps=20,
                current_epoch=0,
                expected_epochs=2,
                trace_dir=str(root),
                world_size=2,
            )
            self.assertTrue(any("interruption" in item for item in errors))
            self.assertTrue(any("optimizer steps" in item for item in errors))
            self.assertTrue(any("missing finalized trace" in item for item in errors))
            self.assertTrue(any("partial trace remains" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
