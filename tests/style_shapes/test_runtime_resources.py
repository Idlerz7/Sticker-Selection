import tempfile
import unittest
from pathlib import Path

from style_shapes.permutations import (
    create_permutation_manifest,
    save_permutation_manifest,
)
from style_shapes.runtime import (
    derive_permutation_world_size,
    resolve_permutation_world_size,
)


class RuntimeResourceTest(unittest.TestCase):
    def test_world_size_derivation_preserves_global_epoch_orders(self):
        source = create_permutation_manifest(13, 3, 8, 2021)
        derived = derive_permutation_world_size(source, 6)

        self.assertEqual(derived["world_size"], 6)
        self.assertEqual(derived["permutations"], source["permutations"])
        self.assertEqual(derived["seed"], 2021)
        self.assertNotEqual(derived["manifest_hash"], source["manifest_hash"])

    def test_resolution_is_named_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "stickerchat_seed2021_ws8.json"
            save_permutation_manifest(
                str(source_path), create_permutation_manifest(11, 2, 8, 2021)
            )

            path1, value1 = resolve_permutation_world_size(str(source_path), 6)
            path2, value2 = resolve_permutation_world_size(str(source_path), 6)

            self.assertTrue(path1.endswith("_ws6.json"))
            self.assertEqual(path1, path2)
            self.assertEqual(value1, value2)
            self.assertEqual(value1["world_size"], 6)


if __name__ == "__main__":
    unittest.main()
