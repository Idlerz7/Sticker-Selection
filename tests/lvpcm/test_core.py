import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lvpcm.io import atomic_torch_save, atomic_write_json, hash_mapping, refuse_incompatible_manifest, sha256_file
from lvpcm.images import perturb_image
from lvpcm.lpc import build_training_lpc, exact_topk, exact_topk_with_group_filtered, query_training_lpc
from lvpcm.metrics import jaccard_rows, paired_bootstrap_delta, ranking_metrics
from lvpcm.scorers import ConditionalScorer, UnconditionalScorer, trainable_parameter_count
from lvpcm.stage_b import centered_pca_condition, epoch_permutations, split_safe_shuffle
from lvpcm.vpd import fit_reducer, multi_vpd, patch_distribution, single_vpd


class VPDTests(unittest.TestCase):
    def test_block_mapping_cls_exclusion_shapes_and_finite(self):
        hidden = [torch.zeros(2, 50, 768) for _ in range(13)]
        for block in (3, 6, 9):
            hidden[block][:, 0] = 1000000.0
            hidden[block][:, 1:] = float(block)
        single = single_vpd(hidden)
        multi = multi_vpd(hidden)
        self.assertEqual(tuple(single.shape), (2, 1536))
        self.assertEqual(tuple(multi.shape), (2, 4608))
        self.assertTrue(torch.isfinite(multi).all())
        self.assertTrue(torch.allclose(single[:, :768], torch.full((2, 768), 6.0)))
        self.assertTrue(torch.equal(single[:, 768:], torch.zeros(2, 768)))

    def test_reducer_is_train_only_and_reproducible(self):
        generator = torch.Generator().manual_seed(7)
        values = torch.randn(12, 8, generator=generator)
        ids = torch.arange(12)
        one = fit_reducer(values, ids, range(8), n_components=4, seed=2021)
        changed = values.clone(); changed[8:] += 10000
        two = fit_reducer(changed, ids, range(8), n_components=4, seed=2021)
        self.assertTrue(np.array_equal(one.scaler_mean, two.scaler_mean))
        self.assertTrue(np.array_equal(one.components, two.components))
        transformed = one.transform(values)
        self.assertEqual(tuple(transformed.shape), (12, 4))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(transformed, dim=1), torch.ones(12), atol=1e-5))

    def test_fixed_image_perturbations_and_crop_eligibility(self):
        image = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        for x in range(8, 32):
            for y in range(8, 32): image.putpixel((x, y), (255, 0, 0, 255))
        buffer = BytesIO(); image.save(buffer, format="PNG"); data = buffer.getvalue()
        resize, resize_ok = perturb_image(data, "resize75")
        jpeg, jpeg_ok = perturb_image(data, "jpeg75")
        crop, crop_ok = perturb_image(data, "alpha_bbox")
        self.assertEqual(resize.size, (40, 40)); self.assertTrue(resize_ok)
        self.assertEqual(jpeg.size, (40, 40)); self.assertTrue(jpeg_ok)
        self.assertEqual(crop.size, (24, 24)); self.assertTrue(crop_ok)


class LPCTests(unittest.TestCase):
    def test_tie_break_is_integer_id(self):
        query = torch.tensor([[1.0, 0.0]])
        index = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        ids = torch.tensor([5, 3, 4])
        result, _ = exact_topk(query, index, ids, 2, query_batch_size=1, index_batch_size=2)
        self.assertEqual(result.tolist(), [[3, 4]])

    def test_group_filtered_topk_reuses_scores_but_removes_exact_group(self):
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
        ids = torch.arange(4); groups = torch.tensor([5, 5, 6, 7])
        ordinary, filtered = exact_topk_with_group_filtered(
            features, features, ids, groups, groups, k=2, query_ids=ids, exclude_equal_id=True,
            query_batch_size=2, index_batch_size=2,
        )
        self.assertEqual(ordinary[0].tolist(), [1, 2])
        self.assertEqual(filtered[0].tolist(), [2, 3])

    def test_mutual_fallback_and_query_isolation(self):
        features = torch.tensor([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]])
        ids = torch.tensor([0, 1, 2])
        train = build_training_lpc(features, ids, k=1, query_batch_size=2, index_batch_size=2)
        self.assertEqual(train["degree"].tolist(), [1, 1, 0])
        self.assertEqual(train["fallback"].tolist(), [False, False, True])
        queries = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
        query_ids = torch.tensor([10, 11])
        together = query_training_lpc(queries, query_ids, features, train, k=1, query_batch_size=2, index_batch_size=2)
        alone = query_training_lpc(queries[:1], query_ids[:1], features, train, k=1, query_batch_size=1, index_batch_size=2)
        self.assertTrue(torch.equal(together["neighbors"][:1], alone["neighbors"]))
        self.assertTrue(torch.equal(together["features"][:1], alone["features"]))

    def test_jaccard_and_hash(self):
        a = torch.tensor([[1, 2, -1], [3, 4, 5]])
        b = torch.tensor([[2, 1, -1], [3, 6, 7]])
        self.assertEqual(jaccard_rows(a, b).tolist(), [1.0, 0.2])
        self.assertEqual(hash_mapping({"b": 1, "a": 2}), hash_mapping({"a": 2, "b": 1}))


class ScorerTests(unittest.TestCase):
    def test_step_zero_bound_gradients_and_parameter_counts(self):
        torch.manual_seed(2021)
        base = torch.randn(3, 10)
        h = torch.randn(3, 10, 768)
        z = torch.randn(3, 10, 256)
        conditional = ConditionalScorer(2.5)
        unconditional = UnconditionalScorer(2.5)
        self.assertEqual(trainable_parameter_count(conditional), 16384)
        self.assertEqual(trainable_parameter_count(unconditional), 16149)
        self.assertTrue(torch.equal(conditional(base, h, z), base))
        self.assertTrue(torch.equal(unconditional(base, h), base))
        with torch.no_grad():
            conditional.C.normal_(); unconditional.v.normal_()
        self.assertLessEqual(float(conditional.delta(h, z).abs().max()), 2.5)
        self.assertLessEqual(float(unconditional.delta(h).abs().max()), 2.5)
        loss = conditional(base, h, z).sum() + unconditional(base, h).sum()
        loss.backward()
        self.assertIsNotNone(conditional.A.grad)
        self.assertIsNotNone(conditional.C.grad)
        self.assertIsNotNone(unconditional.U.grad)
        self.assertIsNotNone(unconditional.v.grad)

    def test_candidate_permutation_equivariance(self):
        torch.manual_seed(12)
        scorer = ConditionalScorer(1.0)
        with torch.no_grad(): scorer.C.normal_()
        b = torch.randn(2, 5); h = torch.randn(2, 5, 768); z = torch.randn(2, 5, 256)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        expected = scorer(b, h, z)[:, permutation]
        actual = scorer(b[:, permutation], h[:, permutation], z[:, permutation])
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))


class MetricsAndSerializationTests(unittest.TestCase):
    def test_ranking_and_paired_bootstrap(self):
        scores = torch.tensor([[0.1, 0.9, 0.3], [0.5, 0.4, 0.3]])
        metrics = ranking_metrics(scores, torch.tensor([1, 2]))
        self.assertAlmostEqual(metrics["mrr"], (1 + 1 / 3) / 2)
        boot = paired_bootstrap_delta([1, 2, 3], [0, 1, 2], replicates=100, seed=2021)
        self.assertEqual(boot["estimate"], 1.0)
        self.assertEqual(boot["ci95"], [1.0, 1.0])

    def test_atomic_serialization_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.pt"
            atomic_torch_save(path, {"ids": torch.arange(3), "features": torch.eye(3)})
            first = sha256_file(path)
            value = torch.load(path)
            self.assertTrue(torch.equal(value["ids"], torch.arange(3)))
            self.assertEqual(first, sha256_file(path))
            json_path = Path(directory) / "manifest.json"
            atomic_write_json(json_path, {"sha256": first})
            self.assertEqual(json.loads(json_path.read_text())["sha256"], first)

    def test_manifest_resume_and_mismatch_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            atomic_write_json(path, {"status": "complete", "input_config_hash": "abc"})
            self.assertTrue(refuse_incompatible_manifest(path, "abc"))
            with self.assertRaises(RuntimeError):
                refuse_incompatible_manifest(path, "different")


class StageBUtilityTests(unittest.TestCase):
    def test_centered_pca_train_only_and_shuffle_split_isolation(self):
        generator = torch.Generator().manual_seed(4)
        bundle = {
            "ids": torch.arange(12), "features": torch.randn(12, 8, generator=generator),
            "train_ids": torch.arange(8),
        }
        reduced = centered_pca_condition(bundle, n_components=4, seed=2021)
        self.assertEqual(tuple(reduced["features"].shape), (12, 4))
        shuffled, mapping = split_safe_shuffle({**reduced, "degree": torch.ones(12), "fallback": torch.zeros(12, dtype=torch.bool)}, 2021)
        self.assertEqual(set(mapping), set(range(12)))
        self.assertTrue(all((source < 8) == (destination < 8) for destination, source in mapping.items()))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(shuffled["features"], dim=1), torch.ones(12), atol=1e-5))

    def test_epoch_orders_are_reproducible_and_complete(self):
        first = epoch_permutations(23, 3, 2021)
        second = epoch_permutations(23, 3, 2021)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertTrue(all(sorted(value.tolist()) == list(range(23)) for value in first))


if __name__ == "__main__":
    unittest.main()
