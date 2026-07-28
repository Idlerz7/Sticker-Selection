import unittest
from pathlib import Path

import torch

from lvpcm.catalog import normalized_candidate_manifest, read_json
from lvpcm.clip_intermediate import CLIPIntermediateExtractor
from lvpcm.images import project_rgb
from lvpcm.io import load_yaml
from lvpcm.legacy import load_dstc_plmodel
from lvpcm.pair_cache import extract_pair_cache, validate_against_legacy


class RealAssetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for real-asset integration")
        cls.config = load_yaml("configs/lvpcm/dstc.yaml")
        cls.model, cls.args = load_dstc_plmodel(cls.config, "cuda:0")

    def test_tokenizer_checkpoint_and_cache_alignment(self):
        self.assertEqual(len(self.model.model.bert_tokenizer), 21130)
        self.assertEqual(self.model.model.bert.get_input_embeddings().weight.shape[0], 21130)
        cache = torch.load(self.config["final_clip_cache"], map_location="cpu")
        self.assertEqual(tuple(cache.shape), (307, 512))
        self.assertTrue(torch.isfinite(cache).all())
        id2img = read_json(self.config["id2img"])
        self.assertEqual(sorted(int(key) for key in id2img), list(range(307)))

    def test_clip_single_image_repeat(self):
        extractor = CLIPIntermediateExtractor("ckpt/clip-ViT-B-32/0_CLIPModel", "cuda:0")
        id2img = read_json(self.config["id2img"])
        data = (Path(self.config["image_root"]) / id2img["000"]).read_bytes()
        a1, a2 = extractor.extract([project_rgb(data)])
        b1, b2 = extractor.extract([project_rgb(data)])
        self.assertTrue(torch.equal(a1, b1))
        self.assertTrue(torch.equal(a2, b2))
        self.assertEqual(tuple(a1.shape), (1, 1536))
        self.assertEqual(tuple(a2.shape), (1, 4608))
        pixels = extractor.processor(images=[project_rgb(data)], return_tensors="pt")["pixel_values"].cuda()
        with torch.no_grad(): fresh_final = extractor.model.get_image_features(pixel_values=pixels).cpu()[0]
        cached_final = torch.load(self.config["final_clip_cache"], map_location="cpu")
        similarities = torch.nn.functional.normalize(cached_final, dim=1).matmul(torch.nn.functional.normalize(fresh_final, dim=0))
        self.assertEqual(int(torch.argmax(similarities)), 0)
        self.assertGreater(float(similarities[0]), 0.999999)

    def test_ten_candidate_h_and_positive_logit_exact_reconstruction(self):
        source = read_json(self.config["validation_candidates"])
        manifest = normalized_candidate_manifest(source[:1], "validation", "fixed_r10", self.config["validation_candidates"])
        cache = extract_pair_cache(self.model, self.args, source[:1], manifest, pair_batch_size=10)
        self.assertEqual(tuple(cache["h"].shape), (1, 10, 768))
        self.assertEqual(tuple(cache["b"].shape), (1, 10))
        exact = validate_against_legacy(self.model, self.args, source[0], manifest[0], cache["h"][0], cache["b"][0])
        self.assertTrue(exact["candidate_ids_equal"])
        self.assertEqual(exact["h_max_abs"], 0.0)
        self.assertEqual(exact["b_max_abs"], 0.0)
        classifier = self.model.model.bert.classifier
        reconstructed = classifier(self.model.model.bert.dropout(cache["h"][0].cuda()))[:, 1].cpu()
        self.assertTrue(torch.equal(reconstructed, cache["b"][0]))


if __name__ == "__main__":
    unittest.main()
