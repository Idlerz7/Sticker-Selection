import unittest
from pathlib import Path

import torch
import yaml
from torch import nn

from vigem.config import load_legacy_shared_initialization


class _TinyNewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.shared = nn.Linear(2, 2)
        self.model.instance_query_head = nn.Linear(2, 2)
        self.model.instance_residual_head = nn.Linear(3, 2)
        self.model.instance_temperature_raw = nn.Parameter(torch.tensor(0.0))


class CompatibilityTest(unittest.TestCase):
    def test_legacy_load_allows_only_new_instance_keys(self):
        model = _TinyNewModel()
        legacy = {
            "state_dict": {
                "model.shared.weight": torch.ones(2, 2),
                "model.shared.bias": torch.zeros(2),
            }
        }
        with self.subTest("allowed missing keys"):
            path = Path(self.id().replace(".", "_") + ".pt")
            try:
                torch.save(legacy, path)
                audit = load_legacy_shared_initialization(model, str(path))
            finally:
                path.unlink(missing_ok=True)
            self.assertTrue(audit["missing_new_keys"])
            self.assertTrue(
                torch.equal(model.model.shared.weight, torch.ones(2, 2))
            )

    def test_new_configs_do_not_replace_legacy_paths(self):
        dstc_path = Path(
            "configs/vigem/dstc_vpd_multi_instance_residual.yaml"
        )
        stickerchat_path = Path(
            "configs/vigem/stickerchat_vpd_pack_fixed_r10_instance_residual.yaml"
        )
        self.assertTrue(dstc_path.exists())
        self.assertTrue(stickerchat_path.exists())
        dstc = yaml.safe_load(dstc_path.read_text())
        stickerchat = yaml.safe_load(stickerchat_path.read_text())
        self.assertEqual(dstc["instance_score_weight"], 0.3)
        self.assertEqual(stickerchat["instance_loss_weight"], 0.3)
        self.assertTrue(
            dstc["output_dir"].startswith("artifacts/vigem/")
        )
        self.assertTrue(
            stickerchat["output_dir"].startswith("artifacts/vigem/")
        )
        self.assertEqual(
            stickerchat["fixed_candidates"]["manifest_path"],
            "artifacts/style_shapes/candidates/"
            "stickerchat_fixed_same_pack_r10/manifest.json",
        )

    def test_original_entrypoint_and_configs_remain_present(self):
        self.assertTrue(Path("scripts/style_shapes/run_pilot.py").exists())
        self.assertTrue(
            Path("configs/style_shapes/dstc_vpd_multi.yaml").exists()
        )
        self.assertTrue(
            Path(
                "configs/style_shapes/"
                "stickerchat_vpd_pack_fixed_same_pack_r10.yaml"
            ).exists()
        )
