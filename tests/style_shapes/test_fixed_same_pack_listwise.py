import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from structured_retrieval_factorized import (
    PrototypeReasoner,
    StructuredFactorizedPLModel,
    StructuredFactorizedStickerModel,
    _fixed_listwise_log_preview,
)
from style_shapes.fixed_same_pack import (
    GRAY_SENTINEL_ID,
    FixedSamePackRuntime,
    build_split_candidates,
    fixed_candidates_for_gold,
    flatten_query_major,
    hardest_expression_rank_loss,
    listwise_match_loss,
    load_pack_orders_from_zip,
    parse_emoji_mapping,
    validate_candidate_row,
    validate_fixed_trace_record,
)


class FixedSamePackCandidatesTest(unittest.TestCase):
    def test_mapping_order_gold_removal_and_gray_padding(self):
        mapping = parse_emoji_mapping("11\tA\n12\tB\n13\tC\n")
        candidates, gray = fixed_candidates_for_gold(
            7,
            "12.npy",
            mapping,
            {"7-11": 101, "7-12": 102, "7-13": 103},
        )
        self.assertEqual(candidates[:3], [102, 101, 103])
        self.assertEqual(candidates[3:], [GRAY_SENTINEL_ID] * 7)
        self.assertEqual(gray, [False, False, False] + [True] * 7)
        validate_candidate_row(candidates, gray, 102)

    def test_real_candidates_are_unique(self):
        with self.assertRaises(ValueError):
            validate_candidate_row(
                [4, 5, 5] + [GRAY_SENTINEL_ID] * 7,
                [False, False, False] + [True] * 7,
                4,
            )

    def test_missing_mapping_fails_instead_of_falling_back(self):
        with self.assertRaises(KeyError):
            fixed_candidates_for_gold(
                7, 12, [11, 12, 13], {"7-11": 101, "7-12": 102}
            )

    def test_zip_release_and_mapping_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "raw.zip"
            row = {
                "context": [{"text": "hello"}],
                "current": {"sticker_set_id": 7, "sticker_id": 12},
            }
            with zipfile.ZipFile(str(archive_path), "w") as archive:
                archive.writestr(
                    "stickerchat/npy_stickers/7/emoji_mapping.txt",
                    "11\tA\n12\tB\n13\tC\n",
                )
                archive.writestr(
                    "stickerchat/release_train.json",
                    json.dumps(row) + "\n",
                )
            with zipfile.ZipFile(str(archive_path), "r") as archive:
                orders, content_hash, fallback_packs = load_pack_orders_from_zip(
                    archive
                )
                rows, candidates, gray, stats = build_split_candidates(
                    archive,
                    "train",
                    orders,
                    {"7-11": 101, "7-12": 102, "7-13": 103},
                )
            self.assertEqual(len(content_hash), 64)
            self.assertEqual(fallback_packs, [])
            self.assertEqual(rows[0]["source_row"], 0)
            self.assertEqual(candidates.shape, (1, 10))
            self.assertEqual(candidates[0, :3].tolist(), [102, 101, 103])
            self.assertEqual(gray[0].sum().item(), 7)
            self.assertEqual(stats["gray_rows"], 1)
            self.assertEqual(stats["gray_slots"], 7)

    def test_vpd_and_semsp_configs_share_candidates_init_and_order(self):
        values = []
        for path in (
            "configs/style_shapes/stickerchat_vpd_pack_fixed_same_pack_r10.yaml",
            "configs/style_shapes/stickerchat_semsp_fixed_same_pack_r10.yaml",
        ):
            with open(path, "r", encoding="utf-8") as handle:
                values.append(yaml.safe_load(handle))
        for key in (
            "init_checkpoint_path",
            "permutation_manifest",
            "fixed_candidates",
        ):
            self.assertEqual(values[0][key], values[1][key])
        self.assertNotEqual(values[0]["group_bank"], values[1]["group_bank"])
        self.assertEqual(
            values[0]["model_overrides"]["factorized_train_mode"],
            "fixed_same_pack_listwise",
        )
        self.assertEqual(
            values[0]["model_overrides"]["factorized_candidate_forward_chunk_size"],
            10,
        )

    def test_4090_profiles_are_fp16_only_and_leave_a800_configs_unchanged(self):
        pairs = (
            (
                "configs/style_shapes/stickerchat_vpd_pack_fixed_same_pack_r10.yaml",
                "configs/style_shapes/"
                "stickerchat_vpd_pack_fixed_same_pack_r10_4090_24g.yaml",
            ),
            (
                "configs/style_shapes/stickerchat_semsp_fixed_same_pack_r10.yaml",
                "configs/style_shapes/"
                "stickerchat_semsp_fixed_same_pack_r10_4090_24g.yaml",
            ),
        )
        for a800_path, rtx_path in pairs:
            with open(a800_path, "r", encoding="utf-8") as handle:
                a800 = yaml.safe_load(handle)
            with open(rtx_path, "r", encoding="utf-8") as handle:
                rtx = yaml.safe_load(handle)
            self.assertNotIn("hardware_profile", a800)
            self.assertNotIn("trainer_precision", a800["model_overrides"])
            self.assertEqual(rtx["hardware_profile"], "rtx4090_24gb_fp16")
            self.assertEqual(a800["conda_env"], "stickr-select")
            self.assertEqual(rtx["conda_env"], "sticker-select")
            self.assertEqual(rtx["model_overrides"]["trainer_precision"], 16)
            self.assertEqual(rtx["model_overrides"]["train_batch_size"], 16)
            self.assertEqual(
                rtx["model_overrides"]["gradient_accumulation_steps"], 1
            )
            self.assertEqual(
                rtx["model_overrides"]["factorized_candidate_forward_chunk_size"],
                10,
            )
            self.assertEqual(a800["group_bank"], rtx["group_bank"])
            self.assertEqual(
                a800["fixed_candidates"], rtx["fixed_candidates"]
            )
            self.assertEqual(
                a800["init_checkpoint_path"], rtx["init_checkpoint_path"]
            )
            self.assertEqual(
                a800["permutation_manifest"], rtx["permutation_manifest"]
            )
            self.assertNotEqual(a800["output_dir"], rtx["output_dir"])


class ListwiseTrainingMathTest(unittest.TestCase):
    def test_minimal_training_step_logs_listwise_debug_at_step_zero(self):
        scalar = torch.tensor(1.0)
        output = SimpleNamespace(
            loss=scalar,
            match_loss=scalar,
            style_proto_loss=scalar,
            orth_loss=scalar,
            expr_rank_loss=scalar,
            debug_info={
                "_train_scalars": torch.tensor([0.25, 0.5]),
                "negative_policy": "fixed_same_pack_listwise",
                "candidate_ids": torch.arange(20).reshape(2, 10),
                "hardest_expression_ids": torch.tensor([7, 18]),
            },
        )

        class FakeMinimalPL:
            def __init__(self):
                self.model = SimpleNamespace(uses_full_variant=lambda: False)
                self.args = SimpleNamespace(
                    lambda_style_proto=0.4,
                    lambda_expr=0.3,
                    lambda_expr_rank_loss=None,
                    lambda_orth=0.5,
                    base_only=False,
                    train_prog_bar_mode="compact",
                    factorized_log_interval=200,
                )
                self.global_step = 0
                self._style_proto_acc_ema = None

            def run_train_batch(self, batch):
                return output

            def log(self, *args, **kwargs):
                return None

            def _update_ema(self, old, value, alpha=0.05):
                return value if old is None else alpha * value + (1.0 - alpha) * old

        loss = StructuredFactorizedPLModel.training_step(
            FakeMinimalPL(), {}, 0
        )
        self.assertIs(loss, scalar)

    def test_listwise_log_preview_does_not_require_legacy_triplet_fields(self):
        preview = _fixed_listwise_log_preview(
            {
                "negative_policy": "fixed_same_pack_listwise",
                "candidate_ids": torch.tensor(
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15, 16, 17, 18, 19, -1],
                        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                    ]
                ),
                "hardest_expression_ids": torch.tensor([7, 19, 24]),
            }
        )
        self.assertEqual(
            preview,
            (
                [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15, 16, 17, 18, 19, -1],
                ],
                [7, 19, 24],
            ),
        )
        self.assertIsNone(
            _fixed_listwise_log_preview({"negative_policy": "legacy_triplet"})
        )

    def test_query_major_flatten_round_trip(self):
        dialogue = torch.tensor([[10, 11], [20, 21]])
        mask = torch.ones_like(dialogue)
        candidates = torch.tensor([[1, 2, 3], [4, 5, 6]])
        flat_dialogue, flat_mask, flat_candidates = flatten_query_major(
            dialogue, mask, candidates
        )
        self.assertEqual(
            flat_dialogue.tolist(),
            [[10, 11], [10, 11], [10, 11], [20, 21], [20, 21], [20, 21]],
        )
        self.assertEqual(flat_mask.shape, flat_dialogue.shape)
        self.assertEqual(flat_candidates.tolist(), [1, 2, 3, 4, 5, 6])
        fake_vectorized = flat_dialogue[:, 0].float() + flat_candidates.float()
        fake_serial = torch.stack(
            [
                dialogue[:, 0].float() + candidates[:, column].float()
                for column in range(candidates.size(1))
            ],
            dim=1,
        )
        self.assertTrue(
            torch.equal(
                fake_vectorized.reshape(candidates.shape), fake_serial
            )
        )

    def test_listwise_ce_gives_every_negative_probability_weighted_gradient(self):
        scores = torch.tensor(
            [[2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5]],
            requires_grad=True,
        )
        listwise_match_loss(scores).backward()
        negative_gradients = scores.grad[0, 1:]
        self.assertTrue(torch.all(negative_gradients > 0))
        self.assertTrue(
            torch.all(negative_gradients[:-1] > negative_gradients[1:])
        )

    def test_expression_aux_only_backprops_to_hardest_negative(self):
        scores = torch.tensor(
            [[2.0, 1.0, 3.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]],
            requires_grad=True,
        )
        loss, hardest = hardest_expression_rank_loss(scores, margin=0.2)
        loss.backward()
        self.assertEqual(hardest.tolist(), [2])
        self.assertLess(scores.grad[0, 0].item(), 0)
        self.assertGreater(scores.grad[0, 2].item(), 0)
        untouched = torch.cat([scores.grad[0, 1:2], scores.grad[0, 3:]])
        self.assertTrue(torch.equal(untouched, torch.zeros_like(untouched)))

    def test_gray_embedding_is_substituted_and_group_score_is_zero(self):
        model = object.__new__(StructuredFactorizedStickerModel)
        object.__setattr__(
            model, "_gray_candidate_embedding_cpu", torch.full((4,), 127.0)
        )
        bank = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        embedded = StructuredFactorizedStickerModel._candidate_embeddings_from_bank(
            model, bank, [2, GRAY_SENTINEL_ID, 0]
        )
        self.assertTrue(torch.equal(embedded[0], bank[2]))
        self.assertTrue(
            torch.equal(embedded[1], torch.full((4,), 127.0))
        )
        self.assertTrue(torch.equal(embedded[2], bank[0]))

        object.__setattr__(model, "_sticker_to_proto_cpu", torch.tensor([0, 1, 1]))
        object.__setattr__(model, "_sticker_to_proto_by_device", {})
        object.__setattr__(model, "prototype_reasoner", PrototypeReasoner(2, 0.0))
        proto_logits = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        group = StructuredFactorizedStickerModel._gather_proto_scores_for_batch(
            model, proto_logits, [1, GRAY_SENTINEL_ID]
        )
        self.assertEqual(group.tolist(), [3.0, 0.0])

    def test_fixed_trace_checks_manifest_hardest_and_gray_group_mask(self):
        candidates = torch.tensor(
            [[4, 5, 6] + [GRAY_SENTINEL_ID] * 7],
            dtype=torch.long,
        )
        runtime = FixedSamePackRuntime(
            {},
            candidates,
            candidates.eq(GRAY_SENTINEL_ID),
        )
        expression = [1.0, 2.0, 3.0] + [0.0] * 7
        record = {
            "negative_policy": "fixed_same_pack_listwise",
            "source_row": 0,
            "positive": 4,
            "candidate_ids": candidates[0].tolist(),
            "gray_mask": candidates[0].eq(GRAY_SENTINEL_ID).tolist(),
            "base_scores": [0.0] * 10,
            "expression_scores": expression,
            "group_scores": [1.0, 1.0, 1.0] + [0.0] * 7,
            "final_scores": [0.0] * 10,
            "hardest_expression_index": 2,
            "hardest_expression_id": 6,
        }
        validate_fixed_trace_record(record, runtime)
        record["group_scores"][-1] = 1.0
        with self.assertRaises(ValueError):
            validate_fixed_trace_record(record, runtime)


if __name__ == "__main__":
    unittest.main()
