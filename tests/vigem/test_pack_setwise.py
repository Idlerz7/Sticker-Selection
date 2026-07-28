import types
import unittest

import torch
from torch import nn

from vigem.pack_model import PackRelativeSetwiseStickerModel


class _SetwiseHarness(nn.Module):
    def __init__(self):
        super().__init__()
        self.instance_residual_head = nn.Linear(768, 2, bias=False)
        self.instance_residual_norm = nn.LayerNorm(
            2, elementwise_affine=False
        )
        self.instance_set_scorer = nn.Sequential(
            nn.Linear(10, 2),
            nn.Tanh(),
            nn.Dropout(0.0),
            nn.Linear(2, 1),
        )
        self.instance_temperature = torch.tensor(2.0)
        with torch.no_grad():
            self.instance_residual_head.weight.copy_(
                torch.cat(
                    [
                        torch.eye(2),
                        torch.zeros(2, 766),
                    ],
                    dim=1,
                )
            )

    def _pack_rows(self, candidate_ids, device):
        table = torch.zeros(4, 768, device=device)
        table[:, :2] = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.0, -1.0]],
            device=device,
        )
        real = candidate_ids.ge(0)
        safe = candidate_ids.clamp_min(0)
        residual = table.index_select(0, safe.reshape(-1)).reshape(
            *candidate_ids.shape, 768
        )
        residual = residual.masked_fill(~real.unsqueeze(-1), 0.0)
        pack_ids = torch.zeros_like(candidate_ids).masked_fill(~real, -1)
        pack_sizes = torch.full_like(candidate_ids, 4).masked_fill(~real, 0)
        vpd = torch.ones_like(candidate_ids).masked_fill(~real, -1)
        return residual, pack_ids, pack_sizes, vpd, pack_sizes, real

    compute = PackRelativeSetwiseStickerModel.compute_setwise_instance_scores


class _FakeBertCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)

    def forward(self, inputs_embeds, attention_mask, return_dict):
        del attention_mask, return_dict
        return types.SimpleNamespace(
            last_hidden_state=self.proj(inputs_embeds)
        )


class _QueryHarness(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = types.SimpleNamespace(bert=_FakeBertCore())
        self.add_module("bert_core", self.bert.bert)
        self.style_query_head = nn.Linear(3, 2, bias=False)
        self.instance_query_head = nn.Linear(3, 2, bias=False)
        self.instance_query_norm = nn.LayerNorm(
            2, elementwise_affine=False
        )

    def _get_text_word_embeddings(self, input_ids):
        return torch.nn.functional.one_hot(
            input_ids, num_classes=3
        ).float()

    encode = PackRelativeSetwiseStickerModel.encode_group_and_instance_queries


class PackSetwiseTest(unittest.TestCase):
    def test_permutation_equivariance_and_gray_mask(self):
        torch.manual_seed(11)
        model = _SetwiseHarness().eval()
        query = torch.tensor([[0.5, -0.5]])
        candidates = torch.tensor([[0, 1, 2, -1]])
        first = model.compute(query, candidates)
        permutation = torch.tensor([2, 0, 3, 1])
        second = model.compute(query, candidates[:, permutation])
        self.assertTrue(
            torch.allclose(
                first["scores"][:, permutation],
                second["scores"],
                atol=1e-6,
            )
        )
        self.assertEqual(float(first["scores"][0, 3].item()), 0.0)
        self.assertFalse(bool(first["informative_mask"][0, 3]))

    def test_instance_listwise_gradients_reach_all_real_candidates(self):
        torch.manual_seed(12)
        model = _SetwiseHarness().train()
        query = torch.tensor([[0.5, -0.5]], requires_grad=True)
        output = model.compute(query, torch.tensor([[0, 1, 2, 3]]))
        loss = torch.nn.functional.cross_entropy(
            output["scores"], torch.zeros(1, dtype=torch.long)
        )
        loss.backward()
        self.assertIsNotNone(query.grad)
        self.assertGreater(float(query.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(model.instance_set_scorer[0].weight.grad)

    def test_group_query_detaches_shared_bert_only(self):
        model = _QueryHarness()
        input_ids = torch.tensor([[0, 1]])
        mask = torch.ones_like(input_ids)
        group, instance = model.encode(input_ids, mask)
        group.sum().backward(retain_graph=True)
        self.assertIsNone(model.bert_core.proj.weight.grad)
        self.assertGreater(
            float(model.style_query_head.weight.grad.abs().sum().item()), 0.0
        )
        model.zero_grad(set_to_none=True)
        instance.square().sum().backward()
        self.assertIsNotNone(model.bert_core.proj.weight.grad)
        self.assertGreater(
            float(model.bert_core.proj.weight.grad.abs().sum().item()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
