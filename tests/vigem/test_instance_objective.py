import unittest

import torch

from vigem.residuals import masked_instance_listwise_loss


class InstanceObjectiveTest(unittest.TestCase):
    def test_all_valid_negatives_receive_probability_weighted_gradient(self):
        scores = torch.tensor(
            [[2.0, 1.5, 1.0, 0.5]], requires_grad=True
        )
        valid = torch.ones_like(scores, dtype=torch.bool)
        loss, eligible = masked_instance_listwise_loss(scores, valid)
        loss.backward()
        self.assertEqual(eligible.tolist(), [True])
        self.assertLess(scores.grad[0, 0].item(), 0.0)
        self.assertTrue(torch.all(scores.grad[0, 1:] > 0.0))
        self.assertTrue(
            torch.all(scores.grad[0, 1:-1] > scores.grad[0, 2:])
        )

    def test_invalid_and_gray_slots_have_no_gradient(self):
        scores = torch.tensor(
            [[2.0, 1.0, 5.0, 4.0]], requires_grad=True
        )
        valid = torch.tensor([[True, True, False, False]])
        loss, _ = masked_instance_listwise_loss(scores, valid)
        loss.backward()
        self.assertEqual(scores.grad[0, 2:].tolist(), [0.0, 0.0])

    def test_row_without_real_negative_is_skipped(self):
        scores = torch.tensor(
            [[2.0, 8.0], [1.0, 0.0]], requires_grad=True
        )
        valid = torch.tensor([[True, False], [True, True]])
        loss, eligible = masked_instance_listwise_loss(scores, valid)
        loss.backward()
        self.assertEqual(eligible.tolist(), [False, True])
        self.assertEqual(scores.grad[0].tolist(), [0.0, 0.0])

    def test_all_rows_ineligible_returns_differentiable_zero(self):
        scores = torch.randn(2, 3, requires_grad=True)
        valid = torch.tensor(
            [[True, False, False], [True, False, False]]
        )
        loss, eligible = masked_instance_listwise_loss(scores, valid)
        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(eligible.tolist(), [False, False])
        loss.backward()
        self.assertTrue(torch.equal(scores.grad, torch.zeros_like(scores)))
