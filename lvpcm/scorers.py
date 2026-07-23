"""Frozen conditional and parameter-matched unconditional scoring controls."""

from __future__ import annotations

import math

import torch
from torch import nn


class ConditionalScorer(nn.Module):
    def __init__(self, sigma_b: float, hidden_dim: int = 768, condition_dim: int = 256, rank: int = 16):
        super().__init__()
        self.sigma_b = float(sigma_b)
        self.rank = int(rank)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.A = nn.Parameter(torch.empty(rank, hidden_dim))
        self.C = nn.Parameter(torch.zeros(rank, condition_dim))
        nn.init.xavier_uniform_(self.A)

    def delta(self, h: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        left = torch.einsum("...d,rd->...r", self.norm(h), self.A)
        right = torch.einsum("...d,rd->...r", condition, self.C)
        return self.sigma_b * torch.tanh((left * right).sum(dim=-1) / math.sqrt(self.rank))

    def forward(self, base_logits: torch.Tensor, h: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return base_logits + self.delta(h, condition)


class UnconditionalScorer(nn.Module):
    def __init__(self, sigma_b: float, hidden_dim: int = 768, rank: int = 21):
        super().__init__()
        self.sigma_b = float(sigma_b)
        self.rank = int(rank)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.U = nn.Parameter(torch.empty(rank, hidden_dim))
        self.v = nn.Parameter(torch.zeros(rank))
        nn.init.xavier_uniform_(self.U)

    def delta(self, h: torch.Tensor) -> torch.Tensor:
        projection = torch.einsum("...d,rd->...r", self.norm(h), self.U)
        return self.sigma_b * torch.tanh(torch.einsum("...r,r->...", projection, self.v) / math.sqrt(self.rank))

    def forward(self, base_logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return base_logits + self.delta(h)


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
