"""Predictive variational bottleneck for local transport descriptors."""

from __future__ import annotations

import torch
from torch import nn


class VariationalBottleneck(nn.Module):
    """Map encoder state to a low-dimensional stochastic descriptor."""

    def __init__(self, input_dim: int, z_dim: int = 4) -> None:
        super().__init__()
        self.mu = nn.Linear(input_dim, z_dim)
        self.logvar = nn.Linear(input_dim, z_dim)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-8.0, max=8.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        return {"z": z, "mu": mu, "logvar": logvar}


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x,c) || N(0, I)) averaged over the batch."""
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
