"""Transport property prediction head."""

from __future__ import annotations

import torch
from torch import nn


class TransportPropertyHead(nn.Module):
    """MLP mapping system distribution representation plus condition to targets."""

    def __init__(self, system_repr_dim: int, condition_dim: int, hidden_dims=(128, 64, 32), target_dim: int = 5) -> None:
        super().__init__()
        dims = [system_repr_dim + condition_dim, *hidden_dims, target_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, system_repr: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Return transport predictions [num_systems, 5]."""
        return self.net(torch.cat([system_repr.float(), condition.float()], dim=-1))
