"""Descriptor time-series encoders for LD-TDN."""

from __future__ import annotations

import torch
from torch import nn


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        pad = kernel_size - 1
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.ReLU(),
        )
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        y = y[..., : x.size(-1)]
        return x + y


class TimeSeriesEncoder(nn.Module):
    """Encode descriptor windows `x` with shape [B, T, F] into [B, H]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        encoder_type: str = "gru",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif encoder_type == "tcn":
            layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
            self.input = nn.Sequential(*layers)
            self.blocks = nn.ModuleList([_TCNBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)])
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.encoder_type == "gru":
            _, h_n = self.encoder(x)
            return h_n[-1]
        h = self.input(x).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        return self.norm(h.transpose(1, 2)[:, -1, :])
