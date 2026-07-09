"""Local future-dynamics prediction heads for LD-TDN."""

from __future__ import annotations

import torch
from torch import nn


class LocalDynamicsHeads(nn.Module):
    """Predict future local classes and short-horizon regression targets."""

    def __init__(self, z_dim: int, hidden_dim: int = 64, class_dims: dict[str, int] | None = None) -> None:
        super().__init__()
        class_dims = class_dims or {
            "mobility": 3,
            "contact": 3,
            "residence": 3,
            "escape": 3,
            "relax": 3,
        }

        def cls_head(name: str) -> nn.Sequential:
            return nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, class_dims[name]))

        self.mobility = cls_head("mobility")
        self.contact = cls_head("contact")
        self.residence = cls_head("residence")
        self.escape = cls_head("escape")
        self.relax = cls_head("relax")
        self.reg = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 13))

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.reg(z)
        return {
            "mobility_logits": self.mobility(z),
            "contact_logits": self.contact(z),
            "residence_logits": self.residence(z),
            "escape_logits": self.escape(z),
            "relax_logits": self.relax(z),
            "disp_mu": raw[:, 0:3],
            "disp_logvar": raw[:, 3:6].clamp(min=-8.0, max=8.0),
            "short_msd_mu": raw[:, 6:8],
            "short_msd_logvar": raw[:, 8:10].clamp(min=-8.0, max=8.0),
            "contact_survival": torch.sigmoid(raw[:, 10]),
            "wall_contact_survival": torch.sigmoid(raw[:, 11]),
            "free_volume_opening": torch.sigmoid(raw[:, 12]),
        }
