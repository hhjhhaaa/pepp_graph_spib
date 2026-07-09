"""Physics-informed transport head for LD-TDN."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PhysicsTransportHead(nn.Module):
    """Predict positive/bounded transport factors and derived pore outputs."""

    def __init__(
        self,
        system_repr_dim: int,
        condition_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        dims = [system_repr_dim + condition_dim, *hidden_dims, 7]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, system_repr: torch.Tensor, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(torch.cat([system_repr.float(), condition.float()], dim=-1))
        d_local = F.softplus(raw[:, 0]) + self.eps
        p_entry = torch.sigmoid(raw[:, 1])
        c_axis = torch.sigmoid(raw[:, 2])
        tau_wall = F.softplus(raw[:, 3]) + self.eps
        tau_move = F.softplus(raw[:, 4]) + self.eps
        p_access = torch.sigmoid(raw[:, 5])
        active_site_residence_fraction = torch.sigmoid(raw[:, 6])
        wall_residence_fraction = tau_wall / (tau_wall + tau_move + self.eps)
        transport_score = p_entry * c_axis / (1.0 + tau_wall / (tau_move + self.eps))
        d_eff = d_local * transport_score
        reaction = p_entry * p_access * active_site_residence_fraction
        return {
            "D_local": d_local,
            "P_entry": p_entry,
            "C_axis": c_axis,
            "tau_wall": tau_wall,
            "tau_move": tau_move,
            "P_access": p_access,
            "wall_residence_fraction": wall_residence_fraction,
            "active_site_residence_fraction": active_site_residence_fraction,
            "transport_score": transport_score,
            "D_eff": d_eff + self.eps,
            "reaction_opportunity_index": reaction,
            "log_D_self": torch.log(d_local + self.eps),
            "log_D_parallel": torch.log(d_local * c_axis + self.eps),
            "log_D_eff": torch.log(d_eff + self.eps),
            "log_tau_segmental": torch.log(tau_move + self.eps),
            "log_tau_res": torch.log(tau_wall + self.eps),
        }
