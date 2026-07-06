"""System-level pooling for local bottleneck outputs."""

from __future__ import annotations

import torch


def aggregate_system_embeddings(
    z: torch.Tensor,
    mobility_probs: torch.Tensor,
    residence_probs: torch.Tensor,
    accessibility_probs: torch.Tensor,
    system_ids: torch.Tensor,
    center_segment_types: torch.Tensor,
    metadata: dict[str, torch.Tensor],
    composition_hist_bins: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate local window outputs into one row per system."""
    unique = torch.unique(system_ids.cpu(), sorted=True)
    rows = []
    for sid in unique:
        mask = (system_ids.cpu() == sid).to(z.device)
        zi = z[mask]
        mob = mobility_probs[mask]
        res = residence_probs[mask]
        acc = accessibility_probs[mask]
        scalars = torch.stack([
            mob[:, 0].mean(),
            mob[:, 2].mean(),
            res[:, 2].mean(),
            acc[:, 2].mean(),
            metadata["local_PE_fraction"].to(z.device)[mask].mean(),
            metadata["local_PP_fraction"].to(z.device)[mask].mean(),
            metadata["local_PC_fraction"].to(z.device)[mask].mean(),
            metadata["polymer_wall_contact_fraction"].to(z.device)[mask].mean(),
            metadata["mean_local_density"].to(z.device)[mask].mean(),
            metadata["mean_free_volume_proxy"].to(z.device)[mask].mean(),
        ])
        rows.append(torch.cat([zi.mean(dim=0), zi.var(dim=0, unbiased=False), scalars], dim=0))
    return torch.stack(rows, dim=0), unique.long()


def system_repr_dim(z_dim: int, hist_bins: int = 5) -> int:
    """Return dimensionality of aggregate_system_embeddings output."""
    return 2 * z_dim + 10
