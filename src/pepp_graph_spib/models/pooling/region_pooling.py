"""Region-wise pooling from local descriptors to system transport descriptors."""

from __future__ import annotations

import torch
from torch import nn


REGION_SCALAR_NAMES = [
    "fraction_fast",
    "fraction_slow",
    "fraction_persistent_contact",
    "fraction_wall_resident",
    "fraction_escape_ready",
    "mean_free_volume_proxy",
    "mean_wall_distance",
    "mean_local_density",
    "mean_polymer_wall_contact_fraction",
    "mean_local_PE_fraction",
    "mean_local_PP_fraction",
    "mean_local_PS_fraction",
    "mean_radial_bin",
    "mean_axial_bin",
]


def region_repr_dim(z_dim: int) -> int:
    """Return the fixed system representation dimension."""
    return 2 * z_dim + len(REGION_SCALAR_NAMES) + 8


class RegionPooling(nn.Module):
    """Aggregate local descriptors by system and simple pore bins."""

    def forward(
        self,
        z: torch.Tensor,
        local_outputs: dict[str, torch.Tensor],
        metadata: dict[str, torch.Tensor],
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        system_ids = metadata["system_id"].detach().cpu()
        unique = torch.unique(system_ids, sorted=True)
        rows = []
        cond_rows = []
        device = z.device
        mobility = local_outputs["mobility_logits"].softmax(dim=-1)
        contact = local_outputs["contact_logits"].softmax(dim=-1)
        escape = local_outputs["escape_logits"].softmax(dim=-1)
        for sid in unique:
            mask_cpu = system_ids == sid
            mask = mask_cpu.to(device)
            zi = z[mask]
            mean_z = zi.mean(dim=0)
            var_z = zi.var(dim=0, unbiased=False)
            md = {key: value.to(device)[mask] for key, value in metadata.items() if value.ndim == 1}
            fast = mobility[mask, -1]
            slow = mobility[mask, 0]
            persistent = contact[mask, -1]
            escape_ready = escape[mask, -1]
            wall_contact = md.get("polymer_wall_contact_fraction", fast.new_zeros(fast.shape))
            scalars = torch.stack(
                [
                    fast.mean(),
                    slow.mean(),
                    persistent.mean(),
                    wall_contact.mean(),
                    escape_ready.mean(),
                    md.get("mean_free_volume_proxy", fast.new_zeros(fast.shape)).mean(),
                    md.get("mean_wall_distance", fast.new_zeros(fast.shape)).mean(),
                    md.get("mean_local_density", fast.new_zeros(fast.shape)).mean(),
                    wall_contact.mean(),
                    md.get("local_PE_fraction", fast.new_zeros(fast.shape)).mean(),
                    md.get("local_PP_fraction", fast.new_zeros(fast.shape)).mean(),
                    md.get("local_PS_fraction", fast.new_zeros(fast.shape)).mean(),
                    md.get("radial_bin", fast.new_zeros(fast.shape)).mean(),
                    md.get("axial_bin", fast.new_zeros(fast.shape)).mean(),
                ]
            )
            radial = md.get("radial_bin", fast.new_zeros(fast.shape)).float().clamp(0, 3).long().cpu()
            axial = md.get("axial_bin", fast.new_zeros(fast.shape)).float().clamp(0, 3).long().cpu()
            radial_hist = torch.bincount(radial, minlength=4).float().to(device)
            radial_hist = radial_hist / radial_hist.sum().clamp_min(1.0)
            axial_hist = torch.bincount(axial, minlength=4).float().to(device)
            axial_hist = axial_hist / axial_hist.sum().clamp_min(1.0)
            rows.append(torch.cat([mean_z, var_z, scalars, radial_hist, axial_hist], dim=0))
            cond_rows.append(condition[mask][0])
        return torch.stack(rows, dim=0), unique.to(device).long(), torch.stack(cond_rows, dim=0)
