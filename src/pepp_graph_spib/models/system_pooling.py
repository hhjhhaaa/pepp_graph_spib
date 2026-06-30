"""System-level distribution pooling for local Graph-SPIB outputs."""

from __future__ import annotations

import torch


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """Categorical entropy for probabilities [N, C]."""
    return -(probs.clamp_min(1.0e-8) * probs.clamp_min(1.0e-8).log()).sum(dim=-1)


def _safe_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    if mask.any():
        return x[mask].mean(dim=0)
    return x.new_zeros(dim)


def aggregate_system_embeddings(
    z: torch.Tensor,
    mobility_probs: torch.Tensor,
    relax_probs: torch.Tensor,
    contact_probs: torch.Tensor,
    system_ids: torch.Tensor,
    center_segment_types: torch.Tensor,
    metadata: dict[str, torch.Tensor],
    pe_hist_bins: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate local windows into system representations.

    z has shape [N, z_dim]. Returned system_repr has one row per unique system.
    """
    pe_hist_bins = pe_hist_bins or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    unique = torch.unique(system_ids.cpu(), sorted=True)
    rows = []
    for sid in unique:
        mask = system_ids.cpu() == sid
        idx = mask.to(z.device)
        zi = z[idx]
        mob = mobility_probs[idx]
        con = contact_probs[idx]
        center_type = center_segment_types.to(z.device)[idx]
        local_pe = metadata["local_PE_fraction"].to(z.device)[idx]
        pe_pe = metadata["PE_PE_contact_fraction"].to(z.device)[idx]
        pp_pp = metadata["PP_PP_contact_fraction"].to(z.device)[idx]
        pe_pp = metadata["PE_PP_contact_fraction"].to(z.device)[idx]
        mean_z = zi.mean(dim=0)
        var_z = zi.var(dim=0, unbiased=False)
        pe_mean = _safe_mean(zi, center_type == 0, zi.size(1))
        pp_mean = _safe_mean(zi, center_type == 1, zi.size(1))
        interface = (local_pe > 0.2) & (local_pe < 0.8)
        pe_rich = local_pe >= 0.8
        pp_rich = local_pe <= 0.2
        interface_mean = _safe_mean(zi, interface, zi.size(1))
        slow_prob = mob[:, 0]
        fast_prob = mob[:, 2]
        persistent = con[:, 2]
        pe_hist = torch.histc(local_pe.float().cpu(), bins=len(pe_hist_bins) - 1, min=0.0, max=1.0).to(z.device)
        pe_hist = pe_hist / pe_hist.sum().clamp_min(1.0)
        local_pp = 1.0 - local_pe
        pp_hist = torch.histc(local_pp.float().cpu(), bins=len(pe_hist_bins) - 1, min=0.0, max=1.0).to(z.device)
        pp_hist = pp_hist / pp_hist.sum().clamp_min(1.0)
        parts = [
            mean_z,
            var_z,
            pe_mean,
            pp_mean,
            interface_mean,
            torch.stack(
                [
                    slow_prob.mean(),
                    fast_prob.mean(),
                    persistent.mean(),
                    entropy(mob).mean(),
                    entropy(con).mean(),
                    pe_pe.mean(),
                    pp_pp.mean(),
                    pe_pp.mean(),
                    slow_prob[pe_rich].mean() if pe_rich.any() else slow_prob.new_tensor(0.0),
                    slow_prob[pp_rich].mean() if pp_rich.any() else slow_prob.new_tensor(0.0),
                    slow_prob[interface].mean() if interface.any() else slow_prob.new_tensor(0.0),
                ]
            ),
            pe_hist,
            pp_hist,
        ]
        rows.append(torch.cat(parts, dim=0))
    return torch.stack(rows, dim=0), unique.long()


def system_repr_dim(z_dim: int, hist_bins: int = 5) -> int:
    """Return dimensionality of aggregate_system_embeddings output."""
    return 5 * z_dim + 11 + 2 * hist_bins
