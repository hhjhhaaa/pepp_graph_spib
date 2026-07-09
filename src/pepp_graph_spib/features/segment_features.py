"""History-only local dynamic descriptor features for LD-TDN."""

from __future__ import annotations

import numpy as np
import torch


SEGMENT_FEATURE_NAMES = [
    "delta_r_parallel",
    "delta_r_radial",
    "delta_r_norm",
    "short_time_msd_parallel",
    "short_time_msd_radial",
    "velocity_autocorrelation_proxy",
    "displacement_persistence",
    "local_PE_fraction",
    "local_PP_fraction",
    "local_PC_fraction",
    "PE_PE_contact_fraction",
    "PP_PP_contact_fraction",
    "PC_PC_contact_fraction",
    "PE_PP_contact_fraction",
    "PE_PC_contact_fraction",
    "PP_PC_contact_fraction",
    "polymer_wall_contact_fraction",
    "contact_persistence",
    "neighbor_persistence",
    "mean_local_density",
    "local_density_fluctuation",
    "packing_heterogeneity",
    "mean_free_volume_proxy",
    "free_volume_lifetime_proxy",
    "free_volume_opening_frequency",
    "axial_free_volume_connectivity_proxy",
    "mean_wall_distance",
    "wall_contact_persistence",
    "near_wall_flag",
    "pore_axis_position",
    "pore_mouth_distance",
    "radial_bin",
    "axial_bin",
    "silanol_contact_count",
    "siloxane_contact_count",
    "dihedral_state_proxy",
    "dihedral_transition_proxy",
    "orientation_relative_to_pore_axis",
    "chain_end_distance_proxy",
    "aromatic_orientation_persistence",
    "carbonate_state_proxy",
]


def feature_dim() -> int:
    """Return the descriptor feature dimension."""
    return len(SEGMENT_FEATURE_NAMES)


def feature_index(name: str) -> int:
    """Return the index of a descriptor feature."""
    try:
        return SEGMENT_FEATURE_NAMES.index(name)
    except ValueError as exc:
        raise KeyError(f"Unknown segment feature: {name}") from exc


def make_feature_vector(values: dict[str, float | int]) -> torch.Tensor:
    """Create a feature vector from named values, filling absent fields with zero."""
    return torch.tensor([float(values.get(name, 0.0)) for name in SEGMENT_FEATURE_NAMES], dtype=torch.float32)


def rolling_descriptor_sequence(
    frame_metadata: list[dict[str, float | int]],
    center_displacements: np.ndarray,
) -> torch.Tensor:
    """Build a descriptor sequence from per-frame history metadata.

    This utility is intentionally lightweight. Real MLFF-MD preprocessing should
    replace the proxy inputs with measured segment, pore, and wall descriptors.
    """
    if len(frame_metadata) == 0:
        raise ValueError("frame_metadata must be non-empty")
    disp = np.asarray(center_displacements, dtype=np.float32)
    if disp.ndim != 2 or disp.shape[1] != 3:
        raise ValueError("center_displacements must have shape [T, 3]")
    rows = []
    prev_norm = 0.0
    for t, meta in enumerate(frame_metadata):
        d = disp[t]
        parallel = float(d[2])
        radial = float(np.linalg.norm(d[:2]))
        norm = float(np.linalg.norm(d))
        persistence = float(np.dot(disp[t], disp[t - 1]) / ((norm * np.linalg.norm(disp[t - 1])) + 1.0e-6)) if t > 0 else 0.0
        density = float(meta.get("mean_local_density", 0.0))
        free_volume = float(meta.get("mean_free_volume_proxy", 0.0))
        wall_distance = float(meta.get("mean_wall_distance", 0.0))
        local_pe = float(meta.get("local_PE_fraction", 0.0))
        local_pp = float(meta.get("local_PP_fraction", 0.0))
        local_pc = float(meta.get("local_PC_fraction", 0.0))
        values = dict(meta)
        values.update(
            {
                "delta_r_parallel": parallel,
                "delta_r_radial": radial,
                "delta_r_norm": norm,
                "short_time_msd_parallel": parallel * parallel,
                "short_time_msd_radial": radial * radial,
                "velocity_autocorrelation_proxy": persistence,
                "displacement_persistence": 0.5 * (persistence + 1.0),
                "local_PE_fraction": local_pe,
                "local_PP_fraction": local_pp,
                "local_PC_fraction": local_pc,
                "mean_local_density": density,
                "local_density_fluctuation": abs(density - float(meta.get("system_density", density))),
                "packing_heterogeneity": abs(local_pe - local_pp),
                "mean_free_volume_proxy": free_volume,
                "free_volume_lifetime_proxy": free_volume * (0.5 + 0.5 * persistence),
                "free_volume_opening_frequency": max(0.0, norm - prev_norm),
                "mean_wall_distance": wall_distance,
                "near_wall_flag": 1.0 if wall_distance < 1.0 else 0.0,
            }
        )
        prev_norm = norm
        rows.append(make_feature_vector(values))
    return torch.stack(rows, dim=0)
