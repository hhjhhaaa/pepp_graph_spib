"""Local graph construction from segment positions and metadata."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from pepp_graph_spib.features.pbc import (
    choose_neighbors,
    displacement,
    pairwise_distances,
    radial_basis,
    radial_shell_id,
)


def environment_type(local_pe_fraction: float) -> int:
    """Encode environment as 0 PP-rich, 1 interface, 2 PE-rich."""
    if local_pe_fraction >= 0.8:
        return 2
    if local_pe_fraction <= 0.2:
        return 0
    return 1


def build_local_graph_at_frame(
    segment_positions: np.ndarray,
    previous_positions: np.ndarray,
    segment_types: np.ndarray,
    chain_ids: np.ndarray,
    segment_indices: np.ndarray,
    center_idx: int,
    box: np.ndarray | None,
    r_cut: float,
    max_neighbors: int,
    shell_edges: list[float],
    torsion_proxy: np.ndarray | None = None,
) -> tuple[Data, dict[str, float]]:
    """Build one segment-centered local graph.

    The graph stores x [N, 16], edge_index [2, E], edge_attr [E, 12].
    All distances and displacements use minimum image convention.
    """
    node_ids = choose_neighbors(segment_positions, center_idx, r_cut, max_neighbors, box)
    pos = segment_positions[node_ids]
    prev = previous_positions[node_ids]
    types = segment_types[node_ids].astype(np.int64)
    chains = chain_ids[node_ids].astype(np.int64)
    seg_idx = segment_indices[node_ids].astype(np.float32)
    center_pos = segment_positions[center_idx]
    center_chain = int(chain_ids[center_idx])
    center_seg_idx = float(segment_indices[center_idx])

    disp = displacement(center_pos[None, :], pos, box)
    recent_disp = displacement(prev, pos, box)
    disp_norm = np.linalg.norm(disp, axis=1)
    recent_mob = np.linalg.norm(recent_disp, axis=1)
    local_density = float(len(node_ids) / max((4.0 / 3.0) * np.pi * r_cut**3, 1.0e-6))
    local_pe = float(np.mean(types == 0))
    local_pp = 1.0 - local_pe
    free_volume = float(np.clip(1.0 - local_density / 8.0, 0.0, 1.0))
    torsion = torsion_proxy[node_ids] if torsion_proxy is not None else np.sin(seg_idx * 0.13)

    x = np.zeros((len(node_ids), 16), dtype=np.float32)
    x[:, 0] = types == 0
    x[:, 1] = types == 1
    x[0, 2] = 1.0
    x[:, 3] = chains == center_chain
    x[:, 4] = (seg_idx - center_seg_idx) / 100.0
    x[:, 5:8] = disp
    x[:, 8] = disp_norm
    x[:, 9] = local_density
    x[:, 10] = local_pe
    x[:, 11] = local_pp
    x[:, 12] = free_volume
    x[:, 13] = torsion.astype(np.float32)
    x[:, 14] = recent_mob
    x[:, 15] = types.astype(np.float32)

    rows: list[int] = []
    cols: list[int] = []
    attrs: list[list[float]] = []
    pe_pe = pp_pp = pe_pp = 0
    contact_edges = 0
    id_to_local = {int(gid): i for i, gid in enumerate(node_ids)}
    for i in range(len(node_ids)):
        for j in range(len(node_ids)):
            if i == j:
                continue
            dvec = displacement(pos[i], pos[j], box)
            dist = float(np.linalg.norm(dvec))
            same_chain = int(chains[i] == chains[j])
            bonded = int(same_chain and abs(seg_idx[i] - seg_idx[j]) <= 1.1)
            same_nonbonded = int(same_chain and not bonded and dist <= r_cut)
            inter_contact = int((not same_chain) and dist <= r_cut)
            center_edge = int(i == 0 or j == 0)
            if not (center_edge or bonded or same_nonbonded or inter_contact):
                continue
            type_pair = {int(types[i]), int(types[j])}
            is_pepe = int(type_pair == {0})
            is_pppp = int(type_pair == {1})
            is_pepp = int(type_pair == {0, 1})
            contact_persist = float(np.exp(-dist / max(r_cut, 1.0e-6)) * inter_contact)
            rb = radial_basis(dist)
            attrs.append(
                [
                    dist,
                    1.0 / (dist + 1.0e-6),
                    float(rb[0]),
                    float(rb[1]),
                    radial_shell_id(dist, shell_edges),
                    float(bonded),
                    float(same_nonbonded),
                    float(inter_contact),
                    float(is_pepe),
                    float(is_pppp),
                    float(is_pepp),
                    contact_persist,
                ]
            )
            rows.append(i)
            cols.append(j)
            if inter_contact:
                contact_edges += 1
                pe_pe += is_pepe
                pp_pp += is_pppp
                pe_pp += is_pepp

    if not rows:
        rows, cols = [0], [0]
        attrs = [[0.0] * 12]
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        center_index=torch.tensor([0], dtype=torch.long),
        segment_type=torch.tensor(types, dtype=torch.long),
        local_pe_fraction=torch.tensor([local_pe], dtype=torch.float32),
        local_pp_fraction=torch.tensor([local_pp], dtype=torch.float32),
    )
    denom = max(contact_edges, 1)
    metadata = {
        "local_PE_fraction": local_pe,
        "local_PP_fraction": local_pp,
        "PE_PE_contact_fraction": pe_pe / denom,
        "PP_PP_contact_fraction": pp_pp / denom,
        "PE_PP_contact_fraction": pe_pp / denom,
        "mean_local_density": local_density,
        "mean_free_volume_proxy": free_volume,
        "mean_displacement_norm": float(np.mean(disp_norm)),
        "mean_dihedral_transition_proxy": float(np.mean(np.abs(torsion))),
        "environment_type": environment_type(local_pe),
    }
    return data, metadata


def local_labels_from_history_and_future(
    center_positions: np.ndarray,
    history_end: int,
    future_tau: int,
    box: np.ndarray | None,
    local_density: float,
    free_volume_proxy: float,
    pepp_contact_fraction: float,
    temperature: float,
    density: float,
) -> dict[str, int | float]:
    """Generate LD-TDN future labels using only future coordinates and proxies.

    Future quantities are never written to graph or descriptor history features.
    """
    now = center_positions[history_end]
    future = center_positions[min(history_end + future_tau, len(center_positions) - 1)]
    fdisp = float(np.linalg.norm(displacement(now, future, box)))
    mobility_score = fdisp + 0.8 * free_volume_proxy - 0.15 * local_density - 0.6 * pepp_contact_fraction
    mobility_score += 0.004 * (temperature - 550.0) - 1.8 * (density - 0.85)
    residence_score = pepp_contact_fraction + 0.6 * density - 0.35 * free_volume_proxy - 0.25 * fdisp
    accessibility_score = 0.9 * free_volume_proxy + 0.45 * fdisp - 0.35 * local_density
    accessibility_score += 0.002 * (temperature - 550.0) - 0.4 * pepp_contact_fraction
    contact_score = pepp_contact_fraction + 0.4 * density - 0.2 * fdisp
    escape_score = fdisp + free_volume_proxy - 0.4 * pepp_contact_fraction - 0.2 * density

    def bin3(value: float, lo: float, hi: float) -> int:
        if value < lo:
            return 0
        if value < hi:
            return 1
        return 2

    return {
        "mobility_class": bin3(mobility_score, 0.35, 0.95),
        "contact_class": bin3(contact_score, 0.2, 0.55),
        "residence_class": bin3(residence_score, 0.35, 0.85),
        "escape_class": bin3(escape_score, 0.15, 0.55),
        "relax_class": bin3(accessibility_score, 0.15, 0.55),
        "future_disp_parallel": float(abs(displacement(now, future, box)[2])),
        "future_disp_radial": float(np.linalg.norm(displacement(now, future, box)[:2])),
        "future_disp_norm": fdisp,
        "short_msd_parallel": float(displacement(now, future, box)[2] ** 2),
        "short_msd_radial": float(np.linalg.norm(displacement(now, future, box)[:2]) ** 2),
        "contact_survival": float(1.0 / (1.0 + np.exp(-(2.0 * contact_score - fdisp)))),
        "wall_contact_survival": float(1.0 / (1.0 + np.exp(-(density + pepp_contact_fraction - fdisp)))),
        "free_volume_opening": float(np.clip(free_volume_proxy + 0.25 * fdisp - 0.1 * density, 0.0, 1.0)),
    }
