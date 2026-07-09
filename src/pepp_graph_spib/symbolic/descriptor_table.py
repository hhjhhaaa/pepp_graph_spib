"""Build LD-TDN system descriptor tables for sparse distillation."""

from __future__ import annotations

import pandas as pd
import torch

from pepp_graph_spib.data.sample import CONDITION_NAMES, SYSTEM_TARGET_NAMES
from pepp_graph_spib.models.heads.physics_transport_head import PhysicsTransportHead
from pepp_graph_spib.models.pooling.region_pooling import REGION_SCALAR_NAMES, RegionPooling, region_repr_dim


REQUIRED_DESCRIPTOR_COLUMNS = [
    "system_id",
    *CONDITION_NAMES,
    "mean_z1",
    "mean_z2",
    "mean_z3",
    "mean_z4",
    "var_z1",
    "var_z2",
    "var_z3",
    "var_z4",
    "fraction_fast",
    "fraction_slow",
    "fraction_persistent_contact",
    "fraction_wall_resident",
    "fraction_escape_ready",
    "mean_free_volume_proxy",
    "mean_wall_distance",
    "mean_local_density",
    "mean_PE_fraction",
    "mean_PP_fraction",
    "mean_PC_fraction",
    "P_entry",
    "C_axis",
    "tau_wall",
    "tau_move",
    "P_access",
    "wall_residence_fraction",
    "active_site_residence_fraction",
    "transport_score",
    "D_eff",
    "reaction_opportunity_index",
    *[f"target_{name}" for name in SYSTEM_TARGET_NAMES],
]


def build_descriptor_table(
    collected: dict,
    condition_names: list[str] | None = None,
    transport_head: PhysicsTransportHead | None = None,
) -> pd.DataFrame:
    """Return one descriptor row per system."""
    condition_names = condition_names or CONDITION_NAMES
    z = collected.get("mu", collected["z"])
    pooling = RegionPooling()
    system_repr, unique_ids, system_condition = pooling(
        z,
        collected["local_outputs"],
        collected["metadata"],
        collected["condition"],
    )
    if transport_head is None:
        transport_head = PhysicsTransportHead(region_repr_dim(z.size(1)), system_condition.size(1))
    transport_head.eval()
    with torch.no_grad():
        physics = transport_head(system_repr, system_condition)
    rows = []
    for row_idx, sid in enumerate(unique_ids.detach().cpu().tolist()):
        mask = collected["metadata"]["system_id"] == sid
        zi = z[mask]
        row = {"system_id": int(sid)}
        for i, name in enumerate(condition_names):
            row[name] = float(system_condition[row_idx, i].detach().cpu())
        for i in range(zi.size(1)):
            row[f"z{i + 1}"] = float(zi[:, i].mean())
            row[f"mean_z{i + 1}"] = float(zi[:, i].mean())
            row[f"var_z{i + 1}"] = float(zi[:, i].var(unbiased=False))
        offset = 2 * zi.size(1)
        for i, name in enumerate(REGION_SCALAR_NAMES):
            export_name = name
            if name == "mean_local_PE_fraction":
                export_name = "mean_PE_fraction"
            elif name == "mean_local_PP_fraction":
                export_name = "mean_PP_fraction"
            elif name == "mean_local_PC_fraction":
                export_name = "mean_PC_fraction"
            row[export_name] = float(system_repr[row_idx, offset + i].detach().cpu())
        for i in range(4):
            row[f"radial_bin_fraction_{i}"] = float(system_repr[row_idx, offset + len(REGION_SCALAR_NAMES) + i].detach().cpu())
            row[f"axial_bin_fraction_{i}"] = float(system_repr[row_idx, offset + len(REGION_SCALAR_NAMES) + 4 + i].detach().cpu())
        for key, value in physics.items():
            row[key] = float(value[row_idx].detach().cpu())
        target = collected["system_targets"][mask][0]
        target_mask = collected["target_mask"][mask][0]
        for i, name in enumerate(SYSTEM_TARGET_NAMES):
            if float(target_mask[i]) > 0.5:
                row[f"target_{name}"] = float(target[i])
        rows.append(row)
    return pd.DataFrame(rows)
