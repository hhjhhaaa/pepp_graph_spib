
"""Build system-level physical descriptor tables from Graph-SPIB outputs."""

from __future__ import annotations

import pandas as pd
import torch

from pepp_graph_spib.models.system_pooling import aggregate_system_embeddings


REQUIRED_DESCRIPTOR_COLUMNS = [
    "system_id",
    "z1",
    "z2",
    "z3",
    "z4",
    "mean_z1",
    "mean_z2",
    "mean_z3",
    "mean_z4",
    "var_z1",
    "var_z2",
    "var_z3",
    "var_z4",
    "density",
    "temperature",
    "PE_fraction",
    "PP_fraction",
    "PE_chain_length",
    "PP_chain_length",
    "mean_dynamic_local_density",
    "mean_dynamic_free_volume_proxy",
    "mean_dynamic_displacement_norm",
    "mean_dynamic_dihedral_transition_proxy",
    "mean_dynamic_PEPP_contact_fraction",
    "mean_dynamic_local_PE_fraction",
    "mean_dynamic_local_PP_fraction",
    "mean_dynamic_interface_flag",
    "mean_local_density",
    "mean_free_volume_proxy",
    "mean_PEPP_contact_fraction",
    "mean_displacement_norm",
    "mean_dihedral_transition_proxy",
    "fraction_fast",
    "fraction_slow",
    "fraction_persistent_residence",
    "fraction_high_accessibility",
    "PE_PE_contact_fraction",
    "PP_PP_contact_fraction",
    "PE_PP_contact_fraction",
    "log_D",
    "log_tau_relax",
    "log_D_eff",
    "tau_res",
    "P_access",
]


def build_descriptor_table(collected: dict, pe_hist_bins: list[float]) -> pd.DataFrame:
    """Return one descriptor row per system.

    The table contains z summaries, historical physical proxy descriptors,
    local PE/PP composition histograms, contact fractions, future-state probabilities,
    global conditions, and system-level transport targets.
    """
    _, unique_ids = aggregate_system_embeddings(
        collected["z"],
        collected["mobility_probs"],
        collected["residence_probs"],
        collected["accessibility_probs"],
        collected["metadata"]["system_id"],
        collected["metadata"]["center_segment_type"],
        collected["metadata"],
        pe_hist_bins,
    )
    rows = []
    hist_bins = len(pe_hist_bins) - 1
    for sid in unique_ids.tolist():
        mask = collected["metadata"]["system_id"] == sid
        z = collected["z"][mask]
        dynamic = collected["dynamic_descriptors"][mask]
        md = {k: v[mask] for k, v in collected["metadata"].items()}
        cond = collected["condition"][mask][0]
        y = collected["y_property"][mask][0]
        pe_hist = torch.histc(md["local_PE_fraction"].float(), bins=hist_bins, min=0.0, max=1.0)
        pe_hist = pe_hist / pe_hist.sum().clamp_min(1.0)
        pp_hist = torch.histc(md["local_PP_fraction"].float(), bins=hist_bins, min=0.0, max=1.0)
        pp_hist = pp_hist / pp_hist.sum().clamp_min(1.0)
        row = {
            "system_id": sid,
            "density": float(cond[0]),
            "temperature": float(cond[1] * 1000.0),
            "PE_fraction": float(cond[2]),
            "PP_fraction": float(cond[3]),
            "PE_chain_length": float(cond[4] * 200.0),
            "PP_chain_length": float(cond[5] * 200.0),
            "mean_dynamic_local_density": float(dynamic[:, 0].mean()),
            "mean_dynamic_free_volume_proxy": float(dynamic[:, 1].mean()),
            "mean_dynamic_displacement_norm": float(dynamic[:, 2].mean()),
            "mean_dynamic_dihedral_transition_proxy": float(dynamic[:, 3].mean()),
            "mean_dynamic_PEPP_contact_fraction": float(dynamic[:, 4].mean()),
            "mean_dynamic_local_PE_fraction": float(dynamic[:, 5].mean()),
            "mean_dynamic_local_PP_fraction": float(dynamic[:, 6].mean()),
            "mean_dynamic_interface_flag": float(dynamic[:, 7].mean()),
            "mean_local_density": float(md["mean_local_density"].mean()),
            "mean_free_volume_proxy": float(md["mean_free_volume_proxy"].mean()),
            "mean_PEPP_contact_fraction": float(md["PE_PP_contact_fraction"].mean()),
            "mean_displacement_norm": float(md["mean_displacement_norm"].mean()),
            "mean_dihedral_transition_proxy": float(md["mean_dihedral_transition_proxy"].mean()),
            "fraction_fast": float(collected["mobility_probs"][mask][:, 2].mean()),
            "fraction_slow": float(collected["mobility_probs"][mask][:, 0].mean()),
            "fraction_persistent_residence": float(collected["residence_probs"][mask][:, 2].mean()),
            "fraction_high_accessibility": float(collected["accessibility_probs"][mask][:, 2].mean()),
            "PE_PE_contact_fraction": float(md["PE_PE_contact_fraction"].mean()),
            "PP_PP_contact_fraction": float(md["PP_PP_contact_fraction"].mean()),
            "PE_PP_contact_fraction": float(md["PE_PP_contact_fraction"].mean()),
            "log_D": float(y[0]),
            "log_tau_relax": float(y[1]),
            "log_D_eff": float(y[2]),
            "tau_res": float(y[3]),
            "P_access": float(y[4]),
        }
        for i in range(z.size(1)):
            row[f"z{i + 1}"] = float(z[:, i].mean())
            row[f"mean_z{i + 1}"] = float(z[:, i].mean())
            row[f"var_z{i + 1}"] = float(z[:, i].var(unbiased=False))
        for class_idx, name in enumerate(["slow", "medium", "fast"]):
            row[f"mean_mobility_prob_{name}"] = float(collected["mobility_probs"][mask][:, class_idx].mean())
        for class_idx, name in enumerate(["short", "intermediate", "persistent"]):
            row[f"mean_residence_prob_{name}"] = float(collected["residence_probs"][mask][:, class_idx].mean())
        for class_idx, name in enumerate(["low", "medium", "high"]):
            row[f"mean_accessibility_prob_{name}"] = float(
                collected["accessibility_probs"][mask][:, class_idx].mean()
            )
        for i, value in enumerate(pe_hist.tolist()):
            row[f"local_PE_fraction_hist_bin_{i}"] = float(value)
        for i, value in enumerate(pp_hist.tolist()):
            row[f"local_PP_fraction_hist_bin_{i}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)
