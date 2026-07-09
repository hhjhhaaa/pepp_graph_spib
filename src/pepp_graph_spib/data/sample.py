"""Canonical LD-TDN local-window sample schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch_geometric.data import Data


CONDITION_NAMES = [
    "density",
    "temperature",
    "PE_fraction",
    "PP_fraction",
    "PS_fraction",
    "PE_chain_length",
    "PP_chain_length",
    "PS_chain_length",
    "PE_repeat_units",
    "PP_repeat_units",
    "PS_repeat_units",
    "mean_chain_length",
    "chain_length_polydispersity",
    "pore_diameter",
    "pore_length",
    "silanol_density",
    "wall_type_id",
    "surface_hydroxylation_fraction",
]

CHAIN_LENGTH_CONDITION_NAMES = [
    "PE_chain_length",
    "PP_chain_length",
    "PS_chain_length",
    "PE_repeat_units",
    "PP_repeat_units",
    "PS_repeat_units",
    "mean_chain_length",
    "chain_length_polydispersity",
]

COMPOSITION_CONDITION_NAMES = ["PE_fraction", "PP_fraction", "PS_fraction"]

WALL_CONDITION_NAMES = [
    "pore_diameter",
    "pore_length",
    "silanol_density",
    "wall_type_id",
    "surface_hydroxylation_fraction",
]

LOCAL_CLASS_LABELS = [
    "mobility_class",
    "contact_class",
    "residence_class",
    "escape_class",
    "relax_class",
]

LOCAL_REGRESSION_LABELS = [
    "future_disp_parallel",
    "future_disp_radial",
    "future_disp_norm",
    "short_msd_parallel",
    "short_msd_radial",
    "contact_survival",
    "wall_contact_survival",
    "free_volume_opening",
]

SYSTEM_TARGET_NAMES = [
    "log_D_self",
    "log_D_parallel",
    "log_D_eff",
    "log_tau_segmental",
    "log_tau_res",
    "P_access",
]


@dataclass
class LocalWindowSample:
    """One center-local short trajectory window for LD-TDN.

    `feature_sequence` contains history-only local dynamic descriptors with
    shape [T, F]. `graph_sequence` is optional and must only contain small local
    ego graphs. `local_labels` are derived from the future window and are never
    used as input features.
    """

    system_id: int
    center_id: int
    center_type: int
    feature_sequence: torch.Tensor | None
    graph_sequence: list[Data] | None
    condition: torch.Tensor
    local_labels: dict[str, int | float]
    system_targets: torch.Tensor
    metadata: dict[str, float | int]


def condition_from_metadata(metadata: dict[str, float | int], names: list[str] | None = None) -> torch.Tensor:
    """Build a condition tensor from metadata using the configured schema."""
    names = names or CONDITION_NAMES
    return torch.tensor([float(metadata.get(name, 0.0)) for name in names], dtype=torch.float32)


def target_tensor_from_metadata(
    metadata: dict[str, float | int],
    names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense target values plus a mask for available system targets."""
    names = names or SYSTEM_TARGET_NAMES
    values = []
    mask = []
    for name in names:
        if name in metadata:
            values.append(float(metadata[name]))
            mask.append(1.0)
        else:
            values.append(0.0)
            mask.append(0.0)
    return torch.tensor(values, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def condition_index(names: list[str], key: str) -> int:
    """Return the index of a named condition variable."""
    try:
        return names.index(key)
    except ValueError as exc:
        raise KeyError(f"Condition variable not configured: {key}") from exc


def numeric_metadata_union(samples: list[LocalWindowSample]) -> dict[str, torch.Tensor]:
    """Collate numeric metadata keys into tensors."""
    keys: set[str] = set()
    for sample in samples:
        for key, value in sample.metadata.items():
            if isinstance(value, (float, int)):
                keys.add(key)
    out: dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        out[key] = torch.tensor([float(sample.metadata.get(key, 0.0)) for sample in samples], dtype=torch.float32)
    out["system_id"] = torch.tensor([int(sample.system_id) for sample in samples], dtype=torch.long)
    out["center_id"] = torch.tensor([int(sample.center_id) for sample in samples], dtype=torch.long)
    out["center_type"] = torch.tensor([int(sample.center_type) for sample in samples], dtype=torch.long)
    return out


def sample_to_legacy_dict(sample: LocalWindowSample) -> dict[str, Any]:
    """Small debugging helper for inspecting saved samples."""
    return {
        "system_id": sample.system_id,
        "center_id": sample.center_id,
        "center_type": sample.center_type,
        "feature_sequence_shape": None if sample.feature_sequence is None else tuple(sample.feature_sequence.shape),
        "has_graph_sequence": sample.graph_sequence is not None,
        "condition_dim": int(sample.condition.numel()),
        "local_labels": dict(sample.local_labels),
        "metadata": dict(sample.metadata),
    }
