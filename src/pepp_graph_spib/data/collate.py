"""Collate LD-TDN local-window batches."""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Batch

from pepp_graph_spib.data.sample import (
    LOCAL_CLASS_LABELS,
    LOCAL_REGRESSION_LABELS,
    SYSTEM_TARGET_NAMES,
    LocalWindowSample,
    numeric_metadata_union,
)


def _stack_feature_sequence(samples: list[LocalWindowSample]) -> torch.Tensor:
    if any(sample.feature_sequence is None for sample in samples):
        raise ValueError("All samples need feature_sequence for descriptor mode")
    return torch.stack([sample.feature_sequence.float() for sample in samples], dim=0)  # type: ignore[union-attr]


def _batch_graph_sequence(samples: list[LocalWindowSample]) -> list[Batch] | None:
    if any(sample.graph_sequence is None for sample in samples):
        return None
    history_len = len(samples[0].graph_sequence or [])
    return [
        Batch.from_data_list([sample.graph_sequence[t] for sample in samples if sample.graph_sequence is not None])
        for t in range(history_len)
    ]


def collate_local_windows(samples: list[LocalWindowSample]) -> dict[str, Any]:
    """Batch local dynamic windows without leaking future labels into features."""
    condition = torch.stack([sample.condition.float() for sample in samples], dim=0)
    local_labels: dict[str, torch.Tensor] = {}
    for key in LOCAL_CLASS_LABELS:
        local_labels[key] = torch.tensor([int(sample.local_labels[key]) for sample in samples], dtype=torch.long)
    for key in LOCAL_REGRESSION_LABELS:
        local_labels[key] = torch.tensor([float(sample.local_labels[key]) for sample in samples], dtype=torch.float32)
    for sample in samples:
        if int(sample.system_targets.numel()) != len(SYSTEM_TARGET_NAMES):
            raise ValueError(
                f"system_targets length {sample.system_targets.numel()} does not match "
                f"{len(SYSTEM_TARGET_NAMES)} target names"
            )
    system_targets = torch.stack([sample.system_targets.float() for sample in samples], dim=0)
    target_mask_rows = []
    for sample in samples:
        target_mask_rows.append(
            torch.tensor(
                [float(sample.metadata.get(f"mask_{target}", 0.0)) for target in SYSTEM_TARGET_NAMES],
                dtype=torch.float32,
            )
        )
    target_mask = torch.stack(target_mask_rows, dim=0)
    graph_sequence = _batch_graph_sequence(samples)
    return {
        "feature_sequence": _stack_feature_sequence(samples),
        "graph_sequence": graph_sequence,
        "batch_graphs_by_time": graph_sequence,
        "condition": condition,
        "local_labels": local_labels,
        "system_targets": system_targets,
        "target_mask": target_mask,
        "metadata": numeric_metadata_union(samples),
    }


collate_graph_windows = collate_local_windows

__all__ = ["collate_local_windows", "collate_graph_windows"]
