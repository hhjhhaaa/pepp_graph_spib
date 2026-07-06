"""Dataset and collate helpers for local dynamic graph windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


@dataclass
class GraphWindowSample:
    """One segment-centered local graph trajectory window.

    graph_sequence is a list of PyG Data objects, length history_len.
    dynamic_descriptors stores history-only trajectory descriptors.
    condition has shape [condition_dim].
    future_labels stores integer class labels for mobility, residence, accessibility.
    metadata stores per-window history-only scalar descriptors for pooling/export.
    """

    system_id: int
    center_segment_id: int
    center_segment_type: int
    graph_sequence: list[Data]
    dynamic_descriptors: torch.Tensor
    condition: torch.Tensor
    future_labels: dict[str, int]
    metadata: dict[str, float | int]
    property_targets: torch.Tensor


class GraphWindowDataset(Dataset):
    """Load graph windows saved by preprocessing scripts."""

    def __init__(
        self,
        path: str,
        system_ids: set[int] | None = None,
        limit_systems: int | None = None,
        limit_samples: int | None = None,
        transform: str | None = None,
    ) -> None:
        payload = torch.load(path, map_location="cpu")
        samples = payload["samples"] if isinstance(payload, dict) else payload
        if system_ids is not None:
            samples = [s for s in samples if int(s.system_id) in system_ids]
        if limit_systems is not None:
            picked = sorted({int(s.system_id) for s in samples})[:limit_systems]
            samples = [s for s in samples if int(s.system_id) in set(picked)]
        if limit_samples is not None:
            samples = samples[:limit_samples]
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphWindowSample:
        sample = self.samples[idx]
        if self.transform is None:
            return sample
        return transform_sample(sample, self.transform)


def transform_sample(sample: GraphWindowSample, transform: str) -> GraphWindowSample:
    """Return a transformed sample for ablation controls."""
    import copy

    out = copy.deepcopy(sample)
    if transform == "static_graph_only":
        last = out.graph_sequence[-1]
        out.graph_sequence = [copy.deepcopy(last) for _ in out.graph_sequence]
    elif transform == "shuffled_history":
        idx = torch.randperm(len(out.graph_sequence)).tolist()
        out.graph_sequence = [out.graph_sequence[i] for i in idx]
    elif transform == "no_composition_edges":
        for graph in out.graph_sequence:
            if graph.edge_attr is not None and graph.edge_attr.size(1) >= 15:
                graph.edge_attr[:, 9:15] = 0.0
            if graph.x is not None and graph.x.size(1) >= 4:
                graph.x[:, 0:4] = 0.0
    elif transform == "no_dynamic_descriptors":
        out.dynamic_descriptors = torch.zeros_like(out.dynamic_descriptors)
    else:
        raise ValueError(f"Unknown transform: {transform}")
    return out


def collate_graph_windows(samples: list[GraphWindowSample]) -> dict[str, Any]:
    history_len = len(samples[0].graph_sequence)
    batch_graphs_by_time = [
        Batch.from_data_list([sample.graph_sequence[t] for sample in samples]) for t in range(history_len)
    ]
    condition = torch.stack([sample.condition.float() for sample in samples], dim=0)
    dynamic_descriptors = torch.stack([sample.dynamic_descriptors.float() for sample in samples], dim=0)
    labels = {
        "y_mobility": torch.tensor([sample.future_labels["mobility"] for sample in samples], dtype=torch.long),
        "y_residence": torch.tensor([sample.future_labels["residence"] for sample in samples], dtype=torch.long),
        "y_accessibility": torch.tensor([sample.future_labels["accessibility"] for sample in samples], dtype=torch.long),
        "y_property": torch.stack([sample.property_targets.float() for sample in samples], dim=0),
    }
    metadata_keys = [
        "local_PE_fraction",
        "local_PP_fraction",
        "local_PC_fraction",
        "local_wall_fraction",
        "PE_PP_contact_fraction",
        "PE_PC_contact_fraction",
        "PP_PC_contact_fraction",
        "polymer_wall_contact_fraction",
        "mean_local_density",
        "mean_free_volume_proxy",
        "mean_displacement_norm",
        "mean_dihedral_transition_proxy",
        "mean_wall_distance",
        "wall_contact_persistence",
        "environment_type",
    ]
    metadata = {
        key: torch.tensor([float(sample.metadata.get(key, 0.0)) for sample in samples], dtype=torch.float32)
        for key in metadata_keys
    }
    metadata["system_id"] = torch.tensor([int(sample.system_id) for sample in samples], dtype=torch.long)
    metadata["center_segment_type"] = torch.tensor([int(sample.center_segment_type) for sample in samples], dtype=torch.long)
    metadata["center_segment_id"] = torch.tensor([int(sample.center_segment_id) for sample in samples], dtype=torch.long)
    return {
        "batch_graphs_by_time": batch_graphs_by_time,
        "dynamic_descriptors": dynamic_descriptors,
        "condition": condition,
        "labels": labels,
        "metadata": metadata,
    }
