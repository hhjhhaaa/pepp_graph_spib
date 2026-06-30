"""Real trajectory preprocessing for PE/PP local graph windows.

The module intentionally separates raw trajectory reading from PE/PP metadata
adaptation because MDAnalysis cannot infer polymer identity or segment mapping
without project-specific metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import torch

from pepp_graph_spib.data.graph_window import GraphWindowSample
from pepp_graph_spib.features.graphs import build_local_graph_at_frame, future_labels_from_history_and_future


@dataclass
class SegmentMetadata:
    """Segment identity arrays loaded from metadata.yaml."""

    segment_type: np.ndarray
    chain_id: np.ndarray
    segment_index: np.ndarray
    condition: np.ndarray
    chain_lengths: dict[str, float]
    property_targets: np.ndarray


def load_trajectory(topology_path: str, trajectory_path: str):
    """Load topology and trajectory with MDAnalysis and return a Universe."""
    import MDAnalysis as mda

    return mda.Universe(topology_path, trajectory_path)


def load_pepp_metadata(metadata_path: str | Path) -> SegmentMetadata:
    """Load PE/PP segment and condition metadata from YAML."""
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    return SegmentMetadata(
        segment_type=np.asarray(data["segment_type"], dtype=np.int64),
        chain_id=np.asarray(data["chain_id"], dtype=np.int64),
        segment_index=np.asarray(data["segment_index"], dtype=np.int64),
        condition=np.asarray(data["condition"], dtype=np.float32),
        chain_lengths=dict(data.get("chain_lengths", {})),
        property_targets=np.asarray(data.get("property_targets", [0, 0, 0, 0, 0]), dtype=np.float32),
    )


def define_segments(universe, segment_scheme: str, metadata: SegmentMetadata | None = None):
    """Convert atoms/beads into polymer segment centers.

    For first-version CG/united-atom input, segment_scheme='bead' maps each atom
    or bead to one segment. Atomistic grouping can be added via metadata-defined
    groups without changing the Graph-SPIB model.
    """
    if segment_scheme != "bead":
        raise ValueError("First version supports segment_scheme='bead' or metadata-precomputed centers.")
    centers = []
    boxes = []
    for ts in universe.trajectory:
        centers.append(universe.atoms.positions.astype(np.float32) / 10.0)
        boxes.append(ts.dimensions[:3].astype(np.float32) / 10.0)
    if metadata is None:
        n = centers[0].shape[0]
        metadata = SegmentMetadata(
            segment_type=np.zeros(n, dtype=np.int64),
            chain_id=np.arange(n, dtype=np.int64),
            segment_index=np.arange(n, dtype=np.int64),
            condition=np.zeros(6, dtype=np.float32),
            chain_lengths={},
            property_targets=np.zeros(5, dtype=np.float32),
        )
    return np.stack(centers), np.stack(boxes), metadata


def build_graph_window(
    trajectory_cache: dict[str, Any],
    center_idx: int,
    time_index: int,
    history_len: int,
    future_tau: int,
    r_cut: float,
    max_neighbors: int,
):
    """Build graph sequence from [t-L, t].

    Future frames are not used for graph features. They are reserved for labels
    in the caller.
    """
    positions = trajectory_cache["positions"]
    boxes = trajectory_cache.get("boxes")
    metadata: SegmentMetadata = trajectory_cache["metadata"]
    shell_edges = trajectory_cache.get("shell_edges", [0.6, 1.2, 2.0, 3.0])
    graphs = []
    for frame in range(time_index - history_len + 1, time_index + 1):
        prev = max(frame - 1, 0)
        graph, _ = build_local_graph_at_frame(
            positions[frame],
            positions[prev],
            metadata.segment_type,
            metadata.chain_id,
            metadata.segment_index,
            center_idx,
            None if boxes is None else boxes[frame],
            r_cut,
            max_neighbors,
            shell_edges,
        )
        graphs.append(graph)
    return graphs


def assign_future_labels(center_positions, local_descriptors, time_index, future_tau):
    """Assign generic future labels from future motion and historical local descriptors.

    This first-version adapter avoids label leakage by returning labels only. It
    does not write future displacement or future contact information into graph
    features. Production runs can replace thresholds through metadata-derived
    labels without changing the downstream model.
    """
    box = local_descriptors.get("box")
    density = float(local_descriptors.get("density", 0.85))
    temperature = float(local_descriptors.get("temperature", 550.0))
    return future_labels_from_history_and_future(
        center_positions,
        time_index,
        future_tau,
        box,
        float(local_descriptors.get("mean_local_density", 0.0)),
        float(local_descriptors.get("mean_free_volume_proxy", 0.0)),
        float(local_descriptors.get("PE_PP_contact_fraction", 0.0)),
        temperature,
        density,
    )


def preprocess_to_graph_windows(
    topology_path: str,
    trajectory_path: str,
    metadata_path: str,
    output_path: str,
    history_len: int,
    future_tau: int,
    r_cut: float,
    max_neighbors: int,
    stride: int = 1,
    segment_scheme: str = "bead",
    max_centers: int | None = None,
    shell_edges: list[float] | None = None,
) -> str:
    """Preprocess a real trajectory into saved GraphWindowSample objects.

    Coordinates are read through MDAnalysis, PE/PP identity and conditions are
    read from metadata.yaml, and all graph features are built only from history
    frames. The saved payload is compatible with GraphWindowDataset.
    """
    universe = load_trajectory(topology_path, trajectory_path)
    metadata = load_pepp_metadata(metadata_path)
    positions, boxes, metadata = define_segments(universe, segment_scheme, metadata)
    shell_edges = shell_edges or [0.6, 1.2, 2.0, 3.0]
    samples: list[GraphWindowSample] = []
    n_frames, n_segments = positions.shape[:2]
    centers = list(range(n_segments))[:max_centers]
    condition = torch.tensor(metadata.condition, dtype=torch.float32)
    property_targets = torch.tensor(metadata.property_targets, dtype=torch.float32)
    for time_index in range(history_len - 1, n_frames - future_tau, stride):
        for center_idx in centers:
            graph_sequence = []
            meta_accum = []
            for frame in range(time_index - history_len + 1, time_index + 1):
                graph, frame_meta = build_local_graph_at_frame(
                    positions[frame],
                    positions[max(frame - 1, 0)],
                    metadata.segment_type,
                    metadata.chain_id,
                    metadata.segment_index,
                    center_idx,
                    boxes[frame],
                    r_cut,
                    max_neighbors,
                    shell_edges,
                )
                graph_sequence.append(graph)
                meta_accum.append(frame_meta)
            window_meta = {
                key: float(np.mean([m[key] for m in meta_accum]))
                for key in meta_accum[0]
                if key != "environment_type"
            }
            window_meta["environment_type"] = int(meta_accum[-1]["environment_type"])
            label_context = {
                **window_meta,
                "box": boxes[time_index],
                "density": float(metadata.condition[0]) if len(metadata.condition) > 0 else 0.85,
                "temperature": float(metadata.condition[1]) * 1000.0 if len(metadata.condition) > 1 and metadata.condition[1] < 10 else float(metadata.condition[1] if len(metadata.condition) > 1 else 550.0),
            }
            labels = assign_future_labels(positions[:, center_idx, :], label_context, time_index, future_tau)
            samples.append(
                GraphWindowSample(
                    system_id=0,
                    center_segment_id=int(center_idx),
                    center_segment_type=int(metadata.segment_type[center_idx]),
                    graph_sequence=graph_sequence,
                    condition=condition,
                    future_labels=labels,
                    metadata=window_meta,
                    property_targets=property_targets,
                )
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples, "source": {"topology": topology_path, "trajectory": trajectory_path}}, output)
    return str(output)
