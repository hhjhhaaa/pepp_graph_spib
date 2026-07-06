"""Dummy local dynamic graph data generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pepp_graph_spib.data.graph_window import GraphWindowSample, dynamic_descriptors_from_metadata
from pepp_graph_spib.features.graphs import build_local_graph_at_frame, future_labels_from_history_and_future


def _random_walk(rng: np.random.Generator, n_frames: int, n_segments: int, box_len: float, mobility: float) -> np.ndarray:
    pos = rng.uniform(0, box_len, size=(n_segments, 3)).astype(np.float32)
    frames = []
    for _ in range(n_frames):
        step = rng.normal(0, mobility, size=(n_segments, 3)).astype(np.float32)
        pos = (pos + step) % box_len
        frames.append(pos.copy())
    return np.stack(frames, axis=0)


def generate_dummy_dataset(cfg: dict, output_path: str | Path, tiny: bool = False) -> tuple[Path, Path]:
    """Generate trainable dummy graph windows without saving hidden label scores."""
    seed = int(cfg["project"]["seed"])
    rng = np.random.default_rng(seed)
    data_cfg = cfg["data"]
    dummy_cfg = cfg["dummy"]
    num_systems = int(dummy_cfg["tiny_num_systems"] if tiny else dummy_cfg["num_systems"])
    samples_per_system = int(dummy_cfg["tiny_samples_per_system"] if tiny else dummy_cfg["samples_per_system"])
    history_len = int(data_cfg["history_len"])
    future_tau = int(data_cfg["future_tau"])
    max_neighbors = int(data_cfg["max_neighbors"])
    r_cut = float(data_cfg["r_cut_nm"])
    box_len = float(dummy_cfg["box_nm"])
    shell_edges = data_cfg["radial_shell_edges_nm"]
    total_frames = history_len + future_tau + samples_per_system + 4

    samples: list[GraphWindowSample] = []
    per_system_proxy: dict[int, list[dict[str, float]]] = defaultdict(list)
    conditions: dict[int, torch.Tensor] = {}
    raw_targets: dict[int, torch.Tensor] = {}

    for system_id in range(num_systems):
        density = float(rng.uniform(0.75, 0.95))
        temperature = float(rng.uniform(450.0, 650.0))
        pe_frac = float(rng.uniform(0.2, 0.8))
        pp_frac = 1.0 - pe_frac
        pe_len = float(rng.integers(50, 201))
        pp_len = float(rng.integers(50, 201))
        condition = torch.tensor(
            [density, temperature / 1000.0, pe_frac, pp_frac, pe_len / 200.0, pp_len / 200.0],
            dtype=torch.float32,
        )
        conditions[system_id] = condition
        n_segments = int(rng.integers(int(dummy_cfg["min_nodes"]), int(dummy_cfg["max_nodes"]) + 1))
        segment_types = (rng.random(n_segments) > pe_frac).astype(np.int64)
        chain_ids = np.repeat(np.arange(max(1, n_segments // 12 + 1)), 12)[:n_segments]
        if len(chain_ids) < n_segments:
            chain_ids = np.pad(chain_ids, (0, n_segments - len(chain_ids)), mode="edge")
        segment_indices = np.arange(n_segments, dtype=np.int64)
        base_mobility = 0.045 + 0.00025 * (temperature - 450.0) + 0.04 * (1.0 - density)
        positions = _random_walk(rng, total_frames, n_segments, box_len, base_mobility)
        box = np.asarray([box_len, box_len, box_len], dtype=np.float32)
        torsion = rng.normal(0, 1, size=(total_frames, n_segments)).astype(np.float32)

        system_records = []
        for sample_idx in range(samples_per_system):
            t_end = history_len + sample_idx
            center_idx = int(rng.integers(0, n_segments))
            graph_sequence = []
            meta_accum: list[dict[str, float]] = []
            for offset in range(history_len):
                frame = t_end - history_len + 1 + offset
                graph, metadata = build_local_graph_at_frame(
                    positions[frame],
                    positions[max(frame - 1, 0)],
                    segment_types,
                    chain_ids,
                    segment_indices,
                    center_idx,
                    box,
                    r_cut,
                    max_neighbors,
                    shell_edges,
                    torsion_proxy=torsion[frame],
                )
                graph_sequence.append(graph)
                meta_accum.append(metadata)
            metadata = {
                key: float(np.mean([m[key] for m in meta_accum]))
                for key in meta_accum[0]
                if key != "environment_type"
            }
            metadata["environment_type"] = int(meta_accum[-1]["environment_type"])
            labels = future_labels_from_history_and_future(
                positions[:, center_idx, :],
                t_end,
                future_tau,
                box,
                metadata["mean_local_density"],
                metadata["mean_free_volume_proxy"],
                metadata["PE_PP_contact_fraction"],
                temperature,
                density,
            )
            system_records.append({**metadata, **{f"label_{k}": v for k, v in labels.items()}})
            sample = GraphWindowSample(
                system_id=system_id,
                center_segment_id=center_idx,
                center_segment_type=int(segment_types[center_idx]),
                graph_sequence=graph_sequence,
                dynamic_descriptors=dynamic_descriptors_from_metadata(metadata),
                condition=condition,
                future_labels=labels,
                metadata=metadata,
                property_targets=torch.zeros(5, dtype=torch.float32),
            )
            samples.append(sample)
        per_system_proxy[system_id] = system_records

    target_rows = []
    for system_id, records in per_system_proxy.items():
        cond = conditions[system_id]
        density = float(cond[0])
        temp = float(cond[1] * 1000.0)
        pe_frac = float(cond[2])
        mean_free = float(np.mean([r["mean_free_volume_proxy"] for r in records]))
        mean_disp = float(np.mean([r["mean_displacement_norm"] for r in records]))
        pepp = float(np.mean([r["PE_PP_contact_fraction"] for r in records]))
        frac_fast = float(np.mean([r["label_mobility"] == 2 for r in records]))
        frac_slow = float(np.mean([r["label_mobility"] == 0 for r in records]))
        log_d = -8.0 + 2.0 * mean_disp + 1.2 * mean_free - 2.0 * density + 0.003 * (temp - 550.0) + 0.4 * frac_fast
        log_tau = 3.0 - 1.5 * mean_disp + 2.2 * density + 0.25 * pepp + 0.2 * frac_slow
        log_d_eff = log_d - 0.7 * pepp - 0.15 * abs(pe_frac - 0.5)
        tau_res = 1.0 + 2.5 * pepp + 1.2 * density
        p_access = 1.0 / (1.0 + np.exp(-(2.0 * mean_free - 1.5 * density + 0.5 * frac_fast)))
        target = torch.tensor([log_d, log_tau, log_d_eff, tau_res, p_access], dtype=torch.float32)
        raw_targets[system_id] = target
        target_rows.append(
            {
                "system_id": system_id,
                "log_D": log_d,
                "log_tau_relax": log_tau,
                "log_D_eff": log_d_eff,
                "tau_res": tau_res,
                "P_access": p_access,
                "density": float(cond[0]),
                "temperature": float(cond[1] * 1000.0),
                "PE_fraction": float(cond[2]),
                "PP_fraction": float(cond[3]),
                "PE_chain_length": float(cond[4] * 200.0),
                "PP_chain_length": float(cond[5] * 200.0),
            }
        )

    for sample in samples:
        sample.property_targets = raw_targets[int(sample.system_id)]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = output_path.parent / "dummy_system_targets.csv"
    torch.save({"samples": samples, "config": {"tiny": tiny}}, output_path)
    pd.DataFrame(target_rows).to_csv(target_path, index=False)
    return output_path, target_path
