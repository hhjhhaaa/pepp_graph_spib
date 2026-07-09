"""Dummy LD-TDN local-window data generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pepp_graph_spib.data.sample import CONDITION_NAMES, SYSTEM_TARGET_NAMES, LocalWindowSample, condition_from_metadata
from pepp_graph_spib.features.graphs import build_local_graph_at_frame
from pepp_graph_spib.features.pbc import displacement
from pepp_graph_spib.features.segment_features import rolling_descriptor_sequence


def _random_walk(rng: np.random.Generator, n_frames: int, n_segments: int, box_len: float, mobility: float) -> np.ndarray:
    pos = rng.uniform(0, box_len, size=(n_segments, 3)).astype(np.float32)
    frames = []
    for _ in range(n_frames):
        step = rng.normal(0, mobility, size=(n_segments, 3)).astype(np.float32)
        pos = (pos + step) % box_len
        frames.append(pos.copy())
    return np.stack(frames, axis=0)


def _bin3(value: float, lo: float, hi: float) -> int:
    if value < lo:
        return 0
    if value < hi:
        return 1
    return 2


def _local_labels(
    center_positions: np.ndarray,
    t_end: int,
    future_tau: int,
    box: np.ndarray,
    metadata: dict[str, float | int],
    condition_meta: dict[str, float],
) -> dict[str, int | float]:
    now = center_positions[t_end]
    future = center_positions[min(t_end + future_tau, len(center_positions) - 1)]
    dvec = displacement(now, future, box)
    parallel = float(abs(dvec[2]))
    radial = float(np.linalg.norm(dvec[:2]))
    norm = float(np.linalg.norm(dvec))
    density = float(condition_meta["density"])
    temp = float(condition_meta["temperature"])
    free_volume = float(metadata.get("mean_free_volume_proxy", 0.0))
    local_density = float(metadata.get("mean_local_density", 0.0))
    wall_contact = float(metadata.get("polymer_wall_contact_fraction", 0.0))
    pepp = float(metadata.get("PE_PP_contact_fraction", 0.0))
    mobility_score = norm + 0.8 * free_volume - 0.12 * local_density + 0.002 * (temp - 550.0)
    contact_score = pepp + 0.4 * wall_contact + 0.25 * density
    residence_score = contact_score + 0.2 * density - 0.35 * norm
    escape_score = norm + free_volume - 0.6 * wall_contact - 0.3 * density
    relax_score = 0.8 * norm + float(metadata.get("dihedral_transition_proxy", 0.0)) - 0.25 * pepp
    contact_survival = float(1.0 / (1.0 + np.exp(-(2.2 * contact_score - 1.0 * norm))))
    wall_contact_survival = float(1.0 / (1.0 + np.exp(-(2.0 * wall_contact + density - norm))))
    opening = float(np.clip(free_volume + 0.35 * norm - 0.15 * density, 0.0, 1.0))
    return {
        "mobility_class": _bin3(mobility_score, 0.35, 0.95),
        "contact_class": _bin3(contact_score, 0.15, 0.55),
        "residence_class": _bin3(residence_score, 0.25, 0.75),
        "escape_class": _bin3(escape_score, 0.15, 0.6),
        "relax_class": _bin3(relax_score, 0.25, 0.8),
        "future_disp_parallel": parallel,
        "future_disp_radial": radial,
        "future_disp_norm": norm,
        "short_msd_parallel": parallel * parallel,
        "short_msd_radial": radial * radial,
        "contact_survival": contact_survival,
        "wall_contact_survival": wall_contact_survival,
        "free_volume_opening": opening,
    }


def _condition_metadata(rng: np.random.Generator) -> dict[str, float]:
    pe_frac = float(rng.uniform(0.2, 0.75))
    pp_frac = 1.0 - pe_frac
    pc_frac = 0.0
    pe_len = float(rng.integers(60, 221))
    pp_len = float(rng.integers(45, 201))
    pc_len = 0.0
    chain_lengths = np.asarray([pe_len, pp_len], dtype=np.float32)
    return {
        "density": float(rng.uniform(0.74, 0.96)),
        "temperature": float(rng.uniform(450.0, 650.0)),
        "PE_fraction": pe_frac,
        "PP_fraction": pp_frac,
        "PC_fraction": pc_frac,
        "PE_chain_length": pe_len,
        "PP_chain_length": pp_len,
        "PC_chain_length": pc_len,
        "PE_repeat_units": pe_len,
        "PP_repeat_units": pp_len,
        "PC_repeat_units": pc_len,
        "mean_chain_length": float(np.average(chain_lengths, weights=[pe_frac, pp_frac])),
        "chain_length_polydispersity": float(chain_lengths.std() / max(chain_lengths.mean(), 1.0)),
        "pore_diameter": float(rng.uniform(4.0, 12.0)),
        "pore_length": float(rng.uniform(12.0, 40.0)),
        "silanol_density": float(rng.uniform(1.0, 5.0)),
        "wall_type_id": float(rng.integers(0, 3)),
        "surface_hydroxylation_fraction": float(rng.uniform(0.2, 0.95)),
    }


def _augment_metadata(
    metadata: dict[str, float | int],
    condition_meta: dict[str, float],
    center_pos: np.ndarray,
    box_len: float,
) -> dict[str, float | int]:
    radial = float(np.linalg.norm(center_pos[:2] - box_len / 2.0))
    axial = float(center_pos[2] / box_len)
    wall_distance = max(0.0, condition_meta["pore_diameter"] * 0.5 - radial)
    out = dict(metadata)
    out.update(
        {
            "local_PC_fraction": 0.0,
            "PC_PC_contact_fraction": 0.0,
            "PE_PC_contact_fraction": 0.0,
            "PP_PC_contact_fraction": 0.0,
            "polymer_wall_contact_fraction": float(np.exp(-wall_distance / 1.5)),
            "contact_persistence": float(metadata.get("PE_PP_contact_fraction", 0.0)),
            "neighbor_persistence": float(np.clip(1.0 - metadata.get("mean_displacement_norm", 0.0) / 2.0, 0.0, 1.0)),
            "mean_wall_distance": wall_distance,
            "wall_contact_persistence": float(np.exp(-wall_distance / 2.0)),
            "near_wall_flag": 1.0 if wall_distance < 1.0 else 0.0,
            "pore_axis_position": axial,
            "pore_mouth_distance": float(min(center_pos[2], box_len - center_pos[2])),
            "radial_bin": float(min(3, int(radial / max(box_len * 0.125, 1.0e-6)))),
            "axial_bin": float(min(3, int(axial * 4))),
            "silanol_contact_count": float(condition_meta["silanol_density"] * np.exp(-wall_distance)),
            "siloxane_contact_count": float((5.0 - condition_meta["silanol_density"]) * np.exp(-wall_distance)),
            "dihedral_state_proxy": float(metadata.get("mean_dihedral_transition_proxy", 0.0)),
            "dihedral_transition_proxy": float(metadata.get("mean_dihedral_transition_proxy", 0.0)),
            "orientation_relative_to_pore_axis": float(abs(center_pos[2] / box_len - 0.5) * 2.0),
            "chain_end_distance_proxy": float(np.clip(1.0 - metadata.get("mean_displacement_norm", 0.0), 0.0, 1.0)),
            "aromatic_orientation_persistence": 0.0,
            "carbonate_state_proxy": condition_meta["PC_fraction"],
            "system_density": condition_meta["density"],
        }
    )
    return out


def generate_dummy_dataset(cfg: dict, output_path: str | Path, tiny: bool = False) -> tuple[Path, Path]:
    """Generate internally consistent LD-TDN dummy windows."""
    seed = int(cfg["project"]["seed"])
    rng = np.random.default_rng(seed)
    data_cfg = cfg["data"]
    dummy_cfg = cfg["dummy"]
    condition_names = cfg.get("conditions", {}).get("names", CONDITION_NAMES)
    num_systems = int(dummy_cfg["tiny_num_systems"] if tiny else dummy_cfg["num_systems"])
    samples_per_system = int(dummy_cfg["tiny_samples_per_system"] if tiny else dummy_cfg["samples_per_system"])
    history_len = int(data_cfg["history_len"])
    future_tau = int(data_cfg["future_tau"])
    max_neighbors = int(data_cfg.get("max_neighbors", 64))
    r_cut = float(data_cfg.get("r_cut_nm", 2.0))
    box_len = float(dummy_cfg.get("box_nm", 8.0))
    shell_edges = data_cfg.get("radial_shell_edges_nm", [0.6, 1.2, 2.0, 3.0])
    total_frames = history_len + future_tau + samples_per_system + 4

    samples: list[LocalWindowSample] = []
    per_system_proxy: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    conditions: dict[int, dict[str, float]] = {}
    raw_targets: dict[int, torch.Tensor] = {}

    for system_id in range(num_systems):
        condition_meta = _condition_metadata(rng)
        conditions[system_id] = condition_meta
        n_segments = int(rng.integers(int(dummy_cfg["min_nodes"]), int(dummy_cfg["max_nodes"]) + 1))
        pe_frac = condition_meta["PE_fraction"]
        pp_threshold = pe_frac + condition_meta["PP_fraction"]
        draws = rng.random(n_segments)
        segment_types = np.where(draws < pe_frac, 0, np.where(draws < pp_threshold, 1, 2)).astype(np.int64)
        segment_types[segment_types == 2] = 1
        chain_ids = np.repeat(np.arange(max(1, n_segments // 12 + 1)), 12)[:n_segments]
        if len(chain_ids) < n_segments:
            chain_ids = np.pad(chain_ids, (0, n_segments - len(chain_ids)), mode="edge")
        segment_indices = np.arange(n_segments, dtype=np.int64)
        base_mobility = 0.035 + 0.00022 * (condition_meta["temperature"] - 450.0)
        base_mobility += 0.035 * (1.0 - condition_meta["density"]) + 0.00015 * condition_meta["mean_chain_length"]
        positions = _random_walk(rng, total_frames, n_segments, box_len, base_mobility)
        box = np.asarray([box_len, box_len, box_len], dtype=np.float32)
        torsion = rng.normal(0, 1, size=(total_frames, n_segments)).astype(np.float32)
        system_records = []

        for sample_idx in range(samples_per_system):
            t_end = history_len + sample_idx
            center_idx = int(rng.integers(0, n_segments))
            graph_sequence = []
            frame_meta = []
            disp_rows = []
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
                metadata = _augment_metadata(metadata, condition_meta, positions[frame, center_idx], box_len)
                graph_sequence.append(graph)
                frame_meta.append(metadata)
                disp_rows.append(displacement(positions[max(frame - 1, 0), center_idx], positions[frame, center_idx], box))

            metadata = {
                key: float(np.mean([float(m.get(key, 0.0)) for m in frame_meta]))
                for key in frame_meta[0]
                if key != "environment_type"
            }
            metadata["environment_type"] = int(frame_meta[-1].get("environment_type", 1))
            metadata["system_id"] = system_id
            labels = _local_labels(positions[:, center_idx, :], t_end, future_tau, box, metadata, condition_meta)
            feature_sequence = rolling_descriptor_sequence(frame_meta, np.asarray(disp_rows, dtype=np.float32))
            system_records.append({**metadata, **{f"label_{k}": v for k, v in labels.items()}})
            sample = LocalWindowSample(
                system_id=system_id,
                center_id=center_idx,
                center_type=int(segment_types[center_idx]),
                feature_sequence=feature_sequence,
                graph_sequence=graph_sequence,
                condition=condition_from_metadata(condition_meta, condition_names),
                local_labels=labels,
                system_targets=torch.zeros(len(SYSTEM_TARGET_NAMES), dtype=torch.float32),
                metadata={**metadata, **condition_meta},
            )
            samples.append(sample)
        per_system_proxy[system_id] = system_records

    target_rows = []
    for system_id, records in per_system_proxy.items():
        cond = conditions[system_id]
        mean_free = float(np.mean([float(r["mean_free_volume_proxy"]) for r in records]))
        mean_disp = float(np.mean([float(r["mean_displacement_norm"]) for r in records]))
        wall = float(np.mean([float(r["polymer_wall_contact_fraction"]) for r in records]))
        frac_fast = float(np.mean([int(r["label_mobility_class"]) == 2 for r in records]))
        frac_escape = float(np.mean([int(r["label_escape_class"]) == 2 for r in records]))
        chain_penalty = np.log1p(cond["mean_chain_length"]) * 0.08
        log_d_self = -7.8 + 1.8 * mean_disp + 1.2 * mean_free - 1.8 * cond["density"] - chain_penalty
        log_d_parallel = log_d_self + 0.15 * cond["pore_length"] / max(cond["pore_diameter"], 1.0)
        log_d_eff = log_d_parallel - 0.8 * wall + 0.35 * frac_escape
        log_tau_segmental = 1.2 - 1.1 * mean_disp + 1.5 * cond["density"] + chain_penalty
        log_tau_res = 1.0 + 1.8 * wall + 0.4 * cond["surface_hydroxylation_fraction"]
        p_access = float(1.0 / (1.0 + np.exp(-(2.2 * mean_free + 1.1 * frac_fast - 0.9 * wall))))
        active_site_residence = float(np.clip(0.35 + 0.45 * wall + 0.2 * frac_escape, 0.0, 1.0))
        roi = float(np.clip(p_access * active_site_residence, 0.0, 1.0))
        target_meta = {
            "log_D_self": log_d_self,
            "log_D_parallel": log_d_parallel,
            "log_D_eff": log_d_eff,
            "log_tau_segmental": log_tau_segmental,
            "log_tau_res": log_tau_res,
            "P_access": p_access,
            "reaction_opportunity_index": roi,
        }
        target = torch.tensor([target_meta[name] for name in SYSTEM_TARGET_NAMES], dtype=torch.float32)
        raw_targets[system_id] = target
        target_rows.append({"system_id": system_id, **cond, **target_meta})

    for sample in samples:
        target = raw_targets[int(sample.system_id)]
        sample.system_targets = target
        for i, name in enumerate(SYSTEM_TARGET_NAMES):
            sample.metadata[name] = float(target[i])
            sample.metadata[f"mask_{name}"] = 1.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = output_path.parent / "dummy_system_targets.csv"
    torch.save({"samples": samples, "condition_names": condition_names, "target_names": SYSTEM_TARGET_NAMES}, output_path)
    pd.DataFrame(target_rows).to_csv(target_path, index=False)
    return output_path, target_path
