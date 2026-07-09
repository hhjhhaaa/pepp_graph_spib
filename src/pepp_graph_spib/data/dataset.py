"""Dataset helpers for LD-TDN local trajectory windows."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pepp_graph_spib.data.sample import (
    CHAIN_LENGTH_CONDITION_NAMES,
    COMPOSITION_CONDITION_NAMES,
    CONDITION_NAMES,
    WALL_CONDITION_NAMES,
    LocalWindowSample,
)
from pepp_graph_spib.features.segment_features import SEGMENT_FEATURE_NAMES


class LocalWindowDataset(Dataset):
    """Load LD-TDN local windows saved by dummy or preprocessing scripts."""

    def __init__(
        self,
        path: str | Path,
        system_ids: set[int] | None = None,
        limit_systems: int | None = None,
        limit_samples: int | None = None,
        transform: str | None = None,
        condition_names: list[str] | None = None,
    ) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples = payload["samples"] if isinstance(payload, dict) else payload
        if system_ids is not None:
            samples = [s for s in samples if int(s.system_id) in system_ids]
        if limit_systems is not None:
            picked = sorted({int(s.system_id) for s in samples})[:limit_systems]
            samples = [s for s in samples if int(s.system_id) in set(picked)]
        if limit_samples is not None:
            samples = samples[:limit_samples]
        self.samples: list[LocalWindowSample] = samples
        self.transform = transform
        self.condition_names = condition_names or CONDITION_NAMES

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> LocalWindowSample:
        sample = self.samples[idx]
        if self.transform is None:
            return sample
        return transform_sample(sample, self.transform, self.condition_names)


def _zero_condition(sample: LocalWindowSample, names: list[str], keys: list[str]) -> None:
    for key in keys:
        if key in names:
            sample.condition[names.index(key)] = 0.0


def _zero_features(sample: LocalWindowSample, keys: list[str]) -> None:
    if sample.feature_sequence is None:
        return
    for key in keys:
        if key in SEGMENT_FEATURE_NAMES:
            sample.feature_sequence[:, SEGMENT_FEATURE_NAMES.index(key)] = 0.0


def transform_sample(
    sample: LocalWindowSample,
    transform: str,
    condition_names: list[str] | None = None,
) -> LocalWindowSample:
    """Return a transformed sample for ablation controls."""
    out = copy.deepcopy(sample)
    condition_names = condition_names or CONDITION_NAMES
    if transform in {"descriptor_time_series_full", "optional_local_gnn"}:
        return out
    if transform == "condition_only":
        if out.feature_sequence is not None:
            out.feature_sequence = torch.zeros_like(out.feature_sequence)
        out.graph_sequence = None
    elif transform == "static_features_only":
        if out.feature_sequence is not None:
            out.feature_sequence = out.feature_sequence[-1:].repeat(out.feature_sequence.size(0), 1)
        out.graph_sequence = None
    elif transform == "shuffled_history":
        if out.feature_sequence is not None:
            idx = torch.randperm(out.feature_sequence.size(0))
            out.feature_sequence = out.feature_sequence[idx]
        if out.graph_sequence is not None:
            idx = torch.randperm(len(out.graph_sequence)).tolist()
            out.graph_sequence = [out.graph_sequence[i] for i in idx]
    elif transform == "no_condition":
        out.condition = torch.zeros_like(out.condition)
    elif transform == "no_chain_length":
        _zero_condition(out, condition_names, CHAIN_LENGTH_CONDITION_NAMES)
    elif transform == "no_composition":
        _zero_condition(out, condition_names, COMPOSITION_CONDITION_NAMES)
        _zero_features(
            out,
            [
                "local_PE_fraction",
                "local_PP_fraction",
                "local_PS_fraction",
                "PE_PE_contact_fraction",
                "PP_PP_contact_fraction",
                "PS_PS_contact_fraction",
                "PE_PP_contact_fraction",
                "PE_PS_contact_fraction",
                "PP_PS_contact_fraction",
            ],
        )
    elif transform == "no_wall_features":
        _zero_condition(out, condition_names, WALL_CONDITION_NAMES)
        _zero_features(
            out,
            [
                "polymer_wall_contact_fraction",
                "wall_contact_persistence",
                "near_wall_flag",
                "mean_wall_distance",
                "pore_axis_position",
                "pore_mouth_distance",
                "radial_bin",
                "axial_bin",
                "silanol_contact_count",
                "siloxane_contact_count",
            ],
        )
    else:
        raise ValueError(f"Unknown transform: {transform}")
    return out
