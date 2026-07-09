from __future__ import annotations

import copy
import torch

from pepp_graph_spib.data.collate import collate_local_windows
from pepp_graph_spib.data.sample import SYSTEM_TARGET_NAMES


def test_collate_returns_expected_keys_and_shapes(tiny_dataset, cfg):
    batch = collate_local_windows([tiny_dataset[i] for i in range(4)])
    assert set(batch) >= {
        "feature_sequence",
        "graph_sequence",
        "condition",
        "local_labels",
        "system_targets",
        "target_mask",
        "metadata",
    }
    assert batch["feature_sequence"].shape == (4, cfg["data"]["history_len"], cfg["data"]["feature_dim"])
    assert batch["condition"].shape == (4, cfg["data"]["condition_dim"])
    assert batch["system_targets"].shape == (4, len(cfg["system_targets"]["names"]))
    assert batch["target_mask"].shape == batch["system_targets"].shape
    assert torch.all(batch["target_mask"] == 1.0)
    assert batch["local_labels"]["mobility_class"].shape == (4,)
    assert batch["metadata"]["system_id"].shape == (4,)


def test_collate_defaults_missing_target_masks_to_invalid(tiny_dataset):
    sample = copy.deepcopy(tiny_dataset[0])
    for target in SYSTEM_TARGET_NAMES:
        sample.metadata.pop(f"mask_{target}", None)
    batch = collate_local_windows([sample])
    assert torch.all(batch["target_mask"] == 0.0)
