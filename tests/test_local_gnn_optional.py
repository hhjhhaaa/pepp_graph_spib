from __future__ import annotations

import torch

from pepp_graph_spib.data.collate import collate_local_windows
from pepp_graph_spib.training.common import build_model_from_config


def test_optional_local_graph_forward_works_on_tiny_graphs(tiny_dataset, cfg):
    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["use_graph"] = True
    batch = collate_local_windows([tiny_dataset[i] for i in range(3)])
    model = build_model_from_config(cfg)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out["z"].shape == (3, cfg["model"]["z_dim"])
    assert out["contact_logits"].shape == (3, 3)
