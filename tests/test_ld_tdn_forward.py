from __future__ import annotations

import torch

from pepp_graph_spib.data.collate import collate_local_windows
from pepp_graph_spib.training.common import build_model_from_config


def test_descriptor_only_forward_outputs_all_local_heads(tiny_dataset, cfg):
    batch = collate_local_windows([tiny_dataset[i] for i in range(4)])
    model = build_model_from_config(cfg)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out["z"].shape == (4, cfg["model"]["z_dim"])
    assert out["mu"].shape == (4, cfg["model"]["z_dim"])
    assert out["logvar"].shape == (4, cfg["model"]["z_dim"])
    for key in ["mobility_logits", "contact_logits", "residence_logits", "escape_logits", "relax_logits"]:
        assert out[key].shape == (4, 3)
    assert out["disp_mu"].shape == (4, 3)
