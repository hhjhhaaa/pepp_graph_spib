from __future__ import annotations

import torch

from pepp_graph_spib.models.heads.physics_transport_head import PhysicsTransportHead
from pepp_graph_spib.models.pooling.region_pooling import RegionPooling, region_repr_dim
from pepp_graph_spib.training.common import (
    build_model_from_config,
    collect_local_outputs,
    local_descriptor_loss,
    make_loader,
    masked_transport_loss,
    move_batch,
)


def test_one_epoch_local_and_transport_smoke(tiny_dataset, cfg):
    device = torch.device("cpu")
    model = build_model_from_config(cfg).to(device)
    loader = make_loader(tiny_dataset, batch_size=12, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        loss = local_descriptor_loss(out, batch["local_labels"], float(cfg["model"]["beta_kl"]))["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        break
    collected = collect_local_outputs(model, loader, device)
    pooling = RegionPooling()
    system_repr, _, system_condition = pooling(
        collected["mu"],
        collected["local_outputs"],
        collected["metadata"],
        collected["condition"],
    )
    target_rows = []
    mask_rows = []
    for sid in torch.unique(collected["metadata"]["system_id"], sorted=True).tolist():
        mask = collected["metadata"]["system_id"] == sid
        target_rows.append(collected["system_targets"][mask][0])
        mask_rows.append(collected["target_mask"][mask][0])
    head = PhysicsTransportHead(region_repr_dim(cfg["model"]["z_dim"]), cfg["data"]["condition_dim"])
    pred = head(system_repr, system_condition)
    loss = masked_transport_loss(pred, torch.stack(target_rows), torch.stack(mask_rows))
    loss.backward()
    assert bool(torch.isfinite(loss))
