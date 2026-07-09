#!/usr/bin/env python
"""Export LD-TDN system-level descriptor table."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pepp_graph_spib.data.dataset import LocalWindowDataset
from pepp_graph_spib.models.heads.physics_transport_head import PhysicsTransportHead
from pepp_graph_spib.models.pooling.region_pooling import region_repr_dim
from pepp_graph_spib.symbolic.descriptor_table import build_descriptor_table
from pepp_graph_spib.training.common import build_model_from_config, collect_local_outputs, make_loader
from pepp_graph_spib.utils import ensure_dirs, get_device, load_config, resolve_path, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--local-checkpoint", default="outputs/checkpoints/local_descriptor_best.pt")
    parser.add_argument("--transport-checkpoint", default="outputs/checkpoints/transport_head_best.pt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    ensure_dirs(cfg)
    device = get_device(cfg)
    data_path = (
        resolve_path(cfg, "dummy_local_windows_path")
        if cfg["data"]["use_dummy"]
        else resolve_path(cfg, "processed_local_windows_path")
    )
    dataset = LocalWindowDataset(data_path)
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(torch.load(args.local_checkpoint, map_location=device, weights_only=False)["model_state"])
    loader = make_loader(dataset, int(cfg["training"]["batch_size"]), shuffle=False)
    collected = collect_local_outputs(model, loader, device)
    head = PhysicsTransportHead(
        region_repr_dim(int(cfg["model"]["z_dim"])),
        int(cfg["data"]["condition_dim"]),
        tuple(int(x) for x in cfg.get("transport", {}).get("hidden_dims", [128, 64])),
    )
    ckpt_path = Path(args.transport_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(cfg["_root"]) / ckpt_path
    if ckpt_path.exists():
        head.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state"])
    table = build_descriptor_table(collected, cfg["conditions"]["names"], head)
    output = Path(args.output)
    if not output.is_absolute():
        output = Path(cfg["_root"]) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
