#!/usr/bin/env python
"""Run lightweight LD-TDN input ablations on dummy or processed data."""

from __future__ import annotations

import argparse

import pandas as pd
import torch

from pepp_graph_spib.data.dataset import LocalWindowDataset
from pepp_graph_spib.data.splits import make_or_load_split
from pepp_graph_spib.training.common import build_model_from_config, local_descriptor_loss, make_loader, move_batch
from pepp_graph_spib.utils import ensure_dirs, get_device, load_config, resolve_path, set_seed


ABLATIONS = [
    "condition_only",
    "static_features_only",
    "shuffled_history",
    "no_condition",
    "no_chain_length",
    "no_composition",
    "no_wall_features",
    "descriptor_time_series_full",
    "optional_local_gnn",
]


def train_one(cfg: dict, data_path, split: dict, ablation: str, device: torch.device, max_epochs: int) -> float:
    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["use_graph"] = ablation == "optional_local_gnn"
    train = LocalWindowDataset(data_path, set(split["train"]), transform=ablation, condition_names=cfg["conditions"]["names"])
    val = LocalWindowDataset(data_path, set(split["val"]), transform=ablation, condition_names=cfg["conditions"]["names"])
    if len(train) == 0 or len(val) == 0:
        raise ValueError(f"Ablation {ablation} requires non-empty train and val splits")
    train_loader = make_loader(train, int(cfg["training"]["batch_size"]), shuffle=True)
    val_loader = make_loader(val, int(cfg["training"]["batch_size"]), shuffle=False)
    model = build_model_from_config(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
    beta_kl = float(cfg["model"]["beta_kl"])
    for _ in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            out = model(batch)
            loss = local_descriptor_loss(out, batch["local_labels"], beta_kl)["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch(batch, device)
            losses.append(float(local_descriptor_loss(model(batch), batch["local_labels"], beta_kl)["loss"].cpu()))
    if not losses:
        raise ValueError(f"Ablation {ablation} produced no validation batches")
    return float(sum(losses) / len(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_descriptor_only.yaml")
    parser.add_argument("--max-epochs", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    ensure_dirs(cfg)
    device = get_device(cfg)
    data_path = resolve_path(cfg, "dummy_graph_path") if cfg["data"]["use_dummy"] else resolve_path(cfg, "processed_graph_path")
    all_data = LocalWindowDataset(data_path)
    split = make_or_load_split(
        [int(s.system_id) for s in all_data.samples],
        resolve_path(cfg, "split_path"),
        int(cfg["project"]["seed"]),
        cfg["training"]["split_fracs"],
    )
    rows = []
    for ablation in ABLATIONS:
        rows.append({"ablation": ablation, "val_local_loss": train_one(cfg, data_path, split, ablation, device, args.max_epochs)})
    path = resolve_path(cfg, "log_dir") / "ablation_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
