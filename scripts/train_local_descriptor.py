#!/usr/bin/env python
"""Train LD-TDN local dynamic descriptor encoder and local heads."""

from __future__ import annotations

import argparse

import pandas as pd
import torch
from tqdm import tqdm

from pepp_graph_spib.data.dataset import LocalWindowDataset
from pepp_graph_spib.data.splits import make_or_load_split
from pepp_graph_spib.training.common import (
    build_model_from_config,
    local_descriptor_loss,
    make_loader,
    move_batch,
    save_checkpoint,
)
from pepp_graph_spib.utils import ensure_dirs, get_device, load_config, resolve_path, set_seed


def _epoch(model, loader, device, beta_kl, optimizer=None, grad_clip=5.0) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        losses = local_descriptor_loss(out, batch["local_labels"], beta_kl)
        if train:
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        rows.append({k: float(v.detach().cpu()) for k, v in losses.items()})
    return pd.DataFrame(rows).mean().to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-systems", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
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
    all_data = LocalWindowDataset(data_path, limit_systems=args.limit_systems, limit_samples=args.limit_samples)
    split = make_or_load_split(
        [int(s.system_id) for s in all_data.samples],
        resolve_path(cfg, "split_path"),
        int(cfg["project"]["seed"]),
        cfg["training"]["split_fracs"],
    )
    train_data = LocalWindowDataset(data_path, set(split["train"]), args.limit_systems, args.limit_samples)
    val_data = LocalWindowDataset(data_path, set(split["val"]), args.limit_systems, args.limit_samples)
    train_loader = make_loader(train_data, int(cfg["training"]["batch_size"]), shuffle=True)
    val_loader = make_loader(val_data, int(cfg["training"]["batch_size"]), shuffle=False)
    model = build_model_from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    max_epochs = args.max_epochs or int(cfg["training"]["max_epochs_local"])
    beta_kl = float(cfg["model"]["beta_kl"])
    grad_clip = float(cfg["training"]["grad_clip"])
    ckpt = resolve_path(cfg, "checkpoint_dir") / "local_descriptor_best.pt"
    best = float("inf")
    rows = []
    for epoch in tqdm(range(1, max_epochs + 1), desc="LD-TDN local"):
        train_metrics = _epoch(model, train_loader, device, beta_kl, optimizer, grad_clip)
        with torch.no_grad():
            val_metrics = _epoch(model, val_loader, device, beta_kl)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        rows.append(row)
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            save_checkpoint(ckpt, model_state=model.state_dict(), config=cfg, epoch=epoch, val_loss=best)
    pd.DataFrame(rows).to_csv(resolve_path(cfg, "log_dir") / "local_train_metrics.csv", index=False)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
