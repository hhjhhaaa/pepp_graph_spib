#!/usr/bin/env python
"""Train the local multi-scale Graph-SPIB model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pepp_graph_spib.data.graph_window import GraphWindowDataset
from pepp_graph_spib.data.splits import make_or_load_split
from pepp_graph_spib.evaluation.plots import plot_z_embedding
from pepp_graph_spib.models.graph_spib import spib_loss
from pepp_graph_spib.training.common import build_model_from_config, collect_spib_outputs, make_loader, move_batch, save_checkpoint
from pepp_graph_spib.utils import ensure_dirs, get_device, load_config, resolve_path, set_seed


def train_epoch(model, loader, optimizer, device, beta_kl, grad_clip):
    model.train()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["batch_graphs_by_time"], batch["dynamic_descriptors"], batch["condition"])
        losses = spib_loss(out, batch["labels"], beta_kl)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        rows.append({k: float(v.detach().cpu()) for k, v in losses.items()})
    return pd.DataFrame(rows).mean().to_dict()


@torch.no_grad()
def eval_epoch(model, loader, device, beta_kl):
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["batch_graphs_by_time"], batch["dynamic_descriptors"], batch["condition"])
        losses = spib_loss(out, batch["labels"], beta_kl)
        rows.append({k: float(v.detach().cpu()) for k, v in losses.items()})
    return pd.DataFrame(rows).mean().to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-systems", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    ensure_dirs(cfg)
    device = get_device(cfg)
    graph_path = resolve_path(cfg, "dummy_graph_path") if cfg["data"]["use_dummy"] else resolve_path(cfg, "processed_graph_path")
    all_data = GraphWindowDataset(str(graph_path), limit_systems=args.limit_systems, limit_samples=args.limit_samples)
    split = make_or_load_split(
        [int(s.system_id) for s in all_data.samples],
        resolve_path(cfg, "split_path"),
        int(cfg["project"]["seed"]),
        cfg["training"]["split_fracs"],
    )
    train_data = GraphWindowDataset(str(graph_path), set(split["train"]), args.limit_systems, args.limit_samples)
    val_data = GraphWindowDataset(str(graph_path), set(split["val"]), args.limit_systems, args.limit_samples)
    test_data = GraphWindowDataset(str(graph_path), set(split["test"]), args.limit_systems, args.limit_samples)
    batch_size = int(cfg["training"]["batch_size"])
    train_loader = make_loader(train_data, batch_size, shuffle=True)
    val_loader = make_loader(val_data, batch_size, shuffle=False)
    test_loader = make_loader(test_data, batch_size, shuffle=False)

    model = build_model_from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    max_epochs = args.max_epochs or int(cfg["training"]["max_epochs_spib"])
    beta_kl = float(cfg["model"]["beta_kl"])
    grad_clip = float(cfg["training"]["grad_clip"])
    best_val = float("inf")
    log_rows = []
    ckpt_path = resolve_path(cfg, "checkpoint_dir") / "graph_spib_best.pt"
    for epoch in tqdm(range(1, max_epochs + 1), desc="GraphSPIB"):
        train_metrics = train_epoch(model, train_loader, optimizer, device, beta_kl, grad_clip)
        val_metrics = eval_epoch(model, val_loader, device, beta_kl)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        log_rows.append(row)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(ckpt_path, model_state=model.state_dict(), config=cfg, epoch=epoch, val_loss=best_val)
    pd.DataFrame(log_rows).to_csv(resolve_path(cfg, "log_dir") / "graph_spib_train_metrics.csv", index=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"])
    collected = collect_spib_outputs(model, test_loader, device)
    z = collected["z"].numpy()
    fig_dir = resolve_path(cfg, "figure_dir")
    plot_z_embedding(z, collected["labels"]["y_mobility"].numpy(), fig_dir / "z_embedding_by_mobility.png", "z by mobility", "mobility")
    plot_z_embedding(z, collected["metadata"]["local_PE_fraction"].numpy(), fig_dir / "z_embedding_by_pe_fraction.png", "z by PE fraction", "PE fraction")
    plot_z_embedding(
        z,
        collected["labels"]["y_residence"].numpy(),
        fig_dir / "z_embedding_by_residence.png",
        "z by residence",
        "residence",
    )
    plot_z_embedding(
        z,
        collected["labels"]["y_accessibility"].numpy(),
        fig_dir / "z_embedding_by_accessibility.png",
        "z by accessibility",
        "accessibility",
    )
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
