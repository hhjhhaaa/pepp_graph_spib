#!/usr/bin/env python
"""Train system-level transport property head."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

from pepp_graph_spib.data.graph_window import GraphWindowDataset
from pepp_graph_spib.data.splits import make_or_load_split
from pepp_graph_spib.evaluation.plots import plot_pred_vs_true
from pepp_graph_spib.models.property_head import TransportPropertyHead
from pepp_graph_spib.models.system_pooling import aggregate_system_embeddings, system_repr_dim
from pepp_graph_spib.training.common import build_model_from_config, collect_spib_outputs, make_loader, regression_metrics, save_checkpoint
from pepp_graph_spib.utils import ensure_dirs, get_device, load_config, resolve_path, set_seed


def build_system_tensors(cfg, spib, dataset, device):
    loader = make_loader(dataset, int(cfg["training"]["batch_size"]), shuffle=False)
    collected = collect_spib_outputs(spib, loader, device)
    system_repr, unique_ids = aggregate_system_embeddings(
        collected["z"],
        collected["mobility_probs"],
        collected["relax_probs"],
        collected["contact_probs"],
        collected["metadata"]["system_id"],
        collected["metadata"]["center_segment_type"],
        collected["metadata"],
        cfg["data"]["pe_hist_bins"],
    )
    cond_rows = []
    target_rows = []
    for sid in unique_ids.tolist():
        mask = collected["metadata"]["system_id"] == sid
        cond_rows.append(collected["condition"][mask][0])
        target_rows.append(collected["y_property"][mask][0])
    return system_repr, torch.stack(cond_rows), torch.stack(target_rows), unique_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
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
    split = make_or_load_split([int(s.system_id) for s in all_data.samples], resolve_path(cfg, "split_path"), int(cfg["project"]["seed"]), cfg["training"]["split_fracs"])
    train_data = GraphWindowDataset(str(graph_path), set(split["train"]), args.limit_systems, args.limit_samples)
    val_data = GraphWindowDataset(str(graph_path), set(split["val"]), args.limit_systems, args.limit_samples)
    test_data = GraphWindowDataset(str(graph_path), set(split["test"]), args.limit_systems, args.limit_samples)
    spib = build_model_from_config(cfg).to(device)
    spib.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state"])
    spib.eval()
    train_repr, train_cond, train_y, _ = build_system_tensors(cfg, spib, train_data, device)
    val_repr, val_cond, val_y, _ = build_system_tensors(cfg, spib, val_data, device)
    test_repr, test_cond, test_y, test_ids = build_system_tensors(cfg, spib, test_data, device)
    head = TransportPropertyHead(system_repr_dim(int(cfg["model"]["z_dim"]), len(cfg["data"]["pe_hist_bins"]) - 1), int(cfg["data"]["condition_dim"])).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    max_epochs = args.max_epochs or int(cfg["training"]["max_epochs_property"])
    best = float("inf")
    rows = []
    ckpt = resolve_path(cfg, "checkpoint_dir") / "transport_head_best.pt"
    for epoch in tqdm(range(1, max_epochs + 1), desc="Transport"):
        head.train()
        pred = head(train_repr.to(device), train_cond.to(device))
        loss = F.mse_loss(pred, train_y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            val_pred = head(val_repr.to(device), val_cond.to(device))
            val_loss = F.mse_loss(val_pred, val_y.to(device))
        rows.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()), "val_loss": float(val_loss.detach().cpu())})
        if float(val_loss.detach().cpu()) < best:
            best = float(val_loss.detach().cpu())
            save_checkpoint(ckpt, model_state=head.state_dict(), config=cfg, val_loss=best)
    pd.DataFrame(rows).to_csv(resolve_path(cfg, "log_dir") / "transport_train_metrics.csv", index=False)
    head.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model_state"])
    head.eval()
    with torch.no_grad():
        pred = head(test_repr.to(device), test_cond.to(device)).cpu()
    targets = cfg["property_targets"]["names"]
    df = pd.DataFrame({"system_id": test_ids.numpy()})
    for i, target in enumerate(targets):
        df[f"true_{target}"] = test_y[:, i].numpy()
        df[f"pred_{target}"] = pred[:, i].numpy()
    df.to_csv(resolve_path(cfg, "log_dir") / "transport_predictions.csv", index=False)
    fig_dir = resolve_path(cfg, "figure_dir")
    for target in ["log_D", "log_tau_relax", "log_D_eff"]:
        i = targets.index(target)
        plot_pred_vs_true(test_y[:, i].numpy(), pred[:, i].numpy(), fig_dir / f"pred_vs_true_{target}.png", target)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
