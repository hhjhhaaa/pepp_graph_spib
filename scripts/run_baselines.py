#!/usr/bin/env python
"""Run validation baselines for Graph-SPIB controls."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch.nn import functional as F

from pepp_graph_spib.data.graph_window import GraphWindowDataset
from pepp_graph_spib.data.splits import make_or_load_split
from pepp_graph_spib.evaluation.plots import plot_baseline_comparison
from pepp_graph_spib.models.property_head import TransportPropertyHead
from pepp_graph_spib.models.system_pooling import aggregate_system_embeddings, system_repr_dim
from pepp_graph_spib.training.common import (
    build_model_from_config,
    collect_spib_outputs,
    make_loader,
    move_batch,
    regression_metrics,
    save_checkpoint,
)
from pepp_graph_spib.models.graph_spib import spib_loss
from pepp_graph_spib.utils import ensure_dirs, get_device, load_config, resolve_path, set_seed


def build_system_tensors(cfg, spib, dataset, device):
    loader = make_loader(dataset, int(cfg["training"]["batch_size"]), shuffle=False)
    collected = collect_spib_outputs(spib, loader, device)
    system_repr, unique_ids = aggregate_system_embeddings(
        collected["z"], collected["mobility_probs"], collected["residence_probs"], collected["accessibility_probs"],
        collected["metadata"]["system_id"], collected["metadata"]["center_segment_type"], collected["metadata"],
        cfg["data"]["pe_hist_bins"],
    )
    cond, y = [], []
    for sid in unique_ids.tolist():
        mask = collected["metadata"]["system_id"] == sid
        cond.append(collected["condition"][mask][0])
        y.append(collected["y_property"][mask][0])
    return system_repr, torch.stack(cond), torch.stack(y), unique_ids


def train_small_control(cfg, dataset_train, dataset_test, device, transform_name):
    spib = build_model_from_config(cfg).to(device)
    loader = make_loader(dataset_train, int(cfg["training"]["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(spib.parameters(), lr=float(cfg["training"]["lr"]))
    for _ in range(int(cfg["training"]["baseline_control_epochs"])):
        spib.train()
        for batch in loader:
            batch = move_batch(batch, device)
            pred = spib(batch["batch_graphs_by_time"], batch["dynamic_descriptors"], batch["condition"])
            loss = spib_loss(pred, batch["labels"], float(cfg["model"]["beta_kl"]))["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    train_repr, train_cond, train_y, _ = build_system_tensors(cfg, spib, dataset_train, device)
    test_repr, test_cond, test_y, _ = build_system_tensors(cfg, spib, dataset_test, device)
    head = TransportPropertyHead(system_repr_dim(int(cfg["model"]["z_dim"]), len(cfg["data"]["pe_hist_bins"]) - 1), int(cfg["data"]["condition_dim"])).to(device)
    opt_h = torch.optim.AdamW(head.parameters(), lr=float(cfg["training"]["lr"]))
    for _ in range(int(cfg["training"]["baseline_control_epochs"]) * 2):
        pred = head(train_repr.to(device), train_cond.to(device))
        loss = F.mse_loss(pred, train_y.to(device))
        opt_h.zero_grad(set_to_none=True)
        loss.backward()
        opt_h.step()
    with torch.no_grad():
        pred = head(test_repr.to(device), test_cond.to(device)).cpu().numpy()
    return test_y.numpy(), pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--spib-checkpoint", required=True)
    parser.add_argument("--transport-checkpoint", required=True)
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
    test_data = GraphWindowDataset(str(graph_path), set(split["test"]), args.limit_systems, args.limit_samples)
    targets = cfg["property_targets"]["names"]
    rows = []

    cond_train = []
    y_train = []
    seen = set()
    for s in train_data.samples:
        if s.system_id in seen:
            continue
        seen.add(s.system_id)
        cond_train.append(s.condition.numpy())
        y_train.append(s.property_targets.numpy())
    cond_test = []
    y_test = []
    seen = set()
    for s in test_data.samples:
        if s.system_id in seen:
            continue
        seen.add(s.system_id)
        cond_test.append(s.condition.numpy())
        y_test.append(s.property_targets.numpy())
    ridge = Ridge(alpha=1.0).fit(np.asarray(cond_train), np.asarray(y_train))
    pred = ridge.predict(np.asarray(cond_test))
    rows.append(regression_metrics(np.asarray(y_test), pred, targets, "condition_only"))

    spib = build_model_from_config(cfg).to(device)
    spib.load_state_dict(torch.load(args.spib_checkpoint, map_location=device, weights_only=False)["model_state"])
    full_repr, full_cond, full_y, _ = build_system_tensors(cfg, spib, test_data, device)
    head = TransportPropertyHead(system_repr_dim(int(cfg["model"]["z_dim"]), len(cfg["data"]["pe_hist_bins"]) - 1), int(cfg["data"]["condition_dim"])).to(device)
    head.load_state_dict(torch.load(args.transport_checkpoint, map_location=device, weights_only=False)["model_state"])
    with torch.no_grad():
        pred_full = head(full_repr.to(device), full_cond.to(device)).cpu().numpy()
    rows.append(regression_metrics(full_y.numpy(), pred_full, targets, "full_graph_descriptor_spib"))

    for name in ["static_graph_only", "shuffled_history", "no_dynamic_descriptors", "no_composition_edges"]:
        tr = GraphWindowDataset(str(graph_path), set(split["train"]), args.limit_systems, args.limit_samples, transform=name)
        te = GraphWindowDataset(str(graph_path), set(split["test"]), args.limit_systems, args.limit_samples, transform=name)
        y_true, y_pred = train_small_control(cfg, tr, te, device, name)
        rows.append(regression_metrics(y_true, y_pred, targets, name))

    out = pd.concat(rows, ignore_index=True)
    path = resolve_path(cfg, "log_dir") / "baseline_metrics.csv"
    out.to_csv(path, index=False)
    plot_baseline_comparison(path, resolve_path(cfg, "figure_dir") / "baseline_comparison.png")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
