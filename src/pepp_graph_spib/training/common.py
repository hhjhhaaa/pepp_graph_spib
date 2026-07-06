"""Shared training and evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from pepp_graph_spib.data.graph_window import GraphWindowDataset, collate_graph_windows
from pepp_graph_spib.models.graph_spib import GraphSPIB


def move_batch(batch: dict, device: torch.device) -> dict:
    """Move a collated graph-window batch to device."""
    return {
        "batch_graphs_by_time": [g.to(device) for g in batch["batch_graphs_by_time"]],
        "dynamic_descriptors": batch["dynamic_descriptors"].to(device),
        "condition": batch["condition"].to(device),
        "labels": {k: v.to(device) for k, v in batch["labels"].items()},
        "metadata": {k: v.to(device) for k, v in batch["metadata"].items()},
    }


def make_loader(dataset: GraphWindowDataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    """Create DataLoader for GraphWindowDataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graph_windows, num_workers=0)


def build_model_from_config(cfg: dict) -> GraphSPIB:
    """Instantiate GraphSPIB from config."""
    data = cfg["data"]
    model = cfg["model"]
    return GraphSPIB(
        node_feature_dim=int(data["node_feature_dim"]),
        edge_feature_dim=int(data["edge_feature_dim"]),
        condition_dim=int(data["condition_dim"]),
        dynamic_descriptor_dim=int(data["dynamic_descriptor_dim"]),
        gnn_hidden_dim=int(model["gnn_hidden_dim"]),
        gnn_layers=int(model["gnn_layers"]),
        temporal_hidden_dim=int(model["temporal_hidden_dim"]),
        temporal_layers=int(model["temporal_layers"]),
        descriptor_hidden_dim=int(model["descriptor_hidden_dim"]),
        condition_hidden_dim=int(model["condition_hidden_dim"]),
        z_dim=int(model["z_dim"]),
        num_mobility_classes=int(data["num_mobility_classes"]),
        num_residence_classes=int(data["num_residence_classes"]),
        num_accessibility_classes=int(data["num_accessibility_classes"]),
        dropout=float(model["dropout"]),
    )


@torch.no_grad()
def collect_spib_outputs(model: GraphSPIB, loader: Iterable, device: torch.device) -> dict[str, torch.Tensor]:
    """Collect z, probabilities, metadata, conditions, and targets from windows."""
    model.eval()
    out = {
        k: []
        for k in [
            "z",
            "mobility_probs",
            "residence_probs",
            "accessibility_probs",
            "dynamic_descriptors",
            "condition",
            "y_property",
        ]
    }
    meta: dict[str, list[torch.Tensor]] = {}
    labels: dict[str, list[torch.Tensor]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch["batch_graphs_by_time"], batch["dynamic_descriptors"], batch["condition"])
        out["z"].append(pred["mu"].detach().cpu())
        out["mobility_probs"].append(pred["mobility_logits"].softmax(dim=-1).detach().cpu())
        out["residence_probs"].append(pred["residence_logits"].softmax(dim=-1).detach().cpu())
        out["accessibility_probs"].append(pred["accessibility_logits"].softmax(dim=-1).detach().cpu())
        out["dynamic_descriptors"].append(batch["dynamic_descriptors"].detach().cpu())
        out["condition"].append(batch["condition"].detach().cpu())
        out["y_property"].append(batch["labels"]["y_property"].detach().cpu())
        for key, value in batch["metadata"].items():
            meta.setdefault(key, []).append(value.detach().cpu())
        for key, value in batch["labels"].items():
            if key != "y_property":
                labels.setdefault(key, []).append(value.detach().cpu())
    result = {k: torch.cat(v, dim=0) for k, v in out.items()}
    result["metadata"] = {k: torch.cat(v, dim=0) for k, v in meta.items()}
    result["labels"] = {k: torch.cat(v, dim=0) for k, v in labels.items()}
    return result


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str], model_name: str) -> pd.DataFrame:
    """Return MAE/RMSE/R2 metrics for each transport target."""
    rows = []
    for i, target in enumerate(targets):
        rows.append(
            {
                "model_name": model_name,
                "target": target,
                "MAE": mean_absolute_error(y_true[:, i], y_pred[:, i]),
                "RMSE": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
                "R2": r2_score(y_true[:, i], y_pred[:, i]) if len(y_true) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_checkpoint(path: str | Path, **payload) -> None:
    """Save checkpoint and create parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
