"""Shared LD-TDN training and evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pepp_graph_spib.data.collate import collate_local_windows
from pepp_graph_spib.data.dataset import LocalWindowDataset
from pepp_graph_spib.data.sample import LOCAL_CLASS_LABELS, SYSTEM_TARGET_NAMES
from pepp_graph_spib.features.segment_features import feature_dim
from pepp_graph_spib.models.bottleneck import kl_divergence
from pepp_graph_spib.models.ld_tdn import LocalDynamicTransportDescriptorNetwork


def move_batch(batch: dict, device: torch.device) -> dict:
    """Move a collated LD-TDN batch to device."""
    return {
        "feature_sequence": batch["feature_sequence"].to(device),
        "graph_sequence": [g.to(device) for g in batch["graph_sequence"]],
        "condition": batch["condition"].to(device),
        "local_labels": {k: v.to(device) for k, v in batch["local_labels"].items()},
        "system_targets": batch["system_targets"].to(device),
        "target_mask": batch["target_mask"].to(device),
        "metadata": {k: v.to(device) for k, v in batch["metadata"].items()},
    }


def make_loader(dataset: LocalWindowDataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    """Create a DataLoader for LD-TDN samples."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_local_windows, num_workers=0)


def build_model_from_config(cfg: dict) -> LocalDynamicTransportDescriptorNetwork:
    """Instantiate LD-TDN from config."""
    data = cfg["data"]
    model = cfg["model"]
    return LocalDynamicTransportDescriptorNetwork(
        feature_dim=int(data.get("feature_dim", feature_dim())),
        condition_dim=int(data["condition_dim"]),
        descriptor_hidden_dim=int(model.get("descriptor_hidden_dim", 64)),
        graph_hidden_dim=int(model.get("graph_hidden_dim", 64)),
        temporal_layers=int(model.get("temporal_layers", 2)),
        condition_hidden_dim=int(model.get("condition_hidden_dim", 64)),
        z_dim=int(model.get("z_dim", 4)),
        encoder_type=str(model.get("encoder_type", "gru")),
        dropout=float(model.get("dropout", 0.1)),
        node_dim=int(data.get("node_feature_dim", 16)),
        edge_dim=int(data.get("edge_feature_dim", 12)),
        gnn_layers=int(model.get("gnn_layers", 2)),
    )


def _gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logvar + (target - mu).pow(2) / logvar.exp().clamp_min(1.0e-6)).mean()


def local_descriptor_loss(outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor], beta_kl: float) -> dict[str, torch.Tensor]:
    """Compute LD-TDN local predictive bottleneck loss."""
    losses = {
        "mobility": F.cross_entropy(outputs["mobility_logits"], labels["mobility_class"]),
        "contact": F.cross_entropy(outputs["contact_logits"], labels["contact_class"]),
        "residence": F.cross_entropy(outputs["residence_logits"], labels["residence_class"]),
        "escape": F.cross_entropy(outputs["escape_logits"], labels["escape_class"]),
        "relax": F.cross_entropy(outputs["relax_logits"], labels["relax_class"]),
    }
    disp_target = torch.stack(
        [
            labels["future_disp_parallel"],
            labels["future_disp_radial"],
            labels["future_disp_norm"],
        ],
        dim=-1,
    )
    msd_target = torch.stack([labels["short_msd_parallel"], labels["short_msd_radial"]], dim=-1)
    losses["disp"] = _gaussian_nll(outputs["disp_mu"], outputs["disp_logvar"], disp_target)
    losses["msd"] = _gaussian_nll(outputs["short_msd_mu"], outputs["short_msd_logvar"], msd_target)
    losses["contact_survival"] = F.mse_loss(outputs["contact_survival"], labels["contact_survival"])
    losses["wall_contact_survival"] = F.mse_loss(outputs["wall_contact_survival"], labels["wall_contact_survival"])
    losses["free_volume_opening"] = F.mse_loss(outputs["free_volume_opening"], labels["free_volume_opening"])
    losses["kl"] = kl_divergence(outputs["mu"], outputs["logvar"])
    total = sum(losses[k] for k in losses if k != "kl") + beta_kl * losses["kl"]
    return {"loss": total, **losses}


@torch.no_grad()
def collect_local_outputs(model: LocalDynamicTransportDescriptorNetwork, loader: Iterable, device: torch.device) -> dict:
    """Collect descriptors, local outputs, metadata, conditions, and targets."""
    model.eval()
    out: dict[str, list[torch.Tensor]] = {
        "z": [],
        "mu": [],
        "condition": [],
        "system_targets": [],
        "target_mask": [],
    }
    local_keys = [
        "mobility_logits",
        "contact_logits",
        "residence_logits",
        "escape_logits",
        "relax_logits",
        "contact_survival",
        "wall_contact_survival",
        "free_volume_opening",
    ]
    local_out: dict[str, list[torch.Tensor]] = {key: [] for key in local_keys}
    meta: dict[str, list[torch.Tensor]] = {}
    labels: dict[str, list[torch.Tensor]] = {key: [] for key in LOCAL_CLASS_LABELS}
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch)
        out["z"].append(pred["z"].detach().cpu())
        out["mu"].append(pred["mu"].detach().cpu())
        out["condition"].append(batch["condition"].detach().cpu())
        out["system_targets"].append(batch["system_targets"].detach().cpu())
        out["target_mask"].append(batch["target_mask"].detach().cpu())
        for key in local_keys:
            local_out[key].append(pred[key].detach().cpu())
        for key, value in batch["metadata"].items():
            meta.setdefault(key, []).append(value.detach().cpu())
        for key in LOCAL_CLASS_LABELS:
            labels[key].append(batch["local_labels"][key].detach().cpu())
    result = {k: torch.cat(v, dim=0) for k, v in out.items()}
    result["local_outputs"] = {k: torch.cat(v, dim=0) for k, v in local_out.items()}
    result["metadata"] = {k: torch.cat(v, dim=0) for k, v in meta.items()}
    result["local_labels"] = {k: torch.cat(v, dim=0) for k, v in labels.items()}
    return result


def masked_transport_loss(outputs: dict[str, torch.Tensor], targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked system-target loss for physics-head outputs."""
    pieces = []
    for i, name in enumerate(SYSTEM_TARGET_NAMES):
        if name not in outputs:
            continue
        valid = mask[:, i] > 0.5
        if not valid.any():
            continue
        pred = outputs[name][valid]
        target = targets[valid, i]
        if name == "P_access":
            pieces.append(F.mse_loss(pred, target.clamp(0.0, 1.0)))
        else:
            pieces.append(F.smooth_l1_loss(pred, target))
    if not pieces:
        anchor = next(iter(outputs.values()))
        return anchor.sum() * 0.0
    return sum(pieces)


def target_valid_counts(mask: torch.Tensor, names: list[str] | None = None) -> dict[str, int]:
    """Return valid target counts per target name from a target mask."""
    names = names or SYSTEM_TARGET_NAMES
    if mask.ndim != 2 or mask.size(1) != len(names):
        raise ValueError(f"target mask shape {tuple(mask.shape)} does not match {len(names)} target names")
    return {name: int((mask[:, i] > 0.5).sum().item()) for i, name in enumerate(names)}


def save_checkpoint(path: str | Path, **payload) -> None:
    """Save a checkpoint and create its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
