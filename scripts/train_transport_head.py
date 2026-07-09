#!/usr/bin/env python
"""Train LD-TDN region pooling plus physics-informed transport head."""

from __future__ import annotations

import argparse

import pandas as pd
import torch
from tqdm import tqdm

from pepp_graph_spib.data.dataset import LocalWindowDataset
from pepp_graph_spib.data.sample import SYSTEM_TARGET_NAMES
from pepp_graph_spib.data.splits import make_or_load_split
from pepp_graph_spib.models.heads.physics_transport_head import PhysicsTransportHead
from pepp_graph_spib.models.pooling.region_pooling import RegionPooling, region_repr_dim
from pepp_graph_spib.training.common import (
    build_model_from_config,
    collect_local_outputs,
    make_loader,
    masked_transport_loss,
    save_checkpoint,
    target_valid_counts,
)
from pepp_graph_spib.utils import ensure_dirs, get_device, load_config, resolve_path, set_seed


def build_system_tensors(cfg: dict, model, dataset, device: torch.device):
    loader = make_loader(dataset, int(cfg["training"]["batch_size"]), shuffle=False)
    collected = collect_local_outputs(model, loader, device)
    pooling = RegionPooling()
    system_repr, unique_ids, system_condition = pooling(
        collected["mu"].to(device),
        {k: v.to(device) for k, v in collected["local_outputs"].items()},
        {k: v.to(device) for k, v in collected["metadata"].items()},
        collected["condition"].to(device),
    )
    targets = []
    masks = []
    for sid in unique_ids.detach().cpu().tolist():
        mask = collected["metadata"]["system_id"] == sid
        targets.append(collected["system_targets"][mask][0])
        masks.append(collected["target_mask"][mask][0])
    return system_repr.detach(), system_condition.detach(), torch.stack(targets).to(device), torch.stack(masks).to(device), unique_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--local-checkpoint", required=True)
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
    test_data = LocalWindowDataset(data_path, set(split["test"]), args.limit_systems, args.limit_samples)
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(torch.load(args.local_checkpoint, map_location=device, weights_only=False)["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    train_repr, train_cond, train_y, train_mask, _ = build_system_tensors(cfg, model, train_data, device)
    val_repr, val_cond, val_y, val_mask, _ = build_system_tensors(cfg, model, val_data, device)
    test_repr, test_cond, test_y, test_mask, test_ids = build_system_tensors(cfg, model, test_data, device)
    count_rows = []
    for split_name, split_mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        counts = target_valid_counts(split_mask.detach().cpu(), SYSTEM_TARGET_NAMES)
        for target, count in counts.items():
            count_rows.append({"split": split_name, "target": target, "valid_count": count})
            print(f"{split_name} {target}: {count} valid")
    pd.DataFrame(count_rows).to_csv(resolve_path(cfg, "log_dir") / "transport_target_counts.csv", index=False)
    head = PhysicsTransportHead(
        region_repr_dim(int(cfg["model"]["z_dim"])),
        int(cfg["data"]["condition_dim"]),
        tuple(int(x) for x in cfg.get("transport", {}).get("hidden_dims", [128, 64])),
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    max_epochs = args.max_epochs or int(cfg["training"]["max_epochs_transport"])
    ckpt = resolve_path(cfg, "checkpoint_dir") / "transport_head_best.pt"
    best = float("inf")
    rows = []
    for epoch in tqdm(range(1, max_epochs + 1), desc="LD-TDN transport"):
        head.train()
        pred = head(train_repr, train_cond)
        loss = masked_transport_loss(pred, train_y, train_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            val_pred = head(val_repr, val_cond)
            val_loss = masked_transport_loss(val_pred, val_y, val_mask)
        rows.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()), "val_loss": float(val_loss.detach().cpu())})
        if float(val_loss.detach().cpu()) < best:
            best = float(val_loss.detach().cpu())
            save_checkpoint(ckpt, model_state=head.state_dict(), config=cfg, val_loss=best)
    pd.DataFrame(rows).to_csv(resolve_path(cfg, "log_dir") / "transport_train_metrics.csv", index=False)
    head.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model_state"])
    head.eval()
    with torch.no_grad():
        pred = head(test_repr, test_cond)
    df = pd.DataFrame({"system_id": test_ids.detach().cpu().numpy()})
    for i, target in enumerate(SYSTEM_TARGET_NAMES):
        df[f"true_{target}"] = test_y[:, i].detach().cpu().numpy()
        if target in pred:
            df[f"pred_{target}"] = pred[target].detach().cpu().numpy()
    for key in [
        "D_local",
        "P_entry",
        "C_axis",
        "tau_wall",
        "tau_move",
        "P_access",
        "wall_residence_fraction",
        "transport_score",
        "D_eff",
    ]:
        df[key] = pred[key].detach().cpu().numpy()
    df.to_csv(resolve_path(cfg, "log_dir") / "transport_predictions.csv", index=False)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
