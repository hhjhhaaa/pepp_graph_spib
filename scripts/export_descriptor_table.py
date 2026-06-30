#!/usr/bin/env python
"""Export system-level descriptor table from Graph-SPIB outputs."""

from __future__ import annotations

import argparse

import torch
from pepp_graph_spib.data.graph_window import GraphWindowDataset
from pepp_graph_spib.symbolic.descriptor_table import build_descriptor_table
from pepp_graph_spib.training.common import build_model_from_config, collect_spib_outputs, make_loader
from pepp_graph_spib.utils import ensure_dirs, get_device, load_config, resolve_path, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--spib-checkpoint", required=True)
    parser.add_argument("--transport-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    ensure_dirs(cfg)
    device = get_device(cfg)
    graph_path = resolve_path(cfg, "dummy_graph_path") if cfg["data"]["use_dummy"] else resolve_path(cfg, "processed_graph_path")
    dataset = GraphWindowDataset(str(graph_path))
    spib = build_model_from_config(cfg).to(device)
    spib.load_state_dict(torch.load(args.spib_checkpoint, map_location=device, weights_only=False)["model_state"])
    loader = make_loader(dataset, int(cfg["training"]["batch_size"]), shuffle=False)
    collected = collect_spib_outputs(spib, loader, device)
    out = build_descriptor_table(collected, cfg["data"]["pe_hist_bins"])
    output = resolve_path(cfg, "embedding_dir") / "descriptor_table.csv" if args.output == "" else args.output
    out.to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
