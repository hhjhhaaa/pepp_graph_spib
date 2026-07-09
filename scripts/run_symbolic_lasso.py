#!/usr/bin/env python
"""Run missing-column robust LASSO descriptor distillation."""

from __future__ import annotations

import argparse

import pandas as pd

from pepp_graph_spib.evaluation.plots import plot_lasso_weights
from pepp_graph_spib.symbolic.lasso import run_lasso_table
from pepp_graph_spib.utils import ensure_dirs, load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--descriptor-table", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    df = pd.read_csv(args.descriptor_table)
    selected = run_lasso_table(df, list(cfg["symbolic"]["target_names"]), alpha=float(cfg["symbolic"]["lasso_alpha"]))
    path = resolve_path(cfg, "log_dir") / "lasso_selected_descriptors.csv"
    selected.to_csv(path, index=False)
    if not selected.empty:
        plot_lasso_weights(path, resolve_path(cfg, "figure_dir") / "lasso_descriptor_weights.png")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
