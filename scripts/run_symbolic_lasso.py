#!/usr/bin/env python
"""Run LASSO sparse descriptor distillation."""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pepp_graph_spib.evaluation.plots import plot_lasso_weights
from pepp_graph_spib.utils import ensure_dirs, load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--descriptor-table", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    df = pd.read_csv(args.descriptor_table)
    targets = cfg["symbolic"]["target_names"]
    exclude = set(targets + cfg["property_targets"]["names"] + ["system_id"])
    feature_cols = [c for c in df.columns if c not in exclude]
    rows = []
    for target in targets:
        model = make_pipeline(StandardScaler(), Lasso(alpha=float(cfg["symbolic"]["lasso_alpha"]), max_iter=10000))
        model.fit(df[feature_cols], df[target])
        coef = model.named_steps["lasso"].coef_
        for name, weight in zip(feature_cols, coef):
            if abs(weight) > 1.0e-8:
                rows.append({"target": target, "descriptor": name, "weight": float(weight)})
    out = pd.DataFrame(rows)
    path = resolve_path(cfg, "log_dir") / "lasso_selected_descriptors.csv"
    out.to_csv(path, index=False)
    plot_lasso_weights(path, resolve_path(cfg, "figure_dir") / "lasso_descriptor_weights.png")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
