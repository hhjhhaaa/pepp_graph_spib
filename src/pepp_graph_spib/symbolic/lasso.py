"""LASSO sparse descriptor distillation utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_lasso_table(df: pd.DataFrame, targets: list[str], alpha: float = 0.01) -> pd.DataFrame:
    """Fit LASSO models and return nonzero descriptor weights."""
    available_targets = [target for target in targets if target in df.columns]
    target_like = {
        c
        for c in df.columns
        if c.startswith("target_")
        or c.startswith("log_D")
        or c.startswith("log_tau")
        or c in {"P_access", "D_eff", "reaction_opportunity_index"}
    }
    exclude = set(available_targets + ["system_id"]) | target_like
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for target in available_targets:
        clean = df[feature_cols + [target]].dropna()
        if clean.empty or len(feature_cols) == 0:
            continue
        model = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=10000))
        model.fit(clean[feature_cols], clean[target])
        for name, weight in zip(feature_cols, model.named_steps["lasso"].coef_):
            if abs(float(weight)) > 1.0e-8:
                rows.append({"target": target, "descriptor": name, "weight": float(weight)})
    return pd.DataFrame(rows, columns=["target", "descriptor", "weight"])
