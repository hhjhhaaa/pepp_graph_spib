"""LASSO sparse descriptor distillation utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_lasso_table(df: pd.DataFrame, targets: list[str], alpha: float = 0.01) -> pd.DataFrame:
    """Fit LASSO models and return nonzero descriptor weights."""
    exclude = set(targets + ["system_id", "log_D_eff", "tau_res", "P_access"])
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for target in targets:
        model = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=10000))
        model.fit(df[feature_cols], df[target])
        for name, weight in zip(feature_cols, model.named_steps["lasso"].coef_):
            if abs(float(weight)) > 1.0e-8:
                rows.append({"target": target, "descriptor": name, "weight": float(weight)})
    return pd.DataFrame(rows, columns=["target", "descriptor", "weight"])
