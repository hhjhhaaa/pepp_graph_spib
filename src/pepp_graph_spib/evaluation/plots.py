"""Plot helpers for embeddings, transport predictions, and LASSO."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_z_embedding(z: np.ndarray, color: np.ndarray, path: str | Path, title: str, label: str) -> None:
    """Scatter z1/z2 colored by a scalar or class label."""
    plt.figure(figsize=(5, 4))
    sc = plt.scatter(z[:, 0], z[:, 1], c=color, s=10, cmap="viridis", alpha=0.85)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(title)
    plt.colorbar(sc, label=label)
    _save(path)


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path, target: str) -> None:
    """Plot predicted vs true transport property."""
    plt.figure(figsize=(4, 4))
    plt.scatter(y_true, y_pred, s=28, alpha=0.85)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel(f"true {target}")
    plt.ylabel(f"pred {target}")
    plt.title(target)
    _save(path)


def plot_lasso_weights(selected_csv: str | Path, path: str | Path) -> None:
    """Plot absolute selected LASSO weights."""
    df = pd.read_csv(selected_csv)
    if df.empty:
        plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, "No nonzero weights", ha="center")
        _save(path)
        return
    df["abs_weight"] = df["weight"].abs()
    top = df.sort_values("abs_weight", ascending=False).head(20)
    labels = top["target"] + ":" + top["descriptor"]
    plt.figure(figsize=(8, 5))
    plt.barh(labels[::-1], top["weight"].values[::-1])
    plt.xlabel("LASSO weight")
    plt.title("Selected descriptor weights")
    _save(path)
