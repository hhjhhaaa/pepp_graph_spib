"""Project utilities for config, paths, seeding, and device selection."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def project_root() -> Path:
    """Return the repository root from any package module."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and attach the project root path."""
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / path
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_root"] = str(project_root())
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a configured path under the project root."""
    value = cfg["paths"][key]
    path = Path(value)
    if not path.is_absolute():
        path = Path(cfg["_root"]) / path
    return path


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create configured output directories."""
    for key in ("checkpoint_dir", "log_dir", "figure_dir", "embedding_dir"):
        resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)
    for key in ("dummy_local_windows_path", "processed_local_windows_path"):
        resolve_path(cfg, key).parent.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict[str, Any]) -> torch.device:
    """Return CUDA when requested and available, otherwise CPU."""
    requested = cfg.get("project", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
