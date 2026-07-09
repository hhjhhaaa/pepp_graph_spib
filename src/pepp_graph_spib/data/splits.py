"""System-level train/validation/test splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pepp_graph_spib.utils import read_json, write_json


def make_or_load_split(
    system_ids: list[int],
    split_path: str | Path,
    seed: int,
    split_fracs: list[float],
) -> dict[str, list[int]]:
    """Create or read a split where systems never cross partitions."""
    path = Path(split_path)
    current = sorted(set(int(x) for x in system_ids))
    if path.exists():
        loaded = read_json(path)
        loaded_ids = sorted(set(int(x) for part in loaded.values() for x in part))
        if loaded_ids == current:
            return loaded
    rng = np.random.default_rng(seed)
    unique = np.array(current, dtype=np.int64)
    if len(unique) < 3:
        raise ValueError(f"Need at least 3 systems for train/val/test split, got {len(unique)}")
    rng.shuffle(unique)
    n = len(unique)
    n_train = max(1, int(round(split_fracs[0] * n)))
    n_val = max(1, int(round(split_fracs[1] * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    split = {
        "train": unique[:n_train].astype(int).tolist(),
        "val": unique[n_train : n_train + n_val].astype(int).tolist(),
        "test": unique[n_train + n_val :].astype(int).tolist(),
    }
    write_json(path, split)
    return split
