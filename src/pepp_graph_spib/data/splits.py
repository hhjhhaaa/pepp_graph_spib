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
    if path.exists():
        return read_json(path)
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(int(x) for x in system_ids)), dtype=np.int64)
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
