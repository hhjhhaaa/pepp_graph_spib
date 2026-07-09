"""Dataset helpers for LD-TDN local trajectory windows."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from pepp_graph_spib.data.sample import LocalWindowSample


class LocalWindowDataset(Dataset):
    """Load LD-TDN local windows saved by dummy or preprocessing scripts."""

    def __init__(
        self,
        path: str | Path,
        system_ids: set[int] | None = None,
        limit_systems: int | None = None,
        limit_samples: int | None = None,
    ) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples = payload["samples"] if isinstance(payload, dict) else payload
        if system_ids is not None:
            samples = [s for s in samples if int(s.system_id) in system_ids]
        if limit_systems is not None:
            picked = sorted({int(s.system_id) for s in samples})[:limit_systems]
            samples = [s for s in samples if int(s.system_id) in set(picked)]
        if limit_samples is not None:
            samples = samples[:limit_samples]
        self.samples: list[LocalWindowSample] = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> LocalWindowSample:
        return self.samples[idx]
