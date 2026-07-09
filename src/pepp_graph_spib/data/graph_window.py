"""Compatibility names for the old graph-window module.

LD-TDN uses `LocalWindowSample`, descriptor `feature_sequence`, optional
`graph_sequence`, and `collate_local_windows`.
"""

from pepp_graph_spib.data.collate import collate_local_windows as collate_graph_windows
from pepp_graph_spib.data.dataset import LocalWindowDataset as GraphWindowDataset
from pepp_graph_spib.data.sample import LocalWindowSample as GraphWindowSample

__all__ = ["GraphWindowSample", "GraphWindowDataset", "collate_graph_windows"]
