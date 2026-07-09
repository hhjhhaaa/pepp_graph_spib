"""Compatibility exports for LD-TDN region pooling."""

from pepp_graph_spib.models.pooling.region_pooling import REGION_SCALAR_NAMES, RegionPooling, region_repr_dim

system_repr_dim = region_repr_dim

__all__ = ["REGION_SCALAR_NAMES", "RegionPooling", "region_repr_dim", "system_repr_dim"]
