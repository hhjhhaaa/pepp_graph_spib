"""Compatibility exports for the refactored LD-TDN model."""

from pepp_graph_spib.models.bottleneck import kl_divergence
from pepp_graph_spib.models.ld_tdn import LocalDynamicTransportDescriptorNetwork
from pepp_graph_spib.training.common import local_descriptor_loss as spib_loss

GraphSPIB = LocalDynamicTransportDescriptorNetwork

__all__ = ["GraphSPIB", "LocalDynamicTransportDescriptorNetwork", "kl_divergence", "spib_loss"]
