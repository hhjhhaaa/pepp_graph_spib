"""Compatibility alias for LD-TDN physics transport."""

from pepp_graph_spib.models.heads.physics_transport_head import PhysicsTransportHead

TransportPropertyHead = PhysicsTransportHead

__all__ = ["PhysicsTransportHead", "TransportPropertyHead"]
