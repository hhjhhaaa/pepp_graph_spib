"""Local Dynamic Transport Descriptor Network (LD-TDN)."""

from __future__ import annotations

import torch
from torch import nn

from pepp_graph_spib.models.bottleneck import VariationalBottleneck
from pepp_graph_spib.models.encoders.local_gnn_encoder import LocalGNNFrameEncoder
from pepp_graph_spib.models.encoders.timeseries_encoder import TimeSeriesEncoder
from pepp_graph_spib.models.heads.local_dynamics_heads import LocalDynamicsHeads


class LocalDynamicTransportDescriptorNetwork(nn.Module):
    """Learn local dynamic transport descriptors from short trajectory windows."""

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        descriptor_hidden_dim: int = 64,
        graph_hidden_dim: int = 64,
        temporal_layers: int = 2,
        condition_hidden_dim: int = 64,
        z_dim: int = 4,
        encoder_type: str = "gru",
        dropout: float = 0.1,
        node_dim: int = 16,
        edge_dim: int = 12,
        gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.graph_encoder = LocalGNNFrameEncoder(node_dim, edge_dim, graph_hidden_dim, gnn_layers, dropout)
        self.graph_temporal = TimeSeriesEncoder(
            graph_hidden_dim,
            hidden_dim=graph_hidden_dim,
            num_layers=temporal_layers,
            encoder_type=encoder_type,
            dropout=dropout,
        )
        self.descriptor_temporal = TimeSeriesEncoder(
            feature_dim,
            hidden_dim=descriptor_hidden_dim,
            num_layers=temporal_layers,
            encoder_type=encoder_type,
            dropout=dropout,
        )
        self.condition_encoder = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, condition_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(condition_hidden_dim, condition_hidden_dim),
            nn.ReLU(),
        )
        self.bottleneck = VariationalBottleneck(descriptor_hidden_dim + graph_hidden_dim + condition_hidden_dim, z_dim)
        self.local_heads = LocalDynamicsHeads(z_dim, hidden_dim=descriptor_hidden_dim)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        if "feature_sequence" not in batch or "graph_sequence" not in batch or "condition" not in batch:
            raise ValueError("LD-TDN requires feature_sequence, graph_sequence, and condition")
        feature_sequence = batch["feature_sequence"].float()
        graph_sequence = batch["graph_sequence"]
        if graph_sequence is None or len(graph_sequence) == 0:
            raise ValueError("LD-TDN requires a non-empty local graph_sequence")
        desc_h = self.descriptor_temporal(feature_sequence)
        graph_frame_h = [self.graph_encoder(graph) for graph in graph_sequence]
        graph_h = self.graph_temporal(torch.stack(graph_frame_h, dim=1))
        cond_h = self.condition_encoder(batch["condition"].float())
        if desc_h.size(0) != graph_h.size(0) or desc_h.size(0) != cond_h.size(0):
            raise ValueError("feature_sequence, graph_sequence, and condition batch sizes must match")
        bottleneck = self.bottleneck(torch.cat([desc_h, graph_h, cond_h], dim=-1))
        local_outputs = self.local_heads(bottleneck["z"])
        return {**bottleneck, **local_outputs, "local_outputs": local_outputs}
