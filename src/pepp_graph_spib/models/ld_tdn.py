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
        temporal_hidden_dim: int = 64,
        temporal_layers: int = 2,
        condition_hidden_dim: int = 64,
        z_dim: int = 4,
        encoder_type: str = "gru",
        dropout: float = 0.1,
        use_graph: bool = False,
        node_dim: int = 16,
        edge_dim: int = 12,
        gnn_hidden_dim: int = 64,
        gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.use_graph = use_graph
        if use_graph:
            self.graph_encoder = LocalGNNFrameEncoder(node_dim, edge_dim, gnn_hidden_dim, gnn_layers, dropout)
            temporal_input_dim = gnn_hidden_dim
        else:
            self.graph_encoder = None
            temporal_input_dim = feature_dim
        self.temporal = TimeSeriesEncoder(
            temporal_input_dim,
            hidden_dim=temporal_hidden_dim,
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
        self.bottleneck = VariationalBottleneck(temporal_hidden_dim + condition_hidden_dim, z_dim)
        self.local_heads = LocalDynamicsHeads(z_dim, hidden_dim=temporal_hidden_dim)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        if self.use_graph:
            graph_sequence = batch.get("graph_sequence") or batch.get("batch_graphs_by_time")
            if graph_sequence is None:
                raise ValueError("use_graph=True requires graph_sequence/batch_graphs_by_time")
            frame_h = [self.graph_encoder(graph) for graph in graph_sequence]  # type: ignore[misc]
            sequence = torch.stack(frame_h, dim=1)
        else:
            sequence = batch["feature_sequence"].float()
        temporal_h = self.temporal(sequence)
        cond_h = self.condition_encoder(batch["condition"].float())
        bottleneck = self.bottleneck(torch.cat([temporal_h, cond_h], dim=-1))
        local_outputs = self.local_heads(bottleneck["z"])
        return {**bottleneck, **local_outputs, "local_outputs": local_outputs}
