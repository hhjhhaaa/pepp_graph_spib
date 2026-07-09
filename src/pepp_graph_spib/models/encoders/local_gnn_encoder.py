"""Local ego-graph frame encoder for LD-TDN."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class LocalGNNFrameEncoder(nn.Module):
    """Encode small local ego graphs, never full-system graphs."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, batch_graph) -> torch.Tensor:
        x = self.node_encoder(batch_graph.x.float())
        edge_attr = batch_graph.edge_attr.float()
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, batch_graph.edge_index, edge_attr)
            x = norm(F.relu(h) + x)
            x = self.dropout(x)
        pooled = global_mean_pool(x, batch_graph.batch)
        if hasattr(batch_graph, "center_index"):
            centers = x[batch_graph.center_index.long()]
        else:
            centers = pooled
        return self.out(torch.cat([centers, pooled], dim=-1))
