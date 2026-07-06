"""Hybrid graph-descriptor SPIB model for local transport learning."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv


def _mean_by_graph(values: torch.Tensor, graph_id: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = values.new_zeros((num_graphs, values.size(-1)))
    count = values.new_zeros((num_graphs, 1))
    out.index_add_(0, graph_id, values)
    count.index_add_(0, graph_id, torch.ones((values.size(0), 1), device=values.device, dtype=values.dtype))
    return out / count.clamp_min(1.0)


def _masked_mean(values: torch.Tensor, graph_id: torch.Tensor, mask: torch.Tensor, num_graphs: int) -> torch.Tensor:
    if mask.any():
        return _mean_by_graph(values[mask], graph_id[mask], num_graphs)
    return values.new_zeros((num_graphs, values.size(-1)))


class GraphSPIB(nn.Module):
    """Hybrid graph-descriptor SPIB for local dynamic graph windows."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        condition_dim: int,
        dynamic_descriptor_dim: int,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 3,
        temporal_hidden_dim: int = 128,
        temporal_layers: int = 2,
        descriptor_hidden_dim: int = 64,
        condition_hidden_dim: int = 64,
        z_dim: int = 4,
        num_mobility_classes: int = 3,
        num_residence_classes: int = 3,
        num_accessibility_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.edge_feature_dim = edge_feature_dim
        self.num_segment_types = 4
        self.num_radial_shells = 4
        self.contact_summary_dim = 7
        self.radial_shell_summary_dim = self.num_radial_shells * self.num_segment_types
        self.node_encoder = nn.Linear(node_feature_dim, gnn_hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(gnn_layers):
            mlp = nn.Sequential(nn.Linear(gnn_hidden_dim, gnn_hidden_dim), nn.ReLU(), nn.Linear(gnn_hidden_dim, gnn_hidden_dim))
            self.convs.append(GINEConv(mlp, edge_dim=edge_feature_dim))
            self.norms.append(nn.LayerNorm(gnn_hidden_dim))
        self.dropout = nn.Dropout(dropout)
        pooled_node_dim = gnn_hidden_dim * (2 + self.num_segment_types)
        frame_dim = pooled_node_dim + self.contact_summary_dim + self.radial_shell_summary_dim
        self.frame_proj = nn.Sequential(nn.Linear(frame_dim, gnn_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.temporal = nn.GRU(gnn_hidden_dim, temporal_hidden_dim, num_layers=temporal_layers, dropout=dropout if temporal_layers > 1 else 0.0, batch_first=True)
        self.descriptor_encoder = nn.Sequential(nn.Linear(dynamic_descriptor_dim, descriptor_hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(descriptor_hidden_dim, descriptor_hidden_dim), nn.ReLU())
        self.condition_encoder = nn.Sequential(nn.Linear(condition_dim, condition_hidden_dim), nn.ReLU(), nn.Linear(condition_hidden_dim, condition_hidden_dim), nn.ReLU())
        joint_dim = temporal_hidden_dim + descriptor_hidden_dim + condition_hidden_dim
        self.mu_head = nn.Linear(joint_dim, z_dim)
        self.logvar_head = nn.Linear(joint_dim, z_dim)
        self.mobility_head = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, num_mobility_classes))
        self.residence_head = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, num_residence_classes))
        self.accessibility_head = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, num_accessibility_classes))

    def encode_frame(self, batch) -> torch.Tensor:
        x = self.node_encoder(batch.x.float())
        edge_attr = batch.edge_attr.float()
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, batch.edge_index, edge_attr)
            x = norm(F.relu(h) + x)
            x = self.dropout(x)
        graph_id = batch.batch
        num_graphs = int(batch.num_graphs)
        center_index = batch.center_index.long().view(-1)
        center = x[center_index]
        is_center = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        is_center[center_index] = True
        neighbor = ~is_center
        seg_type = batch.segment_type.to(x.device)
        type_pools = [_masked_mean(x, graph_id, neighbor & (seg_type == type_id), num_graphs) for type_id in range(self.num_segment_types)]
        all_pool = _masked_mean(x, graph_id, neighbor, num_graphs)
        edge_graph = graph_id[batch.edge_index[0]]
        edge_attr = batch.edge_attr.float().to(x.device)
        contact_values = torch.stack([edge_attr[:, 7], edge_attr[:, 8], edge_attr[:, 9], edge_attr[:, 10], edge_attr[:, 11], edge_attr[:, 0], edge_attr[:, 6]], dim=-1)
        contact_summary = _mean_by_graph(contact_values, edge_graph, num_graphs)
        src, dst = batch.edge_index[0].to(x.device), batch.edge_index[1].to(x.device)
        center_edge = is_center[src] ^ is_center[dst]
        neighbor_node = torch.where(is_center[src], dst, src)
        neighbor_type = seg_type[neighbor_node].clamp(min=0, max=self.num_segment_types - 1)
        shell_raw = (edge_attr[:, 4].clamp(0, 1) * self.num_radial_shells).long().clamp(max=self.num_radial_shells - 1)
        radial_parts = []
        for shell in range(self.num_radial_shells):
            shell_mask = center_edge & (shell_raw == shell)
            for type_id in range(self.num_segment_types):
                indicator = (neighbor_type == type_id).float().unsqueeze(-1)
                radial_parts.append(_masked_mean(indicator, edge_graph, shell_mask, num_graphs))
        radial_shell_summary = torch.cat(radial_parts, dim=-1)
        frame_features = torch.cat([center, all_pool, *type_pools, contact_summary, radial_shell_summary], dim=-1)
        return self.frame_proj(frame_features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, batch_graphs_by_time: list, dynamic_descriptors: torch.Tensor, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        frame_embeddings = [self.encode_frame(batch) for batch in batch_graphs_by_time]
        sequence = torch.stack(frame_embeddings, dim=1)
        _, h_n = self.temporal(sequence)
        temporal_h = h_n[-1]
        desc_h = self.descriptor_encoder(dynamic_descriptors.float())
        cond_h = self.condition_encoder(condition.float())
        joint = torch.cat([temporal_h, desc_h, cond_h], dim=-1)
        mu = self.mu_head(joint)
        logvar = self.logvar_head(joint).clamp(min=-8.0, max=8.0)
        z = self.reparameterize(mu, logvar)
        return {"z": z, "mu": mu, "logvar": logvar, "mobility_logits": self.mobility_head(z), "residence_logits": self.residence_head(z), "accessibility_logits": self.accessibility_head(z)}


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def spib_loss(outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor], beta_kl: float) -> dict[str, torch.Tensor]:
    ce_m = F.cross_entropy(outputs["mobility_logits"], labels["y_mobility"])
    ce_r = F.cross_entropy(outputs["residence_logits"], labels["y_residence"])
    ce_a = F.cross_entropy(outputs["accessibility_logits"], labels["y_accessibility"])
    kl = kl_divergence(outputs["mu"], outputs["logvar"])
    total = ce_m + 0.5 * ce_r + 0.5 * ce_a + beta_kl * kl
    return {"loss": total, "mobility": ce_m, "residence": ce_r, "accessibility": ce_a, "kl": kl}
