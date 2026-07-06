
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from pepp_graph_spib.data.dummy import generate_dummy_dataset
from pepp_graph_spib.data.graph_window import GraphWindowDataset, collate_graph_windows
from pepp_graph_spib.models.graph_spib import kl_divergence, spib_loss
from pepp_graph_spib.models.property_head import TransportPropertyHead
from pepp_graph_spib.models.system_pooling import aggregate_system_embeddings, system_repr_dim
from pepp_graph_spib.symbolic.descriptor_table import REQUIRED_DESCRIPTOR_COLUMNS, build_descriptor_table
from pepp_graph_spib.symbolic.lasso import run_lasso_table
from pepp_graph_spib.training.common import build_model_from_config, collect_spib_outputs, make_loader, move_batch
from pepp_graph_spib.utils import load_config, set_seed


@pytest.fixture(scope="module")
def cfg():
    cfg = load_config("configs/default.yaml")
    set_seed(123)
    return cfg


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory, cfg):
    out = tmp_path_factory.mktemp("data") / "dummy_graph_windows.pt"
    graph_path, _ = generate_dummy_dataset(cfg, out, tiny=True)
    return GraphWindowDataset(str(graph_path), limit_samples=36)


@pytest.fixture()
def model(cfg):
    model = build_model_from_config(cfg)
    model.eval()
    return model


@pytest.fixture()
def batch(tiny_dataset):
    return collate_graph_windows([tiny_dataset[i] for i in range(4)])


def test_dummy_data_generation(tiny_dataset, cfg):
    assert len(tiny_dataset) == 36
    sample = tiny_dataset[0]
    assert len(sample.graph_sequence) == 8
    assert sample.graph_sequence[0].x.shape[1] == 16
    assert sample.graph_sequence[0].edge_attr.shape[1] == 12
    assert sample.dynamic_descriptors.shape[-1] == cfg["data"]["dynamic_descriptor_dim"]
    assert set(sample.future_labels) == {"mobility", "residence", "accessibility"}
    assert sample.property_targets.shape[-1] == len(cfg["property_targets"]["names"])
    assert "local_PE_fraction" in sample.metadata


def test_dataset_loading_and_collate(batch, cfg):
    assert len(batch["batch_graphs_by_time"]) == cfg["data"]["history_len"]
    assert batch["condition"].shape == (4, 6)
    assert batch["dynamic_descriptors"].shape == (4, cfg["data"]["dynamic_descriptor_dim"])
    assert batch["labels"]["y_mobility"].shape == (4,)
    assert batch["labels"]["y_residence"].shape == (4,)
    assert batch["labels"]["y_accessibility"].shape == (4,)
    assert batch["metadata"]["PE_PP_contact_fraction"].shape == (4,)



def test_pyg_batch_center_index_points_to_center_nodes(batch):
    graph = batch["batch_graphs_by_time"][0]
    centers = graph.center_index.long()
    assert centers.numel() == batch["condition"].shape[0]
    assert torch.all(graph.x[centers, 2] == 1.0)
    for graph_id, center_idx in enumerate(centers.tolist()):
        assert int(graph.batch[center_idx]) == graph_id


def test_pe_pp_neighbor_masks_exclude_center_nodes(batch):
    graph = batch["batch_graphs_by_time"][0]
    centers = graph.center_index.long()
    is_center = torch.zeros(graph.x.size(0), dtype=torch.bool)
    is_center[centers] = True
    pe_neighbor_mask = (~is_center) & (graph.segment_type == 0)
    pp_neighbor_mask = (~is_center) & (graph.segment_type == 1)
    assert not torch.any(pe_neighbor_mask & is_center)
    assert not torch.any(pp_neighbor_mask & is_center)
    assert torch.all(graph.x[centers, 2] == 1.0)

def test_graph_spib_forward_node16_edge12(batch, model):
    with torch.no_grad():
        out = model(batch["batch_graphs_by_time"], batch["dynamic_descriptors"], batch["condition"])
    assert out["z"].shape == (4, 4)
    assert out["mu"].shape == (4, 4)
    assert out["logvar"].shape == (4, 4)
    assert out["mobility_logits"].shape == (4, 3)
    assert out["residence_logits"].shape == (4, 3)
    assert out["accessibility_logits"].shape == (4, 3)
    assert batch["batch_graphs_by_time"][0].edge_attr.shape[1] == 12


def test_edge_attr_changes_output(batch, model):
    changed = copy.deepcopy(batch)
    changed["batch_graphs_by_time"][0].edge_attr[:, 10] = 1.0 - changed["batch_graphs_by_time"][0].edge_attr[:, 10]
    with torch.no_grad():
        base = model(batch["batch_graphs_by_time"], batch["dynamic_descriptors"], batch["condition"])["mu"]
        alt = model(changed["batch_graphs_by_time"], changed["dynamic_descriptors"], changed["condition"])["mu"]
    assert not torch.allclose(base, alt)


def test_pe_pp_pooling_changes_output(batch, model):
    changed = copy.deepcopy(batch)
    graph = changed["batch_graphs_by_time"][0]
    graph.segment_type = 1 - graph.segment_type
    graph.x[:, [0, 1]] = graph.x[:, [1, 0]]
    with torch.no_grad():
        base = model(batch["batch_graphs_by_time"], batch["dynamic_descriptors"], batch["condition"])["mu"]
        alt = model(changed["batch_graphs_by_time"], changed["dynamic_descriptors"], changed["condition"])["mu"]
    assert not torch.allclose(base, alt)


def test_gru_temporal_order_affects_output(batch, model):
    reversed_graphs = list(reversed(batch["batch_graphs_by_time"]))
    with torch.no_grad():
        base = model(batch["batch_graphs_by_time"], batch["dynamic_descriptors"], batch["condition"])["mu"]
        alt = model(reversed_graphs, batch["dynamic_descriptors"], batch["condition"])["mu"]
    assert not torch.allclose(base, alt)


def test_kl_loss_is_nonzero_and_included(batch, model, cfg):
    model.train()
    out = model(batch["batch_graphs_by_time"], batch["dynamic_descriptors"], batch["condition"])
    losses = spib_loss(out, batch["labels"], float(cfg["model"]["beta_kl"]))
    expected = losses["mobility"] + losses["residence"] + losses["accessibility"] + float(cfg["model"]["beta_kl"]) * losses["kl"]
    assert losses["kl"].item() > 0.0
    assert torch.allclose(losses["loss"], expected)


def test_mini_training_system_pooling_and_transport_head(tiny_dataset, cfg):
    device = torch.device("cpu")
    model = build_model_from_config(cfg).to(device)
    loader = make_loader(tiny_dataset, batch_size=8, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        for mini in loader:
            mini = move_batch(mini, device)
            out = model(mini["batch_graphs_by_time"], mini["dynamic_descriptors"], mini["condition"])
            loss = spib_loss(out, mini["labels"], float(cfg["model"]["beta_kl"]))["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    collected = collect_spib_outputs(model, loader, device)
    system_repr, system_ids = aggregate_system_embeddings(
        collected["z"],
        collected["mobility_probs"],
        collected["residence_probs"],
        collected["accessibility_probs"],
        collected["metadata"]["system_id"],
        collected["metadata"]["center_segment_type"],
        collected["metadata"],
        cfg["data"]["pe_hist_bins"],
    )
    assert system_repr.shape[1] == system_repr_dim(4, 5)
    assert system_repr.shape[1] == 48
    cond_rows, target_rows = [], []
    for sid in system_ids.tolist():
        mask = collected["metadata"]["system_id"] == sid
        cond_rows.append(collected["condition"][mask][0])
        target_rows.append(collected["y_property"][mask][0])
    cond = torch.stack(cond_rows)
    target = torch.stack(target_rows)
    head = TransportPropertyHead(system_repr_dim(4, 5), 6)
    for _ in range(2):
        pred = head(system_repr, cond)
        loss = F.mse_loss(pred, target)
        loss.backward()
        for param in head.parameters():
            if param.grad is not None:
                param.data -= 1e-3 * param.grad
                param.grad = None
    assert pred.shape == target.shape


def test_descriptor_table_contains_physical_columns_and_lasso_runs(tiny_dataset, cfg):
    device = torch.device("cpu")
    model = build_model_from_config(cfg).to(device)
    loader = make_loader(tiny_dataset, batch_size=12, shuffle=False)
    collected = collect_spib_outputs(model, loader, device)
    table = build_descriptor_table(collected, cfg["data"]["pe_hist_bins"])
    for column in REQUIRED_DESCRIPTOR_COLUMNS:
        assert column in table.columns
    for column in ["local_PE_fraction_hist_bin_0", "local_PP_fraction_hist_bin_0"]:
        assert column in table.columns
    selected = run_lasso_table(table, ["z1", "z2", "z3", "z4", "log_D", "log_tau_relax"], alpha=0.01)
    assert list(selected.columns) == ["target", "descriptor", "weight"]
