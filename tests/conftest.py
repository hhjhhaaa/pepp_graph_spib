from __future__ import annotations

import pytest

from pepp_graph_spib.data.dataset import LocalWindowDataset
from pepp_graph_spib.data.dummy import generate_dummy_dataset
from pepp_graph_spib.utils import load_config, set_seed


@pytest.fixture(scope="session")
def cfg():
    cfg = load_config("configs/model_descriptor_only.yaml")
    set_seed(123)
    return cfg


@pytest.fixture(scope="session")
def tiny_path(tmp_path_factory, cfg):
    path = tmp_path_factory.mktemp("ld_tdn") / "dummy_graph_windows.pt"
    graph_path, _ = generate_dummy_dataset(cfg, path, tiny=True)
    return graph_path


@pytest.fixture()
def tiny_dataset(tiny_path):
    return LocalWindowDataset(tiny_path, limit_samples=36)
