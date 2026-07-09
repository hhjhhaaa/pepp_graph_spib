from __future__ import annotations

from pepp_graph_spib.data.sample import CONDITION_NAMES
from pepp_graph_spib.symbolic.descriptor_table import build_descriptor_table
from pepp_graph_spib.symbolic.lasso import run_lasso_table
from pepp_graph_spib.training.common import build_model_from_config, collect_local_outputs, make_loader


def test_lasso_runs_on_dummy_descriptor_table(tiny_dataset, cfg):
    model = build_model_from_config(cfg)
    loader = make_loader(tiny_dataset, batch_size=18, shuffle=False)
    collected = collect_local_outputs(model, loader, "cpu")
    table = build_descriptor_table(collected, CONDITION_NAMES)
    selected = run_lasso_table(table, ["z1", "z2", "missing_target", "D_eff"], alpha=0.01)
    assert list(selected.columns) == ["target", "descriptor", "weight"]
