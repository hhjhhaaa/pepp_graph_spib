from __future__ import annotations

from pepp_graph_spib.data.sample import CONDITION_NAMES, SYSTEM_TARGET_NAMES
from pepp_graph_spib.symbolic.descriptor_table import REQUIRED_DESCRIPTOR_COLUMNS, build_descriptor_table
from pepp_graph_spib.training.common import build_model_from_config, collect_local_outputs, make_loader


def test_descriptor_table_contains_conditions_z_physics_and_targets(tiny_dataset, cfg):
    model = build_model_from_config(cfg)
    loader = make_loader(tiny_dataset, batch_size=18, shuffle=False)
    collected = collect_local_outputs(model, loader, "cpu")
    table = build_descriptor_table(collected, CONDITION_NAMES)
    for column in REQUIRED_DESCRIPTOR_COLUMNS:
        assert column in table.columns
    for column in CONDITION_NAMES:
        assert column in table.columns
    for column in SYSTEM_TARGET_NAMES:
        assert f"target_{column}" in table.columns
    assert "mean_z1" in table.columns
    assert "D_eff" in table.columns
