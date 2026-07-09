from __future__ import annotations

from pepp_graph_spib.data.sample import CONDITION_NAMES, LOCAL_CLASS_LABELS, LocalWindowSample


def test_dummy_generation_creates_local_window_samples(tiny_dataset, cfg):
    sample = tiny_dataset[0]
    assert isinstance(sample, LocalWindowSample)
    assert sample.feature_sequence is not None
    assert sample.feature_sequence.shape == (cfg["data"]["history_len"], cfg["data"]["feature_dim"])
    assert sample.condition.shape[-1] == len(CONDITION_NAMES)
    assert set(LOCAL_CLASS_LABELS).issubset(sample.local_labels)
    assert "PE_chain_length" in sample.metadata
    assert "PE_repeat_units" in sample.metadata
    assert "PP_repeat_units" in sample.metadata
    assert sample.system_targets.shape[-1] == len(cfg["system_targets"]["names"])
