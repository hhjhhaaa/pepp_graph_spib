# Architecture Audit and Hardening Report

Project: PE/PP Local Multi-Scale Graph-SPIB  
Path: `/home/jinhao/mlff/pepp_graph_spib`  
Audit target: current dummy-validated architecture after hardening

## Audit Checklist

| Item | Status | Evidence |
| --- | --- | --- |
| `edge_attr` enters message passing | Implemented | `GraphSPIB.encode_frame()` passes `edge_attr` into each `GINEConv`. |
| GINEConv correctly uses edge features | Implemented | `GINEConv(..., edge_dim=edge_feature_dim)` is used with `edge_feature_dim=12`. |
| frame readout includes center embedding | Implemented | `center = x[batch.center_index.long()]`. |
| frame readout includes PE-neighbor pooling | Implemented | PE mask uses `segment_type == 0` and masked mean pooling. |
| frame readout includes PP-neighbor pooling | Implemented | PP mask uses `segment_type == 1` and masked mean pooling. |
| frame readout includes all-neighbor pooling | Implemented | `all_pool` pools all non-center nodes. |
| frame readout includes PE/PP contact edge summary | Hardened | Explicit summary uses PE-PP flag, PE-PP persistence, PE-PE fraction, and PP-PP fraction. |
| frame readout includes radial shell composition summary | Hardened | Explicit per-shell PE/PP neighbor ratios are computed from center edges and radial shell ids. |
| temporal encoder uses GRU over `[t-L, t]` | Implemented | Frame embeddings are stacked as `[B, history_len, H]` and passed to GRU. |
| condition encoder handles six global conditions | Implemented | `condition_dim=6` encodes density, temperature, composition, and chain lengths. |
| SPIB bottleneck outputs `mu`, `logvar`, `z` | Implemented | Forward returns all three tensors. |
| KL loss enters training loss | Implemented | `spib_loss()` adds `beta_kl * KL`. |
| three future-state heads exist | Implemented | Separate mobility, relaxation, and contact heads return logits. |
| system pooling is more than `mean(z)` | Hardened | Representation includes z moments, PE/PP/interface means, state fractions, entropies, PE/PP histograms, and contact fractions. |
| baselines perform real ablations | Implemented | `condition_only`, `static_graph_only`, `shuffled_history`, `no_composition_edges`, and checkpoint-only `full_graph_spib` run. |
| descriptor table includes physical descriptors | Hardened | `symbolic/descriptor_table.py` exports z summaries, conditions, proxy descriptors, contact fractions, composition histograms, and transport targets. |

## Issues Found and Fixed

- `radial shell composition summary` was previously represented only indirectly through shell/contact summary values. It is now an explicit per-shell PE/PP neighbor ratio summary.
- system-level pooling previously included local PE fraction histogram but not local PP fraction histogram. It now includes both.
- descriptor table construction previously lived directly in `scripts/export_descriptor_table.py`. It is now implemented in `src/pepp_graph_spib/symbolic/descriptor_table.py` and called by the export script.
- architecture-sensitive tests were previously concentrated in one integration test. They are now split into focused tests for edge attributes, PE/PP pooling, temporal order, KL loss, system pooling, descriptor table columns, and LASSO.
- `minimum_image()` previously used `np.round`, which triggered a native illegal-instruction crash in this WSL/NumPy environment after `check_env.py`. It now uses an equivalent `floor(x + 0.5)` minimum-image calculation.

## Current GraphSPIB Forward Structure

`local graph sequence -> GINE graph frame encoder -> structured frame readout -> GRU temporal encoder -> condition encoder -> SPIB bottleneck -> mobility/relaxation/contact heads`

The structured frame readout concatenates:

1. center node embedding
2. PE-neighbor pooled embedding
3. PP-neighbor pooled embedding
4. all-neighbor pooled embedding
5. PE/PP contact edge summary
6. radial shell PE/PP composition summary

## System Representation

For `z_dim=2` and five histogram bins, `system_repr_dim = 31`.

Composition:

- `mean(z)`: 2
- `var(z)`: 2
- PE-centered `mean(z)`: 2
- PP-centered `mean(z)`: 2
- interface-centered `mean(z)`: 2
- state/contact/entropy/contact-fraction/scoped slow fractions: 11
- local PE fraction histogram: 5
- local PP fraction histogram: 5

## Descriptor Table

The regenerated descriptor table has 40 columns. Core columns include:

`system_id`, `z1`, `z2`, `mean_z1`, `mean_z2`, `var_z1`, `var_z2`, `density`, `temperature`, `PE_fraction`, `PP_fraction`, `PE_chain_length`, `PP_chain_length`, `mean_local_density`, `mean_free_volume_proxy`, `mean_PEPP_contact_fraction`, `mean_PEPP_contact_persistence`, `mean_displacement_norm`, `mean_dihedral_transition_proxy`, `fraction_fast`, `fraction_slow`, `fraction_persistent_contact`, `PE_PE_contact_fraction`, `PP_PP_contact_fraction`, `PE_PP_contact_fraction`, local PE/PP histogram bins, and transport targets.

LASSO targets remain limited to `z1`, `z2`, `log_D`, and `log_tau_relax`. PySR and SISSO are not included.

## Baseline Validation

`outputs/logs/baseline_metrics.csv` was regenerated with 25 rows: five baseline names times five transport targets.

Baseline names:

- `condition_only`
- `static_graph_only`
- `shuffled_history`
- `no_composition_edges`
- `full_graph_spib`

`full_graph_spib` loads trained Graph-SPIB and transport head checkpoints for evaluation and does not retrain the main model inside the baseline script.

## Validation Commands

The full hard validation sequence passed with `set -e` enabled:

```bash
python scripts/check_env.py
python scripts/make_dummy_graph_data.py --config configs/default.yaml --output data/dummy/dummy_graph_windows.pt
python scripts/train_graph_spib.py --config configs/default.yaml
python scripts/train_transport_head.py --config configs/default.yaml --checkpoint outputs/checkpoints/graph_spib_best.pt
python scripts/run_baselines.py --config configs/default.yaml --spib-checkpoint outputs/checkpoints/graph_spib_best.pt --transport-checkpoint outputs/checkpoints/transport_head_best.pt
python scripts/export_descriptor_table.py --config configs/default.yaml --spib-checkpoint outputs/checkpoints/graph_spib_best.pt --transport-checkpoint outputs/checkpoints/transport_head_best.pt --output outputs/embeddings/descriptor_table.csv
python scripts/run_symbolic_lasso.py --config configs/default.yaml --descriptor-table outputs/embeddings/descriptor_table.csv
pytest -q
```

Final pytest result: `9 passed, 2 warnings`.

All required outputs were regenerated locally and remain excluded from Git tracking.
