# LD-TDN: Local Dynamic Transport Descriptor Network

This repository implements a lightweight **Local Dynamic Transport Descriptor Network (LD-TDN)** for ps-scale local polymer trajectory windows.

The model is deliberately **not** a full-system atomistic GNN. It has one maintained mainline:

```text
short local trajectory window
→ descriptor feature_sequence [T, F] + local graph_sequence + explicit conditions
→ descriptor temporal encoder + local GNN temporal encoder + condition encoder
→ fused dual-stream state
→ predictive variational bottleneck
→ z_i local dynamic transport descriptor
→ local future-dynamics heads
→ region-wise pooling over systems / pore bins
→ physics-informed transport head
→ descriptor table
→ LASSO descriptor distillation
```

The local GNN encodes only segment-centered ego graphs from the same trajectory window; it never builds a graph over thousands of atoms.

For the full Chinese project pipeline summary, see
[`docs/ld_tdn_pipeline_summary.md`](docs/ld_tdn_pipeline_summary.md).

## Methodological Reference

LD-TDN follows the spirit of Chen et al., "Constructing custom thermodynamics using deep learning", *Nature Computational Science* 4, 66-85 (2024), DOI: `10.1038/s43588-023-00581-5`.

The useful lesson is not to make a larger black-box model. The paper learns reduced thermodynamic coordinates and a structured stochastic macroscopic dynamics from microscopic trajectory observations, guided by a generalized Onsager principle. LD-TDN adapts that idea as:

- learned local closure coordinates: `z_i`
- explicit controllable physical conditions: chain length, repeat units, composition, density, temperature, pore geometry, and wall chemistry
- predictive short-future losses so `z_i` captures local dynamic transport state
- positive/bounded physical transport factors instead of an unconstrained property MLP
- exported descriptors and factors for sparse regression and interpretation

## Explicit Conditions

Chain length and repeat-unit counts are first-class model inputs. They are not expected to be inferred from local graphs.

Configured condition variables:

```text
density
temperature
PE_fraction
PP_fraction
PS_fraction
PE_chain_length
PP_chain_length
PS_chain_length
PE_repeat_units
PP_repeat_units
PS_repeat_units
mean_chain_length
chain_length_polydispersity
pore_diameter
pore_length
silanol_density
wall_type_id
surface_hydroxylation_fraction
```

## Local Labels And System Targets

Local classification heads:

```text
mobility_class
contact_class
residence_class
escape_class
relax_class
```

Local regression heads:

```text
future_disp_parallel
future_disp_radial
future_disp_norm
short_msd_parallel
short_msd_radial
contact_survival
wall_contact_survival
free_volume_opening
```

System / pore targets use masks so real datasets may omit some labels:

```text
log_D_self
log_D_parallel
log_D_eff
log_tau_segmental
log_tau_res
P_access
```

## Physics Transport Head

The physics head predicts:

```text
D_local > 0
P_entry in [0, 1]
C_axis in [0, 1]
tau_wall > 0
tau_move > 0
P_access in [0, 1]
```

Then derives:

```text
transport_score = P_entry * C_axis / (1 + tau_wall / tau_move)
D_eff = D_local * transport_score
```

`tau_wall / tau_move` is treated as a transport drag term. LD-TDN is a
pore-confined transport descriptor model, not a reactivity model.

## Smoke Workflow

Dummy v1 is PE/PP-only. PS condition fields are schema placeholders and are set
to zero until real PS preprocessing is implemented.

```bash
python scripts/make_dummy_local_windows.py --config configs/main.yaml --tiny
python scripts/train_local_descriptor.py --config configs/main.yaml --max-epochs 1
python scripts/train_transport_head.py --config configs/main.yaml --max-epochs 1 --local-checkpoint outputs/checkpoints/local_descriptor_best.pt
python scripts/export_descriptor_table.py --config configs/main.yaml --output outputs/embeddings/descriptor_table.csv
python scripts/run_symbolic_lasso.py --config configs/main.yaml --descriptor-table outputs/embeddings/descriptor_table.csv
```

If the package is not installed editable in the active environment, run with `PYTHONPATH=src`.

## Main Config

`configs/main.yaml` is the single maintained LD-TDN configuration. It defines the dual-stream local graph + descriptor + condition architecture, region pooling, and physics-informed transport head.

## Repository Tracking Policy

Source, the main config, scripts, and docs are tracked. Generated data, checkpoints, logs, figures, and descriptor tables are excluded by `.gitignore`.

## Real MLFF-MD Preprocessing TODOs

- Implement metadata adapters for PE/PP/PS composition, chain IDs, chain lengths, repeat units, and molecular-weight descriptors.
- Compute measured pore geometry and wall chemistry fields instead of dummy proxies.
- Compute local descriptor sequences from trajectory history only.
- Derive future local labels strictly from future windows.
- Populate target masks for partially available transport labels.
- Validate physical units and normalization before comparing systems.
