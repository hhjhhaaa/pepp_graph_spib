# PE/PP Local Multi-Scale Graph-SPIB

This project implements one first-version architecture for PE/PP blend trajectory learning:

`local dynamic graph windows -> graph encoder -> GRU temporal encoder -> SPIB bottleneck -> future dynamic-state prediction -> system distribution pooling -> transport-property prediction -> LASSO descriptor distillation`

The project root is `/home/jinhao/mlff/pepp_graph_spib`.

## Why local multi-scale Graph-SPIB

The task is to learn mobility embeddings from short high-resolution PE/PP trajectories. A small graph is not a proxy for the whole PE/PP system. Each local graph is a sampling unit for local segment mobility, packing, free volume, PE/PP contact, and local relaxation. System-level diffusion, relaxation time, effective pore diffusion, residence time, and accessibility are predicted only after pooling the distribution of many local embeddings by `system_id`.

## Input definition

Each sample is a `GraphWindowSample(system_id, center_segment_id, center_segment_type, graph_sequence, condition, future_labels, metadata)`. The graph sequence comes from `[t-L, t]`; future labels come from `[t, t+tau]`.

Node and edge proxy features are computed only from history. Future displacement, future contact persistence, and future relaxation are never written into graph features.

Neighbor construction uses NumPy/Torch distance calculations with minimum image convention when a box is available. The project does not use PyG `knn_graph`, `radius_graph`, `NeighborLoader`, `ClusterLoader`, `SparseTensor`, or advanced compiled sampling extensions.

## Why not single-chain or full raw-box graph

A single chain would remove the PE/PP mixed local environment and contact history. A full simulation-box raw graph would make the model learn system size and sampling artifacts while adding unnecessary graph-sampling complexity. Segment-centered local dynamic windows keep the model focused on transferable local dynamics.

## Why not polyBERT, TransPolymer, or pretrained Graph-SPIB

The first PE/PP task has only two polymer identities, so one-hot PE/PP identity, local composition, contact graph features, and global condition variables are sufficient. Polymer language models may be useful later for many-chemistry extensions, but they are not part of the first architecture.

There is no general pretrained Graph-SPIB model for PE/PP local multi-chain trajectory windows. SimPoly/MLFF provides trajectories, not downstream embeddings, so this model is trained from project data.

## Why not PySR/SISSO in v1

The first version uses LASSO sparse regression only. PySR and SISSO are deliberately excluded to keep the environment reproducible and avoid Julia or Fortran/MPI dependencies in the first validation path.

## Environment

Install Miniforge in WSL, then:

```bash
conda create -n pepp-graph-spib python=3.11 -y
conda activate pepp-graph-spib
python -m pip install --upgrade pip setuptools wheel
conda install -y -c conda-forge numpy scipy pandas scikit-learn matplotlib seaborn tqdm pyyaml h5py joblib mdanalysis freud ripser pytest black ruff tensorboard
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install -e .
python scripts/check_env.py
```

For RTX 5090, use the PyTorch official selector and prefer CUDA 12.8 wheels. If the CUDA 12.8 wheel fails, do not install CUDA 12.4 by default; choose the current official supported CUDA wheel.

## Dummy data and training

```bash
python scripts/make_dummy_graph_data.py --config configs/default.yaml --output data/dummy/dummy_graph_windows.pt
python scripts/train_graph_spib.py --config configs/default.yaml
python scripts/train_transport_head.py --config configs/default.yaml --checkpoint outputs/checkpoints/graph_spib_best.pt
python scripts/run_baselines.py --config configs/default.yaml --spib-checkpoint outputs/checkpoints/graph_spib_best.pt --transport-checkpoint outputs/checkpoints/transport_head_best.pt
python scripts/export_descriptor_table.py --config configs/default.yaml --spib-checkpoint outputs/checkpoints/graph_spib_best.pt --transport-checkpoint outputs/checkpoints/transport_head_best.pt --output outputs/embeddings/descriptor_table.csv
python scripts/run_symbolic_lasso.py --config configs/default.yaml --descriptor-table outputs/embeddings/descriptor_table.csv
pytest -q
```

Use `--tiny` for fast dummy generation and `--max-epochs`, `--limit-systems`, `--limit-samples` for smoke tests.

## Real SimPoly/MD trajectories

Real preprocessing has two layers:

1. MDAnalysis reader loads topology, trajectory, coordinates, and box.
2. PE/PP metadata adapter reads `metadata.yaml` for segment type, chain id, segment index, PE/PP identity, chain length, and system condition.

The code does not assume MDAnalysis can infer polymer identity or torsion definitions automatically.

## Repository Tracking Policy

This repository tracks code, configuration, tests, and documentation only. Raw trajectories, processed datasets, checkpoints, logs, figures, and descriptor tables are excluded from Git tracking.

## Baselines and Input Ablations

The reported `baseline_metrics.csv` contains both a trained baseline and input ablation experiments. `condition_only` is a trained baseline using only system conditions. `static_graph_only`, `shuffled_history`, and `no_composition_edges` are input ablations or stress tests for the local multi-scale Graph-SPIB design. `full_graph_spib` loads the trained Graph-SPIB and transport head checkpoints for evaluation.
