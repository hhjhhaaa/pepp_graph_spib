# Dummy Validation Report

Project: PE/PP Local Multi-Scale Graph-SPIB  
Path: `/home/jinhao/mlff/pepp_graph_spib`  
Version target: `v0.1.0-dummy-validated`

## Environment

- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 5090
- `torch.cuda.is_available()`: True
- PyG: installed
- MDAnalysis: installed
- freud: installed
- ripser: installed

## Validation

The following commands passed:

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

## Required Outputs

All required outputs were generated locally and are intentionally excluded from Git tracking.

## Note

`max_neighbors=64` is an upper bound. The actual number of nodes in dummy local graphs may be smaller depending on available local nodes. This is intentional and consistent with variable-size segment-centered local dynamic graph windows.
