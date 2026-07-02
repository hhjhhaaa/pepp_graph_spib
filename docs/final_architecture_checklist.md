# Final Architecture Checklist

1. PE/PP neighbor pooling excludes center node: confirmed in `GraphSPIB.encode_frame()` via `neighbor = ~is_center`, then `pe_mask = neighbor & (segment_type == 0)` and `pp_mask = neighbor & (segment_type == 1)`.
2. `center_index` after PyG batching is tested by `test_pyg_batch_center_index_points_to_center_nodes`.
3. PE/PP contact edge summary is per graph: `edge_graph = batch.batch[edge_index[0]]`, followed by graph-wise pooling to shape `[B, summary_dim]`.
4. Radial shell summary handles both edge directions: `center_edge = center_mask[src] ^ center_mask[dst]`, and `neighbor_node = torch.where(center_mask[src], dst, src)`. Local graph generation also emits directed edges for all retained pairs.
5. Minimum-image convention is orthorhombic-only in v0.1.x, documented in `features/pbc.py`.
6. `baseline_metrics.csv` contains both baselines and input ablations. `condition_only` is a trained baseline; `static_graph_only`, `shuffled_history`, and `no_composition_edges` are input ablations/stress tests.
7. Git ignores data and output artifacts; tracked files contain no checkpoints, descriptors, figures, logs, or dummy datasets.
