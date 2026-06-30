# Data Schema

## GraphWindowSample

`GraphWindowSample(system_id, center_segment_id, center_segment_type, graph_sequence, condition, future_labels, metadata)` is the training sample for the first version.

## Graph Sequence

`graph_sequence` is a list of PyG `Data` objects built from the historical window `[t-L, t]`. Future frames are used only to create labels.

## Node Features

Node features include PE/PP one-hot identity, center flag, same-chain flag, relative segment index, displacement vector, displacement norm, local density, local PE/PP fraction, free-volume proxy, torsion proxy, and recent mobility proxy.

## Edge Features

Edge features include distance, inverse distance, radial basis encoding, radial shell id, same-chain bonded edge, same-chain nonbonded edge, inter-chain contact edge, PE-PE contact, PP-PP contact, PE-PP contact, and contact persistence proxy.

## Metadata

Per-window metadata includes local PE/PP fraction, PE-PE/PP-PP/PE-PP contact fraction, mean local density, mean free-volume proxy, mean displacement norm, mean dihedral transition proxy, and environment type.

## Future Labels

Future labels are local dynamic-state classes: mobility state, relaxation state, and PE/PP contact persistence state. They are derived from `[t, t+tau]` and are not written into graph features.

## System-Level Targets

Transport targets are `log_D`, `log_tau_relax`, `log_D_eff`, `tau_res`, and `P_access`. They are predicted from system-level pooled local embedding distributions plus system conditions.

## Real Trajectory Metadata Adapter

A real trajectory metadata YAML must provide or construct `segment_type`, `chain_id`, `segment_index`, PE/PP identity, chain lengths, system condition, and optional system-level property targets. MDAnalysis reads coordinates and boxes; it is not assumed to infer PE/PP polymer semantics automatically.
