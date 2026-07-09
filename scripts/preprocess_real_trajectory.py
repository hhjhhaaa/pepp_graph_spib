#!/usr/bin/env python
"""Preprocess real SimPoly/MD trajectories into LD-TDN local windows."""

from __future__ import annotations

import argparse

from pepp_graph_spib.data.preprocess_trajectory import preprocess_to_local_windows
from pepp_graph_spib.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--future-tau", type=int, default=None)
    parser.add_argument("--r-cut-nm", type=float, default=None)
    parser.add_argument("--max-neighbors", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-centers", type=int, default=None)
    parser.add_argument("--segment-scheme", default="bead")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    output = preprocess_to_local_windows(
        topology_path=args.topology,
        trajectory_path=args.trajectory,
        metadata_path=args.metadata,
        output_path=args.output,
        history_len=args.history_len or int(data["history_len"]),
        future_tau=args.future_tau or int(data["future_tau"]),
        r_cut=args.r_cut_nm or float(data["r_cut_nm"]),
        max_neighbors=args.max_neighbors or int(data["max_neighbors"]),
        stride=args.stride or int(data.get("stride", 1)),
        segment_scheme=args.segment_scheme,
        max_centers=args.max_centers,
        shell_edges=data.get("radial_shell_edges_nm", [0.6, 1.2, 2.0, 3.0]),
    )
    print(f"wrote local windows: {output}")


if __name__ == "__main__":
    main()
