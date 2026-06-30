#!/usr/bin/env python
"""Generate dummy local dynamic graph windows."""

from __future__ import annotations

import argparse
import faulthandler

faulthandler.enable()

from pepp_graph_spib.data.dummy import generate_dummy_dataset
from pepp_graph_spib.utils import ensure_dirs, load_config, resolve_path, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    ensure_dirs(cfg)
    output = args.output or resolve_path(cfg, "dummy_graph_path")
    graph_path, target_path = generate_dummy_dataset(cfg, output, tiny=args.tiny)
    print(f"wrote graph windows: {graph_path}")
    print(f"wrote system targets: {target_path}")


if __name__ == "__main__":
    main()
