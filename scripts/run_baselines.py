#!/usr/bin/env python
"""Compatibility wrapper for LD-TDN ablations."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_ablation.py")), run_name="__main__")
