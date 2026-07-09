#!/usr/bin/env python
"""Compatibility wrapper: Graph-SPIB has been refactored to LD-TDN."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_local_descriptor.py")), run_name="__main__")
