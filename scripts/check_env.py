#!/usr/bin/env python
"""Check runtime imports for the PE/PP Graph-SPIB project."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    modules = [
        ("torch", "torch"),
        ("torch_geometric", "torch_geometric"),
        ("MDAnalysis", "MDAnalysis"),
        ("freud", "freud"),
        ("ripser", "ripser"),
    ]
    ok = True
    torch_mod = None
    for name, module_name in modules:
        try:
            mod = importlib.import_module(module_name)
            print(f"{name}: OK {getattr(mod, '__version__', '')}")
            if name == "torch":
                torch_mod = mod
        except Exception as exc:
            ok = False
            print(f"{name}: FAIL {exc}")
    if torch_mod is not None:
        cuda_available = torch_mod.cuda.is_available()
        print(f"torch.cuda.is_available(): {cuda_available}")
        print(f"torch.version.cuda: {torch_mod.version.cuda}")
        if cuda_available:
            try:
                gpu_name = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).splitlines()[0]
                print(f"cuda device: {gpu_name}")
            except Exception:
                print("cuda device: available")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
