"""Real MLFF-MD preprocessing entry points for LD-TDN.

Real data must provide LD-TDN condition variables, history-only descriptor
sequences, local ego-graph sequences, future-only local labels, system targets,
and target masks.
"""

from __future__ import annotations

from pathlib import Path


REAL_PREPROCESSING_TODOS = [
    "Read PE/PP/PS chain metadata, chain lengths, repeat units, and composition.",
    "Compute history-only segment descriptor sequences from local MD/MLFF-MD windows.",
    "Build small local ego-graph sequences; never full-box graphs.",
    "Compute pore geometry, radial/axial bins, silanol density, and wall chemistry.",
    "Derive local labels only from future windows.",
    "Populate masked system targets for partially available transport labels.",
]


def preprocess_to_local_windows(*args, **kwargs) -> Path:
    """Placeholder for the real LD-TDN preprocessing adapter."""
    raise NotImplementedError(
        "Real MLFF-MD LD-TDN preprocessing is not implemented yet. TODOs: "
        + "; ".join(REAL_PREPROCESSING_TODOS)
    )
