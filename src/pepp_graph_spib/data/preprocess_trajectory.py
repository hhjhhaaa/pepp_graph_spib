"""Real MLFF-MD preprocessing entry points for LD-TDN.

The previous Graph-SPIB preprocessor emitted the old graph-window schema and is
intentionally disabled. Real data must provide LD-TDN condition variables,
history-only descriptor sequences, future-only local labels, system targets,
and target masks.
"""

from __future__ import annotations

from pathlib import Path


REAL_PREPROCESSING_TODOS = [
    "Read PE/PP/PS chain metadata, chain lengths, repeat units, and composition.",
    "Compute history-only segment descriptor sequences from local MD/MLFF-MD windows.",
    "Optionally build small local ego-graph sequences; never full-box graphs.",
    "Compute pore geometry, radial/axial bins, silanol density, and wall chemistry.",
    "Derive local labels only from future windows.",
    "Populate masked system targets for partially available transport labels.",
]


def preprocess_to_graph_windows(*args, **kwargs) -> Path:
    """Deprecated name kept only to fail clearly for old scripts."""
    raise NotImplementedError(
        "The legacy Graph-SPIB preprocessor was removed. Implement an LD-TDN "
        "metadata adapter that emits LocalWindowSample objects. TODOs: "
        + "; ".join(REAL_PREPROCESSING_TODOS)
    )


def preprocess_to_local_windows(*args, **kwargs) -> Path:
    """Placeholder for the real LD-TDN preprocessing adapter."""
    raise NotImplementedError(
        "Real MLFF-MD LD-TDN preprocessing is not implemented yet. TODOs: "
        + "; ".join(REAL_PREPROCESSING_TODOS)
    )
