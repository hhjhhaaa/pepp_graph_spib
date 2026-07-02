"""Distance, neighbor, and periodic-boundary helpers.

All functions use the minimum image convention when a box is provided. Box
lengths are expected in nm and orthorhombic in this first implementation.
"""

from __future__ import annotations

import numpy as np


def minimum_image(displacements: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """Apply minimum image convention for orthorhombic boxes to vectors [..., 3]."""
    if box is None:
        return displacements
    box = np.asarray(box, dtype=np.float32)
    lengths = box[:3] if box.shape[0] >= 3 else box
    lengths = np.where(lengths <= 0, 1.0, lengths)
    scaled = displacements / lengths
    nearest = np.floor(scaled + 0.5).astype(np.float32)
    return displacements - lengths * nearest


def displacement(a: np.ndarray, b: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """Return b - a under PBC, shape broadcastable to [..., 3]."""
    return minimum_image(np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32), box)


def pairwise_displacements(positions: np.ndarray, center: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """Return center-to-position displacement vectors with shape [N, 3]."""
    return displacement(center[None, :], positions, box)


def pairwise_distances(positions: np.ndarray, center: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """Return center-to-position distances with shape [N]."""
    return np.linalg.norm(pairwise_displacements(positions, center, box), axis=-1)


def choose_neighbors(
    positions: np.ndarray,
    center_idx: int,
    r_cut: float,
    max_neighbors: int,
    box: np.ndarray | None,
) -> np.ndarray:
    """Choose center plus cutoff/KNN neighbors using NumPy distances.

    PyG knn/radius helpers are intentionally not used.
    """
    dists = pairwise_distances(positions, positions[center_idx], box)
    candidates = np.where((dists > 1.0e-8) & (dists <= r_cut))[0]
    if len(candidates) < max_neighbors:
        order = np.argsort(dists)
        candidates = np.array([idx for idx in order if idx != center_idx], dtype=np.int64)
    else:
        candidates = candidates[np.argsort(dists[candidates])]
    picked = candidates[:max_neighbors]
    return np.concatenate([[center_idx], picked]).astype(np.int64)


def radial_basis(distance: float, centers: tuple[float, float] = (0.5, 1.5), gamma: float = 2.0) -> np.ndarray:
    """Two Gaussian radial basis values for a scalar distance."""
    centers_arr = np.asarray(centers, dtype=np.float32)
    return np.exp(-gamma * (float(distance) - centers_arr) ** 2).astype(np.float32)


def radial_shell_id(distance: float, shell_edges: list[float] | tuple[float, ...]) -> float:
    """Return normalized shell id in [0, 1] for fixed radial shells."""
    shell = int(np.searchsorted(np.asarray(shell_edges, dtype=np.float32), distance, side="right"))
    denom = max(len(shell_edges), 1)
    return float(min(shell, len(shell_edges)) / denom)
