"""
Intra-team cohesion metrics for research export.

Cohesion is the mean pairwise Manhattan (L1) distance among members of a team
in the production feature space. Lower values indicate more similar profiles.
"""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np


def mean_pairwise_l1_within_indices(
    distance_matrix: np.ndarray, member_indices: np.ndarray
) -> float:
    """
    Mean upper-triangle L1 distance for members of one cluster.

    Returns 0.0 for singleton clusters (no pairs to compare).
    """
    idx = np.asarray(member_indices, dtype=int)
    if idx.size < 2:
        return 0.0
    sub = distance_matrix[np.ix_(idx, idx)]
    triu = sub[np.triu_indices(len(idx), k=1)]
    if triu.size == 0:
        return 0.0
    return float(np.mean(triu))


def compute_team_cohesion(
    distance_matrix: np.ndarray, labels: np.ndarray
) -> Dict[int, float]:
    """
    Per-cluster mean pairwise L1 cohesion.

    Returns a dict keyed by cluster label (0-based) → mean pairwise distance.
    """
    D = np.asarray(distance_matrix, dtype=float)
    labels = np.asarray(labels)
    out: Dict[int, float] = {}
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        out[int(g)] = mean_pairwise_l1_within_indices(D, idx)
    return out


def assert_distance_matrix_properties(
    distance_matrix: np.ndarray, *, atol: float = 1e-9
) -> None:
    """Verify symmetry, non-negativity, and zero diagonal for a distance matrix."""
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if not np.allclose(D, D.T, atol=atol, equal_nan=False):
        raise ValueError("Distance matrix is not symmetric.")
    if np.any(D < -atol):
        raise ValueError("Distance matrix has negative entries.")
    if not np.allclose(np.diag(D), 0.0, atol=atol):
        raise ValueError("Distance matrix diagonal is not zero.")


def stable_student_id(name: str, *, salt: str = "groupgen-v1") -> str:
    """Deterministic anonymized student identifier from display name."""
    normalized = str(name).strip().lower()
    payload = f"{salt}:{normalized}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()
