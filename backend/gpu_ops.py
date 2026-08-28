"""
GPU-accelerated distance matrix and K-Medoids helpers.

Results are always returned as NumPy ``float64`` arrays so the rest of the
pipeline (size balancing, invariants) stays unchanged.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from .compute_backend import ResolvedBackend
from .kmedoids import kmedoids_pam


def compute_manhattan_distance_matrix(
    feature_matrix: np.ndarray,
    backend: ResolvedBackend,
) -> tuple[np.ndarray, float]:
    """
    Manhattan (L1) pairwise distance matrix.

    Returns ``(matrix, elapsed_ms)``.
    """
    if backend.name == "cpu" or backend.torch_device is None:
        from sklearn.metrics import pairwise_distances

        t0 = time.perf_counter()
        matrix = pairwise_distances(feature_matrix, metric="manhattan")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return np.asarray(matrix, dtype=np.float64), elapsed_ms

    import torch

    t0 = time.perf_counter()
    features = torch.as_tensor(feature_matrix, dtype=torch.float64, device=backend.torch_device)
    dist = torch.cdist(features, features, p=1)
    matrix = dist.detach().cpu().numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return np.asarray(matrix, dtype=np.float64), elapsed_ms


def run_kmedoids(
    distance_matrix: np.ndarray,
    n_groups: int,
    *,
    random_state: int,
    backend: ResolvedBackend,
) -> tuple[np.ndarray, float]:
    """
    K-Medoids clustering on a precomputed distance matrix.

    GPU path keeps the matrix on device for assignment/cost hot loops.
    Returns ``(labels, elapsed_ms)``.
    """
    if backend.name == "cpu" or backend.torch_device is None:
        t0 = time.perf_counter()
        labels, _ = kmedoids_pam(distance_matrix, n_groups, random_state=random_state)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return labels, elapsed_ms

    t0 = time.perf_counter()
    labels = _kmedoids_pam_torch(
        distance_matrix,
        n_groups,
        device=backend.torch_device,
        random_state=random_state,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return labels, elapsed_ms


def _kmedoids_pam_torch(
    distance_matrix: np.ndarray,
    k: int,
    *,
    device: object,
    random_state: int,
    max_iter: int = 200,
) -> np.ndarray:
    """PAM K-Medoids with distance lookups on a torch device."""
    import torch

    dist = torch.as_tensor(distance_matrix, dtype=torch.float64, device=device)
    n = dist.shape[0]
    if k <= 0 or k > n:
        raise ValueError(f"K must be between 1 and {n}")

    rng = np.random.default_rng(random_state)
    medoid_indices = rng.choice(n, k, replace=False).tolist()

    def assign_labels(medoids: list[int]) -> torch.Tensor:
        medoid_idx = torch.tensor(medoids, device=device, dtype=torch.long)
        d = dist[:, medoid_idx]
        return torch.argmin(d, dim=1)

    def total_cost(labels: torch.Tensor, medoids: list[int]) -> float:
        medoid_idx = torch.tensor(medoids, device=device, dtype=torch.long)
        per_point = dist[torch.arange(n, device=device), medoid_idx[labels]]
        return float(per_point.sum().detach().cpu().item())

    labels = assign_labels(medoid_indices)
    current_cost = total_cost(labels, medoid_indices)

    for _ in range(max_iter):
        improved = False
        for i, _current in enumerate(medoid_indices):
            for candidate in range(n):
                if candidate in medoid_indices:
                    continue
                trial = medoid_indices.copy()
                trial[i] = candidate
                trial_labels = assign_labels(trial)
                trial_cost = total_cost(trial_labels, trial)
                if trial_cost < current_cost:
                    medoid_indices = trial
                    labels = trial_labels
                    current_cost = trial_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return labels.detach().cpu().numpy().astype(np.int64)
