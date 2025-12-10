"""
K-Medoids (PAM) Clustering Module

Implements the PAM (Partitioning Around Medoids) algorithm for clustering.
K-Medoids is useful for:
- Non-Euclidean distance metrics (e.g., Manhattan/L1)
- Robustness to outliers
- Interpretability (medoids are actual data points)

This module is separate from clustering.py to keep original K-Means implementation intact.
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.metrics import pairwise_distances


def kmedoids_pam(
    X: np.ndarray,
    K: int,
    random_state: Optional[int] = None,
    max_iter: int = 200
):
    """
    Simple PAM (Partitioning Around Medoids) implementation.

    K-Medoids clusters data by finding K representative points (medoids) that
    minimize the total within-cluster distance. Unlike K-Means, medoids are
    actual data points, making results more interpretable.

    Args:
        X: NxM feature matrix (numeric)
        K: Number of medoids/clusters
        distance_metric: Distance metric for pairwise_distances
                        ('manhattan', 'euclidean', 'cosine', etc.)
        random_state: Seed for reproducibility
        max_iter: Maximum number of swap iterations

    Returns:
        labels: Length-N array of cluster labels (0 to K-1)
        medoid_indices: Length-K array of indices of chosen medoid points
    """
    if random_state is not None:
        np.random.seed(random_state)

    N = X.shape[0]
    if K <= 0 or K > N:
        raise ValueError(f"K must be between 1 and {N}")

    # Precompute full distance matrix (PAM uses it extensively)
    # D = pairwise_distances(X, metric=distance_metric)

    # Initialize medoids: random unique indices
    medoid_indices = np.random.choice(N, K, replace=False).tolist()

    def assign_labels_to_medoids(medoid_list):
        """Assign each point to the nearest medoid."""
        medoid_arr = np.array(medoid_list)
        distances_to_medoids = X[:, medoid_arr]  # shape (N, K)
        labels = np.argmin(distances_to_medoids, axis=1)
        return labels

    def compute_cost(labels, medoid_list):
        """Compute total within-cluster distance (cost)."""
        medoid_arr = np.array(medoid_list)
        cost = X[np.arange(N), medoid_arr[labels]].sum()
        return cost

    # Initial assignment and cost
    labels = assign_labels_to_medoids(medoid_indices)
    current_cost = compute_cost(labels, medoid_indices)

    # PAM swap phase: try improving by swapping medoids with non-medoids
    for iteration in range(max_iter):
        improved = False

        for i, current_medoid in enumerate(medoid_indices):
            for candidate in range(N):
                # Skip if candidate is already a medoid
                if candidate in medoid_indices:
                    continue

                # Try swapping medoid_indices[i] with candidate
                candidate_medoids = medoid_indices.copy()
                candidate_medoids[i] = candidate

                candidate_labels = assign_labels_to_medoids(candidate_medoids)
                candidate_cost = compute_cost(candidate_labels, candidate_medoids)

                # Accept swap if it improves cost
                if candidate_cost < current_cost:
                    medoid_indices = candidate_medoids
                    labels = candidate_labels
                    current_cost = candidate_cost
                    improved = True
                    break  # Accept first improving swap and restart

            if improved:
                break

        # If no improvement found, algorithm has converged
        if not improved:
            break

    return labels, np.array(medoid_indices)
