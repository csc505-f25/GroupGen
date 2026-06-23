"""
K-Medoids (PAM) on a precomputed distance matrix.

Production uses this with the psychometric Manhattan matrix from
``clustering.compute_psychometric_distance_matrix`` (see ``pipeline.py``).

Properties:
  - Works with non-Euclidean distances (Manhattan/L1)
  - Medoids are real students (interpretable)
  - Default ``random_state=42`` for reproducible studies
"""

import numpy as np

DEFAULT_RANDOM_STATE = 42


def kmedoids_pam(
    X: np.ndarray,
    K: int,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_iter: int = 200,
):
    """
    Simple PAM (Partitioning Around Medoids) implementation.

    Args:
        X: NxN distance matrix. X[i, j] = distance between student i and j.
        K: Number of medoids/clusters
        random_state: Seed for reproducible medoid initialization (default 42)
        max_iter: Maximum number of swap iterations

    Returns:
        labels: Length-N array of cluster labels (0 to K-1)
        medoid_indices: Length-K array of indices of chosen medoid points
    """
    rng = np.random.default_rng(random_state)

    N = X.shape[0]
    if K <= 0 or K > N:
        raise ValueError(f"K must be between 1 and {N}")

    # Pick K students as initial medoids (seeded RNG → reproducible studies).
    medoid_indices = rng.choice(N, K, replace=False).tolist()

    def assign_labels_to_medoids(medoid_list):
        # Each student joins the cluster whose medoid is closest in the distance matrix.
        medoid_arr = np.array(medoid_list)
        distances_to_medoids = X[:, medoid_arr]
        return np.argmin(distances_to_medoids, axis=1)

    def compute_cost(labels, medoid_list):
        # Total distance from each student to their cluster medoid (PAM objective).
        medoid_arr = np.array(medoid_list)
        return float(X[np.arange(N), medoid_arr[labels]].sum())

    labels = assign_labels_to_medoids(medoid_indices)
    current_cost = compute_cost(labels, medoid_indices)

    # SWAP phase: try replacing one medoid at a time; stop when no improvement.
    for _ in range(max_iter):
        improved = False

        for i, _current_medoid in enumerate(medoid_indices):
            for candidate in range(N):
                if candidate in medoid_indices:
                    continue  # medoids must be unique

                candidate_medoids = medoid_indices.copy()
                candidate_medoids[i] = candidate

                candidate_labels = assign_labels_to_medoids(candidate_medoids)
                candidate_cost = compute_cost(candidate_labels, candidate_medoids)

                if candidate_cost < current_cost:
                    medoid_indices = candidate_medoids
                    labels = candidate_labels
                    current_cost = candidate_cost
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return labels, np.array(medoid_indices)
