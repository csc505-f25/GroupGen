"""
K-Medoids (PAM) on a precomputed distance matrix.

Production uses this with the psychometric Manhattan matrix from
``clustering.compute_psychometric_distance_matrix`` (see ``pipeline.py``).

Also provides ``kmedoids_size_constrained``: capacity-aware assignment so group
sizes are enforced *during* clustering (joint compactness + cardinality), not
only via post-hoc repair.

Properties:
  - Works with non-Euclidean distances (Manhattan/L1)
  - Medoids are real students (interpretable)
  - Default ``random_state=42`` for reproducible studies
"""

from __future__ import annotations

from typing import List, Sequence

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


def capacities_for_target_size(n_students: int, target_size: int) -> List[int]:
    """
    Per-group seat counts matching production invariants.

    Uses ``ceil(n / target_size)`` groups; ``remainder`` groups get ``base+1``,
    the rest get ``base`` (e.g. n=17, target=5 → [5, 4, 4, 4]).
    """
    from .group_config import calculate_n_groups

    n_groups = calculate_n_groups(n_students, target_size)
    base = n_students // n_groups
    remainder = n_students % n_groups
    return [base + 1] * remainder + [base] * (n_groups - remainder)


def assign_labels_with_capacities(
    distance_matrix: np.ndarray,
    medoid_indices: Sequence[int],
    capacities: Sequence[int],
) -> np.ndarray:
    """
    Assign every point to a medoid without exceeding capacities.

    Medoids are locked into their own clusters. Remaining students are assigned
    by min-cost (Hungarian) matching to the remaining seats so total
    distance-to-medoid is minimized for fixed medoids/capacities.
    Falls back to greedy closest-first if SciPy is unavailable.
    """
    D = np.asarray(distance_matrix, dtype=float)
    medoids = [int(m) for m in medoid_indices]
    caps = [int(c) for c in capacities]
    n = D.shape[0]
    k = len(medoids)

    if k != len(caps):
        raise ValueError("capacities length must match number of medoids.")
    if sum(caps) != n:
        raise ValueError(
            f"capacities must sum to n_students ({n}), got {sum(caps)}."
        )
    if len(set(medoids)) != k:
        raise ValueError("medoid_indices must be unique.")
    if min(caps) <= 0:
        raise ValueError("All capacities must be positive.")

    labels = np.full(n, -1, dtype=int)
    remaining = list(caps)
    for cluster_id, medoid in enumerate(medoids):
        labels[medoid] = cluster_id
        remaining[cluster_id] -= 1
        if remaining[cluster_id] < 0:
            raise ValueError(
                f"Capacity for cluster {cluster_id} is too small to hold its medoid."
            )

    free_students = [i for i in range(n) if labels[i] < 0]
    n_free = len(free_students)
    if n_free == 0:
        return labels
    if sum(remaining) != n_free:
        raise ValueError("Internal capacity mismatch after locking medoids.")

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _assign_labels_with_capacities_greedy(D, medoids, caps)

    seat_cluster: list[int] = []
    cost = np.zeros((n_free, n_free), dtype=float)
    col = 0
    for cluster_id, medoid in enumerate(medoids):
        for _ in range(remaining[cluster_id]):
            for row, student in enumerate(free_students):
                cost[row, col] = D[student, medoid]
            seat_cluster.append(cluster_id)
            col += 1

    _row_ind, col_ind = linear_sum_assignment(cost)
    for row, col_i in enumerate(col_ind):
        labels[free_students[row]] = seat_cluster[int(col_i)]

    if np.any(labels < 0):
        raise ValueError("Capacity assignment left unassigned students.")
    return labels


def _assign_labels_with_capacities_greedy(
    D: np.ndarray, medoids: list[int], capacities: list[int]
) -> np.ndarray:
    """Closest-first greedy fallback when SciPy is unavailable."""
    remaining = list(capacities)
    n = D.shape[0]
    labels = np.full(n, -1, dtype=int)

    for cluster_id, medoid in enumerate(medoids):
        labels[medoid] = cluster_id
        remaining[cluster_id] -= 1

    medoid_arr = np.asarray(medoids, dtype=int)
    unassigned = [i for i in range(n) if labels[i] < 0]
    unassigned.sort(key=lambda i: (float(D[i, medoid_arr].min()), i))

    for i in unassigned:
        order = np.argsort(D[i, medoid_arr], kind="mergesort")
        placed = False
        for cluster_id in order:
            cid = int(cluster_id)
            if remaining[cid] > 0:
                labels[i] = cid
                remaining[cid] -= 1
                placed = True
                break
        if not placed:
            raise ValueError("Capacity assignment failed: no free seats left.")

    if np.any(labels < 0):
        raise ValueError("Capacity assignment left unassigned students.")
    return labels


def _pam_cost(
    distance_matrix: np.ndarray, labels: np.ndarray, medoid_indices: Sequence[int]
) -> float:
    medoid_arr = np.asarray(medoid_indices, dtype=int)
    n = len(labels)
    return float(distance_matrix[np.arange(n), medoid_arr[labels]].sum())


def _update_medoids_from_labels(
    distance_matrix: np.ndarray, labels: np.ndarray, n_groups: int
) -> List[int]:
    """Geometric medoid of each cluster (argmin within-cluster distance sum)."""
    medoids: List[int] = []
    for g in range(n_groups):
        members = np.where(labels == g)[0]
        if members.size == 0:
            raise ValueError(f"Cluster {g} is empty during medoid update.")
        if members.size == 1:
            medoids.append(int(members[0]))
            continue
        sub = distance_matrix[np.ix_(members, members)]
        medoids.append(int(members[int(np.argmin(sub.sum(axis=1)))]))
    return medoids


def kmedoids_size_constrained(
    distance_matrix: np.ndarray,
    capacities: Sequence[int],
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_iter: int = 200,
    max_assign_rounds: int = 50,
    warm_start_pam: bool = True,
):
    """
    Size-constrained K-Medoids under a fixed distance matrix (e.g. Manhattan).

    Jointly enforces cardinality and compactness:
      1. Initialize medoids (optionally from unconstrained PAM warm-start)
      2. Capacity-aware assignment to medoids
      3. Update each cluster's medoid
      4. PAM-style medoid swaps scored with capacity reassignment

    Returns
    -------
    labels : ndarray
        Cluster id per student (0..K-1), sizes matching ``capacities``.
    medoid_indices : ndarray
        Final medoid row indices.
    """
    D = np.asarray(distance_matrix, dtype=float)
    caps = [int(c) for c in capacities]
    n = D.shape[0]
    k = len(caps)

    if D.shape != (n, n):
        raise ValueError("distance_matrix must be square.")
    if k <= 0 or k > n:
        raise ValueError(f"K must be between 1 and {n}")
    if sum(caps) != n:
        raise ValueError(f"capacities must sum to {n}, got {sum(caps)}.")
    if min(caps) <= 0:
        raise ValueError("All capacities must be positive.")

    rng = np.random.default_rng(random_state)

    if warm_start_pam:
        # Unconstrained PAM finds compact medoids; capacities are applied next.
        _labels_pam, medoids_pam = kmedoids_pam(
            D, k, random_state=random_state, max_iter=max_iter
        )
        medoid_indices = [int(m) for m in medoids_pam]
    else:
        medoid_indices = rng.choice(n, k, replace=False).tolist()

    labels = assign_labels_with_capacities(D, medoid_indices, caps)
    medoid_indices = _update_medoids_from_labels(D, labels, k)
    labels = assign_labels_with_capacities(D, medoid_indices, caps)
    current_cost = _pam_cost(D, labels, medoid_indices)

    # Alternate assign ↔ medoid update until stable (keeps sizes exact).
    for _ in range(max_assign_rounds):
        new_medoids = _update_medoids_from_labels(D, labels, k)
        if new_medoids == medoid_indices:
            break
        medoid_indices = new_medoids
        labels = assign_labels_with_capacities(D, medoid_indices, caps)
        current_cost = _pam_cost(D, labels, medoid_indices)

    # PAM swaps: only accept if capacity reassignment lowers total medoid cost.
    for _ in range(max_iter):
        improved = False
        for slot, _current in enumerate(medoid_indices):
            for candidate in range(n):
                if candidate in medoid_indices:
                    continue
                trial_medoids = medoid_indices.copy()
                trial_medoids[slot] = candidate
                try:
                    trial_labels = assign_labels_with_capacities(D, trial_medoids, caps)
                except ValueError:
                    continue
                # Refresh medoids inside trial clusters, then re-assign once for cost.
                trial_medoids = _update_medoids_from_labels(D, trial_labels, k)
                if len(set(trial_medoids)) != k:
                    continue
                try:
                    trial_labels = assign_labels_with_capacities(D, trial_medoids, caps)
                except ValueError:
                    continue
                trial_cost = _pam_cost(D, trial_labels, trial_medoids)
                if trial_cost + 1e-12 < current_cost:
                    medoid_indices = trial_medoids
                    labels = trial_labels
                    current_cost = trial_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return labels, np.asarray(medoid_indices, dtype=int)
