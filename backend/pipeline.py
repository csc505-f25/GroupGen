"""
Shared grouping pipeline for CLI, API, and evaluation scripts.

Lifecycle (strict order):
  1. Ingest/validate (caller: prepare_for_grouping)
  2. Group count calculation
  3. Psychometric features + Manhattan distance (no Gender/Diversity)
  4. Dual size strategies (post-hoc repair vs size-constrained K-Medoids)
  5. Keep the lower PAM-cost feasible labeling (classroom cohesion guardrail)
  6. Final invariant assertion

Demographic fields (Gender, Diversity) remain in the roster for display and
downstream reporting but are never used for distance, clustering, or post-hoc
rebalancing on this branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .clustering import compute_feature_vector, enforce_group_size
from .cohesion import compute_team_cohesion
from .compute_backend import resolve_backend
from .gpu_ops import compute_manhattan_distance_matrix, run_kmedoids
from .group_config import calculate_n_groups
from .invariants import assert_assignment_invariants
from .kmedoids import capacities_for_target_size, kmedoids_size_constrained

# Fixed seed so API and CLI produce identical groups for the same CSV.
DEFAULT_RANDOM_STATE = 42


@dataclass
class GroupingResult:
    """Cluster labels and pipeline metadata for API/CLI."""

    labels: np.ndarray
    warnings: List[str] = field(default_factory=list)
    n_groups: int = 0
    group_size_range: str = ""
    backend: str = "cpu"
    backend_label: str = "CPU (NumPy / scikit-learn)"
    timing_ms: dict[str, float] = field(default_factory=dict)
    # Which size strategy was kept after the PAM-cost bake-off.
    size_strategy: str = "posthoc_repair"
    pam_cost: Optional[float] = None
    team_cohesion: dict[int, float] = field(default_factory=dict)
    distance_matrix: Optional[np.ndarray] = None


def _assert_finite_matrix(matrix: np.ndarray, name: str) -> None:
    # Last line of defense if bad numeric data slipped past CSV validation.
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"{name} contains invalid values (NaN or infinity). "
            "Check your CSV numeric fields (Motivation, Self_Esteem, Work_Ethic)."
        )


def _group_size_summary(labels: np.ndarray, n_students: int, n_groups: int) -> str:
    from .invariants import expected_group_sizes

    allowed = sorted(expected_group_sizes(n_students, n_groups))
    counts = sorted(int(np.sum(labels == g)) for g in np.unique(labels))
    if len(allowed) == 1:
        return f"{allowed[0]} per group"
    return f"{min(counts)}–{max(counts)} per group (expected {allowed[0]} or {allowed[1]})"


def _pam_cost_for_labels(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Total Manhattan distance to each cluster's geometric medoid (cohesion score)."""
    total = 0.0
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        if idx.size == 1:
            continue
        sub = distance_matrix[np.ix_(idx, idx)]
        medoid_local = int(np.argmin(sub.sum(axis=1)))
        medoid = int(idx[medoid_local])
        total += float(distance_matrix[idx, medoid].sum())
    return total


def _select_better_labels(
    distance_matrix: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    *,
    name_a: str = "posthoc_repair",
    name_b: str = "size_constrained",
) -> Tuple[np.ndarray, str, float, float, float]:
    """
    Keep the labeling with lower PAM cost (higher cohesion).

    If PAM costs tie, prefer higher Manhattan silhouette. If that also ties,
    prefer size_constrained.
    Returns labels, chosen name, cost_a, cost_b, chosen_cost.
    """
    from sklearn.metrics import silhouette_score

    cost_a = _pam_cost_for_labels(distance_matrix, labels_a)
    cost_b = _pam_cost_for_labels(distance_matrix, labels_b)
    if cost_b < cost_a - 1e-9:
        return labels_b, name_b, cost_a, cost_b, cost_b
    if cost_a < cost_b - 1e-9:
        return labels_a, name_a, cost_a, cost_b, cost_a

    # PAM costs equal — break ties with Manhattan silhouette when K >= 2.
    n_unique = len(np.unique(labels_a))
    if n_unique < 2:
        return labels_b, name_b, cost_a, cost_b, cost_b

    sil_a = float(silhouette_score(distance_matrix, labels_a, metric="precomputed"))
    sil_b = float(silhouette_score(distance_matrix, labels_b, metric="precomputed"))
    if sil_b > sil_a + 1e-9:
        return labels_b, name_b, cost_a, cost_b, cost_b
    if sil_a > sil_b + 1e-9:
        return labels_a, name_a, cost_a, cost_b, cost_a
    return labels_b, name_b, cost_a, cost_b, cost_b


def run_grouping_pipeline(
    df: pd.DataFrame,
    target_size: int,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = False,
    backend: str = "cpu",
) -> GroupingResult:
    """
    Run feature-only grouping lifecycle. ``df`` must come from ``prepare_for_grouping``.

    Clustering uses Motivation, Self_Esteem, Work_Ethic, and Learning_Style only.
    Gender and Diversity are ignored for feature construction and assignment swaps.

    Size handling: runs both post-hoc medoid repair (A) and size-constrained
    K-Medoids (B), then keeps the lower PAM-cost result so classroom cohesion is
    never worse than either strategy alone.

    Raises InvariantViolation if final assignments fail structural checks.

    ``backend``: ``cpu``, ``gpu`` (auto-detect), ``cuda``, ``directml``, or ``auto``.
    """
    import time

    resolved = resolve_backend(backend)
    timing_ms: dict[str, float] = {}
    pipeline_t0 = time.perf_counter()

    n_students = len(df)
    n_groups = calculate_n_groups(n_students, target_size)

    if verbose:
        print(
            f"   > Configuration: {n_groups} groups for {n_students} students "
            f"(target size {target_size})."
        )

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(
        "   > [Step 3] Psychometric features "
        "(Motivation, Self-Esteem, Work Ethic, Learning Style; demographics excluded)..."
    )
    feature_matrix = compute_feature_vector(df)
    _assert_finite_matrix(feature_matrix, "Feature matrix")

    log(
        f"   > [Step 4] Manhattan distance matrix (psychometric only) [{resolved.label}]..."
    )
    dist_manhattan, dist_ms = compute_manhattan_distance_matrix(feature_matrix, resolved)
    timing_ms["distance_matrix"] = round(dist_ms, 2)
    if dist_manhattan.shape != (n_students, n_students):
        raise ValueError("Distance matrix size does not match student count.")
    _assert_finite_matrix(dist_manhattan, "Distance matrix")
    # Size-constrained path expects a NumPy matrix even when distance ran on GPU.
    dist_np = np.asarray(dist_manhattan, dtype=float)

    log(f"   > [Step 5a] Strategy A: K-Medoids + medoid size repair [{resolved.label}]...")
    t_a0 = time.perf_counter()
    labels_a, kmedoids_ms = run_kmedoids(
        dist_manhattan,
        n_groups,
        random_state=random_state,
        backend=resolved,
    )
    labels_a = enforce_group_size(
        labels_a,
        target_size,
        distance_matrix=dist_np,
    )
    timing_ms["strategy_a_posthoc_ms"] = round(
        (time.perf_counter() - t_a0) * 1000.0, 2
    )
    timing_ms["kmedoids"] = round(kmedoids_ms, 2)

    log("   > [Step 5b] Strategy B: size-constrained K-Medoids (capacity seats)...")
    t_b0 = time.perf_counter()
    caps = capacities_for_target_size(n_students, target_size)
    labels_b, _medoids_b = kmedoids_size_constrained(
        dist_np, caps, random_state=random_state
    )
    timing_ms["strategy_b_size_constrained_ms"] = round(
        (time.perf_counter() - t_b0) * 1000.0, 2
    )

    labels, strategy, cost_a, cost_b, chosen_cost = _select_better_labels(
        dist_np, labels_a, labels_b
    )
    timing_ms["pam_cost_a"] = round(cost_a, 4)
    timing_ms["pam_cost_b"] = round(cost_b, 4)
    log(
        f"   > [Step 5c] Kept {strategy} "
        f"(PAM cost A={cost_a:.4f}, B={cost_b:.4f}, chosen={chosen_cost:.4f})"
    )

    log("   > [Step 6] Final invariant checks...")
    assert_assignment_invariants(labels, n_students, target_size)

    team_cohesion = compute_team_cohesion(dist_np, labels)

    timing_ms["total_pipeline"] = round((time.perf_counter() - pipeline_t0) * 1000.0, 2)
    return GroupingResult(
        labels=labels,
        warnings=[],
        n_groups=n_groups,
        group_size_range=_group_size_summary(labels, n_students, n_groups),
        backend=resolved.name,
        backend_label=resolved.label,
        timing_ms=timing_ms,
        size_strategy=strategy,
        pam_cost=float(chosen_cost),
        team_cohesion=team_cohesion,
        distance_matrix=dist_np,
    )
