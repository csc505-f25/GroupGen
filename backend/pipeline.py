"""
Shared grouping pipeline for CLI, API, and evaluation scripts.

Lifecycle (strict order):
  1. Ingest/validate (caller: prepare_for_grouping)
  2. Group count calculation
  3. Psychometric features + Manhattan distance (no Gender/Diversity)
  4. K-Medoids clustering
  5. Size balancing
  6. Gender fairness (until stable or cap)
  7. Diversity fairness (until stable or cap)
  8. Bounded gender/diversity reconciliation (prevents tennis-match loops)
  9. Final invariant assertion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from .clustering import (
    check_diversity_isolation,
    check_gender_isolation,
    compute_feature_vector,
    compute_psychometric_distance_matrix,
    enforce_group_size,
)
from .fairness_distribution import (
    MIN_GROUP_SIZE_DIVERSITY,
    MIN_GROUP_SIZE_GENDER,
    has_fairness_donor,
    rebalance_demographic_column,
)
from .group_config import calculate_n_groups
from .invariants import assert_assignment_invariants
from .kmedoids import kmedoids_pam

# Fixed seed so API and CLI produce identical groups for the same CSV.
DEFAULT_RANDOM_STATE = 42
# Hard cap — fairness swaps can ping-pong gender vs diversity; stop if no progress.
MAX_FAIRNESS_ITERATIONS = 40


@dataclass
class GroupingResult:
    """Labels plus optional fairness residuals for API/CLI."""

    labels: np.ndarray
    warnings: List[str] = field(default_factory=list)
    n_groups: int = 0
    group_size_range: str = ""


def _assert_finite_matrix(matrix: np.ndarray, name: str) -> None:
    # Last line of defense if bad numeric data slipped past CSV validation.
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"{name} contains invalid values (NaN or infinity). "
            "Check your CSV numeric fields (Motivation, Self_Esteem, Work_Ethic)."
        )


def run_fairness_passes(
    df: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """
    For each minority gender and ethnicity label in the roster, spread into pairs
    across groups (e.g. 5 → 2+2+1), then alternate gender/diversity rebalance passes.

    Pairwise swaps preserve per-group counts (size invariants from step 5).
    """
    # Work on a copy so callers' arrays are never mutated unexpectedly.
    labels = labels.copy()

    for round_num in range(1, MAX_FAIRNESS_ITERATIONS + 1):
        changed = False

        labels, gender_changed = rebalance_demographic_column(
            df,
            labels,
            distance_matrix,
            "Gender",
            min_group_size=MIN_GROUP_SIZE_GENDER,
            verbose=verbose,
        )
        changed = changed or gender_changed

        labels, diversity_changed = rebalance_demographic_column(
            df,
            labels,
            distance_matrix,
            "Diversity",
            min_group_size=MIN_GROUP_SIZE_DIVERSITY,
            verbose=verbose,
        )
        changed = changed or diversity_changed

        if verbose and (gender_changed or diversity_changed):
            print(f"   > Fairness round {round_num}: demographic rebalance.")

        isolated_gender = check_gender_isolation(df, labels)
        isolated_diversity = check_diversity_isolation(df, labels)

        if not isolated_gender and not isolated_diversity:
            break
        if not changed:
            break  # no improving swap — remaining isolation may be unavoidable

    return labels


def collect_fairness_warnings(df: pd.DataFrame, labels: np.ndarray) -> List[str]:
    """Describe remaining gender/diversity isolation after all swap rounds."""
    warnings: List[str] = []
    gender_left = check_gender_isolation(df, labels)
    if gender_left:
        parts = [
            f"group {g + 1} (1 lone {gender} with others of different genders)"
            for g, gender in sorted(gender_left.items())
        ]
        msg = (
            "Gender balance: could not pair every isolated student — "
            + ", ".join(parts)
            + "."
        )
        no_donor = [
            gender
            for g, gender in gender_left.items()
            if not has_fairness_donor(df, labels, g, gender, "Gender")
        ]
        if no_donor:
            msg += (
                " No other group has 2+ students with gender "
                f"{', '.join(sorted(set(no_donor)))}; automatic swaps cannot fix that."
            )
        else:
            msg += (
                " Swaps could not improve spread further; a manual swap may still help."
            )
        warnings.append(msg)
    diversity_left = check_diversity_isolation(df, labels)
    if diversity_left:
        parts = [
            f"group {g + 1} (1 lone {cat})"
            for g, cat in sorted(diversity_left.items())
        ]
        msg = (
            "Diversity balance: could not pair every isolated student — "
            + ", ".join(parts)
            + "."
        )
        no_donor = [
            cat
            for g, cat in diversity_left.items()
            if not has_fairness_donor(df, labels, g, cat, "Diversity")
        ]
        if no_donor:
            msg += (
                " No other group has 2+ students in "
                f"{', '.join(sorted(set(no_donor)))}; automatic swaps cannot fix that."
            )
        else:
            msg += (
                " Swaps could not improve spread further; a manual swap may still help."
            )
        warnings.append(msg)
    return warnings


def _group_size_summary(labels: np.ndarray, n_students: int, n_groups: int) -> str:
    from .invariants import expected_group_sizes

    allowed = sorted(expected_group_sizes(n_students, n_groups))
    counts = sorted(int(np.sum(labels == g)) for g in np.unique(labels))
    if len(allowed) == 1:
        return f"{allowed[0]} per group"
    return f"{min(counts)}–{max(counts)} per group (expected {allowed[0]} or {allowed[1]})"


def run_grouping_pipeline(
    df: pd.DataFrame,
    target_size: int,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = False,
) -> GroupingResult:
    """
    Run full grouping lifecycle. ``df`` must come from ``prepare_for_grouping``.

    Only ``labels`` are mutated after ingest; student profile columns in ``df``
    are never modified (no leaky writes to behavioral data).

    Raises InvariantViolation if final assignments fail structural checks.
    """
    n_students = len(df)
    # Step 2: how many clusters K-Medoids should form (ceil formula).
    n_groups = calculate_n_groups(n_students, target_size)

    if verbose:
        print(
            f"   > Configuration: {n_groups} groups for {n_students} students "
            f"(target size {target_size})."
        )

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("   > [Step 3] Psychometric features (Motivation, Self-Esteem, Work Ethic, Learning Style)...")
    feature_matrix = compute_feature_vector(df)
    _assert_finite_matrix(feature_matrix, "Feature matrix")

    log("   > [Step 4] Manhattan distance matrix (psychometric only)...")
    dist_manhattan = compute_psychometric_distance_matrix(feature_matrix)
    if dist_manhattan.shape != (n_students, n_students):
        raise ValueError("Distance matrix size does not match student count.")
    _assert_finite_matrix(dist_manhattan, "Distance matrix")

    log("   > [Step 5] Clustering (K-Medoids, Manhattan)...")
    # Each student gets one cluster id 0 .. n_groups-1 (initial sizes may be uneven).
    labels, _ = kmedoids_pam(dist_manhattan, n_groups, random_state=random_state)

    # Step 6: move students between groups until counts match base / base+1 distribution.
    log(f"   > [Step 6] Enforcing group size (target: {target_size})...")
    labels = enforce_group_size(
        labels,
        target_size,
        distance_matrix=dist_manhattan,
    )

    log("   > [Steps 7–9] Gender fairness, then diversity, then bounded reconciliation...")
    labels = run_fairness_passes(df, labels, dist_manhattan, verbose=verbose)

    # Step 10: prove roster is structurally valid before API/CLI returns success.
    log("   > [Step 10] Final invariant checks...")
    assert_assignment_invariants(labels, n_students, target_size)

    # Non-fatal: leftover isolation after swap limits (shown in UI / manifest).
    fairness_warnings = collect_fairness_warnings(df, labels)
    return GroupingResult(
        labels=labels,
        warnings=fairness_warnings,
        n_groups=n_groups,
        group_size_range=_group_size_summary(labels, n_students, n_groups),
    )
