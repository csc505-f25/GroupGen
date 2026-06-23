"""
Post-grouping structural checks for production pipeline output.

Called at the end of ``run_grouping_pipeline`` via ``assert_assignment_invariants``.
Raises ``InvariantViolation`` (API maps to HTTP 422) if:
  - wrong number of groups vs ``calculate_n_groups``
  - any group size outside the expected base / base+1 distribution
  - largest group exceeds ``target_size``
  - students missing from assignments
"""

from __future__ import annotations

from typing import List, Set

import numpy as np

from .group_config import calculate_n_groups


class InvariantViolation(ValueError):
    """Raised when final group assignments fail structural checks."""


def expected_group_sizes(n_students: int, n_groups: int) -> Set[int]:
    """Valid per-group sizes after balanced distribution."""
    if n_groups <= 0:
        raise ValueError("n_groups must be positive.")
    base = n_students // n_groups
    remainder = n_students % n_groups
    if remainder == 0:
        return {base}  # e.g. 30 students, 6 groups → all groups size 5
    return {base, base + 1}  # e.g. 31 students, 7 groups → sizes 4 and 5 only


def validate_assignment_invariants(
    labels: np.ndarray,
    n_students: int,
    target_size: int,
) -> List[str]:
    """
    Verify every student has exactly one group and sizes match the ceiling formula.

    Returns a list of human-readable errors (empty if valid).
    """
    errors: List[str] = []

    # labels[i] = cluster for student at row i — length must match roster.
    if len(labels) != n_students:
        errors.append(
            f"Internal error: expected {n_students} assignments, got {len(labels)}."
        )
        return errors

    if np.any(labels < 0):
        errors.append("Invalid negative group assignment detected.")

    if n_students == 0:
        errors.append("No students to assign.")
        return errors

    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer) and not np.all(
        labels == np.floor(labels)
    ):
        errors.append("Group assignments must be whole-number cluster IDs.")

    n_groups_expected = calculate_n_groups(n_students, target_size)
    unique_labels = np.unique(labels)
    n_groups_actual = len(unique_labels)

    if n_groups_actual != n_groups_expected:
        errors.append(
            f"Expected {n_groups_expected} groups for {n_students} students "
            f"(target size {target_size}), but got {n_groups_actual}."
        )

    allowed_sizes = expected_group_sizes(n_students, n_groups_expected)
    counts = {int(lab): int(np.sum(labels == lab)) for lab in unique_labels}

    if sum(counts.values()) != n_students:
        errors.append("Some students are missing from group assignments.")

    for group_id, size in sorted(counts.items()):
        if size not in allowed_sizes:
            errors.append(
                f"Group {group_id + 1} has {size} students; "
                f"expected only sizes {sorted(allowed_sizes)} "
                f"for target size {target_size}."
            )

    # Teacher-facing cap: no group larger than the requested target (e.g. 5).
    max_size = max(counts.values())
    if max_size > target_size:
        errors.append(
            f"Largest group has {max_size} students, which exceeds "
            f"target size {target_size}."
        )

    return errors


def assert_assignment_invariants(
    labels: np.ndarray,
    n_students: int,
    target_size: int,
) -> None:
    """Raise InvariantViolation if any check fails."""
    errors = validate_assignment_invariants(labels, n_students, target_size)
    if errors:
        raise InvariantViolation("; ".join(errors))
