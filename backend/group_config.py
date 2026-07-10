"""
Shared group-count formula for CLI, API, and pipeline.

``calculate_n_groups(n_students, target_size)`` returns ``ceil(n / target)``
so remainder students are spread across groups (e.g. 31 @ 5 → 7 groups of 4 and 5).
"""

from __future__ import annotations

import numpy as np


def calculate_n_groups(n_students: int, target_size: int) -> int:
    """
    Number of clusters to form.

    Uses ceil(n / target) so remainder students are distributed via
    enforce_group_size (e.g. 31 students, target 5 -> 7 groups).
    """
    if target_size <= 0:
        raise ValueError("Group size must be at least 1.")
    if n_students <= 0:
        raise ValueError("CSV must contain at least one student row.")
    # Entire class fits in one group when target >= headcount.
    if target_size >= n_students:
        return 1
    # e.g. ceil(31/5) = 7 groups, then enforce_group_size spreads 31 seats as 4+4+5+5+5+5+5
    return max(1, int(np.ceil(n_students / target_size)))
