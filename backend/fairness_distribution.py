"""
Demographic spread for fairness swaps (no sklearn).

After psychometric clustering, pair up students who share the same **minority**
gender or ethnicity label — for **every** value in ``Gender`` and ``Diversity``
(Male, Female, Non-binary, Prefer not to say, Hispanic/Latinx, Asian American, …).

Majority labels in a class (more than half the roster) are left alone so their
rebalance does not undo minority pairing. Examples for a minority of size 5:
spread 2+2+1 across groups; for 2 students, one group of 2 together. One leftover
singleton may remain when counts do not divide evenly — that is expected.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MIN_GROUP_SIZE_GENDER = 4
MIN_GROUP_SIZE_DIVERSITY = 3


def category_target_counts(n: int, n_groups: int) -> List[int]:
    """
    Ideal per-group counts for ``n`` students in one demographic category.

    Spread into pairs across groups (e.g. 5 → [2, 2, 1, 0, …], 6 → [2, 2, 2, 0, …]),
    not one mega-group. ``n == 2`` → a single group of 2; ``n == 1`` cannot be paired.
    """
    if n_groups <= 0:
        return []
    if n <= 0:
        return [0] * n_groups
    if n == 1:
        return [1] + [0] * (n_groups - 1)
    buckets: List[int] = []
    remaining = n
    while remaining > 0:
        if remaining >= 2:
            buckets.append(2)
            remaining -= 2
        else:
            buckets.append(1)
            remaining -= 1
    while len(buckets) < n_groups:
        buckets.append(0)
    return buckets[:n_groups]


def assign_category_targets(
    current: Dict[int, int], target_counts: List[int]
) -> Dict[int, int]:
    """Map each group id to a target count (largest targets → groups that already hold more)."""
    groups_sorted = sorted(
        current.keys(), key=lambda g: (current[g], -g), reverse=True
    )
    targets_sorted = sorted(target_counts, reverse=True)
    return {int(g): int(t) for g, t in zip(groups_sorted, targets_sorted)}


def category_distribution_penalty(
    current: Dict[int, int],
    targets: Dict[int, int],
    group_sizes: Dict[int, int],
    *,
    min_group_size: int,
) -> int:
    """Lower is better: penalize lone members and groups above their spread target."""
    penalty = 0
    for g, actual in current.items():
        target = targets.get(g, 0)
        size = group_sizes.get(g, 0)
        if size >= min_group_size and actual == 1:
            penalty += 1000
        if actual > target:
            penalty += 100 * (actual - target)
        if actual < target:
            penalty += 50 * (target - actual)
    return penalty


def current_category_counts(
    df: pd.DataFrame,
    labels: np.ndarray,
    column: str,
    value: str,
    groups: np.ndarray,
) -> Dict[int, int]:
    return {
        int(g): int(
            (df.iloc[np.where(labels == g)[0]][column].astype(str) == value).sum()
        )
        for g in groups
    }


def apply_pair_swap(labels: np.ndarray, a: int, b: int) -> None:
    ga, gb = int(labels[a]), int(labels[b])
    labels[a] = gb
    labels[b] = ga


def has_fairness_donor(
    df: pd.DataFrame,
    labels: np.ndarray,
    group_id: int,
    target_value: str,
    column: str,
) -> bool:
    """True if some other group has 2+ students with ``target_value``."""
    for donor_id in np.unique(labels):
        if int(donor_id) == group_id:
            continue
        donor_indices = np.where(labels == donor_id)[0]
        if int((df.iloc[donor_indices][column] == target_value).sum()) >= 2:
            return True
    return False


def _group_penalty(count: int, target: int, size: int, min_group_size: int) -> int:
    """Penalty contribution for one group (matches ``category_distribution_penalty``)."""
    penalty = 0
    if size >= min_group_size and count == 1:
        penalty += 1000
    if count > target:
        penalty += 100 * (count - target)
    elif count < target:
        penalty += 50 * (target - count)
    return penalty


def rebalance_demographic_column(
    df: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    column: str,
    *,
    min_group_size: int,
    verbose: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Spread each **minority** label in ``column`` (``Gender`` or ``Diversity``) into
    pairs across groups via 1-for-1 swaps. Every distinct label is considered;
    majority labels are skipped. Psychometric similarity is preserved as much as
    possible (minimum swap distance).

    NumPy-only hot path: a productive swap moves exactly one "token" of the target
    label from one group to another, so only two groups' penalties change. This
    avoids re-scanning the whole roster (and pandas overhead) for every candidate.
    """
    labels = np.asarray(labels).copy()
    group_ids = np.unique(labels)
    n_groups = len(group_ids)
    n_students = len(df)
    # Map each student to a contiguous group index 0..k-1 (for fast bincount).
    label_pos = np.searchsorted(group_ids, labels)
    sizes = np.bincount(label_pos, minlength=n_groups)

    col_values = df[column].astype(str).to_numpy()
    changed_any = False
    max_rounds = n_students  # monotonic penalty decrease converges well within this

    for value in pd.unique(col_values):
        if value is None or str(value).strip() == "":
            continue
        is_val = col_values == value
        n_total = int(is_val.sum())
        if n_total < 2:
            continue
        # Strict minority only (fewer than half the class). Skip majority labels so
        # rebalancing e.g. the dominant gender does not undo pairing for smaller groups.
        if 2 * n_total >= n_students:
            continue

        target_sorted = np.array(
            sorted(category_target_counts(n_total, n_groups), reverse=True)
        )

        for _ in range(max_rounds):
            label_pos = np.searchsorted(group_ids, labels)
            # Count target-label holders per group (by group index).
            counts = np.bincount(
                label_pos[is_val], minlength=n_groups
            ).astype(int)

            # Assign targets: groups holding more get the larger target slots.
            order = sorted(range(n_groups), key=lambda i: (counts[i], -i), reverse=True)
            targets = np.zeros(n_groups, dtype=int)
            for slot, i in enumerate(order):
                targets[i] = target_sorted[slot]

            base_pen = np.array(
                [
                    _group_penalty(counts[i], targets[i], sizes[i], min_group_size)
                    for i in range(n_groups)
                ]
            )
            if base_pen.sum() == 0:
                break

            # Precompute holders / non-holders per group (indices into the roster).
            holders = [
                np.where((label_pos == i) & is_val)[0] for i in range(n_groups)
            ]
            non_holders = [
                np.where((label_pos == i) & ~is_val)[0] for i in range(n_groups)
            ]

            best_improvement = 0
            best_cost = float("inf")
            best_move: Optional[tuple[int, int]] = None  # (donor idx, receiver idx)

            # Consider moving one token from donor group ia -> receiver group ib.
            for ia in range(n_groups):
                if counts[ia] < 1:
                    continue
                new_donor_pen = _group_penalty(
                    counts[ia] - 1, targets[ia], sizes[ia], min_group_size
                )
                donor_delta = new_donor_pen - base_pen[ia]
                for ib in range(n_groups):
                    if ia == ib or len(non_holders[ib]) == 0:
                        continue
                    new_recv_pen = _group_penalty(
                        counts[ib] + 1, targets[ib], sizes[ib], min_group_size
                    )
                    improvement = -(donor_delta + (new_recv_pen - base_pen[ib]))
                    if improvement <= 0 or improvement < best_improvement:
                        continue
                    # Cheapest swap realizing this token move (min psychometric cost).
                    sub = distance_matrix[np.ix_(holders[ia], non_holders[ib])]
                    cost = float(sub.min())
                    if improvement > best_improvement or cost < best_cost:
                        best_improvement = improvement
                        best_cost = cost
                        flat = int(sub.argmin())
                        a = int(holders[ia][flat // sub.shape[1]])
                        b = int(non_holders[ib][flat % sub.shape[1]])
                        best_move = (a, b)

            if best_move is None:
                break

            a, b = best_move
            ga, gb = int(labels[a]), int(labels[b])
            labels[a] = gb
            labels[b] = ga
            changed_any = True
            if verbose:
                print(
                    f"   > {column} rebalance ({value}): swapped students {a} <-> {b} "
                    f"(penalty -{best_improvement})"
                )

    return labels, changed_any
