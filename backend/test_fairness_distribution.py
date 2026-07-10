"""Unit tests for demographic spread targets (no sklearn required)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load():
    path = Path(__file__).with_name("fairness_distribution.py")
    spec = importlib.util.spec_from_file_location("fairness_distribution", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fairness_distribution"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_fd = _load()
category_target_counts = _fd.category_target_counts
rebalance_demographic_column = _fd.rebalance_demographic_column


def test_category_target_counts() -> None:
    assert category_target_counts(2, 6) == [2, 0, 0, 0, 0, 0]
    assert category_target_counts(4, 6) == [2, 2, 0, 0, 0, 0]
    assert category_target_counts(5, 6) == [2, 2, 1, 0, 0, 0]
    assert category_target_counts(6, 6) == [2, 2, 2, 0, 0, 0]


def test_rebalance_pairs_minority_ethnicity() -> None:
    n = 15
    df = pd.DataFrame(
        {
            "Gender": ["Male"] * n,
            "Diversity": ["White American"] * 10 + ["Hispanic/Latinx"] * 5,
        }
    )
    labels = np.array([0, 1, 2, 3, 4] * 3)
    dist = np.ones((n, n))
    np.fill_diagonal(dist, 0.0)
    new_labels, changed = rebalance_demographic_column(
        df, labels, dist, "Diversity", min_group_size=3
    )
    assert changed
    hispanic_per_group = [
        int(((df["Diversity"] == "Hispanic/Latinx") & (new_labels == g)).sum())
        for g in range(5)
    ]
    assert sum(1 for x in hispanic_per_group if x >= 2) >= 2
    assert sum(hispanic_per_group) == 5


def test_rebalance_pairs_five_females() -> None:
    n = 15
    df = pd.DataFrame(
        {
            "Gender": ["Female"] * 5 + ["Male"] * 10,
            "Diversity": ["A"] * n,
        }
    )
    labels = np.array([0, 1, 2, 3, 4] * 3)
    dist = np.ones((n, n))
    np.fill_diagonal(dist, 0.0)
    new_labels, changed = rebalance_demographic_column(
        df, labels, dist, "Gender", min_group_size=4
    )
    assert changed
    female_per_group = [
        int(((df["Gender"] == "Female") & (new_labels == g)).sum()) for g in range(5)
    ]
    assert sum(1 for x in female_per_group if x >= 2) >= 2
    assert sum(female_per_group) == 5


if __name__ == "__main__":
    test_category_target_counts()
    test_rebalance_pairs_minority_ethnicity()
    test_rebalance_pairs_five_females()
    print("ok")
