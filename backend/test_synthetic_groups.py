"""Synthetic classroom recovery tests.

Plant students with known psychometric archetypes, run the production
pipeline, and check that GroupGen returns the groups those profiles imply.

Existing unit tests cover plumbing (reproducibility, invariants, crash-free).
These tests ask the ML question: given a fake roster with a known right
answer, does clustering actually put similar students together?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from backend.cohesion import compute_team_cohesion
from backend.pipeline import DEFAULT_RANDOM_STATE, run_grouping_pipeline

# Three well-separated skill profiles. Likert 1–4 + distinct VAK styles.
HIGH_VISUAL = dict(
    Motivation=4, Self_Esteem=4, Work_Ethic=4, Learning_Style="Visual"
)
MID_AUDITORY = dict(
    Motivation=2, Self_Esteem=3, Work_Ethic=2, Learning_Style="Auditory"
)
LOW_KINESTHETIC = dict(
    Motivation=1, Self_Esteem=1, Work_Ethic=1, Learning_Style="Kinesthetic"
)


def _roster_from_archetypes(
    archetypes: list[dict],
    n_per: int,
    *,
    name_prefix: str = "Student",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a fake class. Returns (dataframe, planted archetype id per row)."""
    rows = []
    planted: list[int] = []
    genders = ["Male", "Female"]
    diversity = ["Group A", "Group B", "Group C"]
    i = 0
    for archetype_id, profile in enumerate(archetypes):
        for _ in range(n_per):
            rows.append(
                {
                    "Name": f"{name_prefix} {i}",
                    "Gender": genders[i % 2],
                    "Diversity": diversity[i % 3],
                    **profile,
                }
            )
            planted.append(archetype_id)
            i += 1
    return pd.DataFrame(rows), np.asarray(planted, dtype=int)


def _mean_cohesion(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    scores = compute_team_cohesion(distance_matrix, labels)
    return float(np.mean(list(scores.values())))


def _random_labels_matching_sizes(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle students across seats so group sizes stay the same."""
    return rng.permutation(np.asarray(labels))


def _group_purity(pred_labels: np.ndarray, planted: np.ndarray) -> float:
    """Fraction of students in a group whose archetype matches the group majority."""
    n = len(pred_labels)
    correct = 0
    for g in np.unique(pred_labels):
        _values, counts = np.unique(planted[pred_labels == g], return_counts=True)
        correct += int(counts.max())
    return correct / n


def test_recovers_three_planted_archetypes() -> None:
    """15 students, 3 distinct profiles of 5, target size 5 → ARI of 1.0."""
    df, planted = _roster_from_archetypes(
        [HIGH_VISUAL, MID_AUDITORY, LOW_KINESTHETIC], n_per=5
    )
    result = run_grouping_pipeline(df, target_size=5, random_state=DEFAULT_RANDOM_STATE)
    assert result.n_groups == 3
    ari = adjusted_rand_score(planted, result.labels)
    assert ari == 1.0, f"Expected perfect recovery, got ARI={ari:.3f}"


def test_assigned_groups_are_archetype_pure() -> None:
    """Size constraints may split an archetype; distant profiles stay mostly separate.

    18 students (3 × 6) at target size 5 forces 4 groups of 4 or 5. Three
    clusters of 6 cannot pack into {4, 5} without a leftover remainder
    team, so one group may mix. The other teams should stay archetype-pure.
    """
    df, planted = _roster_from_archetypes(
        [HIGH_VISUAL, MID_AUDITORY, LOW_KINESTHETIC], n_per=6
    )
    result = run_grouping_pipeline(df, target_size=5, random_state=DEFAULT_RANDOM_STATE)
    assert result.n_groups == 4
    counts = [int(np.sum(result.labels == g)) for g in np.unique(result.labels)]
    assert set(counts).issubset({4, 5})
    assert max(counts) <= 5
    # 16/18 is the best packing: two pure 5s, one pure 4, one remainder 4.
    assert _group_purity(result.labels, planted) >= 16 / 18
    pure_groups = 0
    for g in np.unique(result.labels):
        if len(set(planted[result.labels == g])) == 1:
            pure_groups += 1
    assert pure_groups >= 3


def test_noisy_archetypes_still_recover() -> None:
    """Small Likert jitter around three centroids should still recover clusters."""
    rng = np.random.default_rng(0)
    df, planted = _roster_from_archetypes(
        [HIGH_VISUAL, MID_AUDITORY, LOW_KINESTHETIC], n_per=5
    )
    for col in ("Motivation", "Self_Esteem", "Work_Ethic"):
        jitter = rng.integers(-1, 2, size=len(df))
        df[col] = (df[col] + jitter).clip(1, 4)
    result = run_grouping_pipeline(df, target_size=5, random_state=DEFAULT_RANDOM_STATE)
    ari = adjusted_rand_score(planted, result.labels)
    assert ari >= 0.7, f"Noisy recovery too weak: ARI={ari:.3f}"


def test_cohesion_beats_random_assignment() -> None:
    """GroupGen teams should be more similar (lower L1) than random teams."""
    df, _planted = _roster_from_archetypes(
        [HIGH_VISUAL, MID_AUDITORY, LOW_KINESTHETIC], n_per=5
    )
    result = run_grouping_pipeline(df, target_size=5, random_state=DEFAULT_RANDOM_STATE)
    assert result.distance_matrix is not None
    grouped = _mean_cohesion(result.distance_matrix, result.labels)

    rng = np.random.default_rng(123)
    random_scores = [
        _mean_cohesion(
            result.distance_matrix,
            _random_labels_matching_sizes(result.labels, rng),
        )
        for _ in range(30)
    ]
    # Lower cohesion = more similar profiles. Planted structure should
    # beat every random seating with the same group sizes.
    assert grouped < min(random_scores), (
        f"Grouped cohesion {grouped:.4f} did not beat random "
        f"(best random {min(random_scores):.4f})"
    )


def test_demographics_do_not_define_groups() -> None:
    """Gender/ethnicity vary inside each archetype; recovery should ignore them."""
    df, planted = _roster_from_archetypes(
        [HIGH_VISUAL, LOW_KINESTHETIC], n_per=6
    )
    # Alternate gender so it is uncorrelated with archetype membership
    # after the first student of each block.
    df["Gender"] = ["Male", "Female"] * (len(df) // 2)
    result = run_grouping_pipeline(df, target_size=6, random_state=DEFAULT_RANDOM_STATE)
    ari_profile = adjusted_rand_score(planted, result.labels)
    gender_codes = (df["Gender"] == "Female").astype(int).to_numpy()
    ari_gender = adjusted_rand_score(gender_codes, result.labels)
    assert ari_profile == 1.0, f"Profile ARI={ari_profile:.3f}"
    assert ari_gender < 0.2, (
        f"Groups tracked gender (ARI={ari_gender:.3f}); demographics should be ignored"
    )


def test_identical_profiles_stay_together() -> None:
    """Copies of the same profile should not be split across mixed teams."""
    df, planted = _roster_from_archetypes(
        [HIGH_VISUAL, LOW_KINESTHETIC], n_per=8
    )
    result = run_grouping_pipeline(df, target_size=4, random_state=DEFAULT_RANDOM_STATE)
    # 16 students, target 4 → 4 groups. Expect two all-high and two all-low.
    assert result.n_groups == 4
    assert _group_purity(result.labels, planted) == 1.0
    for g in np.unique(result.labels):
        assert len(set(planted[result.labels == g])) == 1
