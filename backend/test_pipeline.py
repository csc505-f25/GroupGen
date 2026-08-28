"""Unit tests for production feature clustering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.clustering import compute_feature_vector, compute_psychometric_distance_matrix
from backend.cohesion import (
    assert_distance_matrix_properties,
    compute_team_cohesion,
    stable_student_id,
)
from backend.data_loader import prepare_for_grouping
from backend.paths import DEFAULT_TEMPLATE_CSV
from backend.pipeline import DEFAULT_RANDOM_STATE, run_grouping_pipeline
from backend.research_export import build_research_dataframe
from backend.vak_answer_catalog import _tie_break_seed, score_learning_style_from_row


def _sample_df(n: int = 12) -> pd.DataFrame:
    styles = ["Visual", "Auditory", "Kinesthetic"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "Name": f"Student {i}",
                "Gender": "Male" if i % 2 == 0 else "Female",
                "Motivation": (i % 4) + 1,
                "Self_Esteem": ((i + 1) % 4) + 1,
                "Work_Ethic": ((i + 2) % 4) + 1,
                "Learning_Style": styles[i % 3],
                "Diversity": "Group A" if i % 2 == 0 else "Group B",
            }
        )
    return pd.DataFrame(rows)


def test_feature_vector_excludes_demographics() -> None:
    df = _sample_df()
    X = compute_feature_vector(df)
    assert X.shape == (len(df), 6)
    assert np.isfinite(X).all()


def test_distance_matrix_properties() -> None:
    df = _sample_df()
    X = compute_feature_vector(df)
    D = compute_psychometric_distance_matrix(X)
    assert_distance_matrix_properties(D)
    assert D.shape == (len(df), len(df))


def test_pipeline_reproducible_labels() -> None:
    df = prepare_for_grouping(DEFAULT_TEMPLATE_CSV)
    a = run_grouping_pipeline(df, target_size=5, random_state=DEFAULT_RANDOM_STATE)
    b = run_grouping_pipeline(df, target_size=5, random_state=DEFAULT_RANDOM_STATE)
    assert np.array_equal(a.labels, b.labels)


def test_team_cohesion_per_group() -> None:
    df = _sample_df()
    result = run_grouping_pipeline(df, target_size=4, random_state=42)
    assert len(result.team_cohesion) == result.n_groups
    for label, score in result.team_cohesion.items():
        assert score >= 0.0
        assert label in result.labels


def test_research_export_columns() -> None:
    df = _sample_df()
    result = run_grouping_pipeline(df, target_size=4, random_state=42)
    assert result.distance_matrix is not None
    out = build_research_dataframe(df, result.labels, result.distance_matrix)
    required = {
        "student_id",
        "assigned_team_id",
        "Motivation",
        "Self_Esteem",
        "Work_Ethic",
        "Learning_Style",
        "Motivation_z",
        "Self_Esteem_z",
        "Work_Ethic_z",
        "intra_team_cohesion_score",
    }
    assert required.issubset(out.columns)
    assert out["student_id"].nunique() == len(df)
    # Same team → same cohesion score
    for team_id in out["assigned_team_id"].unique():
        vals = out.loc[out["assigned_team_id"] == team_id, "intra_team_cohesion_score"]
        assert vals.nunique() == 1


def test_stable_student_id_deterministic() -> None:
    assert stable_student_id("Jane Doe") == stable_student_id("Jane Doe")
    assert stable_student_id("Jane Doe") != stable_student_id("John Doe")


def test_vak_tie_break_seed_stable_across_calls() -> None:
    answers = ("read the instructions first", "discuss with friends")
    a = _tie_break_seed(answers)
    b = _tie_break_seed(answers)
    assert a == b


def test_vak_tie_break_produces_valid_style() -> None:
    # Equal A/B/C counts force tie-break path (use exact catalog strings).
    tied = [
        'I say "it\'s great to see you!"',
        'I say "it\'s great to see you!"',
        'I say "it\'s great to hear from you!"',
        'I say "it\'s great to hear from you!"',
        "move around alot, fiddle with pens and pencils and touch things",
        "move around alot, fiddle with pens and pencils and touch things",
    ]
    style_a = score_learning_style_from_row(tied)
    style_b = score_learning_style_from_row(tied)
    assert style_a == style_b
    assert style_a in {"Visual", "Auditory", "Kinesthetic"}


def test_k1_pipeline_does_not_crash() -> None:
    df = _sample_df(n=5)
    result = run_grouping_pipeline(df, target_size=10, random_state=42)
    assert result.n_groups == 1
    assert len(result.labels) == len(df)
