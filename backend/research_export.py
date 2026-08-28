"""
Research-ready CSV export for GroupGen cohorts.

Produces student-level rows with anonymized IDs, raw + z-scored features,
team assignment, and per-team intra-group cohesion (mean pairwise L1).
Demographics are metadata columns only — never used in clustering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .clustering import VALID_LEARNING_STYLES
from .cohesion import compute_team_cohesion, stable_student_id

Z_FEATURE_COLS = ["Motivation_z", "Self_Esteem_z", "Work_Ethic_z"]
LS_ONEHOT_COLS = ["LS_Visual", "LS_Auditory", "LS_Kinesthetic"]


def _encode_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (full matrix, z-scored numeric block, one-hot learning style block)."""
    scaler = StandardScaler()
    numeric_z = scaler.fit_transform(
        df[["Motivation", "Self_Esteem", "Work_Ethic"]].values
    )
    encoder = OneHotEncoder(categories=[VALID_LEARNING_STYLES], sparse_output=False)
    ls_onehot = encoder.fit_transform(df[["Learning_Style"]])
    matrix = np.hstack([numeric_z, ls_onehot])
    return matrix, numeric_z, ls_onehot


def build_research_dataframe(
    df: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    *,
    team_cohesion: Dict[int, float] | None = None,
) -> pd.DataFrame:
    """
    Student-level research table for regression / archival.

    Columns include student_id, assigned_team_id, raw scores, z-scores,
    learning-style one-hot flags, intra_team_cohesion_score, and demographics.
    """
    labels = np.asarray(labels, dtype=int)
    if team_cohesion is None:
        team_cohesion = compute_team_cohesion(distance_matrix, labels)

    _, numeric_z, ls_onehot = _encode_features(df)

    out = pd.DataFrame()
    out["student_id"] = df["Name"].map(stable_student_id)
    out["Name"] = df["Name"].astype(str).str.strip()
    out["assigned_team_id"] = labels + 1

    out["Motivation"] = df["Motivation"].astype(int)
    out["Self_Esteem"] = df["Self_Esteem"].astype(int)
    out["Work_Ethic"] = df["Work_Ethic"].astype(int)
    out["Learning_Style"] = df["Learning_Style"].astype(str).str.strip()

    for i, col in enumerate(Z_FEATURE_COLS):
        out[col] = np.round(numeric_z[:, i], 6)

    for i, col in enumerate(LS_ONEHOT_COLS):
        out[col] = ls_onehot[:, i].astype(int)

    out["intra_team_cohesion_score"] = pd.Series(labels).map(
        lambda g: round(team_cohesion[int(g)], 6)
    )

    # Metadata for post-hoc controls — not used in clustering.
    if "Gender" in df.columns:
        out["Gender"] = df["Gender"].astype(str).str.strip()
    if "Diversity" in df.columns:
        out["Diversity"] = df["Diversity"].astype(str).str.strip()

    return out


def build_team_summary(
    research_df: pd.DataFrame,
    team_cohesion: Dict[int, float],
) -> pd.DataFrame:
    """One row per team with size and cohesion."""
    rows = []
    for team_id in sorted(research_df["assigned_team_id"].unique()):
        g = research_df[research_df["assigned_team_id"] == team_id]
        label = int(team_id) - 1
        rows.append(
            {
                "assigned_team_id": int(team_id),
                "team_size": int(len(g)),
                "intra_team_cohesion_score": round(team_cohesion[label], 6),
                "avg_motivation": round(float(g["Motivation"].mean()), 3),
                "avg_self_esteem": round(float(g["Self_Esteem"].mean()), 3),
                "avg_work_ethic": round(float(g["Work_Ethic"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def write_research_exports(
    df: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    run_dir: Path,
    *,
    team_cohesion: Dict[int, float] | None = None,
) -> tuple[Path, Path]:
    """Write student-level and team-level research CSVs."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if team_cohesion is None:
        team_cohesion = compute_team_cohesion(distance_matrix, labels)

    research_df = build_research_dataframe(
        df, labels, distance_matrix, team_cohesion=team_cohesion
    )
    team_df = build_team_summary(research_df, team_cohesion)

    student_path = run_dir / "research_export.csv"
    team_path = run_dir / "team_cohesion.csv"
    research_df.to_csv(student_path, index=False)
    team_df.to_csv(team_path, index=False)
    return student_path, team_path
