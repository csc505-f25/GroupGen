"""
Convert raw Google Forms CSV exports into GroupGen's seven clustering columns.

Section boundaries are detected via four **magic wand** divider columns.
Learning style uses the paper VAK catalog (``vak_answer_catalog``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .vak_answer_catalog import score_learning_style_from_row


def _require_finite_mean(mean_val: float, section: str) -> float:
    """Reject empty or non-numeric Likert sections instead of silently bucketing."""
    if pd.isna(mean_val) or not np.isfinite(mean_val):
        raise ValueError(
            f"{section} section has no valid numeric responses. "
            "Ensure every student answered the Likert items in that block."
        )
    return float(mean_val)


def _mean_to_1_4_mot_we(mean_val: float) -> int:
    """Map a section mean (1–4 Likert) to discrete 1–4 buckets."""
    mean_val = _require_finite_mean(mean_val, "Motivation/work ethic")
    if mean_val <= 2.0:
        return 1
    if mean_val <= 2.6:
        return 2
    if mean_val <= 3.2:
        return 3
    return 4


def _mean_to_1_4_se(mean_val: float) -> int:
    """Map a 1–7 self-esteem mean to 1–4 scale."""
    mean_val = _require_finite_mean(mean_val, "Self-esteem")
    if mean_val <= 2.5:
        return 1
    if mean_val <= 4.0:
        return 2
    if mean_val <= 5.5:
        return 3
    return 4


def _header_text(columns: list) -> list[str]:
    return [str(c).strip() for c in columns]


def _find_column_index(headers: list[str], keyword: str) -> int:
    key = keyword.lower()
    for i, h in enumerate(headers):
        if key in h.lower():
            return i
    return -1


def is_raw_google_form(df: pd.DataFrame) -> bool:
    """True when headers look like an unmodified Google Form export."""
    headers = _header_text(df.columns.tolist())
    has_name = any("first and last name" in h.lower() for h in headers)
    has_wand = any("magic wand" in h.lower() for h in headers)
    return has_name and has_wand


def _load_raw_df(input_source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(input_source, pd.DataFrame):
        return input_source.copy()
    return pd.read_csv(input_source, dtype=str)


def _section_slice(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    if start < 0 or end <= start:
        return df.iloc[:, 0:0]
    return df.iloc[:, start:end]


def process_google_form(
    input_source: Union[str, Path, pd.DataFrame],
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Read a raw Google Form CSV and return the seven clustering columns.

    Accepts a file path, file-like object, or an already-loaded DataFrame.
    """
    df = _load_raw_df(input_source)
    headers = _header_text(df.columns.tolist())

    wand_indices = [i for i, h in enumerate(headers) if "magic wand" in h.lower()]
    if len(wand_indices) < 4:
        raise ValueError(
            f"Expected 4 magic-wand divider columns; found {len(wand_indices)}. "
            "See STUDY_SETUP.md for the required Google Form layout."
        )

    name_idx = _find_column_index(headers, "first and last name")
    if name_idx == -1:
        raise ValueError('Could not find a "first and last name" column in the export.')

    email_idx = _find_column_index(headers, "email")
    gender_idx = _find_column_index(headers, "gender identity")
    diversity_idx = _find_column_index(headers, "ethnicity")

    meta_end = name_idx
    if email_idx != -1:
        meta_end = max(meta_end, email_idx)
    ls_start = meta_end + 1
    ls_end = wand_indices[0]
    se_start, se_end = wand_indices[0] + 1, wand_indices[1]
    mot_start, mot_end = wand_indices[1] + 1, wand_indices[2]
    we_start, we_end = wand_indices[2] + 1, wand_indices[3]

    valid_rows_mask = (
        df.iloc[:, name_idx].fillna("").astype(str).str.strip() != ""
    )
    df = df[valid_rows_mask].copy()

    out = pd.DataFrame()
    out["Name"] = df.iloc[:, name_idx].fillna("Unknown").astype(str).str.strip()

    ls_cols = _section_slice(df, ls_start, ls_end)
    out["Learning_Style"] = ls_cols.apply(
        lambda row: score_learning_style_from_row(row.tolist()), axis=1
    )

    se_cols = _section_slice(df, se_start, se_end).apply(
        pd.to_numeric, errors="coerce"
    )
    out["Self_Esteem"] = se_cols.mean(axis=1).apply(_mean_to_1_4_se)

    mot_cols = _section_slice(df, mot_start, mot_end).apply(
        pd.to_numeric, errors="coerce"
    )
    out["Motivation"] = mot_cols.mean(axis=1).apply(_mean_to_1_4_mot_we)

    we_cols = _section_slice(df, we_start, we_end).apply(
        pd.to_numeric, errors="coerce"
    )
    out["Work_Ethic"] = we_cols.mean(axis=1).apply(_mean_to_1_4_mot_we)

    out["Gender"] = (
        df.iloc[:, gender_idx].astype(str).str.strip()
        if gender_idx != -1
        else "Unknown"
    )
    out["Diversity"] = (
        df.iloc[:, diversity_idx].astype(str).str.strip()
        if diversity_idx != -1
        else "Unknown"
    )

    out = out[
        [
            "Name",
            "Gender",
            "Motivation",
            "Self_Esteem",
            "Work_Ethic",
            "Learning_Style",
            "Diversity",
        ]
    ]

    if output_path:
        out.to_csv(output_path, index=False)

    return out


def merge_groups_into_raw_export(
    raw_df: pd.DataFrame, grouped_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach ``Group_ID`` from scored roster back onto the original Form export."""
    merged = raw_df.copy()
    headers = _header_text(merged.columns.tolist())
    name_idx = _find_column_index(headers, "first and last name")
    if name_idx == -1:
        raise ValueError('Raw export is missing a "first and last name" column.')

    raw_names = merged.iloc[:, name_idx].astype(str).str.strip()
    group_map = (
        grouped_df[["Name", "Group_ID"]]
        .assign(Name=lambda d: d["Name"].astype(str).str.strip())
        .drop_duplicates(subset=["Name"], keep="first")
        .set_index("Name")["Group_ID"]
    )
    merged["Group_ID"] = raw_names.map(group_map)
    if merged["Group_ID"].isna().any():
        missing = raw_names[merged["Group_ID"].isna()].tolist()[:5]
        raise ValueError(
            "Could not assign Group_ID for every student in the raw export. "
            f"Unmatched names (sample): {missing}"
        )
    return merged
