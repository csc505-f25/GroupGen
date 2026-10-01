"""Intake tests: Google Form Likert scoring, including frequency labels."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.group_gen_intake import process_google_form
from backend.paths import DEFAULT_GOOGLE_FORM_SAMPLE_CSV, PROJECT_ROOT

WAND = "If you had a magic wand, what would you change about these questions?"


def _frequency_form_df() -> pd.DataFrame:
    """Minimal raw Form export using Always/Usually/Sometimes/Never (real study)."""
    return pd.DataFrame(
        [
            {
                "Timestamp": "t",
                "Email Address": "a@example.edu",
                "What is your first and last name": "Ada Lovelace",
                "When I operate new equipment I generally:": "read the instructions first",
                WAND + " 1": "",
                "I believe I will receive an excellent grade in this class.": "5",
                WAND + " 2": "",
                "I sit near the front of the class if possible.": "Always",
                "I am alert in classes": "Somtimes",
                WAND + " 3": "",
                "I arrive at classes and other meetings on time.": "Usually",
                "I devote sufficient study time to each of my courses.": "Never",
                WAND + " 4": "",
                "To which gender identity do you most identify?": "Female",
                "To which ethnicity do you most identify?": "Group A",
            }
        ]
    )


def test_numeric_google_form_sample_still_scores() -> None:
    scored = process_google_form(DEFAULT_GOOGLE_FORM_SAMPLE_CSV)
    assert scored["Motivation"].between(1, 4).all()
    assert scored["Work_Ethic"].between(1, 4).all()
    assert scored["Self_Esteem"].between(1, 4).all()


def test_email_at_end_of_form_is_kept_and_does_not_break_vak() -> None:
    """CSC211-style export: Email Address is the last column, not the preamble."""
    df = _frequency_form_df().drop(columns=["Email Address"])
    df["Email Address"] = "ada@example.edu"
    scored = process_google_form(df)
    assert scored.loc[0, "Email"] == "ada@example.edu"
    assert scored.loc[0, "Learning_Style"] in {"Visual", "Auditory", "Kinesthetic"}
    assert scored.loc[0, "Motivation"] in {1, 2, 3, 4}


def test_frequency_labels_score_motivation_and_work_ethic() -> None:
    scored = process_google_form(_frequency_form_df())
    assert len(scored) == 1
    assert scored.loc[0, "Name"] == "Ada Lovelace"
    assert scored.loc[0, "Email"] == "a@example.edu"
    assert scored.loc[0, "Motivation"] in {1, 2, 3, 4}
    assert scored.loc[0, "Work_Ethic"] in {1, 2, 3, 4}
    assert scored.loc[0, "Self_Esteem"] in {1, 2, 3, 4}


def test_blank_frequency_block_names_the_student() -> None:
    df = _frequency_form_df()
    df.loc[0, "I sit near the front of the class if possible."] = ""
    df.loc[0, "I am alert in classes"] = ""
    with pytest.raises(ValueError, match="Ada Lovelace"):
        process_google_form(df)


@pytest.mark.parametrize(
    "relpath",
    [
        "docs/Inclass_data/CRPY/CRPY_Get-Features_Survey (Responses) - Form Responses 1.csv",
        "docs/Inclass_data/CSC412/CSC412 GroupGen Teaming Survey (Responses) - Form Responses 1.csv",
        "docs/Inclass_data/CSC110/CSC110_Datainput_6_1_26 - Lab section1.csv",
    ],
)
def test_real_classroom_intake_csvs(relpath: str) -> None:
    path = PROJECT_ROOT / relpath
    if not path.is_file():
        pytest.skip(f"missing {relpath}")
    scored = process_google_form(path)
    assert len(scored) > 0
    assert scored["Motivation"].between(1, 4).all()
    assert scored["Work_Ethic"].between(1, 4).all()
    assert scored["Self_Esteem"].between(1, 4).all()
