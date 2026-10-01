"""
Import smoke test — verifies all entry points load without path/import errors.

Run from repo root:
    python -m backend.check_imports
"""

from __future__ import annotations


def main() -> None:
    from .api import app
    from .data_loader import prepare_for_grouping
    from .group_gen_intake import (
        is_raw_google_form,
        merge_groups_into_raw_export,
        process_google_form,
    )
    from .data_loader import _read_csv_repaired
    from .vak_answer_catalog import (
        classify_vak_letter,
        is_known_vak_answer,
        learning_style_from_abc_counts,
        score_learning_style_from_row,
    )
    from .paths import DEFAULT_GOOGLE_FORM_SAMPLE_CSV, DEFAULT_TEMPLATE_CSV, PROJECT_ROOT
    from .pipeline import run_grouping_pipeline

    # Pre-formatted classroom CSV (direct path)
    df = prepare_for_grouping(DEFAULT_TEMPLATE_CSV)
    result = run_grouping_pipeline(df, target_size=5, verbose=False)
    assert len(result.labels) == len(df)

    # Raw Google Form export → clustering format → groups → merge back to raw
    from .data_loader import read_csv_source

    raw_df = read_csv_source(DEFAULT_GOOGLE_FORM_SAMPLE_CSV)
    assert is_raw_google_form(raw_df), "sample Google Form CSV should be detected as raw export"

    # Real Google Forms export: full option text (no a/b/c prefix)
    meal = (
        "try to get together whilst doing something else, such as an activity or a meal"
    )
    assert is_known_vak_answer(meal)
    hdr = (
        "Timestamp,Email,What is your first and last name,VAK1,VAK2,"
        "MagicWand,SE1,Gender,Diversity\n"
    )
    # Option 22c split across two fields (unquoted comma in export)
    row = (
        "t,e,Student,read the instructions first,"
        "try to get together whilst doing something else,"
        " such as an activity or a meal,"
        "look at a map,,4,Male,Asian\n"
    )
    repaired = _read_csv_repaired(hdr + row)
    assert len(repaired.columns) == 9
    assert meal in repaired.iloc[0].astype(str).tolist()

    assert classify_vak_letter('I say "it\'s great to see you!"') == "A"
    assert classify_vak_letter('I say "it\'s great to hear from you!"') == "B"
    assert classify_vak_letter('I say "its great to see you"') == "A"
    assert classify_vak_letter("discuss them with my friends") == "B"
    assert (
        classify_vak_letter(
            "move around alot, fiddle with pens and pencils and touch things"
        )
        == "C"
    )
    assert (
        classify_vak_letter("playing sports, going to the gym or doing DIY") == "C"
    )
    assert classify_vak_letter("imagine what the food would look like") == "A"
    assert classify_vak_letter("read reviews in magazines or online") == "A"
    assert classify_vak_letter("stand and more") == "C"
    assert (
        classify_vak_letter(
            "talk through the options with my head or with someone else"
        )
        == "B"
    )
    assert (
        classify_vak_letter(
            "can't stand still, fiddle and move around constantly"
        )
        == "C"
    )
    assert (
        classify_vak_letter(
            "talk through the options in my head or with someone else"
        )
        == "B"
    )
    assert (
        classify_vak_letter(
            "can't sit still, I fiddle and move around constantly"
        )
        == "C"
    )

    assert classify_vak_letter("read the instructions first") == "A"
    assert classify_vak_letter("give them a verbal explanation") == "B"
    assert classify_vak_letter("go ahead and have a go, I can figure it out as I use it") == "C"
    # Paper: 2 A's, 1 B, 0 C → Visual
    assert (
        score_learning_style_from_row(
            [
                "read the instructions first",
                "follow a written recipe",
                "give them a verbal explanation",
            ]
        )
        == "Visual"
    )
    assert learning_style_from_abc_counts({"A": 10, "B": 12, "C": 8}) == "Auditory"
    # Tie: pick one of the tied styles (stable for a fixed seed)
    tie_style = learning_style_from_abc_counts({"A": 10, "B": 10, "C": 8}, tie_seed=42)
    assert tie_style in ("Visual", "Auditory")
    assert learning_style_from_abc_counts({"A": 10, "B": 10, "C": 8}, tie_seed=42) == tie_style

    scored = process_google_form(raw_df)
    expected_cols = [
        "Name",
        "Email",
        "Gender",
        "Motivation",
        "Self_Esteem",
        "Work_Ethic",
        "Learning_Style",
        "Diversity",
    ]
    assert list(scored.columns) == expected_cols
    assert scored["Motivation"].between(1, 4).all()
    assert scored["Self_Esteem"].between(1, 4).all()
    assert scored["Work_Ethic"].between(1, 4).all()
    assert set(scored["Learning_Style"].unique()).issubset({"Visual", "Auditory", "Kinesthetic"})

    form_df = prepare_for_grouping(DEFAULT_GOOGLE_FORM_SAMPLE_CSV)
    # 100-student sample needs group_size=10 (K=20 collapses with repeated score patterns).
    form_result = run_grouping_pipeline(form_df, target_size=10, verbose=False)
    assert len(form_result.labels) == len(form_df)

    grouped = form_df.copy()
    grouped["Group_ID"] = form_result.labels + 1
    merged = merge_groups_into_raw_export(raw_df, grouped)
    assert "Group_ID" in merged.columns
    assert merged["Group_ID"].notna().all()
    assert len(merged) == len(form_df)

    assert app.title == "GroupGen API"
    print(f"OK — template ({len(df)} students) + Google Form intake ({len(form_df)} students)")
    print(f"    project root: {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
