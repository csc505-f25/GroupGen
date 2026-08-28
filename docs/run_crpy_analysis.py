"""
Run GroupGen study analysis for CRPY only.

Usage (from repo root):
  python docs/run_crpy_analysis.py

Notebook:
  from run_crpy_analysis import run_crpy
  run_crpy(ga)   # after notebook_bootstrap.init_session()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

import groupgen_analysis as ga

DOCS_ROOT = Path(__file__).resolve().parent
CRPY_DIR = DOCS_ROOT / "Inclass_data" / "CRPY"


@dataclass
class CrpyResults:
    surveys: list
    panel: pd.DataFrame
    availability: pd.DataFrame
    change: pd.DataFrame
    change_tests: pd.DataFrame
    student_teacher: pd.DataFrame
    student_teacher_summary: pd.DataFrame
    teacher_groupgen: pd.DataFrame
    teacher_groupgen_summary: pd.DataFrame
    regroup_linked: pd.DataFrame


def _load_crpy_surveys(crpy_dir: Path | None = None) -> list:
    root = crpy_dir or CRPY_DIR
    crpy_files = sorted(root.glob("*.csv"))
    if not crpy_files:
        raise FileNotFoundError(f"No CSV files found in {root}")
    return [ga.load_survey(path) for path in crpy_files]


CRPY_TABLE_FILES: dict[str, str] = {
    "panel": "crpy_student_panel.csv",
    "availability": "crpy_data_availability.csv",
    "change": "crpy_confidence_change.csv",
    "change_tests": "crpy_confidence_change_tests.csv",
    "student_teacher": "crpy_student_to_teacher_change.csv",
    "student_teacher_summary": "crpy_student_to_teacher_summary.csv",
    "teacher_groupgen": "crpy_teacher_to_groupgen_change.csv",
    "teacher_groupgen_summary": "crpy_teacher_to_groupgen_summary.csv",
    "regroup_linked": "crpy_regroup_to_teacher_outcomes.csv",
}

CRPY_FIGURE_FILES: dict[str, str] = {
    "confidence_distribution": "crpy_confidence_change_distribution.png",
    "confidence_before_after": "crpy_confidence_change_before_vs_after.png",
    "student_teacher_outcomes": "crpy_student_to_teacher_outcomes.png",
    "teacher_groupgen_outcomes": "crpy_teacher_to_groupgen_outcomes.png",
}


def _save_tables(results: CrpyResults) -> None:
    """Write fixed CRPY table names, overwriting existing files."""
    table_data = {
        "panel": results.panel,
        "availability": results.availability,
        "change": results.change,
        "change_tests": results.change_tests,
        "student_teacher": results.student_teacher,
        "student_teacher_summary": results.student_teacher_summary,
        "teacher_groupgen": results.teacher_groupgen,
        "teacher_groupgen_summary": results.teacher_groupgen_summary,
        "regroup_linked": results.regroup_linked,
    }
    for key, filename in CRPY_TABLE_FILES.items():
        df = table_data[key]
        if df is not None and not df.empty:
            ga.write_table(df, filename)
        else:
            ga.remove_table(filename)


def _save_figures(results: CrpyResults) -> list[Path]:
    """Write fixed CRPY figure names, overwriting existing files."""
    saved: list[Path] = []
    if not results.change.empty:
        saved.extend(
            ga.make_confidence_change_figure(results.change, stem="crpy_confidence_change")
        )
    else:
        ga.remove_figure(CRPY_FIGURE_FILES["confidence_distribution"])
        ga.remove_figure(CRPY_FIGURE_FILES["confidence_before_after"])

    if not results.student_teacher_summary.empty:
        saved.extend(
            ga.make_group_outcome_change_figure(
                results.student_teacher_summary,
                filename=CRPY_FIGURE_FILES["student_teacher_outcomes"],
                title="CRPY: student-choice vs teacher groups",
            )
        )
    else:
        ga.remove_figure(CRPY_FIGURE_FILES["student_teacher_outcomes"])

    if not results.teacher_groupgen_summary.empty:
        saved.extend(
            ga.make_group_outcome_change_figure(
                results.teacher_groupgen_summary,
                filename=CRPY_FIGURE_FILES["teacher_groupgen_outcomes"],
                title="CRPY: teacher groups vs GroupGen groups",
            )
        )
    else:
        ga.remove_figure(CRPY_FIGURE_FILES["teacher_groupgen_outcomes"])

    return saved


def compute_crpy(crpy_dir: Path | None = None) -> CrpyResults:
    surveys = _load_crpy_surveys(crpy_dir)
    change = ga.confidence_change_table(surveys)
    student_teacher = ga.group_outcome_change_table(surveys)
    teacher_groupgen = ga.group_outcome_change_table(
        surveys, wave_pair=ga.CRPY_TEACHER_TO_GROUPGEN_PAIR
    )
    return CrpyResults(
        surveys=surveys,
        panel=ga.build_student_panel(surveys),
        availability=ga.data_availability_report(surveys),
        change=change,
        change_tests=ga.confidence_change_tests(change),
        student_teacher=student_teacher,
        student_teacher_summary=ga.group_outcome_change_summary(student_teacher),
        teacher_groupgen=teacher_groupgen,
        teacher_groupgen_summary=ga.group_outcome_change_summary(teacher_groupgen),
        regroup_linked=ga.regroup_features_to_teacher_outcomes(surveys),
    )


def _inventory_table(surveys: list) -> pd.DataFrame:
    rows = []
    for sv in surveys:
        rows.append(
            {
                "study_wave": sv.study_wave,
                "survey": sv.survey_label,
                "n_responses": len(sv.df),
                "scales": ", ".join(s.name for s in sv.present_scales) or "(none)",
            }
        )
    return pd.DataFrame(rows)


def _show_figures(results: CrpyResults, show: Callable[..., Any]) -> None:
    try:
        from IPython.display import Image, display
    except ImportError:
        return

    for path in _save_figures(results):
        if path.exists():
            display(Image(filename=str(path)))


def run_crpy(
    ga_module: Any | None = None,
    *,
    crpy_dir: Path | None = None,
    save: bool = True,
    display: bool = True,
    show: Optional[Callable[..., Any]] = None,
) -> CrpyResults:
    """
    Run full CRPY pipeline and optionally render notebook-friendly output.

    Pass the ``ga`` module from ``notebook_bootstrap.init_session()`` so paths
    match the active kernel.
    """
    global ga
    if ga_module is not None:
        ga = ga_module

    if crpy_dir is None:
        crpy_dir = Path(ga.DATA_DIR) / "CRPY"

    results = compute_crpy(crpy_dir)
    if save:
        _save_tables(results)
        if not display:
            _save_figures(results)

    if display:
        show_fn = show
        if show_fn is None:
            try:
                from IPython.display import display as show_fn
            except ImportError:
                show_fn = print

        print("=" * 70)
        print("CRPY SURVEYS")
        print("=" * 70)
        for sv in results.surveys:
            print(f"  [{sv.study_wave:22}] n={len(sv.df):2}  {sv.survey_label}")

        show_fn(_inventory_table(results.surveys))

        if not results.availability.empty:
            print("\nData availability:")
            show_fn(results.availability)

        if not results.change.empty:
            print("\nConfidence change (check-in -> get-features):")
            show_fn(results.change.head(12))
            if not results.change_tests.empty:
                show_fn(results.change_tests)

        if not results.student_teacher_summary.empty:
            print("\nStudent-choice -> teacher groups (mean change):")
            show_fn(results.student_teacher_summary)

        if not results.teacher_groupgen_summary.empty:
            print("\nTeacher groups -> GroupGen groups (mean change):")
            show_fn(results.teacher_groupgen_summary)

        if not results.regroup_linked.empty:
            print("\nGet-features linked to teacher outcomes (sample):")
            show_fn(results.regroup_linked.head())

        _show_figures(results, show_fn)
        print(f"\nSaved tables:  {ga.OUT_DIR}")
        print(f"Saved figures: {ga.OUT_FIGURES}")

    return results


def main() -> None:
    ga.configure_study_paths(DOCS_ROOT)
    run_crpy(ga, save=True, display=True, show=print)


if __name__ == "__main__":
    main()
