"""
GroupGen — Study Analysis Pipeline
==================================

A reusable analysis pipeline for the GroupGen classroom study. It is written to
work on the *pilot* data collected so far (multiple classes, multiple survey
types) and to drop straight onto next semester's data with no code changes:
just add the new CSV exports under ``docs/Inclass_data/<COURSE>/``.

Folder layout under ``docs/``::

  Inclass_data/<COURSE>/   survey CSV exports (input — do not edit by scripts)
  output/tables/           generated CSV tables from analysis
  output/figures/          generated PNG charts from analysis
  groupgen_analysis.py     reusable analysis pipeline
  GG_analysis.ipynb        interactive notebook
  DataAnalysis.py          legacy single-class CSC412 script

What it does
------------
1. Discovers every survey CSV under ``docs/Inclass_data`` and classifies it as an
   INTAKE/baseline survey, a CHECK-IN survey, or a POST/outcome survey based on
   which questions it contains.
2. Cleans Google Forms quirks (trailing spaces, text Likert scales, the
   "Somtimes" typo, reverse-worded items).
3. Scores each validated construct (subscale = mean of its items) and computes
   scale reliability (Cronbach's alpha) per class.
4. Writes tidy, paper-ready tables (CSV) and figures to ``docs/output/tables``
   and ``docs/output/figures``.
5. Where data allow, builds a student panel and runs longitudinal analyses:
   intake→check-in confidence change (paired) and intake→post outcome linkage.

Why these constructs
--------------------
The surveys map onto recognizable instruments:

* Baseline (1-7):  Academic self-efficacy (MSLQ-style, 8 items)
* Baseline (1-5 frequency text): Classroom engagement (7), Study habits (8)
* Outcomes (1-5):  Group belonging (3), Group satisfaction (3),
                   Psychological safety (Edmondson, 7, with reverse items),
                   Team learning behaviors (7)

IMPORTANT methodological notes
------------------------------
* Psychological safety has REVERSE-worded items; they are reverse-scored before
  the subscale mean is taken. Averaging them raw (as the old script did) is wrong.
* Scales live on different ranges (1-7 vs 1-5). Means are reported per scale;
  do NOT pool different scales into one average without standardizing first.
* Outcome surveys (post) often lack a name/ID, so they cannot yet be linked to a
  student's demographics or baseline. The demographic-equity analysis therefore
  runs only where the linkage exists. See the printed LIMITATIONS section.

Run:  python docs/groupgen_analysis.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Matplotlib/seaborn are optional: the tables are the core deliverable.
def _running_in_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell in ("ZMQInteractiveShell", "GoogleColabShell")
    except NameError:
        return False


try:
    import matplotlib

    if not _running_in_notebook():
        matplotlib.use("Agg")  # headless-safe for CLI scripts
    import matplotlib.pyplot as plt
    import seaborn as sns

    _HAVE_PLOTS = True
except Exception:  # pragma: no cover - plotting is a convenience, not required
    _HAVE_PLOTS = False


# =====================================================================
# PATHS
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Inclass_data"
OUTPUT_ROOT = SCRIPT_DIR / "output"
OUT_DIR = OUTPUT_ROOT / "tables"      # CSV tables
OUT_FIGURES = OUTPUT_ROOT / "figures"  # PNG charts


def configure_study_paths(root: Path | None = None) -> tuple[Path, Path, Path]:
    """
    Point the pipeline at a docs root (for notebooks / Colab).

    Returns ``(data_dir, tables_dir, figures_dir)`` and updates module globals.
    """
    global DATA_DIR, OUT_DIR, OUT_FIGURES, OUTPUT_ROOT

    base = Path(root).resolve() if root else SCRIPT_DIR
    DATA_DIR = base / "Inclass_data"
    OUTPUT_ROOT = base / "output"
    OUT_DIR = OUTPUT_ROOT / "tables"
    OUT_FIGURES = OUTPUT_ROOT / "figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    return DATA_DIR, OUT_DIR, OUT_FIGURES


def ensure_output_dirs() -> None:
    """Create output subfolders if missing."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)


def clear_outputs(*, tables: bool = True, figures: bool = True) -> int:
    """Delete generated CSV/PNG files under ``output/``. Returns files removed."""
    ensure_output_dirs()
    removed = 0
    if tables:
        for path in OUT_DIR.iterdir():
            if path.is_file():
                path.unlink()
                removed += 1
    if figures:
        for path in OUT_FIGURES.iterdir():
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


def write_table(df: pd.DataFrame, filename: str) -> Optional[Path]:
    """Write a CSV to ``output/tables/``, overwriting any existing file."""
    ensure_output_dirs()
    path = OUT_DIR / filename
    df.to_csv(path, index=False)
    return path


def remove_table(filename: str) -> None:
    """Remove a table output if it exists (e.g. when a run no longer produces it)."""
    path = OUT_DIR / filename
    if path.is_file():
        path.unlink()


def remove_figure(filename: str) -> None:
    """Remove a figure output if it exists."""
    path = OUT_FIGURES / filename
    if path.is_file():
        path.unlink()


# =====================================================================
# TEXT NORMALIZATION + LIKERT MAPPING
# =====================================================================
def normalize(text: str) -> str:
    """Lowercase, collapse whitespace/newlines, strip punctuation noise."""
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# Text frequency scale used by engagement + study-habit items (5-point).
FREQUENCY_MAP = {
    "always": 5,
    "usually": 4,
    "sometimes": 3,
    "somtimes": 3,  # common export typo
    "rarely": 2,
    "never": 1,
}


def to_numeric_likert(series: pd.Series) -> pd.Series:
    """Coerce a survey column to numeric, mapping text frequency labels first."""
    mapped = series.astype(str).str.strip().str.lower().map(FREQUENCY_MAP)
    return mapped.fillna(pd.to_numeric(series, errors="coerce"))


# =====================================================================
# SCALE DEFINITIONS
# =====================================================================
@dataclass
class Scale:
    """A psychometric construct: a set of items that are averaged together."""

    name: str
    kind: str  # 'baseline' or 'outcome'
    likert_max: int  # 7 for self-efficacy, 5 for the rest
    # Each item is a distinctive substring (normalized) that identifies its column.
    items: list[str]
    # Substrings of items that are reverse-worded (scored as likert_max+1 - x).
    reverse: list[str] = field(default_factory=list)


SCALES: list[Scale] = [
    # ---- Baseline constructs (intake + check-in) ----
    Scale(
        name="academic_self_efficacy",
        kind="baseline",
        likert_max=7,
        items=[
            "i believe i will receive an excellent grade",
            "understand the most difficult material presented in the readings",
            "understand the basic concepts taught in this course",
            "understand the most complex material presented by the instructor",
            "do an excellent job on the assignments and tests",
            "i expect to do well in this class",
            "master the skills being taught in this class",
            "considering the difficulty of this course",
        ],
    ),
    Scale(
        name="classroom_engagement",
        kind="baseline",
        likert_max=5,
        items=[
            "i sit near the front of the class",
            "i am alert in classes",
            "i ask the instructor questions when clarification",
            "i volunteer answers to questions posed by instructors",
            "i participate in meaningful class discussions",
            "i take the initiative in group activities",
            "i use a study method which helps me develop an interest",
        ],
    ),
    Scale(
        name="study_habits",
        kind="baseline",
        likert_max=5,
        items=[
            "i arrive at classes and other meetings on time",
            "i devote sufficient study time to each of my courses",
            "i schedule definite times and outline specific goals",
            "i prepare a",  # "...to do list daily" (encoding-mangled quotes)
            "i avoid activities which tend to interfere",
            "i use prime time when i am most alert for study",
            "at the beginning of the term, i make up daily activity",
            "i begin major course assignments well in advance",
        ],
    ),
    # ---- Outcome constructs (post / reflection / M2) ----
    Scale(
        name="group_belonging",
        kind="outcome",
        likert_max=5,
        items=[
            "i feel that i belong to this group",
            "i see myself as part of this group",
            "i feel that i am a member of this group",
        ],
    ),
    Scale(
        name="group_satisfaction",
        kind="outcome",
        likert_max=5,
        items=[
            "i am happy to be part of this group",
            "i am content to be part of this group",
            "this group is one of the best anywhere",
        ],
    ),
    Scale(
        name="psychological_safety",
        kind="outcome",
        likert_max=5,
        items=[
            "when someone makes a mistake, it is often held against",
            "it is easy to discuss issues and problems in this group",
            "participants are sometimes rejected for having different opinions",
            "it is completely safe to take a risk in group discussions",
            "it is difficult to ask other participants for help",
            "no participant deliberately undermines another participant",
            "participants value and respect other's contributions",
        ],
        reverse=[
            "when someone makes a mistake, it is often held against",
            "participants are sometimes rejected for having different opinions",
            "it is difficult to ask other participants for help",
        ],
    ),
    Scale(
        name="team_learning",
        kind="outcome",
        likert_max=5,
        items=[
            "participants seek feedback from one another",
            "a difference of opinion is held as a teaching/learning opportunity",
            "problems or errors are communicated appropriately",
            "participants actively seek new information from each other",
            "participants talk openly about mistakes or misconceptions",
            "participants raise concerns about plans or decisions",
            "multidisciplinary/diverse views are presented and respected",
        ],
    ),
]

# Demographic + identifier columns (baseline surveys only).
DEMO_COLS = {
    "gender": "to which gender identity do you most identify",
    "ethnicity": "to which ethnicity do you most identify",
    "year": "what year are you",
}
NAME_KEY = "what is your first and last name"
EMAIL_PATTERNS = ("email address", "email")
TIMESTAMP_KEY = "timestamp"
ACCEPT_RECOMMEND_KEY = "would you recommend a system like groupgen"
STABILITY_KEY = "did you have the same group"
CONFIDENCE_SCALE = "academic_self_efficacy"


# =====================================================================
# COLUMN MATCHING
# =====================================================================
def find_column(df_norm_cols: dict[str, str], pattern: str) -> Optional[str]:
    """Return the ORIGINAL column whose normalized text contains ``pattern``."""
    for norm_col, orig_col in df_norm_cols.items():
        if pattern in norm_col:
            return orig_col
    return None


def item_columns(df: pd.DataFrame, scale: Scale) -> dict[str, str]:
    """Map each present item-pattern -> the matching original column name."""
    norm_map = {normalize(c): c for c in df.columns}
    found: dict[str, str] = {}
    for item in scale.items:
        col = find_column(norm_map, item)
        if col is not None:
            found[item] = col
    return found


# =====================================================================
# RELIABILITY
# =====================================================================
def cronbach_alpha(items: pd.DataFrame) -> tuple[float, int]:
    """
    Cronbach's alpha for an (n_respondents x k_items) numeric frame.

    Returns (alpha, n_used). Rows with any missing item are dropped listwise.
    Alpha is NaN when k < 2 or n < 2 or total variance is zero.
    """
    data = items.apply(pd.to_numeric, errors="coerce").dropna()
    n, k = data.shape
    if k < 2 or n < 2:
        return float("nan"), n
    item_var = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan"), n
    alpha = (k / (k - 1)) * (1 - item_var.sum() / total_var)
    return float(alpha), n


# =====================================================================
# SURVEY LOADING + SCORING
# =====================================================================
@dataclass
class Survey:
    course: str
    survey_label: str
    path: Path
    df: pd.DataFrame
    present_scales: list[Scale]
    survey_type: str  # 'intake' | 'checkin' | 'outcome' | 'mixed'
    study_wave: str = "unknown"


# Course-specific wave labels for longitudinal analysis.
CRPY_CONFIDENCE_PAIR = ("group_zero_baseline", "post_regroup_features")
CRPY_OUTCOME_PAIR = ("group_zero_outcome", "teacher_group_outcome")
CRPY_TEACHER_TO_GROUPGEN_PAIR = ("teacher_group_outcome", "groupgen_group_outcome")
CRPY_ALL_WAVES = (
    "group_zero_baseline",
    "group_zero_outcome",
    "post_regroup_features",
    "teacher_group_outcome",
    "groupgen_group_outcome",
)


def infer_study_wave(course: str, label: str, survey_type: str) -> str:
    """Map a survey export to a study wave (CRPY has a multi-phase design)."""
    key = normalize(label)
    if course == "CRPY":
        if "check-in" in key or "check in" in key:
            return "group_zero_baseline"
        if "student-reflection" in key or "student reflection" in key:
            return "group_zero_outcome"
        if "get-features" in key or "get features" in key:
            return "post_regroup_features"
        if "groupgen" in key and "reflection" in key:
            return "groupgen_group_outcome"
        if "teacher" in key and "reflection" in key:
            return "teacher_group_outcome"
    if survey_type == "intake":
        return "intake"
    if survey_type == "checkin":
        return "checkin"
    if survey_type == "outcome":
        return "outcome"
    return survey_type


def load_survey(path: Path) -> Survey:
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()

    present = [s for s in SCALES if item_columns(df, s)]
    has_outcome = any(s.kind == "outcome" for s in present)
    has_baseline = any(s.kind == "baseline" for s in present)
    has_vak = any("when i cook a new dish" in normalize(c) for c in df.columns)

    if has_outcome and has_baseline:
        stype = "mixed"
    elif has_outcome:
        stype = "outcome"
    elif has_vak:
        stype = "intake"  # full intake form (VAK + baseline scales)
    elif has_baseline:
        stype = "checkin"  # baseline scales only, no VAK
    else:
        stype = "unknown"

    course = path.parent.name
    label = path.stem.split("(Responses)")[0].strip(" -_")
    wave = infer_study_wave(course, label, stype)
    return Survey(course, label, path, df, present, stype, wave)


def score_survey(survey: Survey) -> pd.DataFrame:
    """Add a column per present scale (subscale mean) to a copy of the data."""
    df = survey.df.copy()
    for scale in survey.present_scales:
        cols = item_columns(df, scale)
        item_df = pd.DataFrame(
            {item: to_numeric_likert(df[col]) for item, col in cols.items()}
        )
        # Reverse-score where needed (e.g. negatively-worded safety items).
        for item in cols:
            if any(rev in item for rev in scale.reverse):
                item_df[item] = (scale.likert_max + 1) - item_df[item]
        df[f"score_{scale.name}"] = item_df.mean(axis=1)
    return df


# =====================================================================
# ANALYSES
# =====================================================================
def reliability_table(surveys: list[Survey]) -> pd.DataFrame:
    rows = []
    for sv in surveys:
        for scale in sv.present_scales:
            cols = item_columns(sv.df, scale)
            item_df = pd.DataFrame(
                {it: to_numeric_likert(sv.df[c]) for it, c in cols.items()}
            )
            for it in cols:
                if any(rev in it for rev in scale.reverse):
                    item_df[it] = (scale.likert_max + 1) - item_df[it]
            alpha, n_used = cronbach_alpha(item_df)
            scored = item_df.mean(axis=1)
            rows.append(
                {
                    "course": sv.course,
                    "survey": sv.survey_label,
                    "survey_type": sv.survey_type,
                    "scale": scale.name,
                    "kind": scale.kind,
                    "likert_max": scale.likert_max,
                    "n_items_found": len(cols),
                    "n_items_expected": len(scale.items),
                    "n_respondents": int(scored.notna().sum()),
                    "cronbach_alpha": round(alpha, 3) if pd.notna(alpha) else np.nan,
                    "mean": round(float(scored.mean()), 3),
                    "sd": round(float(scored.std(ddof=1)), 3),
                }
            )
    return pd.DataFrame(rows)


def pooled_scale_summary(rel: pd.DataFrame) -> pd.DataFrame:
    """Sample-size-weighted pooled mean per scale across classes."""
    out = []
    for scale, g in rel.groupby("scale", sort=False):
        n = g["n_respondents"].sum()
        if n == 0:
            continue
        wmean = np.average(g["mean"], weights=g["n_respondents"])
        out.append(
            {
                "scale": scale,
                "kind": g["kind"].iloc[0],
                "likert_max": g["likert_max"].iloc[0],
                "classes": g["course"].nunique(),
                "total_n": int(n),
                "pooled_mean": round(float(wmean), 3),
                "median_alpha": round(float(g["cronbach_alpha"].median()), 3),
            }
        )
    return pd.DataFrame(out)


def demographic_equity(surveys: list[Survey]) -> pd.DataFrame:
    """
    Baseline construct means split by gender/ethnicity, where demographics exist.
    Highlights whether minority groups START at a disadvantage (the fairness aim
    is to ensure outcomes don't track these baseline gaps).
    """
    rows = []
    for sv in surveys:
        norm_map = {normalize(c): c for c in sv.df.columns}
        scored = score_survey(sv)
        baseline_scales = [s for s in sv.present_scales if s.kind == "baseline"]
        if not baseline_scales:
            continue
        for demo_name, pat in DEMO_COLS.items():
            demo_col = find_column(norm_map, pat)
            if demo_col is None:
                continue
            grp = scored.copy()
            grp[demo_col] = grp[demo_col].astype(str).str.strip()
            for scale in baseline_scales:
                col = f"score_{scale.name}"
                agg = grp.groupby(demo_col)[col].agg(["count", "mean"])
                for level, r in agg.iterrows():
                    if not level or level.lower() == "nan":
                        continue
                    rows.append(
                        {
                            "course": sv.course,
                            "demographic": demo_name,
                            "level": level,
                            "scale": scale.name,
                            "n": int(r["count"]),
                            "mean": round(float(r["mean"]), 3),
                        }
                    )
    return pd.DataFrame(rows)


def acceptance_summary(surveys: list[Survey]) -> pd.DataFrame:
    rows = []
    for sv in surveys:
        norm_map = {normalize(c): c for c in sv.df.columns}
        rec = find_column(norm_map, ACCEPT_RECOMMEND_KEY)
        if rec is not None:
            vc = sv.df[rec].astype(str).str.strip()
            vc = vc[~vc.str.lower().isin(["", "nan"])]
            for level, count in vc.value_counts().items():
                rows.append(
                    {
                        "course": sv.course,
                        "question": "would_recommend_groupgen",
                        "response": level,
                        "count": int(count),
                    }
                )
        stab = find_column(norm_map, STABILITY_KEY)
        if stab is not None:
            vc = sv.df[stab].astype(str).str.strip()
            vc = vc[~vc.str.lower().isin(["", "nan"])]
            for level, count in vc.value_counts().items():
                rows.append(
                    {
                        "course": sv.course,
                        "question": "same_group_whole_time",
                        "response": level,
                        "count": int(count),
                    }
                )
    return pd.DataFrame(rows)


# =====================================================================
# LONGITUDINAL / CHANGE ANALYSIS
# =====================================================================
def find_email_column(df: pd.DataFrame) -> Optional[str]:
    norm_map = {normalize(c): c for c in df.columns}
    for pattern in EMAIL_PATTERNS:
        col = find_column(norm_map, pattern)
        if col is not None:
            return col
    return None


def make_student_key(name: object, email: Optional[object] = None) -> Optional[str]:
    """Normalize a student identifier; prefer email when present."""
    if email is not None and pd.notna(email):
        email_str = str(email).strip().lower()
        if email_str and email_str != "nan":
            return email_str
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    name_str = re.sub(r"\s+", " ", str(name).strip().lower())
    return name_str or None


def survey_has_student_id(survey: Survey) -> bool:
    norm_map = {normalize(c): c for c in survey.df.columns}
    return find_column(norm_map, NAME_KEY) is not None


def build_student_panel(surveys: list[Survey]) -> pd.DataFrame:
    """
    Stack scored surveys into a long student panel (one row per student × wave).

    Duplicate exports for the same course/type/student are collapsed to the
    earliest timestamp when available.
    """
    rows: list[dict] = []
    baseline_scales = [s.name for s in SCALES if s.kind == "baseline"]
    outcome_scales = [s.name for s in SCALES if s.kind == "outcome"]

    for sv in surveys:
        norm_map = {normalize(c): c for c in sv.df.columns}
        name_col = find_column(norm_map, NAME_KEY)
        email_col = find_email_column(sv.df)
        ts_col = find_column(norm_map, TIMESTAMP_KEY)
        scored = score_survey(sv)

        for idx in range(len(scored)):
            student_key = None
            if name_col is not None:
                student_key = make_student_key(
                    scored.at[idx, name_col],
                    scored.at[idx, email_col] if email_col else None,
                )

            row: dict = {
                "course": sv.course,
                "survey": sv.survey_label,
                "survey_type": sv.survey_type,
                "study_wave": sv.study_wave,
                "student_key": student_key,
                "has_student_id": student_key is not None,
            }
            if ts_col is not None:
                row["survey_date"] = scored.at[idx, ts_col]

            for demo_name, pat in DEMO_COLS.items():
                demo_col = find_column(norm_map, pat)
                if demo_col is not None:
                    row[demo_name] = scored.at[idx, demo_col]

            for scale_name in baseline_scales + outcome_scales:
                col = f"score_{scale_name}"
                if col in scored.columns:
                    row[col] = scored.at[idx, col]

            rows.append(row)

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel

    if "survey_date" in panel.columns:
        panel["survey_date"] = pd.to_datetime(panel["survey_date"], errors="coerce")
        panel = panel.sort_values(["course", "study_wave", "student_key", "survey_date"])
    else:
        panel = panel.sort_values(["course", "study_wave", "student_key", "survey"])

    dedupe_cols = ["course", "study_wave", "student_key"]
    panel = panel.dropna(subset=["student_key"]).drop_duplicates(
        subset=dedupe_cols, keep="first"
    )
    return panel.reset_index(drop=True)


def data_availability_report(surveys: list[Survey]) -> pd.DataFrame:
    """Summarize which longitudinal analyses are possible per course."""
    rows = []
    for course in sorted({sv.course for sv in surveys}):
        course_surveys = [sv for sv in surveys if sv.course == course]
        waves = {sv.study_wave for sv in course_surveys}
        wave_counts = (
            pd.Series([sv.study_wave for sv in course_surveys])
            .value_counts()
            .to_dict()
        )

        if course == "CRPY":
            conf_ok = all(w in waves for w in CRPY_CONFIDENCE_PAIR)
            outcome_ok = all(w in waves for w in CRPY_OUTCOME_PAIR)
            groupgen_ok = all(w in waves for w in CRPY_TEACHER_TO_GROUPGEN_PAIR)
            full_timeline = all(w in waves for w in CRPY_ALL_WAVES)
            notes = (
                "CRPY multi-phase: group-zero baseline -> post-regroup features; "
                "group-zero outcomes -> teacher-group outcomes -> GroupGen outcomes."
            )
        else:
            conf_ok = "intake" in waves and "checkin" in waves
            outcome_ok = "intake" in waves and "outcome" in waves and any(
                survey_has_student_id(sv)
                for sv in course_surveys
                if sv.study_wave == "outcome"
            )
            groupgen_ok = False
            full_timeline = False
            notes = _availability_note_legacy(course_surveys)

        rows.append(
            {
                "course": course,
                "n_surveys": len(course_surveys),
                "waves_present": ", ".join(sorted(waves)),
                "confidence_change_possible": conf_ok,
                "group_outcome_change_possible": outcome_ok if course == "CRPY" else False,
                "teacher_to_groupgen_change_possible": groupgen_ok if course == "CRPY" else False,
                "crpy_full_timeline": full_timeline if course == "CRPY" else False,
                "intake_to_outcome_linkage_possible": (
                    course != "CRPY"
                    and "intake" in waves
                    and "outcome" in waves
                    and any(
                        survey_has_student_id(sv)
                        for sv in course_surveys
                        if sv.study_wave == "outcome"
                    )
                ),
                **{f"n_{wave}": wave_counts.get(wave, 0) for wave in sorted(waves)},
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def _availability_note_legacy(course_surveys: list[Survey]) -> str:
    waves = {sv.study_wave for sv in course_surveys}
    intake = "intake" in waves
    checkin = "checkin" in waves
    outcome = "outcome" in waves
    outcome_named = any(
        survey_has_student_id(sv) for sv in course_surveys if sv.study_wave == "outcome"
    )
    if intake and checkin:
        return "Paired intake->check-in confidence change available."
    if intake and outcome and outcome_named:
        return "No check-in; can link intake baseline to post outcomes by name."
    if intake and outcome:
        return "No check-in; post survey lacks names - class-level summaries only."
    if intake:
        return "Intake baseline only."
    return "Limited survey coverage."


def _availability_note(
    intake: Optional[Survey],
    checkin: Optional[Survey],
    outcome: Optional[Survey],
) -> str:
    return _availability_note_legacy(
        [s for s in (intake, checkin, outcome) if s is not None]
    )


def _confidence_wave_pair(course: str) -> Optional[tuple[str, str]]:
    if course == "CRPY":
        return CRPY_CONFIDENCE_PAIR
    return ("intake", "checkin")


def confidence_change_table(
    surveys: list[Survey],
    scale_name: str = CONFIDENCE_SCALE,
) -> pd.DataFrame:
    """
    Pair pre/post confidence scores using study-wave labels.

    CRPY: group_zero_baseline (check-in) -> post_regroup_features (Get-Features).
    Other courses: intake -> check-in when both exist.
    """
    panel = build_student_panel(surveys)
    score_col = f"score_{scale_name}"
    if panel.empty or score_col not in panel.columns:
        return pd.DataFrame()

    rows = []
    for course in sorted(panel["course"].unique()):
        pair = _confidence_wave_pair(course)
        if pair is None:
            continue
        before_wave, after_wave = pair
        before = panel[
            (panel["course"] == course) & (panel["study_wave"] == before_wave)
        ].set_index("student_key")
        after = panel[
            (panel["course"] == course) & (panel["study_wave"] == after_wave)
        ].set_index("student_key")
        if before.empty or after.empty:
            continue

        linked_keys = before.index.intersection(after.index)
        for student_key in sorted(linked_keys):
            before_score = before.at[student_key, score_col]
            after_score = after.at[student_key, score_col]
            if pd.isna(before_score) or pd.isna(after_score):
                continue
            row = {
                "course": course,
                "student_key": student_key,
                "scale": scale_name,
                "before_wave": before_wave,
                "after_wave": after_wave,
                "before_score": round(float(before_score), 3),
                "after_score": round(float(after_score), 3),
                "change": round(float(after_score - before_score), 3),
            }
            if "survey_date" in before.columns:
                row["before_date"] = before.at[student_key, "survey_date"]
            if "survey_date" in after.columns:
                row["after_date"] = after.at[student_key, "survey_date"]
            rows.append(row)

    return pd.DataFrame(rows)


def group_outcome_change_table(
    surveys: list[Survey],
    *,
    wave_pair: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """
    Compare group-outcome constructs across two CRPY waves (paired by name).

    Default: group_zero_outcome -> teacher_group_outcome.
    Use ``wave_pair=CRPY_TEACHER_TO_GROUPGEN_PAIR`` for teacher -> GroupGen week.
    """
    panel = build_student_panel(surveys)
    if panel.empty:
        return pd.DataFrame()

    outcome_scales = [s.name for s in SCALES if s.kind == "outcome"]
    before_wave, after_wave = wave_pair or CRPY_OUTCOME_PAIR
    rows = []

    crpy = panel[panel["course"] == "CRPY"]
    before = crpy[crpy["study_wave"] == before_wave].set_index("student_key")
    after = crpy[crpy["study_wave"] == after_wave].set_index("student_key")
    if before.empty or after.empty:
        return pd.DataFrame()

    linked_keys = before.index.intersection(after.index)
    for student_key in sorted(linked_keys):
        row: dict = {
            "course": "CRPY",
            "student_key": student_key,
            "before_wave": before_wave,
            "after_wave": after_wave,
        }
        if "survey_date" in before.columns:
            row["before_date"] = before.at[student_key, "survey_date"]
        if "survey_date" in after.columns:
            row["after_date"] = after.at[student_key, "survey_date"]
        for scale_name in outcome_scales:
            col = f"score_{scale_name}"
            if col not in before.columns or col not in after.columns:
                continue
            b = before.at[student_key, col]
            a = after.at[student_key, col]
            if pd.isna(b) or pd.isna(a):
                continue
            row[f"{scale_name}_before"] = round(float(b), 3)
            row[f"{scale_name}_after"] = round(float(a), 3)
            row[f"{scale_name}_change"] = round(float(a - b), 3)
        if any(k.endswith("_change") for k in row):
            rows.append(row)

    return pd.DataFrame(rows)


def group_outcome_change_summary(change_df: pd.DataFrame) -> pd.DataFrame:
    """Mean outcome change per construct for CRPY teacher vs group-zero groups."""
    if change_df.empty:
        return pd.DataFrame()

    rows = []
    change_cols = [c for c in change_df.columns if c.endswith("_change")]
    for col in change_cols:
        scale = col.removesuffix("_change")
        before_col = f"{scale}_before"
        after_col = f"{scale}_after"
        delta = change_df[col].astype(float)
        rows.append(
            {
                "course": change_df["course"].iloc[0],
                "outcome_scale": scale,
                "n_paired": int(delta.notna().sum()),
                "mean_before": round(float(change_df[before_col].mean()), 3),
                "mean_after": round(float(change_df[after_col].mean()), 3),
                "mean_change": round(float(delta.mean()), 3),
                "pct_improved": round(float((delta > 0).mean() * 100), 1),
            }
        )
    return pd.DataFrame(rows)


def regroup_features_to_teacher_outcomes(surveys: list[Survey]) -> pd.DataFrame:
    """Link post-regroup self-efficacy (Get-Features) to teacher-group outcomes."""
    panel = build_student_panel(surveys)
    if panel.empty:
        return pd.DataFrame()

    baseline_scales = [s.name for s in SCALES if s.kind == "baseline"]
    outcome_scales = [s.name for s in SCALES if s.kind == "outcome"]
    features = panel[panel["study_wave"] == "post_regroup_features"].set_index("student_key")
    teacher = panel[panel["study_wave"] == "teacher_group_outcome"].set_index("student_key")
    if features.empty or teacher.empty:
        return pd.DataFrame()

    rows = []
    linked_keys = features.index.intersection(teacher.index)
    for student_key in sorted(linked_keys):
        if features.at[student_key, "course"] != teacher.at[student_key, "course"]:
            continue
        row = {"course": features.at[student_key, "course"], "student_key": student_key}
        for scale_name in baseline_scales:
            col = f"score_{scale_name}"
            if col in features.columns and pd.notna(features.at[student_key, col]):
                row[f"regroup_{scale_name}"] = round(float(features.at[student_key, col]), 3)
        for scale_name in outcome_scales:
            col = f"score_{scale_name}"
            if col in teacher.columns and pd.notna(teacher.at[student_key, col]):
                row[f"teacher_{scale_name}"] = round(float(teacher.at[student_key, col]), 3)
        rows.append(row)

    return pd.DataFrame(rows)


def confidence_change_tests(change_df: pd.DataFrame) -> pd.DataFrame:
    """Paired t-test and Wilcoxon signed-rank test per course."""
    if change_df.empty:
        return pd.DataFrame()

    try:
        from scipy import stats
    except ImportError:  # pragma: no cover
        return pd.DataFrame()

    rows = []
    for course, group in change_df.groupby("course", sort=False):
        before = group["before_score"].astype(float)
        after = group["after_score"].astype(float)
        delta = group["change"].astype(float)
        n = len(group)
        if n < 2:
            continue
        t_stat, t_p = stats.ttest_rel(after, before)
        try:
            w_stat, w_p = stats.wilcoxon(after, before)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        rows.append(
            {
                "course": course,
                "scale": group["scale"].iloc[0],
                "before_wave": group["before_wave"].iloc[0],
                "after_wave": group["after_wave"].iloc[0],
                "n_paired": n,
                "mean_before": round(float(before.mean()), 3),
                "mean_after": round(float(after.mean()), 3),
                "mean_change": round(float(delta.mean()), 3),
                "sd_change": round(float(delta.std(ddof=1)), 3) if n > 1 else np.nan,
                "pct_improved": round(float((delta > 0).mean() * 100), 1),
                "paired_t_p": round(float(t_p), 4),
                "wilcoxon_p": round(float(w_p), 4) if pd.notna(w_p) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def intake_outcome_linkage(surveys: list[Survey]) -> pd.DataFrame:
    """
    Link intake baseline scores to post outcome scores by student identifier.

    Works for CSC412 (post surveys include names). CRPY uses wave-specific
    linkage functions instead.
    """
    panel = build_student_panel(surveys)
    if panel.empty:
        return pd.DataFrame()

    baseline_scales = [s.name for s in SCALES if s.kind == "baseline"]
    outcome_scales = [s.name for s in SCALES if s.kind == "outcome"]
    intake = panel[panel["study_wave"] == "intake"].set_index("student_key")
    outcome = panel[panel["study_wave"] == "outcome"].set_index("student_key")

    rows = []
    for course in sorted(intake["course"].unique()):
        if course == "CRPY":
            continue
        course_outcome = outcome[outcome["course"] == course]
        if course_outcome.empty or not course_outcome["has_student_id"].any():
            continue
        course_intake = intake[intake["course"] == course]
        linked_keys = course_intake.index.intersection(course_outcome.index)
        for student_key in sorted(linked_keys):
            row = {"course": course, "student_key": student_key}
            for scale_name in baseline_scales:
                col = f"score_{scale_name}"
                if col in course_intake.columns:
                    val = course_intake.at[student_key, col]
                    if pd.notna(val):
                        row[f"intake_{scale_name}"] = round(float(val), 3)
            for scale_name in outcome_scales:
                col = f"score_{scale_name}"
                if col in course_outcome.columns:
                    val = course_outcome.at[student_key, col]
                    if pd.notna(val):
                        row[f"post_{scale_name}"] = round(float(val), 3)
            for demo_name in DEMO_COLS:
                if demo_name in course_intake.columns:
                    row[demo_name] = course_intake.at[student_key, demo_name]
            rows.append(row)

    return pd.DataFrame(rows)


def intake_outcome_correlations(linked: pd.DataFrame) -> pd.DataFrame:
    """Correlate intake academic self-efficacy with post outcome constructs."""
    if linked.empty:
        return pd.DataFrame()

    intake_col = f"intake_{CONFIDENCE_SCALE}"
    if intake_col not in linked.columns:
        return pd.DataFrame()

    post_cols = [c for c in linked.columns if c.startswith("post_")]
    rows = []
    for course, group in linked.groupby("course", sort=False):
        for post_col in post_cols:
            subset = group[[intake_col, post_col]].dropna()
            n = len(subset)
            if n < 3:
                continue
            r = subset[intake_col].corr(subset[post_col])
            rows.append(
                {
                    "course": course,
                    "intake_predictor": CONFIDENCE_SCALE,
                    "post_outcome": post_col.removeprefix("post_"),
                    "n_linked": n,
                    "pearson_r": round(float(r), 3) if pd.notna(r) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_confidence_change_figure(
    change_df: pd.DataFrame,
    *,
    stem: str = "confidence_change",
) -> list[Path]:
    if not _HAVE_PLOTS or change_df.empty:
        return []
    saved = []

    plt.figure(figsize=(8, 4))
    sns.histplot(data=change_df, x="change", bins=8, kde=True, color="steelblue")
    plt.axvline(0, color="grey", ls="--", lw=1)
    plt.title("Self-efficacy change (after minus before)")
    plt.xlabel("Change score")
    plt.ylabel("Students")
    plt.tight_layout()
    p = OUT_FIGURES / f"{stem}_distribution.png"
    plt.savefig(p, dpi=120)
    plt.close()
    saved.append(p)

    plot_df = change_df.melt(
        id_vars=["course", "student_key"],
        value_vars=["before_score", "after_score"],
        var_name="wave",
        value_name="score",
    )
    plt.figure(figsize=(7, 4))
    sns.pointplot(data=plot_df, x="wave", y="score", hue="course", dodge=False, errorbar="se")
    plt.title("Self-efficacy: before vs after")
    plt.ylabel("Mean academic self-efficacy (1-7)")
    plt.xlabel("")
    plt.tight_layout()
    p = OUT_FIGURES / f"{stem}_before_vs_after.png"
    plt.savefig(p, dpi=120)
    plt.close()
    saved.append(p)

    return saved


def make_group_outcome_change_figure(
    summary_df: pd.DataFrame,
    *,
    filename: str = "group_outcome_before_vs_after.png",
    title: str = "CRPY group outcomes: student-choice vs teacher groups",
) -> list[Path]:
    if not _HAVE_PLOTS or summary_df.empty:
        return []
    plt.figure(figsize=(9, 5))
    plot_df = summary_df.melt(
        id_vars=["outcome_scale"],
        value_vars=["mean_before", "mean_after"],
        var_name="wave",
        value_name="mean_score",
    )
    sns.barplot(data=plot_df, x="outcome_scale", y="mean_score", hue="wave")
    plt.title(title)
    plt.ylabel("Mean score (1-5)")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p = OUT_FIGURES / filename
    plt.savefig(p, dpi=120)
    plt.close()
    return [p]


def aggregate_class_means(rel: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple survey waves per course into one weighted mean per scale."""
    rows = []
    for (course, scale, kind), group in rel.groupby(["course", "scale", "kind"], sort=False):
        n_total = int(group["n_respondents"].sum())
        if n_total == 0:
            continue
        wmean = np.average(group["mean"], weights=group["n_respondents"])
        rows.append(
            {
                "course": course,
                "scale": scale,
                "kind": kind,
                "likert_max": group["likert_max"].iloc[0],
                "n_respondents": n_total,
                "mean": round(float(wmean), 3),
                "cronbach_alpha": round(float(group["cronbach_alpha"].median()), 3),
            }
        )
    return pd.DataFrame(rows)


def class_construct_summary(rel: pd.DataFrame) -> pd.DataFrame:
    """Per-class subscale means (aggregated across survey waves)."""
    agg = aggregate_class_means(rel)
    cols = ["course", "scale", "kind", "likert_max", "n_respondents", "mean", "cronbach_alpha"]
    return agg[cols].copy()


def cross_class_mean_matrix(rel: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Pivot table: courses x scales for baseline or outcome constructs."""
    agg = aggregate_class_means(rel)
    sub = agg[agg["kind"] == kind].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot(index="course", columns="scale", values="mean")


def class_profile_correlation(rel: pd.DataFrame) -> pd.DataFrame:
    """
    Correlate classes based on their construct-mean profiles.

    Each class is treated as one observation; columns are construct means.
    High correlation means two classes show a similar pattern across scales.
    """
    agg = aggregate_class_means(rel)
    wide = agg.pivot(index="scale", columns="course", values="mean")
    wide = wide.dropna(how="all")
    if wide.shape[1] < 2:
        return pd.DataFrame()
    return wide.corr().round(3)


def pooled_construct_correlation(surveys: list[Survey], kind: str) -> pd.DataFrame:
    """
    Pool students across classes and correlate construct scores (same kind only).

    Uses intake/baseline waves where present; does not mix 1-7 and 1-5 scales
    across kinds, but can correlate within baseline or within outcome sets.
    """
    panel = build_student_panel(surveys)
    if panel.empty:
        return pd.DataFrame()

    scales = [s.name for s in SCALES if s.kind == kind]
    score_cols = [f"score_{name}" for name in scales if f"score_{name}" in panel.columns]
    if len(score_cols) < 2:
        return pd.DataFrame()

    if kind == "baseline":
        waves = {"intake", "checkin", "group_zero_baseline", "post_regroup_features"}
    else:
        waves = {"outcome", "group_zero_outcome", "teacher_group_outcome", "groupgen_group_outcome"}

    subset = panel[panel["study_wave"].isin(waves)][score_cols].dropna(how="any")
    if len(subset) < 3:
        return pd.DataFrame()

    corr = subset.corr().round(3)
    corr.index = corr.index.str.removeprefix("score_")
    corr.columns = corr.columns.str.removeprefix("score_")
    return corr


def make_cross_class_figures(rel: pd.DataFrame) -> list[Path]:
    if not _HAVE_PLOTS or rel.empty:
        return []
    saved = []

    for kind, title, fname in [
        ("baseline", "Baseline constructs by class", "baseline_by_class.png"),
        ("outcome", "Outcome constructs by class", "outcome_by_class_compare.png"),
    ]:
        agg = aggregate_class_means(rel)
        sub = agg[agg["kind"] == kind]
        if sub.empty:
            continue
        plt.figure(figsize=(10, 5))
        sns.barplot(data=sub, x="scale", y="mean", hue="course")
        plt.title(title)
        plt.ylabel("Mean score")
        plt.xlabel("")
        plt.xticks(rotation=20, ha="right")
        plt.legend(title="Course", fontsize=8)
        plt.tight_layout()
        p = OUT_FIGURES / fname
        plt.savefig(p, dpi=120)
        plt.close()
        saved.append(p)

    class_corr = class_profile_correlation(rel)
    if not class_corr.empty and class_corr.shape[0] >= 2:
        plt.figure(figsize=(5, 4))
        sns.heatmap(class_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
        plt.title("Class profile correlation (similar scale patterns?)")
        plt.tight_layout()
        p = OUT_FIGURES / "class_profile_correlation.png"
        plt.savefig(p, dpi=120)
        plt.close()
        saved.append(p)

    return saved


# =====================================================================
# FIGURES
# =====================================================================
def make_figures(rel: pd.DataFrame) -> list[Path]:
    if not _HAVE_PLOTS or rel.empty:
        return []
    saved = []

    outcomes = rel[rel["kind"] == "outcome"]
    if not outcomes.empty:
        plt.figure(figsize=(9, 5))
        sns.barplot(data=outcomes, x="scale", y="mean", hue="course")
        plt.title("Outcome constructs by class (subscale mean, 1-5)")
        plt.ylabel("Mean score (1-5)")
        plt.xlabel("")
        plt.ylim(1, 5)
        plt.xticks(rotation=20, ha="right")
        plt.legend(title="Course", fontsize=8)
        plt.tight_layout()
        p = OUT_FIGURES / "outcomes_by_class.png"
        plt.savefig(p, dpi=120)
        plt.close()
        saved.append(p)

    if not rel.empty:
        plt.figure(figsize=(9, 5))
        rel_valid = rel.dropna(subset=["cronbach_alpha"])
        if not rel_valid.empty:
            sns.barplot(data=rel_valid, x="scale", y="cronbach_alpha", hue="course")
            plt.axhline(0.70, ls="--", color="grey", lw=1, label="alpha = .70")
            plt.title("Scale reliability (Cronbach's alpha) by class")
            plt.ylabel("Cronbach's alpha")
            plt.xlabel("")
            plt.xticks(rotation=20, ha="right")
            plt.legend(title="Course", fontsize=8)
            plt.tight_layout()
            p = OUT_FIGURES / "reliability_by_class.png"
            plt.savefig(p, dpi=120)
            plt.close()
            saved.append(p)
    return saved


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    ensure_output_dirs()
    files = sorted(DATA_DIR.rglob("*.csv"))
    if not files:
        print(f"No CSVs found under {DATA_DIR}")
        return

    surveys = [load_survey(f) for f in files]

    print("=" * 70)
    print("DISCOVERED SURVEYS")
    print("=" * 70)
    for sv in surveys:
        scales = ", ".join(s.name for s in sv.present_scales) or "(no known scales)"
        print(f"  [{sv.survey_type:7}|{sv.study_wave:22}] {sv.course} / {sv.survey_label}")
        print(f"            n={len(sv.df)} | scales: {scales}")

    rel = reliability_table(surveys)
    pooled = pooled_scale_summary(rel)
    equity = demographic_equity(surveys)
    accept = acceptance_summary(surveys)

    rel.to_csv(OUT_DIR / "reliability_and_means.csv", index=False)
    pooled.to_csv(OUT_DIR / "pooled_scale_summary.csv", index=False)
    if not equity.empty:
        equity.to_csv(OUT_DIR / "baseline_equity_by_demographic.csv", index=False)
    if not accept.empty:
        accept.to_csv(OUT_DIR / "acceptance_and_stability.csv", index=False)

    availability = data_availability_report(surveys)
    panel = build_student_panel(surveys)
    change = confidence_change_table(surveys)
    change_tests = confidence_change_tests(change)
    group_outcome_change = group_outcome_change_table(surveys)
    group_outcome_summary = group_outcome_change_summary(group_outcome_change)
    teacher_groupgen_change = group_outcome_change_table(
        surveys, wave_pair=CRPY_TEACHER_TO_GROUPGEN_PAIR
    )
    teacher_groupgen_summary = group_outcome_change_summary(teacher_groupgen_change)
    regroup_linked = regroup_features_to_teacher_outcomes(surveys)
    linked = intake_outcome_linkage(surveys)
    correlations = intake_outcome_correlations(linked)

    availability.to_csv(OUT_DIR / "data_availability.csv", index=False)
    if not panel.empty:
        panel.to_csv(OUT_DIR / "student_panel.csv", index=False)
    if not change.empty:
        change.to_csv(OUT_DIR / "confidence_change.csv", index=False)
    if not change_tests.empty:
        change_tests.to_csv(OUT_DIR / "confidence_change_tests.csv", index=False)
    if not group_outcome_change.empty:
        group_outcome_change.to_csv(OUT_DIR / "group_outcome_change.csv", index=False)
    if not group_outcome_summary.empty:
        group_outcome_summary.to_csv(OUT_DIR / "group_outcome_change_summary.csv", index=False)
    if not teacher_groupgen_change.empty:
        teacher_groupgen_change.to_csv(
            OUT_DIR / "teacher_to_groupgen_outcome_change.csv", index=False
        )
    if not teacher_groupgen_summary.empty:
        teacher_groupgen_summary.to_csv(
            OUT_DIR / "teacher_to_groupgen_outcome_summary.csv", index=False
        )
    if not regroup_linked.empty:
        regroup_linked.to_csv(OUT_DIR / "regroup_to_teacher_outcomes.csv", index=False)
    if not linked.empty:
        linked.to_csv(OUT_DIR / "intake_outcome_linked.csv", index=False)
    if not correlations.empty:
        correlations.to_csv(OUT_DIR / "intake_outcome_correlations.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n" + "=" * 70)
    print("RELIABILITY + SUBSCALE MEANS (per class)")
    print("=" * 70)
    print(rel.to_string(index=False))

    print("\n" + "=" * 70)
    print("POOLED SCALE SUMMARY (sample-size weighted across classes)")
    print("=" * 70)
    print(pooled.to_string(index=False))

    if not equity.empty:
        print("\n" + "=" * 70)
        print("BASELINE CONSTRUCTS BY DEMOGRAPHIC (where demographics exist)")
        print("=" * 70)
        print(equity.to_string(index=False))

    if not accept.empty:
        print("\n" + "=" * 70)
        print("GROUPGEN ACCEPTANCE + GROUP STABILITY")
        print("=" * 70)
        print(accept.to_string(index=False))

    figs = make_figures(rel)
    figs.extend(make_confidence_change_figure(change))
    figs.extend(make_group_outcome_change_figure(group_outcome_summary))

    print("\n" + "=" * 70)
    print("DATA AVAILABILITY (what longitudinal analyses are possible)")
    print("=" * 70)
    print(availability.to_string(index=False))

    if not change.empty:
        print("\n" + "=" * 70)
        print("CONFIDENCE CHANGE (group-zero baseline -> post-regroup features)")
        print("=" * 70)
        print(change.to_string(index=False))
        if not change_tests.empty:
            print("\nPaired tests:")
            print(change_tests.to_string(index=False))
    else:
        print("\n(No paired confidence-change waves found.)")

    if not group_outcome_summary.empty:
        print("\n" + "=" * 70)
        print("GROUP OUTCOME CHANGE (student-choice -> teacher groups, CRPY)")
        print("=" * 70)
        print(group_outcome_summary.to_string(index=False))

    if not teacher_groupgen_summary.empty:
        print("\n" + "=" * 70)
        print("GROUP OUTCOME CHANGE (teacher groups -> GroupGen groups, CRPY)")
        print("=" * 70)
        print(teacher_groupgen_summary.to_string(index=False))

    if not regroup_linked.empty:
        print("\n" + "=" * 70)
        print("POST-REGROUP FEATURES -> TEACHER-GROUP OUTCOMES (CRPY)")
        print("=" * 70)
        print(f"Linked students: {len(regroup_linked)}")

    if not linked.empty:
        print("\n" + "=" * 70)
        print("INTAKE -> POST OUTCOME LINKAGE (paired by student ID)")
        print("=" * 70)
        print(f"Linked students: {len(linked)} across {linked['course'].nunique()} course(s)")
        if not correlations.empty:
            print("\nBaseline self-efficacy vs post outcomes (Pearson r):")
            print(correlations.to_string(index=False))

    print("\n" + "=" * 70)
    print("LIMITATIONS / DATA-LINKAGE NOTES")
    print("=" * 70)
    linkage_issues = []
    for sv in surveys:
        if sv.survey_type in ("outcome",):
            norm_map = {normalize(c): c for c in sv.df.columns}
            if find_column(norm_map, NAME_KEY) is None:
                linkage_issues.append(f"{sv.course}/{sv.survey_label}")
    if linkage_issues:
        print("- Outcome surveys with NO name/ID (cannot link to demographics or")
        print("  baseline scores; outcome equity-by-demographic not computable):")
        for x in linkage_issues:
            print(f"    * {x}")
    print("- Scales use different ranges (self-efficacy 1-7; others 1-5):")
    print("  compare within a scale, or z-standardize before combining.")
    print("- No group/team ID in outcome exports: cannot yet model nesting of")
    print("  students within teams (needed for multilevel outcome models).")

    print("\nOutputs written to:")
    print(f"  tables:  {OUT_DIR}")
    print(f"  figures: {OUT_FIGURES}")
    for f in sorted(OUT_DIR.glob("*.csv")):
        print("  -", f.name)
    for f in sorted(OUT_FIGURES.glob("*.png")):
        print("  -", f.name)


if __name__ == "__main__":
    main()
