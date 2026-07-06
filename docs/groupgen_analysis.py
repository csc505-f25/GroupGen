"""
GroupGen — Study Analysis Pipeline
==================================

A reusable analysis pipeline for the GroupGen classroom study. It is written to
work on the *pilot* data collected so far (multiple classes, multiple survey
types) and to drop straight onto next semester's data with no code changes:
just add the new CSV exports under ``docs/Inclass_data/<COURSE>/``.

What it does
------------
1. Discovers every survey CSV under ``docs/Inclass_data`` and classifies it as an
   INTAKE/baseline survey, a CHECK-IN survey, or a POST/outcome survey based on
   which questions it contains.
2. Cleans Google Forms quirks (trailing spaces, text Likert scales, the
   "Somtimes" typo, reverse-worded items).
3. Scores each validated construct (subscale = mean of its items) and computes
   scale reliability (Cronbach's alpha) per class.
4. Writes tidy, paper-ready tables (CSV) and figures to ``docs/analysis_output``.

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
try:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
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
OUT_DIR = SCRIPT_DIR / "analysis_output"


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
ACCEPT_RECOMMEND_KEY = "would you recommend a system like groupgen"
STABILITY_KEY = "did you have the same group"


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
    return Survey(course, label, path, df, present, stype)


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
        p = OUT_DIR / "outcomes_by_class.png"
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
            p = OUT_DIR / "reliability_by_class.png"
            plt.savefig(p, dpi=120)
            plt.close()
            saved.append(p)
    return saved


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
        print(f"  [{sv.survey_type:7}] {sv.course} / {sv.survey_label}")
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

    print("\nOutputs written to:", OUT_DIR)
    for f in sorted(OUT_DIR.glob("*.csv")):
        print("  -", f.name)
    for f in figs:
        print("  -", f.name)


if __name__ == "__main__":
    main()
