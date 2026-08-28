"""
RQ-aligned GroupGen study analysis (pilot + next-semester ready).

Research Question 1
-------------------
IV: Grouping method (GroupGen vs teacher control).
DVs:
  (a) Feature levels over the semester (Get-Features / Check-in trajectories)
  (b) Collaboration satisfaction & related group outcomes (reflection surveys)

Research Question 2
-------------------
Primary predictor: within-team feature similarity (NOT grouping method).
Covariates: age variance, demographic isolation, group size.
DV: collaboration / project satisfaction.

Pilot note
----------
CRPY is sequential (student-choice → teacher → GroupGen), not a parallel RCT.
Use these analyses for instrument checks, effect-size planning, and code paths —
not confirmatory causal claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import groupgen_analysis as ga

DOCS_ROOT = Path(__file__).resolve().parent
ROSTER_TEMPLATE_PATH = DOCS_ROOT / "Inclass_data" / "team_roster_template.csv"
ROSTER_CANDIDATES = (
    DOCS_ROOT / "Inclass_data" / "team_rosters.csv",
    DOCS_ROOT / "Inclass_data" / "CRPY" / "team_rosters.csv",
)

# Map CRPY outcome waves → grouping condition labels for RQ1b.
CONDITION_BY_WAVE = {
    "group_zero_outcome": "student_choice",
    "teacher_group_outcome": "teacher",
    "groupgen_group_outcome": "groupgen",
}

FEATURE_SCALES = [
    "academic_self_efficacy",
    "classroom_engagement",
    "study_habits",
]
OUTCOME_SCALES = [
    "group_belonging",
    "group_satisfaction",
    "psychological_safety",
    "team_learning",
]

# Primary collaboration DV for RQ hypotheses.
PRIMARY_SATISFACTION_SCALE = "group_satisfaction"

ROSTER_REQUIRED_COLS = ("student_key", "team_id", "condition")
ROSTER_OPTIONAL_COLS = ("group_size", "course", "age")


@dataclass
class RqStudyResults:
    surveys: list
    panel: pd.DataFrame
    inventory: pd.DataFrame
    design_readiness: pd.DataFrame
    # RQ1a
    feature_change: pd.DataFrame
    feature_change_tests: pd.DataFrame
    # RQ1b
    outcomes_by_condition: pd.DataFrame
    teacher_vs_groupgen: pd.DataFrame
    teacher_vs_groupgen_summary: pd.DataFrame
    # RQ2
    rq2_status: dict[str, Any]
    roster: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_composition: pd.DataFrame = field(default_factory=pd.DataFrame)
    rq2_model_ready: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Instruments
    reliability: pd.DataFrame = field(default_factory=pd.DataFrame)


# =====================================================================
# LOADING / INVENTORY
# =====================================================================
def load_course_surveys(course: str = "CRPY", data_dir: Path | None = None) -> list:
    root = Path(data_dir) if data_dir is not None else Path(ga.DATA_DIR) / course
    files = sorted(root.glob("*.csv"))
    # Ignore roster files sitting next to survey exports.
    files = [p for p in files if "roster" not in p.name.lower()]
    if not files:
        raise FileNotFoundError(f"No survey CSVs found in {root}")
    return [ga.load_survey(p) for p in files]


def survey_inventory(surveys: list) -> pd.DataFrame:
    rows = []
    for sv in surveys:
        condition = CONDITION_BY_WAVE.get(sv.study_wave, "")
        rows.append(
            {
                "course": sv.course,
                "study_wave": sv.study_wave,
                "condition": condition or "(not an outcome condition wave)",
                "survey": sv.survey_label,
                "n_responses": len(sv.df),
                "has_student_id": ga.survey_has_student_id(sv),
                "scales": ", ".join(s.name for s in sv.present_scales) or "(none)",
            }
        )
    return pd.DataFrame(rows)


def design_readiness_table(surveys: list, roster: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Checklist of what the current data can support for the new RQs.
    """
    waves = {sv.study_wave for sv in surveys}
    courses = {sv.course for sv in surveys}
    has_features = "post_regroup_features" in waves or "intake" in waves
    has_checkin = "group_zero_baseline" in waves or "checkin" in waves
    has_teacher = "teacher_group_outcome" in waves
    has_groupgen = "groupgen_group_outcome" in waves
    roster_ok = roster is not None and not roster.empty
    has_age = roster_ok and "age" in roster.columns and roster["age"].notna().any()
    has_team_id = roster_ok and "team_id" in roster.columns
    has_condition = roster_ok and "condition" in roster.columns

    rows = [
        {
            "requirement": "RQ1 IV: parallel GroupGen vs teacher arms",
            "status": "partial_pilot",
            "detail": (
                "CRPY has teacher AND GroupGen outcome waves, but students saw them "
                "sequentially (not randomized parallel arms)."
            ),
        },
        {
            "requirement": "RQ1a: Get-Features / baseline feature scores",
            "status": "ready" if has_features else "missing",
            "detail": "Found Get-Features or intake wave." if has_features else "No feature intake found.",
        },
        {
            "requirement": "RQ1a: Check-in / mid-semester feature scores",
            "status": "ready" if has_checkin else "missing",
            "detail": "Found check-in wave." if has_checkin else "No check-in wave found.",
        },
        {
            "requirement": "RQ1b: collaboration outcomes under teacher grouping",
            "status": "ready" if has_teacher else "missing",
            "detail": "Teacher reflection wave present." if has_teacher else "Missing teacher outcome wave.",
        },
        {
            "requirement": "RQ1b: collaboration outcomes under GroupGen",
            "status": "ready" if has_groupgen else "missing",
            "detail": "GroupGen reflection wave present." if has_groupgen else "Missing GroupGen outcome wave.",
        },
        {
            "requirement": "RQ2: team roster (student_key, team_id, condition)",
            "status": "ready" if has_team_id and has_condition else "blocked",
            "detail": (
                f"Roster loaded ({len(roster)} rows)."
                if roster_ok
                else f"Add a roster CSV (see {ROSTER_TEMPLATE_PATH.name})."
            ),
        },
        {
            "requirement": "RQ2 covariates: group_size on roster",
            "status": (
                "ready"
                if roster_ok and "group_size" in roster.columns and roster["group_size"].notna().any()
                else "blocked"
            ),
            "detail": "Needed to test team-size effects.",
        },
        {
            "requirement": "RQ2 covariates: age (for within-team age variance)",
            "status": "ready" if has_age else "blocked",
            "detail": "Age not in current Get-Features demographics; collect next wave or add to roster.",
        },
        {
            "requirement": "RQ2: gender/ethnicity for isolation metrics",
            "status": "partial_pilot" if any(sv.course in courses for sv in surveys) else "missing",
            "detail": "Gender/ethnicity exist on Get-Features; join to roster via student_key when roster exists.",
        },
    ]
    return pd.DataFrame(rows)


# =====================================================================
# RQ1a — FEATURE TRAJECTORIES
# =====================================================================
def feature_change_long(surveys: list, scales: list[str] | None = None) -> pd.DataFrame:
    """
    Paired feature change for every baseline scale with a pre/post wave pair.

    Pilot CRPY pairing (existing labels):
      before = group_zero_baseline (Check-in)
      after  = post_regroup_features (Get-Features)

    Next-semester design will reverse the *substantive* roles
    (Get-Features = T0 baseline used for grouping; Check-in = T1+ follow-up),
    but the same pairing code path applies once wave labels are configured.
    """
    scales = scales or FEATURE_SCALES
    frames = []
    for scale in scales:
        part = ga.confidence_change_table(surveys, scale_name=scale)
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def feature_change_tests(change_df: pd.DataFrame) -> pd.DataFrame:
    """Paired tests per course × scale (supports multi-scale RQ1a tables)."""
    if change_df.empty:
        return pd.DataFrame()

    try:
        from scipy import stats
    except ImportError:
        return pd.DataFrame()

    rows = []
    group_cols = ["course", "scale"] if "scale" in change_df.columns else ["course"]
    for keys, group in change_df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        before = group["before_score"].astype(float)
        after = group["after_score"].astype(float)
        delta = group["change"].astype(float)
        n = len(group)
        if n < 2:
            continue
        _t_stat, t_p = stats.ttest_rel(after, before)
        try:
            _w_stat, w_p = stats.wilcoxon(after, before)
        except ValueError:
            w_p = float("nan")
        rows.append(
            {
                **key_map,
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


def make_feature_trajectory_figure(
    change_df: pd.DataFrame,
    *,
    filename: str = "rq1_feature_trajectories.png",
) -> list[Path]:
    if not ga._HAVE_PLOTS or change_df.empty:
        return []

    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = change_df.melt(
        id_vars=["course", "student_key", "scale"],
        value_vars=["before_score", "after_score"],
        var_name="timepoint",
        value_name="score",
    )
    plot_df["timepoint"] = plot_df["timepoint"].map(
        {"before_score": "earlier wave", "after_score": "later wave"}
    )

    g = sns.catplot(
        data=plot_df,
        x="timepoint",
        y="score",
        col="scale",
        kind="point",
        errorbar="se",
        height=3.5,
        aspect=0.9,
        sharey=False,
    )
    g.figure.suptitle("RQ1a (pilot): feature levels across available waves", y=1.03)
    g.set_axis_labels("", "Mean score")
    g.set_titles("{col_name}")
    path = ga.OUT_FIGURES / filename
    g.savefig(path, dpi=120)
    plt.close("all")
    return [path]


# =====================================================================
# RQ1b — COLLABORATION OUTCOMES BY CONDITION
# =====================================================================
def outcomes_by_condition(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Class-level mean outcomes tagged by grouping condition.

    Pilot: student_choice / teacher / groupgen from CRPY reflection waves.
    Future: same table, but each student belongs to one parallel arm.
    """
    if panel.empty:
        return pd.DataFrame()

    rows = []
    for wave, condition in CONDITION_BY_WAVE.items():
        sub = panel[panel["study_wave"] == wave]
        if sub.empty:
            continue
        for scale in OUTCOME_SCALES:
            col = f"score_{scale}"
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "course": sub["course"].iloc[0],
                    "study_wave": wave,
                    "condition": condition,
                    "scale": scale,
                    "n": int(vals.shape[0]),
                    "mean": round(float(vals.mean()), 3),
                    "sd": round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def teacher_vs_groupgen_change(surveys: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired RQ1b contrast using pilot sequential teacher → GroupGen weeks."""
    change = ga.group_outcome_change_table(
        surveys, wave_pair=ga.CRPY_TEACHER_TO_GROUPGEN_PAIR
    )
    summary = ga.group_outcome_change_summary(change)
    return change, summary


def make_outcomes_by_condition_figure(
    outcomes_df: pd.DataFrame,
    *,
    filename: str = "rq1_outcomes_by_condition.png",
) -> list[Path]:
    if not ga._HAVE_PLOTS or outcomes_df.empty:
        return []

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Focus on the two arms that match the planned study.
    plot_df = outcomes_df[outcomes_df["condition"].isin(["teacher", "groupgen"])].copy()
    if plot_df.empty:
        plot_df = outcomes_df.copy()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="scale", y="mean", hue="condition")
    plt.title("RQ1b (pilot): collaboration outcomes by grouping condition")
    plt.ylabel("Mean score (1-5)")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(1, 5)
    plt.tight_layout()
    path = ga.OUT_FIGURES / filename
    plt.savefig(path, dpi=120)
    plt.close()
    return [path]


def make_teacher_vs_groupgen_figure(
    summary_df: pd.DataFrame,
    *,
    filename: str = "rq1_teacher_vs_groupgen.png",
) -> list[Path]:
    return ga.make_group_outcome_change_figure(
        summary_df,
        filename=filename,
        title="RQ1b (pilot): teacher groups vs GroupGen groups (paired)",
    )


# =====================================================================
# RQ2 — TEAM COMPOSITION (ROSTER-GATED)
# =====================================================================
def find_roster_path(explicit: Path | None = None) -> Optional[Path]:
    if explicit is not None and explicit.exists():
        return explicit
    for path in ROSTER_CANDIDATES:
        if path.exists():
            return path
    return None


def load_roster(path: Path | None = None) -> pd.DataFrame:
    roster_path = find_roster_path(path)
    if roster_path is None:
        return pd.DataFrame()
    df = pd.read_csv(roster_path)
    df.columns = df.columns.str.strip().str.lower()
    if "student_key" in df.columns:
        df["student_key"] = (
            df["student_key"].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        )
    if "condition" in df.columns:
        df["condition"] = df["condition"].astype(str).str.strip().str.lower()
    return df


def rq2_status(roster: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Any]:
    missing = [c for c in ROSTER_REQUIRED_COLS if c not in roster.columns]
    ready = roster is not None and not roster.empty and not missing
    demos_available = any(c in panel.columns for c in ("gender", "ethnicity"))
    return {
        "roster_loaded": not roster.empty,
        "n_roster_rows": int(len(roster)) if not roster.empty else 0,
        "missing_required_columns": missing,
        "ready_for_regression": ready,
        "has_group_size": "group_size" in roster.columns if not roster.empty else False,
        "has_age": "age" in roster.columns if not roster.empty else False,
        "demographics_on_panel": demos_available,
        "template_path": str(ROSTER_TEMPLATE_PATH),
        "blocked_reason": (
            None
            if ready
            else (
                "No team roster CSV found. Create Inclass_data/team_rosters.csv "
                f"using {ROSTER_TEMPLATE_PATH.name} (student_key, team_id, condition)."
            )
        ),
    }


def _isolation_flag(series: pd.Series) -> pd.Series:
    """True when a student is the only person with their demographic level on the team."""
    counts = series.fillna("unknown").astype(str).str.strip().value_counts()
    return series.fillna("unknown").astype(str).str.strip().map(lambda x: counts.get(x, 0) == 1)


def build_team_composition(
    roster: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    feature_wave: str = "post_regroup_features",
) -> pd.DataFrame:
    """
    Team-level covariates for RQ2.

    Requires roster with student_key, team_id, condition.
    Feature similarity uses Get-Features (or configured feature_wave) scores.
    """
    if roster.empty:
        return pd.DataFrame()
    missing = [c for c in ROSTER_REQUIRED_COLS if c not in roster.columns]
    if missing:
        return pd.DataFrame()

    feat = panel[panel["study_wave"] == feature_wave].copy()
    if feat.empty:
        # Fall back to any intake-like wave.
        feat = panel[panel["study_wave"].isin(["intake", "post_regroup_features"])].copy()

    merged = roster.merge(feat, on="student_key", how="left", suffixes=("", "_panel"))
    if "course" not in merged.columns and "course_panel" in merged.columns:
        merged["course"] = merged["course_panel"]

    score_cols = [f"score_{s}" for s in FEATURE_SCALES if f"score_{s}" in merged.columns]
    rows = []
    for (team_id, condition), g in merged.groupby(["team_id", "condition"], dropna=False):
        row: dict[str, Any] = {
            "team_id": team_id,
            "condition": condition,
            "n_members_roster": int(len(g)),
            "n_members_with_features": int(g[score_cols[0]].notna().sum()) if score_cols else 0,
        }
        if "group_size" in g.columns and g["group_size"].notna().any():
            row["group_size"] = float(g["group_size"].dropna().iloc[0])
        else:
            row["group_size"] = float(len(g))

        if "age" in g.columns and g["age"].notna().sum() >= 2:
            row["age_variance"] = float(pd.to_numeric(g["age"], errors="coerce").var(ddof=1))
        else:
            row["age_variance"] = np.nan

        if "gender" in g.columns:
            row["pct_gender_isolated"] = float(_isolation_flag(g["gender"]).mean())
        if "ethnicity" in g.columns:
            row["pct_ethnicity_isolated"] = float(_isolation_flag(g["ethnicity"]).mean())

        # Mean pairwise Euclidean distance on available feature scores (higher = less similar).
        if score_cols and g[score_cols].dropna().shape[0] >= 2:
            mat = g[score_cols].dropna().to_numpy(dtype=float)
            dists = []
            for i in range(len(mat)):
                for j in range(i + 1, len(mat)):
                    dists.append(float(np.linalg.norm(mat[i] - mat[j])))
            row["feature_dissimilarity"] = float(np.mean(dists)) if dists else np.nan
            row["feature_similarity"] = (
                float(-np.mean(dists)) if dists else np.nan
            )  # higher = more similar
        else:
            row["feature_dissimilarity"] = np.nan
            row["feature_similarity"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def build_rq2_student_table(
    roster: pd.DataFrame,
    panel: pd.DataFrame,
    team_comp: pd.DataFrame,
    *,
    outcome_wave: str = "groupgen_group_outcome",
    outcome_scale: str = PRIMARY_SATISFACTION_SCALE,
) -> pd.DataFrame:
    """
    Student-level analysis table for:
      satisfaction ~ feature_similarity + age_variance + isolation + group_size
    """
    if roster.empty or team_comp.empty:
        return pd.DataFrame()

    outcomes = panel[panel["study_wave"] == outcome_wave].copy()
    score_col = f"score_{outcome_scale}"
    if outcomes.empty or score_col not in outcomes.columns:
        return pd.DataFrame()

    base = roster.merge(
        outcomes[["student_key", score_col]],
        on="student_key",
        how="inner",
    ).rename(columns={score_col: "satisfaction"})
    out = base.merge(team_comp, on=["team_id", "condition"], how="left")
    return out


# =====================================================================
# ORCHESTRATION
# =====================================================================
def compute_rq_study(
    *,
    course: str = "CRPY",
    roster_path: Path | None = None,
    ga_module: Any | None = None,
) -> RqStudyResults:
    global ga
    if ga_module is not None:
        ga = ga_module

    surveys = load_course_surveys(course)
    panel = ga.build_student_panel(surveys)
    roster = load_roster(roster_path)
    status = rq2_status(roster, panel)

    feature_change = feature_change_long(surveys)
    feature_tests = feature_change_tests(feature_change)
    outcomes = outcomes_by_condition(panel)
    tvg, tvg_sum = teacher_vs_groupgen_change(surveys)

    team_comp = pd.DataFrame()
    rq2_table = pd.DataFrame()
    if status["ready_for_regression"]:
        team_comp = build_team_composition(roster, panel)
        rq2_table = build_rq2_student_table(roster, panel, team_comp)

    return RqStudyResults(
        surveys=surveys,
        panel=panel,
        inventory=survey_inventory(surveys),
        design_readiness=design_readiness_table(surveys, roster),
        feature_change=feature_change,
        feature_change_tests=feature_tests,
        outcomes_by_condition=outcomes,
        teacher_vs_groupgen=tvg,
        teacher_vs_groupgen_summary=tvg_sum,
        rq2_status=status,
        roster=roster,
        team_composition=team_comp,
        rq2_model_ready=rq2_table,
        reliability=ga.reliability_table(surveys),
    )


def save_rq_outputs(results: RqStudyResults) -> list[Path]:
    ga.ensure_output_dirs()
    saved: list[Path] = []

    table_map = {
        "rq_survey_inventory.csv": results.inventory,
        "rq_design_readiness.csv": results.design_readiness,
        "rq1_feature_change.csv": results.feature_change,
        "rq1_feature_change_tests.csv": results.feature_change_tests,
        "rq1_outcomes_by_condition.csv": results.outcomes_by_condition,
        "rq1_teacher_vs_groupgen_change.csv": results.teacher_vs_groupgen,
        "rq1_teacher_vs_groupgen_summary.csv": results.teacher_vs_groupgen_summary,
        "rq_reliability.csv": results.reliability,
    }
    if not results.team_composition.empty:
        table_map["rq2_team_composition.csv"] = results.team_composition
    if not results.rq2_model_ready.empty:
        table_map["rq2_student_model_table.csv"] = results.rq2_model_ready

    for name, df in table_map.items():
        if df is not None and not df.empty:
            path = ga.write_table(df, name)
            if path:
                saved.append(path)

    saved.extend(make_feature_trajectory_figure(results.feature_change))
    saved.extend(make_outcomes_by_condition_figure(results.outcomes_by_condition))
    if not results.teacher_vs_groupgen_summary.empty:
        saved.extend(make_teacher_vs_groupgen_figure(results.teacher_vs_groupgen_summary))

    return saved


def ensure_roster_template() -> Path:
    """Write a blank roster template if missing (does not overwrite)."""
    ROSTER_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ROSTER_TEMPLATE_PATH.exists():
        template = pd.DataFrame(
            [
                {
                    "student_key": "first last",
                    "team_id": "T1",
                    "condition": "groupgen",  # or teacher
                    "group_size": 4,
                    "course": "CRPY",
                    "age": "",
                }
            ]
        )
        template.to_csv(ROSTER_TEMPLATE_PATH, index=False)
    return ROSTER_TEMPLATE_PATH


def run_rq_study(
    ga_module: Any | None = None,
    *,
    course: str = "CRPY",
    save: bool = True,
    display: bool = True,
) -> RqStudyResults:
    """Notebook / CLI entry point."""
    global ga
    if ga_module is not None:
        ga = ga_module

    ensure_roster_template()
    results = compute_rq_study(course=course, ga_module=ga)
    paths = save_rq_outputs(results) if save else []

    if display:
        try:
            from IPython.display import Image, display as ipy_display
        except ImportError:
            ipy_display = print
            Image = None

        print("=" * 70)
        print("STUDY DESIGN READINESS (pilot data vs new RQs)")
        print("=" * 70)
        ipy_display(results.design_readiness)

        print("\nSurvey inventory:")
        ipy_display(results.inventory)

        print("\n--- RQ1a: Feature trajectories ---")
        if results.feature_change_tests.empty:
            print("No paired feature waves found.")
        else:
            ipy_display(results.feature_change_tests)

        print("\n--- RQ1b: Outcomes by condition ---")
        if results.outcomes_by_condition.empty:
            print("No condition-tagged outcome waves found.")
        else:
            ipy_display(results.outcomes_by_condition)

        if not results.teacher_vs_groupgen_summary.empty:
            print("\nTeacher vs GroupGen (paired change summary):")
            ipy_display(results.teacher_vs_groupgen_summary)

        print("\n--- RQ2: Composition model status ---")
        for k, v in results.rq2_status.items():
            print(f"  {k}: {v}")
        if not results.team_composition.empty:
            ipy_display(results.team_composition.head())

        if Image is not None:
            for p in paths:
                if str(p).endswith(".png") and Path(p).exists():
                    ipy_display(Image(filename=str(p)))

        print(f"\nSaved outputs under:\n  {ga.OUT_DIR}\n  {ga.OUT_FIGURES}")

    return results


def main() -> None:
    ga.configure_study_paths(DOCS_ROOT)
    run_rq_study(ga, save=True, display=True)


if __name__ == "__main__":
    main()
