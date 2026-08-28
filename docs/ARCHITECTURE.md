# GroupGen — System Architecture

How data flows through GroupGen for developers and researchers auditing the classroom study pipeline.

> **Branch `feature-only-clustering`:** `run_grouping_pipeline` does **not** run demographic fairness swaps. Gender/Diversity are ingest/display fields only.

## Design goals

1. **Homogeneous skill groups** — cluster on motivation, self-esteem, work ethic, and learning style (Manhattan distance).
2. **Predictable sizes** — `ceil(n / target_size)` groups, then balance (e.g. 31 @ 5 → seven groups of 4 and 5).
3. **One production path** — Web UI, API, and CLI: `prepare_for_grouping` → `run_grouping_pipeline`.
4. **Raw Form support** — teachers upload Google Forms CSV without a manual scoring step.

## Entry points

| Entry | Module | Persistence |
|-------|--------|-------------|
| Web UI | `frontend/src/app/page.tsx` → `POST /generate-groups` | None |
| REST API | `backend/api.py` | None |
| CLI | `backend/generate_groups.py` | `backend/output/runs/<timestamp>_<uuid>/` |
| Research | `backend/run_full_evaluation.py` | `backend/output_plots/runs/<timestamp>_<uuid>/` |

## Lifecycle diagram

```
CSV bytes / file path
        │
        ▼
┌────────────────────────────────────────────┐
│  prepare_for_grouping (data_loader)        │
│  read (delimiter sniff)                     │
│  if raw Google Form → group_gen_intake      │
│  normalize → drop blanks → validate         │
│  → preprocess → validate again              │
└────────────────────────────────────────────┘
        │  DataFrame: 7 profile columns
        ▼
┌────────────────────────────────────────────┐
│  run_grouping_pipeline (pipeline)           │
│  features + Manhattan matrix                │
│  kmedoids_pam (seed 42)                     │
│  enforce_group_size → invariants            │
└────────────────────────────────────────────┘
        │
        ▼
   API JSON  |  CLI: final_groups.csv, report, manifest
               |  raw_responses_with_groups.csv (raw input only)
```

## Raw Google Form detection

`group_gen_intake.is_raw_google_form(df)` is true when headers include **both**:

- `first and last name`
- `magic wand` (divider questions)

Otherwise `prepare_for_grouping` expects the seven clustering columns (or aliases via `COLUMN_ALIASES`).

## `backend/group_gen_intake.py`

**Role:** Convert survey export → `Name`, `Gender`, `Motivation`, `Self_Esteem`, `Work_Ethic`, `Learning_Style`, `Diversity`.

| Phase | Logic |
|-------|--------|
| Section bounds | Four magic wand column indices; fallbacks for keyword anchors |
| Learning style | All columns between email/name and first wand → VAK scoring |
| Self-esteem | After wand 1 until wand 2 | 1–7 mean → `_mean_to_1_4_se` |
| Motivation | After wand 2 until wand 3 | mean → `_mean_to_1_4_mot_we` |
| Work ethic | After wand 3 until wand 4 | mean → `_mean_to_1_4_mot_we` |
| Demographics | Gender / ethnicity columns after last wand |

`merge_groups_into_raw_export(raw_df, grouped_df)` — CLI adds `Group_ID` to the original export.

## `backend/vak_answer_catalog.py`

**Role:** Paper-accurate VAK scoring when cells contain **full option text**.

1. `_VAK_OPTIONS` — 30 questions × 3 strings, each tagged `A`, `B`, or `C` per option (a)/(b)/(c) in the instrument.
2. `classify_vak_letter(cell)` — text → `A` | `B` | `C` (exact match after normalize).
3. `score_learning_style_from_row` — count A/B/C; majority → Visual / Auditory / Kinesthetic.
4. Ties → `learning_style_from_abc_counts` picks one tied letter (seed from that row’s answers).

Does **not** guess style from keywords; only catalog text (and bare `A`/`B`/`C` if exported).

## `backend/data_loader.py`

- `prepare_for_grouping` — **production ingest**; calls intake when needed.
- `read_csv_source` — `utf-8-sig`, delimiter auto-detect.
- `validate_data` / `preprocess_data` — shared validation after intake or for pre-scored CSV.

## `backend/pipeline.py`

Orchestrator; mutates only `labels`, not profile columns.

Production lifecycle: features → Manhattan matrix → K-Medoids → size balance → invariants → per-team cohesion. Demographic fields in the roster are never fed into distance or swap logic on this branch.

## `backend/cohesion.py` / `backend/research_export.py`

- `compute_team_cohesion` — per-team mean pairwise L1 (lower = more similar profiles); attached to `GroupingResult.team_cohesion`.
- `build_research_dataframe` / `write_research_exports` — CLI writes `research_export.csv` (student_id, z-features, cohesion IV) and `team_cohesion.csv`.

## `backend/clustering.py` / `kmedoids.py` / `invariants.py`

Features (psychometric only), PAM, size enforcement, `InvariantViolation` → HTTP 422.

`fairness_distribution.py` and isolation helpers in `clustering.py` remain for research/tests but are not invoked by `run_grouping_pipeline`.

## `backend/api.py`

- `POST /generate-groups` — multipart CSV + `group_size` query param.
- Ingest errors → HTTP 400; invariant failures → 422.
- `to_json_safe` on full response tree.

## `backend/generate_groups.py` + `run_manifest.py`

Per-run folder under `backend/output/runs/`:

- `final_groups.csv` — teacher roster + `Group_ID`
- `research_export.csv` — anonymized `student_id`, z-scored features, `intra_team_cohesion_score`
- `team_cohesion.csv` — one row per team
- `run_manifest.json` — seed, strategy, PAM cost, git commit

Raw Form inputs also write `raw_responses_with_groups.csv`.

## Sample data

| Path | Rows | Notes |
|------|------|--------|
| `backend/data/templates/classroom_template.csv` | 30 | Pre-scored; default CLI input |
| `backend/data/templates/google_form_sample.csv` | 30 | Abbreviated raw form (3 VAK items + wands) for intake tests |

Rebuild: `python backend/data/templates/_build_samples.py`

## Research vs production

`run_full_evaluation.py` uses `prepare_for_grouping` on the classroom template but **not** `run_grouping_pipeline`. Do not use evaluation plots as live classroom assignments.

## API response shape

```json
{
  "status": "success",
  "total_students": 30,
  "total_groups": 6,
  "target_group_size": 5,
  "configured_groups": 6,
  "group_size_summary": "4–5 per group (expected 4 or 5)",
  "warnings": [],
  "groups": [
    {
      "id": 1,
      "members": [{ "Name": "...", "Gender": "...", "Learning_Style": "Visual", ... }],
      "stats": { "size": 5, "avg_motivation": 3.2, "intra_team_cohesion_score": 1.20, ... }
    }
  ]
}
```

## Reading the code

1. `backend/data_loader.py` — ingest branch  
2. `backend/group_gen_intake.py` + `backend/vak_answer_catalog.py` — survey conversion  
3. `backend/pipeline.py` — grouping  
4. `backend/api.py` or `backend/generate_groups.py` — boundaries  

Smoke test: `python -m backend.check_imports`
