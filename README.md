# GroupGen: Automated Student Grouping System

GroupGen forms classroom project groups by placing students with **similar psychometric profiles** together. It clusters on four behavioral dimensions — **Learning Style**, **Work Ethic**, **Motivation**, and **Self-Esteem** — using **K-Medoids (PAM)** with **Manhattan (L1)** distance. Gender and ethnicity are collected for reporting and post-hoc analysis but are **never** fed into the distance matrix or clustering algorithm.

> **Research goal:** Produce homogeneous skill-profile teams (students who work at a similar pace and share learning preferences) while enforcing realistic group-size constraints. Each run exports a per-team **intra-group cohesion score** (mean pairwise feature distance) for downstream regression analysis.

> **Quick demo for reviewers:** Install dependencies (Steps 1–2), then run the **Web UI** (Step 4) and upload `backend/data/templates/classroom_template.csv` to see grouped students visually.

**Deep dive for code reviewers:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Algorithm at a glance

| Component | Implementation |
|-----------|----------------|
| **Features (6-D)** | `StandardScaler` on Motivation, Self_Esteem, Work_Ethic (1–4) + one-hot Learning_Style (Visual / Auditory / Kinesthetic) |
| **Distance** | Manhattan L1 |
| **Clustering** | K-Medoids (PAM) on precomputed distance matrix, `random_state=42` |
| **Size constraints** | Two strategies run in parallel; lower PAM cost wins: (A) post-hoc medoid repair, (B) capacity-constrained PAM with Hungarian assignment |
| **Cohesion IV** | Per team: mean upper-triangle pairwise L1 in feature space (lower = more similar) |
| **Demographics** | Gender, Diversity — metadata only, excluded from `compute_feature_vector` |

---

## Step-by-step: run the system

> **For professors and reviewers:** Start with **Step 4 (Web UI)**. It is the easiest way to upload a survey CSV, see balanced groups on screen, and inspect per-group motivation, work ethic, and learning-style breakdowns. The CLI (Step 5) is for saved audit files and research exports.

### Prerequisites

- Python 3.10+ (3.11 recommended)
- Node.js 18+ (required for the Web UI)
- Git clone of this repository

### Step 1 — Install Python dependencies

From the project root (`GroupGen/`):

```bash
python -m venv venv
```

**Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`  
If `python` is not found, try `py -3 -m venv venv` instead.  
**Mac/Linux:** `source venv/bin/activate`

```bash
pip install -r requirements.txt
```

### Step 2 — Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 3 — Verify the pipeline (optional)

```bash
python -m backend.check_imports
```

Expected output: `OK` (smoke test on sample CSV + Google Form intake).

```bash
python -m pytest backend/test_pipeline.py -v
```

### Step 4 — Run the Web UI (recommended)

Use two terminals. Keep both running while you review groups.

**Terminal 1 — API** (project root, venv active):

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

Health check: [http://127.0.0.1:8000/](http://127.0.0.1:8000/) should return `{"status":"GroupGen API is running"}`.

**Terminal 2 — Frontend:**

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

**In the UI:**

1. Upload a CSV (try `backend/data/templates/classroom_template.csv` for a quick demo).
2. Set **Group Size** (e.g. 5).
3. Click **Generate Groups**.

**What you will see:** one card per group with student names, group size, average Motivation / Self-Esteem / Work Ethic, and learning-style breakdown (Visual / Auditory / Kinesthetic). Use **Download PDF** for a printable roster.

Raw Google Form exports work the same way — upload the downloaded CSV as-is. See [STUDY_SETUP.md](STUDY_SETUP.md) for required form layout.

### Step 5 — CLI (research exports and audit files)

Use the CLI when you need saved files (`final_groups.csv`, `run_manifest.json`, `research_export.csv`) rather than the on-screen view.

```bash
# Windows
set GROUPGEN_INPUT_CSV=backend\data\templates\classroom_template.csv
set GROUPGEN_GROUP_SIZE=5
python -m backend.generate_groups

# Mac/Linux
export GROUPGEN_INPUT_CSV=backend/data/templates/classroom_template.csv
export GROUPGEN_GROUP_SIZE=5
python -m backend.generate_groups
```

**Outputs** (`backend/output/runs/<timestamp>_<uuid>/`):

| File | Purpose |
|------|---------|
| `final_groups.csv` | Teacher roster: Name, scores, demographics, `Group_ID` |
| `research_export.csv` | Anonymized student table + cohesion scores |
| `team_cohesion.csv` | Per-team cohesion and trait averages |
| `final_groups_report.txt` | Printable text report |
| `run_manifest.json` | Seed, strategy, PAM cost, git commit |

**Real Google Form export:**

```bash
set GROUPGEN_INPUT_CSV=C:\path\to\form_responses.csv
set GROUPGEN_GROUP_SIZE=5
python -m backend.generate_groups
```

**Direct API call** (same backend as the UI):

```bash
curl -X POST "http://127.0.0.1:8000/generate-groups?group_size=5" \
  -F "file=@backend/data/templates/classroom_template.csv"
```

### Step 6 — Inspect clustering quality (diagnostics)

```bash
python -m backend.diagnose_feature_clusters backend/data/templates/classroom_template.csv --group-size 5
```

Prints silhouette scores, PAM cost, mean within-cluster pairwise L1, and confirms demographics are excluded from features.

---

## Program structure (where to read the code)

Read in this order to follow one request end-to-end:

```
CSV upload
    │
    ▼
data_loader.prepare_for_grouping     ← ingest, validate, normalize
    │   └── group_gen_intake         ← raw Google Form → 7 columns
    │   └── vak_answer_catalog       ← VAK option text → A/B/C → style
    ▼
pipeline.run_grouping_pipeline     ← production orchestrator
    │   ├── clustering.compute_feature_vector      (6-D psychometric matrix)
    │   ├── gpu_ops / sklearn                      (Manhattan distance matrix)
    │   ├── kmedoids.kmedoids_pam                  (unconstrained PAM)
    │   ├── kmedoids.kmedoids_size_constrained     (capacity seats)
    │   ├── clustering.enforce_group_size          (post-hoc repair)
    │   ├── cohesion.compute_team_cohesion         (per-team IV)
    │   └── invariants.assert_assignment_invariants
    ▼
generate_groups / api                ← export
    └── research_export              ← research_export.csv + team_cohesion.csv
```

| Module | Role |
|--------|------|
| `backend/data_loader.py` | CSV read, column aliases, two-pass validation |
| `backend/group_gen_intake.py` | Raw Form → Motivation, Self_Esteem, Work_Ethic, Learning_Style, Gender, Diversity |
| `backend/vak_answer_catalog.py` | Paper-accurate VAK scoring from full option text |
| `backend/clustering.py` | Feature engineering, Manhattan matrix, size balancing |
| `backend/kmedoids.py` | PAM + size-constrained K-Medoids (Hungarian assignment) |
| `backend/pipeline.py` | Dual size-strategy bake-off, cohesion, invariants |
| `backend/cohesion.py` | Per-team mean pairwise L1 cohesion metric |
| `backend/research_export.py` | Anonymized student_id, z-features, cohesion CSV |
| `backend/invariants.py` | Post-assignment structural checks (→ HTTP 422 if fail) |
| `backend/api.py` | FastAPI REST endpoint |
| `backend/generate_groups.py` | CLI with per-run audit folder |
| `backend/diagnose_feature_clusters.py` | Cluster quality diagnostics |

Legacy modules (`fairness_distribution.py`, Gower distance in `evaluate_clustering.py`) exist for research comparison only and are **not** called by the production pipeline.

---

## How grouping works (detailed)

### 1. Ingest — `prepare_for_grouping`

1. Read CSV (`utf-8-sig`; auto-repair comma-split VAK answers in Form exports)
2. If raw Google Form: score VAK + Likert sections via `group_gen_intake`
3. Normalize column names, drop blank rows, validate ranges (1–4 scores, unique names)
4. Preprocess strings (M/F → Male/Female, title-case learning style)
5. Reject empty Likert sections (no silent NaN → score 4)

### 2. Feature vector (6 dimensions)

| Source column | Transform | In distance matrix? |
|---------------|-----------|---------------------|
| Motivation | `StandardScaler` (z-score) | Yes |
| Self_Esteem | `StandardScaler` (z-score) | Yes |
| Work_Ethic | `StandardScaler` (z-score) | Yes |
| Learning_Style | One-hot [Visual, Auditory, Kinesthetic] | Yes |
| Gender | Metadata only | **No** |
| Diversity | Metadata only | **No** |

### 3. Clustering lifecycle

1. `n_groups = ceil(n_students / target_size)`
2. Build Manhattan distance matrix on 6-D features
3. **Strategy A:** Unconstrained PAM → `enforce_group_size` (move students closest to receiver medoid)
4. **Strategy B:** Size-constrained PAM (fixed capacities + Hungarian seat assignment)
5. Keep labeling with **lower PAM cost** (total distance to medoids)
6. Assert invariants: every student assigned, sizes ∈ {⌊n/k⌋, ⌊n/k⌋+1}, no group > target
7. Compute per-team cohesion (mean pairwise L1)

### 4. Research export schema (`research_export.csv`)

| Column | Description |
|--------|-------------|
| `student_id` | Anonymized blake2b hash of name (stable across runs) |
| `assigned_team_id` | 1-based team ID |
| `Motivation`, `Self_Esteem`, `Work_Ethic` | Raw 1–4 scores |
| `Motivation_z`, `Self_Esteem_z`, `Work_Ethic_z` | Z-scored values used in clustering |
| `Learning_Style` | Visual / Auditory / Kinesthetic |
| `LS_Visual`, `LS_Auditory`, `LS_Kinesthetic` | One-hot flags |
| `intra_team_cohesion_score` | Mean pairwise L1 within team (lower = more similar) |
| `Gender`, `Diversity` | Metadata for post-hoc controls only |

---

## Sample data

| File | Rows | Use |
|------|------|-----|
| `backend/data/templates/classroom_template.csv` | 30 | Pre-scored; default CLI input |
| `backend/data/templates/google_form_sample.csv` | 30 | Raw-style Form export (tests intake) |
| `backend/data/templates/low_scores_template.csv` | 30 | All low Motivation/SE/WE (edge-case test) |

Regenerate: `python backend/data/templates/_build_samples.py`

---

## Reproducibility guarantees

- **Deterministic clustering:** Same CSV + group size + seed (42) → identical groups
- **Stable VAK ties:** Blake2b digest of normalized answers (not Python `hash()`)
- **Audit trail:** CLI writes `run_manifest.json` with seed, strategy, PAM cost, git commit
- **Validation gates:** Invalid data → HTTP 400; broken assignments → HTTP 422

**Quick verification:**

```bash
python -m backend.check_imports
python -m pytest backend/test_pipeline.py -v
```

---

## Google Form requirements

Teachers upload the **unmodified** Google Forms CSV. Required structure:

1. Name column (`What is your first and last name`)
2. 30 VAK learning-style items (option text must match `vak_answer_catalog.py`)
3. Four **magic wand** divider questions between sections
4. Self-esteem (1–7), Motivation, Work Ethic Likert blocks
5. Gender identity and ethnicity columns

Full layout: [STUDY_SETUP.md](STUDY_SETUP.md)

---

## Optional: algorithm comparison (research only)

Compares K-Means, K-Medoids, Gower on template data. **Does not** use the production pipeline.

```bash
python -m backend.run_full_evaluation
```

Outputs under `backend/output_plots/runs/<timestamp>_<id>/`.

---

## Known limitations

- Learning-style one-hot and z-scored traits share the same L1 space (implicit weighting — document in methods)
- Small classes with repeated score patterns may produce uneven clusters before size balancing
- Web UI does not download CSV yet — use CLI for research exports
- API is stateless; no server-side file persistence

---

## Documentation index

| File | Audience | Purpose |
|------|----------|---------|
| **README.md** (this file) | ML reviewers, researchers | Goals, run steps, algorithm summary |
| [STUDY_SETUP.md](STUDY_SETUP.md) | Teachers | Google Form layout, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Developers | Full lifecycle, module map, API contract |
| `backend/pipeline.py` | Code readers | Annotated production orchestrator |

---

## License

See [LICENSE](LICENSE).
