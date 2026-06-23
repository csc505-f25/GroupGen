# GroupGen: Automated Student Grouping System

GroupGen forms classroom project groups by placing students with **similar** motivation, self-esteem, work ethic, and learning style together (homogeneous skill profiles). After clustering, the system balances group sizes and swaps students when needed so **every minority gender and ethnicity label** in that class is **spread into pairs** across groups (e.g. 5 students who share the same underrepresented label → two groups with 2 each plus one remainder), not left as singletons in every group and not piled into one group. The dominant gender or ethnicity in the roster is not rebalanced (that would undo minority pairing). One leftover singleton can still happen when counts do not divide evenly.

**Production algorithm:** K-Medoids (PAM) with **Manhattan (L1)** distance on scaled behavioral features (`random_state=42`).

**Developer docs:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — full data lifecycle, module map, and API contract.

---

## Quick start (classroom study)

Teachers export a CSV from Google Forms and upload it **as-is** through the web UI (no manual rescoring).

1. Install dependencies ([Installation](#installation)).
2. **API:** `uvicorn backend.api:app --reload --port 8000` (from project root).
3. **UI:** `cd frontend && npm install && npm run dev`
4. Open [http://localhost:3000](http://localhost:3000), upload CSV, set group size, click **Generate Groups**.

Teacher guide (form layout, VAK scoring, troubleshooting): **[STUDY_SETUP.md](STUDY_SETUP.md)**

**Sample CSVs** (30 students each, in `backend/data/templates/`):

| File | Use |
|------|-----|
| `google_form_sample.csv` | Raw-style Google Form export (tests intake) |
| `classroom_template.csv` | Pre-scored 7 columns (fast smoke test / evaluation) |
| `low_scores_template.csv` | Pre-scored 7 columns, all low Motivation/Self_Esteem/Work_Ethic (1–2) |

Regenerate samples: `python backend/data/templates/_build_samples.py`

---

## How it works

### Input (two supported formats)

| Format | Who uses it | What happens |
|--------|-------------|----------------|
| **Raw Google Form CSV** | Teachers (study) | Auto-detected → `group_gen_intake` scores survey → 7 columns |
| **Pre-scored CSV** | Tests, spreadsheets | Seven columns already present → validate only |

See [Data format](#data-format) and [Google Form intake](#google-form-intake).

### Ingest — `prepare_for_grouping` (`data_loader.py`)

All production paths use the same ingest step **before** any math runs:

1. Read CSV (`utf-8-sig`, auto-detect comma / semicolon / tab)
2. **If raw Google Form:** `group_gen_intake.process_google_form` (VAK + Likert + demographics)
3. Normalize column names (e.g. `Ethnicity` → `Diversity`)
4. Drop blank rows (common at end of Form exports)
5. Validate required fields, 1–4 scores, unique names, non-empty gender/diversity
6. Preprocess (`M`/`F` → Male/Female, trim strings, title-case learning style)
7. Validate again (catch values that could not be coerced)

Failures return clear `ValueError` messages → HTTP 400 in the API.

### Grouping — `run_grouping_pipeline` (`pipeline.py`)

Shared by **API** and **CLI**. Only cluster **labels** are updated after ingest; student survey columns in the DataFrame are never modified.

| Step | What it does |
|------|----------------|
| 1. Ingest | Done by caller (`prepare_for_grouping`) |
| 2. Group count | `n_groups = ceil(students / target_size)` |
| 3. Features | Motivation, Self_Esteem, Work_Ethic + one-hot Learning_Style (standardized) |
| 4. Distances | Pairwise **Manhattan** on features only — **no Gender/Diversity** |
| 5. Clustering | **K-Medoids (PAM)** on distance matrix (`random_state=42`) |
| 6. Size balance | Move students until each group size is valid (raises if impossible) |
| 7–9. Fairness | For each **minority** gender and ethnicity label, spread into pairs across groups (low-cost swaps); alternate gender/diversity passes until stable |
| 10. Invariants | Prove every student assigned, sizes match formula, no group over target |

If structural checks fail → HTTP 422 (API) or CLI exit with error. Residual fairness issues → `warnings` in API/CLI/manifest, not silent success.

### Outputs

| Path | How to run | Output |
|------|------------|--------|
| **Web UI** | `npm run dev` + API | Group cards, summary line, optional warnings banner |
| **API** | `POST /generate-groups?group_size=5` | JSON (`groups`, `warnings`, `group_size_summary`, …) |
| **CLI** | `python -m backend.generate_groups` | `backend/output/runs/<timestamp>_<id>/` (CSV, report, manifest) |
| **Research** | `python -m backend.run_full_evaluation` | `backend/output_plots/runs/<timestamp>_<id>/` — **not** live grouping |

---

## Project structure

```
GroupGen/
├── backend/
│   ├── api.py                 # FastAPI: upload → JSON
│   ├── pipeline.py            # Production grouping lifecycle
│   ├── data_loader.py         # CSV ingest, validation, normalization
│   ├── group_gen_intake.py    # Raw Google Form → 7 clustering columns
│   ├── vak_answer_catalog.py  # VAK option text → A/B/C → learning style
│   ├── paths.py               # Canonical paths (template CSV, output dirs)
│   ├── group_config.py        # ceil(n / target_size) group count
│   ├── check_imports.py       # Smoke test: python -m backend.check_imports
│   ├── clustering.py          # Features, distances, size balancing, isolation checks
│   ├── fairness_distribution.py # Spread minority gender/diversity into pairs (NumPy)
│   ├── kmedoids.py            # K-Medoids (PAM) on distance matrix
│   ├── invariants.py          # Post-pipeline structural checks
│   ├── json_util.py           # NumPy/pandas → JSON-safe types
│   ├── http_util.py           # Readable API error messages
│   ├── generate_groups.py     # CLI + per-run manifest
│   ├── run_manifest.py        # Audit JSON per CLI run
│   ├── run_full_evaluation.py # Research algorithm comparison
│   ├── evaluate_clustering.py
│   ├── skill_variance.py
│   └── data/templates/        # classroom_template.csv, google_form_sample.csv, low_scores_template.csv (30 students each)
├── frontend/                  # Next.js upload UI (src/app/page.tsx)
├── docs/
│   └── ARCHITECTURE.md        # Deep dive for code readers
├── requirements.txt
├── STUDY_SETUP.md
└── README.md
```

---

## Installation

```bash
git clone <your-repo-url>
cd GroupGen
```

### Python (API + CLI)

```bash
python -m venv venv
```

**Windows:** `.\venv\Scripts\activate`  
**Mac/Linux:** `source venv/bin/activate`

```bash
pip install -r requirements.txt
```

Core packages: `numpy`, `pandas`, `scikit-learn`, `fastapi`, `uvicorn`, `matplotlib`.  
`gower` is optional — used only by `run_full_evaluation`, not classroom grouping.

### Frontend (web UI)

```bash
cd frontend
npm install
```

---

## Running the system

### Web UI + API (recommended for teachers)

**Terminal 1 — API** (project root):

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

Optional: `NEXT_PUBLIC_API_URL` if the API is not on `http://127.0.0.1:8000`.

### CLI only

```bash
python -m backend.generate_groups
```

Defaults to `backend/data/templates/classroom_template.csv`. Override:

```bash
# Windows
set GROUPGEN_INPUT_CSV=C:\path\to\responses.csv
set GROUPGEN_OUTPUT_CSV=backend\output\my_run\final_groups.csv
python -m backend.generate_groups
```

Default writes (unique folder per run):

- `backend/output/runs/<timestamp>_<id>/final_groups.csv` — scored columns + `Group_ID`
- `backend/output/runs/<timestamp>_<id>/final_groups_report.txt`
- `backend/output/runs/<timestamp>_<id>/run_manifest.json` — parameters, warnings, paths
- `backend/output/runs/<timestamp>_<id>/raw_responses_with_groups.csv` — **only when input was a raw Google Form export** (original columns + `Group_ID`)

### API directly

```bash
curl -X POST "http://127.0.0.1:8000/generate-groups?group_size=5" \
  -F "file=@path/to/responses.csv"
```

Health check: `GET http://127.0.0.1:8000/`

---

## Research: algorithm evaluation (optional)

Compares clustering variants on the template dataset. **Does not** use `run_grouping_pipeline`.

```bash
python -m backend.run_full_evaluation
```

Includes K-Means / K-Medoids (Euclidean, Manhattan) and optional K-Medoids (Gower).  
Outputs under `backend/output_plots/runs/<timestamp>_<id>/`.

---

## Data format

| Column | Type | Description |
| :--- | :--- | :--- |
| `Name` | String | Unique student name or ID |
| `Gender` | String | Male / Female (`M` / `F` accepted); other values allowed (e.g. Non-binary) |
| `Motivation` | Int (1–4) | 1 = low, 4 = high |
| `Self_Esteem` | Int (1–4) | 1 = low, 4 = high |
| `Work_Ethic` | Int (1–4) | 1 = low, 4 = high |
| `Learning_Style` | String | Visual, Auditory, or Kinesthetic |
| `Diversity` | String | Race/ethnicity category (consistent spelling) |

Header aliases (pre-scored CSV only): `backend/data_loader.py` → `COLUMN_ALIASES`.

### Google Form intake

Teachers upload the **unmodified** Google Forms CSV. No spreadsheet rescoring required.

| Module | Role |
|--------|------|
| `group_gen_intake.py` | Finds sections via **magic wand** dividers; scores Likert blocks; builds 7 columns |
| `vak_answer_catalog.py` | Maps each VAK option text to paper letter **A / B / C** |

**Learning style (VAK)** — same rule as the research paper:

1. Each of the 30 items has options **(a)**, **(b)**, **(c)** → count as **A**, **B**, **C**.
2. Whichever letter has the **highest count** wins: A → Visual, B → Auditory, C → Kinesthetic.
3. If two or three letters **tie**, the system picks **one** tied style (stable per student’s answers).

Option text in Google Forms must match the official inventory in `vak_answer_catalog.py` (Google exports the full sentence the student selected, not the letter alone).

**Other scored fields:**

- **Self-esteem block** — 1–7 Likert mean → 1–4 scale  
- **Motivation & work ethic blocks** — section mean → 1–4 scale  
- **Gender / diversity** — from end-of-form columns  

**Form requirements:** `What is your first and last name`, four **magic wand** divider questions between sections, `To which gender identity…`, `To which ethnicity…`. See [STUDY_SETUP.md](STUDY_SETUP.md).

---

## Safety and study guarantees

- **Determinism:** Same CSV + group size → same groups (seed 42, stable row order).
- **No NaN in clustering:** Ingest validation + finite-matrix checks before K-Medoids.
- **No invalid success:** Imbalanced sizes or broken assignments block the response.
- **Stateless API:** No server-side files; each request is isolated.
- **CLI audit trail:** UUID run folders + `run_manifest.json`.

**Verify setup:** from repo root, `python -m backend.check_imports` should print `OK`.

---

## Known limitations

- Fairness runs until no improving swap remains (bounded iteration cap); some isolation may remain when counts don't divide evenly → check `warnings`.
- Fairness applies to **every** `Gender` and `Diversity` value with 2+ students and fewer than half the class; lone-member warnings use groups of **4+** (gender) or **3+** (diversity).
- Small classes may not have enough donors for every swap.
- Web UI does not download CSV yet — use CLI or print from the browser.

---

## Documentation

| File | Purpose |
|------|---------|
| [STUDY_SETUP.md](STUDY_SETUP.md) | Teacher setup, Google Form layout, VAK, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Code map, intake lifecycle, API JSON, research vs production |
| [workspace_cleanup_report.md](workspace_cleanup_report.md) | Directory flattening log (frontend + backend alignment) |
| Inline comments in `backend/pipeline.py`, `group_gen_intake.py`, … | Step-by-step logic while reading source |
| `backend/data/templates/*.csv` | 30-student sample files for tests and demos |

---

## License

See [LICENSE](LICENSE).
