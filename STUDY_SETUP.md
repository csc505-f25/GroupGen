# GroupGen — Classroom Study Setup

Students are grouped by **similar** motivation, self-esteem, work ethic, and learning style. After clustering, the system balances group sizes and swaps students when needed so no one is the sole representative of their **gender** or **diversity** category in a group.

Teachers upload the **raw CSV** from Google Forms. GroupGen converts it automatically—no manual spreadsheet step.

---

## 1. One-time install

From the project root (`GroupGen/`):

```bash
python -m venv venv
```

**Windows:** `.\venv\Scripts\activate`  
**Mac/Linux:** `source venv/bin/activate`

```bash
pip install -r requirements.txt
```

---

## 2. Google Form layout

Your form should follow the **research VAK inventory** (30 learning-style items, three options each) plus Likert sections for self-esteem, motivation, and work ethic.

### Required structure

1. **Timestamp / Email** (optional but typical)
2. **Name** — e.g. `What is your first and last name`
3. **Learning style (VAK)** — 30 multiple-choice questions; option text must match the paper exactly (see `backend/vak_answer_catalog.py`)
4. **Magic wand** divider question (first)
5. **Self-esteem / self-efficacy** items (1–7 scale)
6. **Magic wand** divider (second)
7. **Motivation / engagement** items
8. **Magic wand** divider (third)
9. **Work ethic / study habits** items
10. **Magic wand** divider (fourth)
11. **Gender** — e.g. `To which gender identity do you most identify?`
12. **Ethnicity / diversity** — e.g. `To which ethnicity do you most identify?`

The intake script (`backend/group_gen_intake.py`) locates sections using the **four magic wand** columns and keyword headers. If you add or remove a divider, update the form consistently.

### Export and upload

**Responses → ⋮ → Download responses (.csv)** — upload that file in the web UI or CLI. Do not reformat columns in Excel unless you save as **CSV UTF-8 (comma delimited)**.

### Learning style scoring (paper method)

For each student, GroupGen:

1. Maps each selected option to **A**, **B**, or **C** (option (a) = A, (b) = B, (c) = C).
2. Counts **A’s**, **B’s**, and **C’s** across all VAK items.
3. Assigns the style with the **highest count**:
   - mostly **A** → **Visual**
   - mostly **B** → **Auditory**
   - mostly **C** → **Kinesthetic**
4. If counts **tie**, picks one of the tied styles (same student → same pick on re-run).

Google Forms stores the **full answer text** (e.g. `read the instructions first`), not the letter. The catalog in `vak_answer_catalog.py` lists all 90 official option strings.

### Columns used for clustering (after conversion)

| Column | Values |
|--------|--------|
| `Name` | Unique student name or ID |
| `Gender` | Free text; `M`/`F` normalized to Male/Female |
| `Motivation` | 1–4 |
| `Self_Esteem` | 1–4 |
| `Work_Ethic` | 1–4 |
| `Learning_Style` | Visual, Auditory, or Kinesthetic |
| `Diversity` | Race/ethnicity category (consistent spelling helps) |

---

## 3. Sample files (30 students)

| File | Purpose |
|------|---------|
| `backend/data/templates/google_form_sample.csv` | Shortened raw-form shape for testing intake |
| `backend/data/templates/classroom_template.csv` | Already-scored 7 columns (default CLI / evaluation) |

Regenerate both: `python backend/data/templates/_build_samples.py`

---

## 4. Run the API (required for the web UI)

From project root:

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

Health check: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 5. Run the web UI

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), upload the CSV, set group size, click **Generate Groups**.

Optional: `frontend/.env.local` with `NEXT_PUBLIC_API_URL` if the API is not on `http://127.0.0.1:8000`.

---

## 6. CLI alternative (no browser)

```bash
# Default: classroom_template.csv (30 students, pre-scored)
python -m backend.generate_groups

# Your class Google Form export:
set GROUPGEN_INPUT_CSV=C:\path\to\responses.csv
python -m backend.generate_groups
```

Each run creates `backend/output/runs/<timestamp>_<id>/`:

| File | Contents |
|------|----------|
| `final_groups.csv` | Scored columns + `Group_ID` |
| `final_groups_report.txt` | Printable roster |
| `run_manifest.json` | Audit metadata |
| `raw_responses_with_groups.csv` | **Only if input was raw Google Form** — original survey + `Group_ID` |

---

## 7. What the algorithm does (API and CLI)

1. **Ingest** — raw form → seven columns (or validate pre-scored CSV)  
2. **K-Medoids (Manhattan)** on Motivation, Self_Esteem, Work_Ethic, Learning_Style  
3. **Size balancing** — e.g. 31 students @ target 5 → seven groups of 4 and 5  
4. **Gender fairness** — for each gender label that is a **minority** in this class (fewer than half the students), spread into pairs across groups (e.g. 2 together, 5 → 2+2+1); applies to Female, Male, Non-binary, Prefer not to say, etc.  
5. **Diversity fairness** — same pairing rule for each **minority** ethnicity/diversity label (Hispanic/Latinx, Asian American, …) in groups of 3+  
6. **Invariants + warnings** — block bad rosters; flag swap limits  

Gender and Diversity are **not** in clustering distance; they only guide swaps.

---

## 8. Study tips

- Use the **same target group size** across sections when comparing runs.  
- Keep Google Form option wording **identical** to the research instrument.  
- Save CLI output folders with section and date in the name.  
- Read API **`warnings`** even on success—some fairness cases may need a manual check.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Missing required columns | Upload raw Form CSV (with magic wands), or match `classroom_template.csv` |
| Could not classify learning-style answer | Option text must match `vak_answer_catalog.py` exactly |
| File parsed as one column | Re-export CSV from Google Forms; avoid `.xlsx` renamed to `.csv` |
| `Expected N fields in line X, saw M` | A **learning-style answer contains commas** (e.g. option 22c: *\"…such as an activity or a meal\"*). Re-download CSV from Google Forms; avoid Excel round-trips. GroupGen rejoins known VAK phrases when possible. |
| `ModuleNotFoundError: gower` | `pip install -r requirements.txt` (gower is evaluation-only) |
| Could not connect to backend | Start uvicorn on port 8000 |
| Values outside 1–4 after intake | Check Likert sections export numbers or agree/disagree labels |
| Duplicate names | Each student name must be unique |
| Warnings after success | Review listed groups; swap budget was exhausted |

---

## For developers

| Doc | Content |
|-----|---------|
| [README.md](README.md) | Install, intake overview, data format |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module map, lifecycle, API contract |
| `backend/group_gen_intake.py` | Section detection and Likert scoring |
| `backend/vak_answer_catalog.py` | 30×3 option text → A/B/C |
| `backend/pipeline.py` | Clustering and fairness |

Verify: `python -m backend.check_imports` from repo root → `OK`.
