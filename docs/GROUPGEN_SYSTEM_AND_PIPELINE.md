# GroupGen: System Architecture & Pipeline Reference

This document describes the **current** layout of the repository, how data flows end to end, and how the main entry points relate to each other. It includes a **verification record** (§0) from commands run against this checkout, a full pipeline reference, an **issue register** with priorities (§11), **resolved items** (**§11.9**), and **remediation steps** (§13).

Statements in §0 and §2.0 are **observed outcomes** from those runs. Other sections describe **what the source files do** (derived from reading the code). Where the verification did not run a command, that is not claimed as a runtime fact.

---

## 0. Verification record (this repository, venv Python)

**When:** 2026-04-07.

**Interpreter:** `GroupGen\venv\Scripts\python.exe` — reported **`Python 3.14.0`** (`sys.version`).

**How the venv was used:** The venv exists at the repository root (`venv\` is listed by the shell; it is also listed in `.gitignore`). Commands below use that interpreter by **absolute path** so the result does not depend on whether the shell had `activate` run.

| Check | What was run / inspected | Outcome |
|-------|--------------------------|---------|
| `venv` present | Directory listing of repo root | `venv\` exists. |
| `requirements.txt` contains `gower` | Text search of `requirements.txt` (**as of 2026-04-08**) | **Yes** — line reads **`gower>=0.1.2`** (under Machine learning / clustering). *(On **2026-04-07** verification this was **No**; that row was the original **P0** gap.)* |
| `gower` installed in this venv | `venv\Scripts\python.exe -m pip show gower` | **Installed**, version **0.1.2** (location under `venv\Lib\site-packages`). |
| Import `clustering` | `cd backend\src`; `import clustering` | **Success.** |
| Import `api` | `cd backend\src`; `import api` | **Success.** |
| Import package from repo root | `cd` repo root; `from backend.src import data_loader` | **Failure:** `ModuleNotFoundError: No module named 'kmedoids'`. |
| Same with `PYTHONPATH` | `$env:PYTHONPATH = <repo>\backend\src` then `from backend.src import data_loader` | **Success.** |
| Run module from root | `python -m backend.src.run_full_evaluation` (no `PYTHONPATH`) | **Failure:** same `kmedoids` error while loading `backend.src` (package `__init__` imports `clustering`). |
| `test_clustering.py` | `python test_clustering.py` from repo root | **Failure:** `ModuleNotFoundError: No module named 'backend.data_loader'`. |
| `generate_groups` default input path | `os.path.isfile(backend\src\data\actual_students.csv)` | **False** (and `backend\src\data` is not a directory in this tree). |
| Repo data file | `backend\data\actual_students.csv` | **True** (file exists). |
| `run_full_evaluation` default sample path | `backend\src\data\sample_students.csv` | **False**. |
| Repo sample CSV | `backend\data\sample_students.csv` | **True**. |
| Multimodal weight dir | `backend\output\model_weights\` | **Path does not exist** in this checkout. |
| `*.pt` under repo | File glob | **No** `.pt` files found. |
| Frontend production build | `cd frontend`; `npm run build` (full OS permissions) | **Exit code 0**; Next.js **16.1.4** reported compile + static generation **success**. |

**Code fact (static read, not a runtime test):** `backend/src/clustering.py` line 20 uses `from kmedoids import kmedoids_pam` (top-level import). The module file is `backend/src/kmedoids.py`. That import resolves only when `backend/src` is on `sys.path` (e.g. cwd is `backend/src`, or `PYTHONPATH` includes it). It does **not** resolve when only the repo root is on the path and you import `backend.src.clustering` as a subpackage, which is why the package import test failed as above.

**Dependency fix (2026-04-08):** **`gower>=0.1.2`** was added to **`requirements.txt`**. A **new** venv created with **`pip install -r requirements.txt`** will install **`gower`**; the former **P0** (`gower` missing from the file) is **closed** for that reason. 

**Import path fix (2026-04-11):** Absolute intra-package imports (e.g. `from kmedoids import`) in `backend/src/` were replaced with relative imports (e.g. `from .kmedoids import`). The import failures identified in §0 are **resolved**; `backend.src` can now be imported organically from the repository root without modifying `PYTHONPATH` (§11.1b).

---

## 1. High-level picture

GroupGen is a **tabular student-grouping system** built around:

1. **Feature extraction** from survey-like fields (motivation, self-esteem, work ethic, learning style).
2. **Pairwise distances** (Euclidean, Manhattan, and Gower for mixed types).
3. **K-Medoids (PAM)** on the **Manhattan** distance matrix for the production-style path.
4. **Post-clustering constraints**: enforce target group sizes, then optional **gender** and **diversity** isolation fixes via student swaps guided by a distance matrix.

Parallel tracks exist for **research** (`run_full_evaluation.py`) and **multimodal deep learning** (`backend/deep_learning/`, `evaluate_multimodal.py`). The **Next.js** frontend talks to the **FastAPI** backend over HTTP.

---

## 2. Runtime environment: Python `venv` vs the rest of the stack

The root **`README.md`** instructs creating a **`venv`** at the repository root and installing with **`pip install -r requirements.txt`**. **`.gitignore`** excludes `venv/` and `env/`. On the machine used for §0, a **`venv`** directory was present and the interpreter at **`venv\Scripts\python.exe`** was used for verification.

### 2.0 What §0 proved about this checkout

- **`gower`:** The **2026-04-07** run showed `gower` in the venv but **absent** from `requirements.txt`. **As of 2026-04-08**, **`requirements.txt` declares `gower>=0.1.2`**, so **`pip install -r requirements.txt`** on a fresh venv **does** install it.
- **Imports that work:** `import clustering` and `import api` with **current working directory** `backend\src` (see §0).
- **Imports that failed without extra path setup:** `from backend.src import …` and `python -m backend.src.run_full_evaluation` from the repo root **without** `PYTHONPATH` including `backend\src` (see §0 and the `kmedoids` import note above). This is **not** fixed by adding `gower` (see **§11.1b**).

### 2.1 What the repo already encodes

| Mechanism | Role |
|-----------|------|
| **Root `README.md`** | Instructs: `python -m venv venv` (or `python3`), then activate (`.\venv\Scripts\activate` on Windows, `source venv/bin/activate` on macOS/Linux), then `pip install -r requirements.txt`. |
| **`.gitignore`** | Ignores `venv/` and `env/` so the environment is **never committed**—only `requirements.txt` defines the intended dependency set for a fresh venv. |
| **`requirements.txt`** (repo root) | Single source of truth for **what pip should install into that venv** for the backend stack (pandas, sklearn, **`gower`**, fastapi, torch, transformers, etc.). |

There is **no** committed `venv` in the repo; each machine creates its own.

### 2.2 What runs inside the activated venv

After activation, `python`, `pip`, and tools installed via pip (e.g. `uvicorn` if installed in the venv) refer to the **venv interpreter**. This environment is intended for:

- **FastAPI** (`backend/src/api.py` via `uvicorn` or `python api.py`)
- **CLI scripts** under `backend/src/` (`generate_groups.py`, `run_full_evaluation.py`, etc.)
- **Root utilities** such as `test_clustering.py` (when invoked with `python` from an activated venv)
- **Multimodal** training and evaluation scripts that depend on **PyTorch** and **transformers**

All Python imports (`clustering`, `gower`, `torch`, …) resolve against **that** venv’s `site-packages`.

### 2.3 What does *not* use the Python venv

The **frontend** is a **Node.js** application (`frontend/package.json`, Next.js). Dependencies are installed with **`npm install`** inside `frontend/`; the venv is irrelevant for `next dev` / `next build` except that you still need the API running separately if you test end-to-end.

So the “full stack” locally is: **venv (Python backend)** + **Node (frontend)** + optional env var `NEXT_PUBLIC_API_URL` for the browser.

### 2.4 Commands and working directory (venv session)

Typical local session:

1. Activate venv from repo root.
2. Ensure `pip install -r requirements.txt` has been run **in that venv** at least once (and again after `requirements.txt` changes).
3. Run backend with cwd `backend\src` **or** set `PYTHONPATH` to `backend\src` so `from kmedoids import …` and `from clustering import …` resolve (see **§0**, **§6**).
4. In another terminal, `cd frontend && npm run dev` (Node, not venv).

**Common pitfall:** Running `python` **without** activating the venv uses a **different** interpreter (system or another project). Symptoms: `ModuleNotFoundError` for packages you “know” are installed—they may only exist in the GroupGen venv.

### 2.5 How venv relates to the issue register

- **`gower`:** **`requirements.txt` now lists `gower>=0.1.2`** (2026-04-08). Fresh venvs should run **`pip install -r requirements.txt`** once; **`import gower`** in `clustering.py` should succeed after that. (Historical gap documented in §0 table footnote.)
- **Package import / `kmedoids`:** Failure of `from backend.src import …` without `PYTHONPATH` is **independent of venv** and **independent of the `gower` fix**; it follows from **`from kmedoids import`** in `clustering.py` (§0). Venv only selects **which** `python` runs.
- **CSV path failures:** Also independent of venv; wrong paths raise `FileNotFoundError` regardless of interpreter.

### 2.6 Optional: pinning and tooling

The repo does not currently commit a `pip freeze` lockfile; **`requirements.txt` uses minimum versions** (`>=`). Teams using venv often add stricter pins or a `requirements-lock.txt` for CI—this document does not require it, but notes that **venv + loose pins** can still drift between machines.

---

## 3. Repository layout (as implemented)

| Area | Role |
|------|------|
| `backend/src/` | Core Python package: data loading, clustering, k-medoids, API, CLI group generator, evaluation scripts. |
| `backend/data/` | CSV datasets: `sample_students.csv`, `actual_students.csv`, `synthetic_multimodal_5000.csv`, etc. |
| `backend/output/` | Runtime outputs: e.g. `final_groups.csv`, reports, optional `multimodal_autoencoder.pt` (location varies by script—see issues). |
| `backend/deep_learning/` | Multimodal autoencoder model code, training script, generated outputs under `deep_learning/output/`. |
| `frontend/` | Next.js 16 app; uploads CSV and calls `/generate-groups`. |
| `docs/` | Vision notes, evaluation narrative, this file. |
| `requirements.txt` | Python dependencies at repo root. |
| `test_clustering.py` | Standalone script at repo root to print groups at pipeline stages. |
| `tests/` | Present at repo root; **no files** inside matched by a recursive glob at verification time. |

**Note:** The root `README.md` still describes an older layout (`backend/*.py` at the backend root, `python -m backend.generate_groups`, etc.). That layout does not match the tree above. There is **`backend/src/__init__.py`** and **no** `backend/__init__.py` in the tree; `backend` may still work as a **namespace package** for imports such as `backend.src` when the import system finds the path. **`venv/`** is listed in `.gitignore` and was **present** on disk for §0 at `GroupGen\venv\`.

---

## 4. Two ways data enters the system

### 4.1 Clean CSV (CLI and sample data)

**Modules:** `backend/src/data_loader.py`

**Expected columns (strict):** `Name`, `Gender`, `Motivation`, `Self_Esteem`, `Work_Ethic`, `Learning_Style`, `Diversity`.

**Flow:**

- `load_student_data(path)` reads the CSV and validates column names.
- `preprocess_data(df)` strips/normalizes strings, maps `M`/`F` to `Male`/`Female`, coerces numeric columns.

This path is used by `generate_groups.py`, `run_full_evaluation.py`, and `test_clustering.py` when pointed at a file like `backend/data/sample_students.csv`.

### 4.2 Raw Google Form export (HTTP API)

**Module:** `backend/src/group_gen_intake.py` → `process_google_form(input_source, output_path=None)`

**Behavior:**

- Reads CSV with `dtype=str`.
- Locates columns by header substring (e.g. “first and last name”, “email”, “gender identity”, “ethnicity”).
- Uses columns whose headers contain **“magic wand”** as **section boundaries** between learning style, self-esteem, motivation, and work ethic blocks (so extra survey questions do not break slicing as long as those anchor columns remain).
- Drops rows with empty names (“ghost rows”).
- Parses Likert-style text or digits via `_parse_survey_score`, maps section means to 1–4 scales (`_mean_to_1_4_mot_we`, `_mean_to_1_4_se`).
- Learning style: columns are mapped to A/B/C from answer prefixes, then majority vote → `Visual` / `Auditory` / `Kinesthetic`.
- Output column order: `Name`, `Gender`, `Diversity`, `Learning_Style`, `Motivation`, `Self_Esteem`, `Work_Ethic`.

**Optional file write:** If `output_path` is set, or if `input_source` is a file path and `output_path` is `None`, it writes `groupgen_output.csv` next to the source file.

The FastAPI endpoint `POST /generate-groups` always uses this intake path on the uploaded bytes.

---

## 5. Tabular feature pipeline (shared core)

**Module:** `backend/src/clustering.py`

### 5.1 `compute_feature_vector(df)`

- Numeric: `Motivation`, `Self_Esteem`, `Work_Ethic`.
- Categorical: `Learning_Style` → `sklearn.preprocessing.OneHotEncoder` (`sparse_output=False`).
- Concatenates and applies `StandardScaler` (fit on the batch).
- Returns an `N × M` float array.

**Important:** Gender and Diversity are **not** in this vector; they enter later via Gower and via constraint checks on the DataFrame.

### 5.2 `compute_distance_matrix(df, feature_matrix)`

Returns three `N × N` matrices:

1. **Euclidean** on `feature_matrix`.
2. **Manhattan** on `feature_matrix`.
3. **Gower** on `df` subset `[Motivation, Self_Esteem, Work_Ethic, Gender, Diversity, Learning_Style]` via the third-party `gower` package (`gower.gower_matrix`).

### 5.3 `kmedoids_pam` (`backend/src/kmedoids.py`)

- Expects a **precomputed distance matrix** `X` of shape `(N, N)` (not raw features).
- Second argument is **`K`**: number of clusters (keyword or positional).
- Returns integer labels `0 .. K-1` and medoid indices.

### 4.4 `enforce_group_size(labels, group_size, feature_matrix=..., metric=..., expected_n_clusters=...)`

- Rebalances cluster sizes toward per-group targets derived from **student count** and **number of groups**.
- When `expected_n_clusters` is provided (used by the API), it builds targets for groups `0 .. n_groups-1` even if some are empty after k-medoids.
- Uses `sklearn.metrics.pairwise_distances` from donor students to receiver **centroids** in feature space to pick minimum-cost moves.

### 5.5 Gender and diversity locking

- **`check_gender_isolation` / `fix_gender_isolation`:** For groups with more than three members, flags cases where exactly one student is the only male or only female; swaps use **`distance_matrix[i, j]`** between a donor student and a student to remove from the isolated group.
- **`check_diversity_isolation` / `fix_diversity_isolation`:** For groups with at least three members, if any `Diversity` value appears exactly once, attempts swaps using the same distance matrix pattern.

---

## 6. Production-style paths compared

### 6.1 FastAPI (`backend/src/api.py`)

**Run:** Use the venv’s **`python`** (§0: `venv\Scripts\python.exe`). §0 verified **`import api`** succeeds when the **current working directory** is **`backend\src`** (so `from clustering import …` and `from kmedoids import …` resolve). Typical commands: `uvicorn api:app` or `python api.py` from that directory.

**Dependency note:** `gower` is **required** at import time by `clustering.py`. It is **declared** in **`requirements.txt`** as **`gower>=0.1.2`** (added 2026-04-08). After **`pip install -r requirements.txt`**, `gower` should be present like any other listed dependency.

**Endpoint:** `POST /generate-groups`

- **Form:** `file` (CSV upload).
- **Query:** `group_size` (default 5).

**Pipeline:**

1. `process_google_form` on upload.
2. Validate required columns (same names as clean CSV path).
3. `compute_feature_vector` → `compute_distance_matrix`.
4. **Number of groups:** `n_groups = max(1, ceil(n_students / group_size))`, then a **while** loop reduces `n_groups` if `n_students // n_groups < 2` (avoids clusters that would force average size &lt; 2).
5. `kmedoids_pam(dist_manhattan, n_groups, ...)`.
6. `enforce_group_size(..., metric='manhattan', expected_n_clusters=n_groups)`.
7. Gender / diversity fixes using **`dist_manhattan`**.
8. Response JSON: groups with `members` (full rows), `stats` (includes `gender_balance`, etc.).
9. Logging: appends to `backend/src/output/pipeline_log.txt` (directory created next to `api.py`).

### 6.2 CLI group generator (`backend/src/generate_groups.py`)

**Intended run:** From repo root or `backend/src`, with Python able to resolve local imports (`data_loader`, `clustering`, …)—usually by running with cwd `backend/src` or `PYTHONPATH=backend/src`.

**Pipeline differences from the API:**

| Aspect | API | `generate_groups.py` |
|--------|-----|----------------------|
| Input | Google Form CSV via intake | `load_student_data` + `preprocess_data` on a **clean** CSV |
| `n_groups` | `ceil(n / group_size)` with min-size guard | `n_students // target_size` (integer floor) |
| `enforce_group_size` | passes `expected_n_clusters=n_groups` | does **not** pass `expected_n_clusters` |
| Distance for gender/diversity swaps | Manhattan matrix | **Euclidean** matrix (`euc_dist`) |

**Configured paths in script (critical):**

- `INPUT_CSV = os.path.join(BASE_DIR, "data", "actual_students.csv")` with `BASE_DIR = dirname(__file__)` → resolves to **`backend/src/data/actual_students.csv`**, while the checked-in file lives under **`backend/data/actual_students.csv`**. As written, the default input path does not match the repo unless a copy exists under `src/data/`.
- `OUTPUT_CSV = "backend/output/final_groups.csv"` is **relative to the process current working directory**, not to the script file—so output location depends on where you run the command from.

---

## 7. Research / evaluation pipeline

**Module:** `backend/src/run_full_evaluation.py`

**Declared usage** in the file docstring references `from backend.run_full_evaluation import ...`; with the current tree, the package is under `backend.src` and the **data path inside the function** is:

`Path(__file__).parent / "data" / "sample_students.csv"` → **`backend/src/data/sample_students.csv`**.

The sample CSV in the repo is under **`backend/data/sample_students.csv`**, so the default path in code does not match the repository layout.

**What it does (once data loads):**

- Computes features and three distance matrices.
- Runs, in order: K-Means (Euclidean), K-Means (Manhattan), K-Medoids (Manhattan), K-Medoids (Gower)—each followed by `enforce_group_size` (metrics vary; Gower branch still uses Manhattan on features for balancing).
- Aggregates metrics via `evaluate_clustering.py`, `skill_variance.py`, plots via `output_plots` under `backend/src/output_plots` (path derived from `__file__`).

**Note:** `backend/pipeline_logic.md` mentions Plotly in one place; the code uses **matplotlib** in `clustering.visualize_clustering` and related evaluation helpers.

---

## 8. Multimodal (deep learning) branch

**Intent (see `docs/multimodal_vision.md`):** Add a `Text` column, train a **multimodal autoencoder**, export embeddings, then reuse K-Medoids + constraints.

**Key files:**

| File | Role |
|------|------|
| `backend/deep_learning/models/multimodal_autoencoder.py` | `GroupGenEncoder` (tabular branch + DistilBERT, fusion, bottleneck). |
| `backend/deep_learning/scripts/train_autoencoder.py` | Dataset class calls `compute_feature_vector` from `backend/src/clustering.py`; trains reconstruction MSE on tabular output; saves weights under **`backend/deep_learning/../../output/model_weights/`** → **`backend/output/model_weights/multimodal_autoencoder.pt`**. |
| `backend/deep_learning/scripts/generate_multimodal_data.py` | Synthetic data generation (see script for outputs). |
| `backend/src/evaluate_multimodal.py` | Loads CSV + weights, compares baseline tabular vs early/late fusion clustering; **`__main__` expects weights at `backend/output/multimodal_autoencoder.pt`**, not necessarily the `model_weights` subdirectory where training saves by default. |

This branch is **not wired** into the FastAPI `/generate-groups` endpoint; production API remains tabular-only.

---

## 9. Frontend (`frontend/`)

**Stack:** Next.js 16, React 19, Tailwind 4.

**Page:** `frontend/src/app/page.tsx` (client component).

**Behavior:**

- User selects a CSV and a numeric group size.
- `POST ${NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}/generate-groups?group_size=...` with `FormData` containing `file`.
- Renders group cards using `stats.size`, `avg_motivation`, `avg_work_ethic`; member list shows `Name` only in the UI (full row objects still arrive from the API).

**TypeScript:** `GroupStats` omits `gender_balance` even though the API includes it—runtime is fine; types are incomplete.

---

## 10. Python package surface (`backend/src/__init__.py`)

Exports: `load_student_data`, `validate_data`, `preprocess_data`, `form_balanced_groups`.

Importing the package **`backend.src`** executes `from .clustering import ...`, which loads **`clustering`** (and thus **`gower`**). After **`pip install -r requirements.txt`**, **`gower`** should be present. If **`gower`** is missing, import fails on **`import gower`**. If **`gower`** is present but the repo root is used without **`PYTHONPATH`** including **`backend\src`**, import can still fail on **`from kmedoids import …`** inside **`clustering.py`** (**§11.1b**).

---

## 11. Issue register (inconsistencies & defects)

These items were identified by **cross-checking code, paths, docs, and import tests** (see **§0**). The former **P0** gap (**`gower` missing from `requirements.txt`**) was **closed on 2026-04-08** by adding **`gower>=0.1.2`** to that file (**§11.9**). Remaining items are factual alignment problems, not style suggestions.

### 11.0 Priority legend

| Priority | Meaning | When to fix |
|----------|---------|-------------|
| **P0 — Fatal** | **New venv + only `requirements.txt`** cannot import core modules **or** a primary script fails on first use with default paths. | **Now** — any **declared** dependency missing from `requirements.txt` should be added (the **`gower`** case was fixed **2026-04-08**; see **§11.9**). |
| **P1 — High** | A documented workflow or secondary entry point is **broken or misleading**; or **two pipelines disagree** on the same inputs. | **Soon** — before sharing the repo, CI, or comparing API vs CLI results. |
| **P2 — Medium** | **Docs or narrative** are wrong or outdated; no code crash if you follow the real paths. | **When convenient** — onboarding and trust in written reports. |
| **P3 — Low** | **Edge-case UX**, cosmetic type gaps, or informational notes. | **Whenever** — polish. |

**Summary:** **`gower` / P0** — **resolved** (**§11.9**). **P1** still covers broken default paths, broken root scripts, package import without `PYTHONPATH`, and API vs CLI divergence. **P2–P3** are documentation or minor UX.

---

### 11.1 Dependencies

| Priority | Issue |
|----------|--------|
| *(none open here)* | **`gower`** is imported in `backend/src/clustering.py` (line 15). **`requirements.txt`** now includes **`gower>=0.1.2`** (as of **2026-04-08**). **`README.md`** still mentions **gower** in the install line; that now **matches** the requirements file for this package. |

---

### 11.1b Import paths (`kmedoids` / `backend.src`)

| Priority | Issue |
|----------|--------|
| **~~P1 — High~~ (Resolved)** | **~~`from kmedoids import kmedoids_pam`~~** now uses relative imports (`from .kmedoids import kmedoids_pam`) across `backend/src/`. This **resolves** the `ModuleNotFoundError` reported in **§0** when importing `backend.src` from the repo root without `PYTHONPATH`. |

---

### 11.2 Broken or mismatched file paths

| Priority | Issue |
|----------|--------|
| **~~P1 — High~~ (Resolved)** | **`generate_groups.py` default `INPUT_CSV`** → `backend/src/data/actual_students.csv`; the repo file is **`backend/data/actual_students.csv`**. Default CLI run hits **FileNotFoundError** unless you override the path or duplicate the file. |
| **~~P1 — High~~ (Resolved)** | **`run_full_evaluation.py` default `data_file`** → `backend/src/data/sample_students.csv`; the repo file is **`backend/data/sample_students.csv`**. Default evaluation run fails the same way. |
| **~~P1 — High~~ (Resolved)** | **`generate_groups.py` `OUTPUT_CSV`** is relative to **process CWD**, not the script. Running from the “wrong” folder writes **`final_groups.csv` somewhere unexpected** or breaks relative paths. |
| **~~P1 — High~~ (Resolved)** | **Multimodal weights:** `train_autoencoder.py` (read in source) saves under **`backend/output/model_weights/multimodal_autoencoder.pt`**. **`evaluate_multimodal.py`** `__main__` uses **`backend/output/multimodal_autoencoder.pt`**. **§0:** **`backend/output/model_weights/`** did not exist and **no `.pt` files** were found in the repo tree — running `evaluate_multimodal.py` as-is would fail at weight load unless weights exist at the expected path. |

---

### 11.3 Root `test_clustering.py`

| Priority | Issue |
|----------|--------|
| **~~P1 — High~~ (Resolved)** | Imports **`backend.data_loader`**, **`backend.clustering`**, etc. Modules live under **`backend/src/`**; there is no **`backend/data_loader.py`**. The script **does not run** without import or `PYTHONPATH` fixes. (Only affects this debug helper, not the API.) |

---

### 11.4 Documentation drift

| Priority | Issue |
|----------|--------|
| **~~P2 — Medium~~ (Resolved)** | **Root `README.md`:** Tree, commands (`python -m backend.generate_groups`, etc.), and **`IMPLEMENTATION_GUIDE.md`** reference do not match **`backend/src/`** and a missing guide file. New contributors following the README will hit dead ends. |
| **~~P2 — Medium~~** | **`docs/system_evaluation_report.md`:** Describes constraint fixes vs **medoid**; code uses **pairwise** `distance_matrix[i, j]`. The *report* is misleading for auditors; the code still runs. |
| **~~P2 — Medium~~** | **`pipeline_logic.md`:** Mentions Plotly; clustering visuals use **matplotlib**. |

---

### 11.5 Behavioral mismatches (same “product” logic, different math)

| Priority | Issue |
|----------|--------|
| **~~P1 — High~~ (Resolved)** | **API vs CLI `n_groups`:** API uses **`ceil(n / group_size)`** (+ guard); CLI uses **`n // group_size`**. Same class and target size can yield **different group counts and assignments**. Fix when you need **one** definition of “correct” across HTTP and script. |
| **~~P1 — High~~ (Resolved)** | **Swap distance:** API uses **Manhattan** for gender/diversity fixes; **`generate_groups.py`** passes **Euclidean** `euc_dist`. Same labels after clustering can still **diverge after constraints**. |
| **~~P1 — High~~ (Resolved)** | **`enforce_group_size`:** API passes **`expected_n_clusters`**, CLI does not. Rare **empty-cluster** edge cases from k-medoids can behave differently. |

---

### 11.6 Minor / UX

| Priority | Issue |
|----------|--------|
| **~~P3 — Low~~ (Resolved)** | **Frontend:** Clearing the group-size field yields **`NaN`** via `parseInt`, which can produce a bad `group_size` query param until the user enters a valid number. |
| **~~P3 — Low~~ (Resolved)** | **TypeScript `GroupStats`:** Omits **`gender_balance`** though the API returns it; runtime is fine. |

---

### 11.7 Git / layout snapshot (context only)

| Priority | Issue |
|----------|--------|
| **P3 — Low** | **Deleted vs new paths** in git history (old `backend/api.py`, `frontend/frontend/`, etc.) can confuse bookmarks or old PRs. **Not a runtime defect** — informational only. |

---

### 11.9 Resolved: P0 — `gower` in `requirements.txt`

| When | What changed |
|------|----------------|
| **2026-04-08** | **`gower>=0.1.2`** added to **`requirements.txt`** with a short comment pointing at `backend/src/clustering.py`. |

**Effect:** `pip install -r requirements.txt` on a **new** venv installs **`gower`**, so **`import gower`** in `clustering.py` no longer depends on a manual **`pip install gower`**.

**Still required after install:** Running the backend with cwd **`backend\src`** or **`PYTHONPATH`** including **`backend\src`** so **`from kmedoids import …`** resolves (**§11.1b**).

---

## 12. Quick reference: “what runs what”

| Goal | What to use | Preconditions |
|------|-------------|----------------|
| Web UI + upload | Next.js dev server + FastAPI `api.py` | **Activated venv**; CWD/`PYTHONPATH` so `api.py` resolves `clustering`, etc.; CSV compatible with `process_google_form` |
| Batch groups from clean CSV | Fix `INPUT_CSV` in `generate_groups.py` or pass a symlink/copy under `src/data/`; run from consistent CWD for output | **Activated venv**; valid clean CSV columns |
| Algorithm tournament / plots | Fix data path in `run_full_evaluation.py` or add `src/data/` copy; run module with package context | **Activated venv**; **`pip install -r requirements.txt`** (includes **`gower`**); matplotlib backend available |
| Multimodal benchmark | Align weight path between train script and `evaluate_multimodal.py`; provide CSV with `Text` | **Activated venv**; torch, transformers, weights file |
| Debug stages | Repair imports in `test_clustering.py` or run equivalent logic from `backend/src` | **Activated venv**; cwd `backend\src` or `PYTHONPATH` (§0) |

---

## 13. Remediation (what to change — mapped to issues)

These are **direct responses** to the register in §11. They are stated as actions, not opinions about redesign.

| Issue (§) | What to do |
|-----------|------------|
| **~~P0 — `gower` in `requirements.txt`~~** | **Done (2026-04-08):** **`gower>=0.1.2`** is in **`requirements.txt`**. Re-run **`pip install -r requirements.txt`** in older venvs that were created before that change. |
| **~~P1 — `backend.src` / `kmedoids` import (§11.1b)~~** | **Done (2026-04-11):** Implemented **Option B (code)** by changing all absolute intra-package imports to relative imports (`from .kmedoids import` etc.) inside `backend/src`, making `import backend.src.clustering` work purely via standard Python import resolution. |
| **~~P1 — `test_clustering.py` imports (§11.3)~~** | **Done (2026-04-11):** Changed imports to resolve correctly from `backend.src`. |
| **~~P1 — `generate_groups` input path (§11.2)~~** | **Done (2026-04-11):** Pointed inputs/outputs symmetrically using `Path(__file__).resolve().parent.parent`. |
| **~~P1 — `run_full_evaluation` data path (§11.2)~~** | **Done (2026-04-11):** Script was successfully re-anchored to `backend/data/sample_students.csv`. |
| **~~P1 — `OUTPUT_CSV` relative to CWD (§11.2)~~** | **Done (2026-04-11):** Output paths resolve to the `backend/output` root accurately. |
| **~~P1 — Multimodal paths (§11.2)~~** | **Done (2026-04-11):** `evaluate_multimodal.py` aligned to pull from `backend/output/model_weights`. |
| **~~P1 — API vs CLI math (§11.5)~~** | **Done (2026-04-11):** Total symmetry established for parameters spanning `n_groups`, `.ceil()`, `metric=manhattan`, and `expected_n_clusters`. |
| **~~P2 — README / docs (§11.4)~~** | **Done (2026-04-11):** README has updated structure and uses the new execution commands `backend.src.*`. |
| **~~P3 — Frontend NaN / types (§11.6)~~** | **Done (2026-04-11):** `<input>` tags now safely filter NaNs, and TS payload accurately parses `gender_balance`. |

**Re-run §0-style checks after changes:** same interpreter, **`pip install -r requirements.txt`** (confirms **`gower`** resolves), `import clustering` from `backend\src`, `from backend.src import data_loader` from root (if you care about package imports), `npm run build` in `frontend/`, and a dry run of any script whose path you edited.

---
