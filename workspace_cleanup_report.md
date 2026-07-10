# GroupGen Workspace Cleanup Report

**Date:** 2026-06-01  
**Scope:** Flatten redundant nested directories, align documentation and configuration, verify backend + frontend boot paths.

---

## [Legacy Layout]

```
GroupGen/
├── backend/
│   ├── api.py, pipeline.py, …          # production Python package (correct)
│   ├── backend/                        # ❌ accidental nested folder (cwd bug artifact)
│   │   └── output/
│   │       └── final_groups_report.txt
│   ├── data/templates/
│   └── output/, output_plots/
├── frontend/
│   └── frontend/                       # ❌ redundant nesting (create-next-app in subfolder)
│       ├── src/app/page.tsx
│       ├── public/
│       ├── package.json
│       ├── node_modules/
│       └── .next/
├── docs/ARCHITECTURE.md
├── README.md
├── STUDY_SETUP.md
└── requirements.txt
```

**Problems:**
- Docs and onboarding said `cd frontend/frontend` — non-standard and error-prone.
- A prior CLI run from inside `backend/` created `backend/backend/output/`.
- `.next` build cache contained hard-coded `frontend/frontend` absolute paths.

---

## [Aligned Layout]

```
GroupGen/
├── backend/                            # Python package (unchanged module layout)
│   ├── api.py
│   ├── check_imports.py
│   ├── paths.py                        # canonical filesystem paths
│   ├── pipeline.py, clustering.py, …
│   ├── data/templates/classroom_template.csv
│   └── output/runs/                    # CLI outputs (gitignored)
├── frontend/                           # ✅ flat Next.js app root
│   ├── src/app/page.tsx, layout.tsx
│   ├── public/groupgen-high-resolution-logo.png
│   ├── package.json, tsconfig.json, next.config.ts
│   ├── .env.example
│   └── README.md
├── docs/ARCHITECTURE.md
├── README.md
├── STUDY_SETUP.md
├── workspace_cleanup_report.md         # this file
└── requirements.txt
```

**Removed:**
- `frontend/frontend/` (contents promoted to `frontend/`)
- `backend/backend/` (empty nested tree deleted)

---

## [Modified Files Table]

| File Name | Old Path Location | New Path Location | Key Changes Made |
|-----------|-------------------|-------------------|------------------|
| `page.tsx` | `frontend/frontend/src/app/page.tsx` | `frontend/src/app/page.tsx` | **Moved** (no code changes; API fetch unchanged) |
| `layout.tsx` | `frontend/frontend/src/app/layout.tsx` | `frontend/src/app/layout.tsx` | **Moved** |
| `globals.css` | `frontend/frontend/src/app/globals.css` | `frontend/src/app/globals.css` | **Moved** |
| `package.json` | `frontend/frontend/package.json` | `frontend/package.json` | **Moved** |
| `package-lock.json` | `frontend/frontend/package-lock.json` | `frontend/package-lock.json` | **Moved** |
| `tsconfig.json` | `frontend/frontend/tsconfig.json` | `frontend/tsconfig.json` | **Moved** (`@/*` → `./src/*` unchanged) |
| `next.config.ts` | `frontend/frontend/next.config.ts` | `frontend/next.config.ts` | **Moved** |
| `eslint.config.mjs` | `frontend/frontend/eslint.config.mjs` | `frontend/eslint.config.mjs` | **Moved** |
| `postcss.config.mjs` | `frontend/frontend/postcss.config.mjs` | `frontend/postcss.config.mjs` | **Moved** |
| `next-env.d.ts` | `frontend/frontend/next-env.d.ts` | `frontend/next-env.d.ts` | **Moved** |
| `public/*` | `frontend/frontend/public/` | `frontend/public/` | **Moved** (logo + static assets) |
| `.gitignore` (frontend) | `frontend/frontend/.gitignore` | `frontend/.gitignore` | **Moved** |
| `README.md` (frontend) | `frontend/frontend/README.md` (CRA boilerplate) | `frontend/README.md` | **Replaced** with GroupGen-specific run instructions |
| `.env.example` | *(none)* | `frontend/.env.example` | **Created** — documents `NEXT_PUBLIC_API_URL` |
| `README.md` (root) | `GroupGen/README.md` | same | Updated all `frontend/frontend` → `frontend` |
| `STUDY_SETUP.md` | `GroupGen/STUDY_SETUP.md` | same | Updated cd paths and logo location |
| `ARCHITECTURE.md` | `docs/ARCHITECTURE.md` | same | Updated Web UI path; paths.py note |
| `.gitignore` (root) | `GroupGen/.gitignore` | same | Added `frontend/.next/`, `node_modules/`; removed obsolete `backend/backend/` guard |
| `clustering.py` | `backend/clustering.py` | same | *(prior fix)* `from .kmedoids import kmedoids_pam` |
| `api.py` | `backend/api.py` | same | *(prior fix)* relative imports `from .pipeline import …` |
| `generate_groups.py` | `backend/generate_groups.py` | same | *(prior fix)* relative imports + `paths.py` |
| `paths.py` | `backend/paths.py` | same | *(prior fix)* canonical `BACKEND_DIR`, `PROJECT_ROOT`, output dirs |
| `check_imports.py` | `backend/check_imports.py` | same | *(prior fix)* smoke test for imports + pipeline |
| `final_groups_report.txt` | `backend/backend/output/` | **deleted** | Removed stale nested artifact |
| `.next/` build cache | `frontend/frontend/.next/` | **deleted before move** | Regenerated at `frontend/.next/` on next `npm run dev/build` |

**Python / TypeScript import notes:**
- No `@/` path changes required — `tsconfig.json` still maps `@/*` → `./src/*` relative to `frontend/`.
- Backend package imports use relative `from .module` inside `backend/`; entry via `python -m backend.*` or `uvicorn backend.api:app` from **repo root**.
- UI → API contract unchanged: `POST {NEXT_PUBLIC_API_URL}/generate-groups?group_size=N` (default `http://127.0.0.1:8000`).

---

## Terminal operations executed

These commands were run to perform the physical flattening (PowerShell, repo root):

```powershell
# 1. Remove stale Next.js cache (contained old nested absolute paths)
Remove-Item "frontend\frontend\.next" -Recurse -Force

# 2. Promote inner frontend app to frontend/
Get-ChildItem "frontend\frontend" -Force | Move-Item -Destination "frontend\" -Force

# 3. Remove empty nested shell
Remove-Item "frontend\frontend" -Recurse -Force

# 4. Remove accidental backend nesting
Remove-Item "backend\backend" -Recurse -Force
```

**Equivalent manual steps (if recreating on another machine):**

```bash
# Unix/macOS
rm -rf frontend/frontend/.next
mv frontend/frontend/* frontend/frontend/.[!.]* frontend/ 2>/dev/null || true
rmdir frontend/frontend
rm -rf backend/backend
```

---

## [Verification Checklist]

Run from **repo root** (`GroupGen/`):

### Backend

- [ ] `.\venv\Scripts\Activate.ps1` (or `source venv/bin/activate`)
- [ ] `pip install -r requirements.txt`
- [ ] `python -m backend.check_imports`  
  **Expect:** `OK — imports and pipeline (10 students, 2 groups)`
- [ ] `uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000`  
  **Expect:** `Application startup complete.` (no import traceback)
- [ ] Browser: [http://127.0.0.1:8000](http://127.0.0.1:8000) → `{"status":"GroupGen API is running"}`

### Frontend

- [ ] `cd frontend`
- [ ] `npm install`
- [ ] `npm run dev`  
  **Expect:** `Local: http://localhost:3000`
- [ ] *(Optional)* `npm run build`  
  **Expect:** `✓ Compiled successfully` and route `/` listed (ESLint config warning is non-fatal)
- [ ] Upload `backend/data/templates/classroom_template.csv`, group size **5**  
  **Expect:** 2 groups, 5 students each, summary line at top

### CLI (same pipeline as API)

- [ ] `python -m backend.generate_groups` → enter `5`  
  **Expect:** new folder under `backend/output/runs/<timestamp>_<id>/` with CSV + manifest

### Path hygiene

- [ ] `Test-Path frontend\frontend` → **False**
- [ ] `Test-Path backend\backend` → **False**
- [ ] `Test-Path frontend\src\app\page.tsx` → **True**
- [ ] `Test-Path frontend\package.json` → **True**

---

## Post-migration daily commands (quick reference)

| Task | Command |
|------|---------|
| API | `uvicorn backend.api:app --reload --port 8000` |
| UI | `cd frontend` → `npm run dev` |
| Smoke test | `python -m backend.check_imports` |
| CLI groups | `python -m backend.generate_groups` |

---

## Verification results (automated run on 2026-06-01)

| Check | Result |
|-------|--------|
| `python -m backend.check_imports` | **PASS** |
| `npm run build` (from `frontend/`) | **PASS** (compiled; minor ESLint config warning) |
| Nested `frontend/frontend` | **Removed** |
| Nested `backend/backend` | **Removed** |

---

*End of workspace cleanup report.*
