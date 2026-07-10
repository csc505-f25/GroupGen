# GroupGen Frontend

Next.js upload UI for the classroom study. Teachers upload the **raw Google Forms CSV**; the backend converts and groups students via `POST /generate-groups`.

## Run (from this directory)

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Start the API first from repo root:

```bash
uvicorn backend.api:app --reload --port 8000
```

Optional: copy `.env.example` to `.env.local` and set `NEXT_PUBLIC_API_URL`.

## What the UI sends

- Multipart `file` — `.csv` from Google Forms (Responses → Download CSV)
- Query `group_size` — target students per group (default 5)

The browser does **not** parse or score the survey; all intake runs on the API (`group_gen_intake` + `vak_answer_catalog`).

## Docs

- [../README.md](../README.md) — install, intake overview, sample templates  
- [../STUDY_SETUP.md](../STUDY_SETUP.md) — form layout, VAK scoring, troubleshooting  
- [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) — API JSON contract  
