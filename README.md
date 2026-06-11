# Project October

Movie recommendation web app and public API.

The repository currently contains a Python prototype for movie recommendations. The project is being organized into a full product with backend API, frontend app, user history, and developer-facing recommendation endpoints.

Start here:

- `docs/PROJECT_BRIEF.md` - product goal and MVP scope.
- `docs/PROCESS.md` - how to run the project workflow with Codex.
- `docs/ROADMAP.md` - phased development plan.
- `docs/BACKLOG.md` - task list.
- `docs/ARCHITECTURE.md` - target technical architecture.
- `docs/API_SPEC.md` - draft API contract.

Local Codex guidance may exist in `AGENTS.md` and `docs/prompts/`. These files are local workflow aids by default and are not required for running the application.

## Current prototype

Raw CSV datasets are expected under `data/raw/`; generated recommendation data is expected under `data/processed/`.

Run the current recommendation smoke example from the repository root:

```bash
python backend/recommendations_func.py
```

## Backend API

Install dependencies and run the FastAPI app from the repository root:

```bash
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```
