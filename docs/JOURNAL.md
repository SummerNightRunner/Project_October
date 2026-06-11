# Project Journal

## 2026-06-11

### Context

The repository started as a Python prototype for movie recommendations. It contains raw movie datasets, a processed metadata CSV, and backend scripts for preprocessing, recommendations, registration, and ratings updates.

### Work

- Identified target direction: website plus public movie recommendation API.
- Created project organization plan.
- Added project docs, role prompts, and daily workflow files.
- Added git policy for Codex-assisted work: agents may commit, push, and manage branches when useful, but must keep local AI instruction files, secrets, private data, and ML artifacts out of remote git by default.
- Stabilized the current backend recommendation prototype before FastAPI work.
- Updated backend data paths to use `data/raw/` for source CSVs and `data/processed/` for generated recommendation metadata.
- Fixed recommendation feature loading so `genres_list` and `keywords_list` are parsed back into Python lists after reading processed CSV.
- Added a README smoke-run note for `python backend/recommendations_func.py`.

### Decisions

- Treat Codex as a project secretary and task execution assistant.
- Use Project HQ for planning and separate task threads for implementation.
- Start with manual and CSV history input before external imports.
- Push project documentation to remote, but keep local Codex instruction files local unless explicitly approved.

### Next

- Confirm MVP scope.
- Create FastAPI skeleton.

## 2026-06-12

### Context

The backend recommendation prototype is stabilized enough to begin wrapping it with an API layer. The next roadmap step is Phase 2: Backend API MVP.

### Planning

- Recommended first task: API-001, create a FastAPI skeleton with `GET /health`.
- Keep dependency management, smoke tests, and MVP confirmation close behind API-001.
- Avoid broad data, auth, or frontend work until the API skeleton is in place.
- Updated the daily plan after checking roadmap, backlog, journal, latest daily note, and git status.
- Kept `docs/BACKLOG.md` unchanged because the current task ordering still matches the recommended next work.

### Work

- Completed API-001 by adding the initial FastAPI app entrypoint at `backend/app/main.py`.
- Added `GET /health`, returning `{"status": "ok"}`.
- Added the initial `requirements.txt` with FastAPI and Uvicorn.
- Documented the API launch command and health endpoint.

### Decisions

- Include the initial `requirements.txt` in API-001 instead of splitting it into a separate first dependency task.
- Use PostgreSQL from the start for application persistence direction.
- Treat `docs/prompts/` as exclusively local Codex working files, not project documentation to publish.

### Verification

- Ran a local Uvicorn smoke check for `GET /health`.
- Re-ran `python backend/recommendations_func.py` to confirm the recommendation prototype remains callable.

### Risks

- The working tree contains mixed backend, documentation, and data-path changes, so commits and pushes should be staged deliberately.
- The raw dataset location changed from `data/*.csv` to `data/raw/*.csv`; data commit policy still needs care.
- User, secret, and rating storage are still prototype-local and should move to database-backed persistence later.
- `docs/prompts/` should remain out of remote git unless the user explicitly approves publishing local Codex prompt files.
