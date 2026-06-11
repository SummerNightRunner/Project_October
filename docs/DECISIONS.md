# Architecture Decisions

## ADR-0001: Use Codex Project Docs As The Team Memory

Date: 2026-06-11

Status: accepted

Decision:

Use `AGENTS.md` plus `docs/` as the durable project memory for Codex-assisted development.

Reasoning:

- The project needs a repeatable workflow across multiple threads.
- `AGENTS.md` gives Codex persistent repository guidance.
- `docs/ROADMAP.md`, `docs/BACKLOG.md`, and `docs/JOURNAL.md` make the project easier to resume.

Consequences:

- Significant work should update docs.
- Task threads should reference the relevant prompt from `docs/prompts/`.
- The Project HQ thread should maintain planning and journal state.

## ADR-0002: Build A Web App And API From The Same Backend

Date: 2026-06-11

Status: proposed

Decision:

Use one FastAPI backend for both the website and public developer API.

Reasoning:

- Avoid duplicate recommendation logic.
- Make the website a first-class client of the same API used by external projects.
- Keep OpenAPI documentation close to implementation.

Consequences:

- API schemas must be treated as product contracts.
- Public endpoints should be versioned under `/v1`.
- Internal app endpoints can evolve faster than public endpoints.

## ADR-0003: Support Manual And CSV History Input Before Kinopoisk Import

Date: 2026-06-11

Status: proposed

Decision:

Implement manual history input and CSV import before attempting Kinopoisk integration.

Reasoning:

- A stable official Kinopoisk user-history API is not confirmed.
- Manual and CSV input unlock the core product without external dependency risk.
- Import adapters can be added later behind a clean interface.

Consequences:

- MVP should not depend on Kinopoisk.
- Kinopoisk remains a research task until feasibility is proven.

## ADR-0004: Use PostgreSQL From The Start For App Persistence

Date: 2026-06-12

Status: accepted

Decision:

Use PostgreSQL as the target database from the start for application users, ratings, history events, API keys, and future app state.

Reasoning:

- The product direction includes user accounts, watched history, API keys, and usage logging.
- PostgreSQL avoids an early SQLite-to-PostgreSQL migration for core persistence.
- SQLite can still be used only for isolated local experiments when explicitly scoped that way.

Consequences:

- Backend setup should include PostgreSQL-oriented configuration and documentation.
- API and user-data tasks should avoid adding new app state to local CSV files.
- Local development needs a clear PostgreSQL setup path.

## ADR-0005: Keep Codex Prompt Files Local

Date: 2026-06-12

Status: accepted

Decision:

Treat `docs/prompts/` as exclusively local Codex working files.

Reasoning:

- Prompt files are local operating instructions, not product documentation.
- Keeping them local reduces the risk of publishing personal workflow details or agent-specific process notes.

Consequences:

- Do not push `docs/prompts/` unless the user explicitly reverses this decision.
- Project documentation that should be shared belongs in the main `docs/` files outside `docs/prompts/`.
