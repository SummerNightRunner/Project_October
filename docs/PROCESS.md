# Project Process

## Operating Model

Use one main Codex thread as Project HQ and separate task threads for implementation.

Default language for Codex-facing project work is Russian. Codex should write user-facing plans, statuses, reviews, summaries, and project-management notes in Russian. English is acceptable for code, commands, file names, API fields, package names, and established technical identifiers.

Project HQ is for:

- daily planning;
- roadmap and backlog grooming;
- architecture discussion;
- project journal updates;
- task creation and prioritization;
- end-of-day summaries.

Task threads are for:

- implementing one scoped change;
- reviewing one diff;
- debugging one failure;
- researching one integration;
- producing one design or API contract.

## Daily Loop

1. Open Project HQ.
2. Ask the Project Secretary prompt to read `docs/ROADMAP.md`, `docs/BACKLOG.md`, and `docs/JOURNAL.md`.
3. Generate 3-5 tasks for the day.
4. Pick one task.
5. Start a separate task thread with `docs/prompts/task-implementation.md`.
6. After completion, update the relevant docs.
7. End the day with `docs/prompts/end-of-day.md`.

## Task Format

Every implementation task should include:

- Goal.
- Context.
- Constraints.
- Files likely involved.
- Done when.
- Verification.

## Branch And Worktree Guidance

- Use a separate branch or Codex worktree for large tasks.
- Keep small documentation-only changes in the local checkout if no code work is in progress.
- Do not mix unrelated backend, frontend, ML, and documentation changes in the same task.
- Agents may create branches, commit completed work, push safe changes to `origin`, and clean up local branches when useful.
- Agents should ask the user before a merge request or pull request is needed; the user will create it.
- Avoid force pushes, shared-branch rebases, remote branch deletion, or direct protected-branch updates unless the user explicitly approves.
- Local Codex instruction files can guide all dialogs, but they are not automatically remote-facing project artifacts.

## What Belongs In Remote Git

Commit and push:

- source code;
- tests;
- project documentation;
- API specs;
- architecture decisions;
- templates and setup docs;
- safe sample configuration files.

Do not commit or push:

- secrets, tokens, credentials, or API keys;
- `data/key/secret.key`;
- `data/users.csv`;
- local databases such as `*.db`, `*.sqlite`, `*.sqlite3`;
- local Codex state such as `.codex/` or files from `~/.codex`;
- local neural-network/agent instructions such as `AGENTS.md`, `AGENTS.override.md`, and `docs/prompts/` unless the user explicitly approves publishing them;
- model checkpoints, embeddings, vector indexes, experiment outputs, or generated ML artifacts unless explicitly approved;
- private user history or personal imported watch data.

## Documentation Rules

- Product changes go to `docs/PROJECT_BRIEF.md` or `docs/ROADMAP.md`.
- API changes go to `docs/API_SPEC.md`.
- Architecture choices go to `docs/ARCHITECTURE.md` and `docs/DECISIONS.md`.
- Daily progress goes to `docs/JOURNAL.md` and `docs/daily/`.
- Reusable prompts go to `docs/prompts/` and are local Codex workflow files by default.
