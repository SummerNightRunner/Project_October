# Roadmap

## Phase 0 - Organize The Project

Status: in progress

- Create project docs and role prompts.
- Define MVP scope.
- Document current repository state.
- Establish daily workflow.

## Phase 1 - Stabilize Existing Prototype

Status: planned

- Fix inconsistent data paths.
- Fix parsing of `genres_list` and `keywords_list` after loading processed CSV.
- Add dependency file.
- Add smoke tests for recommendation output.
- Decide whether early persistence uses SQLite or PostgreSQL.

## Phase 2 - Backend API MVP

Status: planned

- Create FastAPI app structure.
- Add health endpoint.
- Add movie search endpoint.
- Add recommendation endpoint.
- Add request and response schemas.
- Add OpenAPI-friendly examples.

## Phase 3 - User Data MVP

Status: planned

- Add users table.
- Add watched history events.
- Add ratings and preference events.
- Exclude watched movies from recommendations.
- Add simple auth.

## Phase 4 - Website MVP

Status: planned

- Create frontend app.
- Add movie search.
- Add watched history management.
- Add recommendation results page.
- Add movie details page.
- Connect frontend to backend API.

## Phase 5 - Public Developer API

Status: planned

- Add API key model.
- Add `/v1/recommendations`.
- Add rate-limit design.
- Add usage logging.
- Add API examples and developer docs.

## Phase 6 - Imports And Enrichment

Status: planned

- Add CSV import.
- Add TMDB metadata enrichment research.
- Research legal and technical options for Kinopoisk history import.
- Add ID mapping between local dataset, TMDB, IMDb, and possible Kinopoisk IDs.

## Phase 7 - Better Recommendation Quality

Status: planned

- Add feedback loop from user ratings.
- Add hybrid recommendation scoring.
- Evaluate recommendation quality with small test profiles.
- Consider embeddings and vector search.
