# Architecture

## Current Architecture

The current repository is a local Python prototype:

- `backend/app/main.py` exposes the initial FastAPI app entrypoint.
- `backend/data_preprocessor.py` prepares movie metadata.
- `backend/recommendations_func.py` computes recommendations.
- `backend/user_registration.py` stores encrypted user data in CSV.
- `backend/ratings_updater.py` appends ratings to CSV and reruns preprocessing.
- `data/raw/` contains source CSV datasets.
- `data/processed/` contains generated recommendation data.

## Target Architecture

```text
frontend app
    |
    v
FastAPI backend
    |
    +-- recommendation service
    +-- movie catalog service
    +-- user profile service
    +-- public API key service
    |
    v
PostgreSQL
    |
    +-- movies
    +-- users
    +-- user_movie_events
    +-- ratings
    +-- api_keys
    +-- api_usage_logs
```

## Backend Modules

Planned structure:

```text
backend/
  app/
    main.py
    api/
      routes/
    core/
    db/
    models/
    schemas/
    services/
      recommendations.py
      movies.py
      users.py
    tests/
```

## Recommendation Strategy

MVP:

- content-based recommendations;
- TF-IDF over movie descriptions;
- genre and keyword similarity;
- adult and animation flags;
- collection bonus;
- exclude already watched movies.

Later:

- collaborative filtering from user ratings;
- hybrid scoring;
- embeddings for descriptions and metadata;
- vector search with FAISS or pgvector;
- recommendation explanations.

## Data Strategy

- Keep large source datasets in `data/raw/`.
- Keep generated artifacts in `data/processed/`.
- Do not store application user state in raw CSV long-term.
- Move users, history, ratings, and API keys to a database.

## External Integrations

- TMDB is the preferred candidate for official movie metadata enrichment.
- Kinopoisk history import is research-only until a stable and legal import path is confirmed.
- CSV import should be the first supported history import path.
