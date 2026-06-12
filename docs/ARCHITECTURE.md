# Architecture

## Текущее состояние

Проект очищен до базового состояния:

- FastAPI entrypoint: `backend/app/main.py`.
- Рекомендательное ядро: `backend/recommendations_func.py`.
- Препроцессинг данных: `backend/data_preprocessor.py`.
- Raw CSV: `data/raw/`.
- Generated metadata: `data/processed/`.

## Целевая архитектура

```text
frontend app
    |
    v
FastAPI backend
    |
    +-- movie catalog service
    +-- recommendation service
    +-- user history service
    +-- public API service
    |
    v
PostgreSQL
```

## Backend

Ближайшая целевая структура:

```text
backend/
  app/
    __init__.py
    main.py
  recommendations_func.py
  data_preprocessor.py
  ratings_updater.py
  user_registration.py
```

Дальше, когда API начнет расти, можно перейти к пакетной структуре:

```text
backend/
  app/
    main.py
    routes/
    schemas/
    services/
    db/
```

Пока не усложняем структуру без необходимости.

## Data Policy

- `data/raw/*.csv` хранится через Git LFS.
- `data/processed/` генерируется локально и не коммитится.
- `data/key/secret.key`, `data/users.csv`, локальные базы и приватная история пользователей не коммитятся.
- Будущие embeddings, vector indexes, checkpoints и experiment runs не коммитятся без явного решения.

## Recommendation Strategy

MVP использует текущую content-based модель:

- TF-IDF по описаниям;
- жанры;
- keywords;
- флаги `adult` и `animation`;
- бонус за коллекцию.

Позже:

- учет пользовательской истории;
- гибридная модель;
- объяснения рекомендаций;
- embeddings/vector search только после стабилизации API и данных.
