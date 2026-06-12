# Архитектура

## Текущее состояние

Проект очищен до базового состояния:

- точка входа FastAPI: `backend/app/main.py`.
- Рекомендательное ядро: `backend/recommendations_func.py`.
- Препроцессинг данных: `backend/data_preprocessor.py`.
- Исходные CSV: `data/raw/`.
- Генерируемые метаданные: `data/processed/`.

## Целевая архитектура

```text
frontend-приложение
    |
    v
FastAPI backend
    |
    +-- сервис каталога фильмов
    +-- сервис рекомендаций
    +-- сервис пользовательской истории
    +-- публичный API-сервис
    |
    v
PostgreSQL
```

## Backend-структура

Текущая ближайшая структура:

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

Когда API начнет расти, можно перейти к более явной пакетной структуре:

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

## Политика данных

- `data/raw/*.csv` хранится через Git LFS.
- `data/processed/` генерируется локально и не коммитится.
- `data/key/secret.key`, `data/users.csv`, локальные базы и приватная история пользователей не коммитятся.
- Будущие embeddings, vector indexes, checkpoints и результаты экспериментов не коммитятся без явного решения.

## Стратегия рекомендаций

MVP использует текущую модель на признаках фильма:

- TF-IDF по описаниям;
- жанры;
- ключевые слова;
- флаги `adult` и `animation`;
- бонус за коллекцию.

Позже:

- учет пользовательской истории;
- гибридная модель;
- объяснения рекомендаций;
- embeddings/vector search только после стабилизации API и данных.
