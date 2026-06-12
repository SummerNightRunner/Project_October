# Project October

Project October - веб-приложение и API для рекомендаций фильмов.

Сейчас проект находится в состоянии очищенного Python-прототипа: есть рабочее рекомендательное ядро на CSV-датасетах, минимальная точка входа FastAPI и дымовые тесты.

## Структура

- `backend/` - Python-код прототипа и будущего API.
- `backend/app/main.py` - FastAPI-приложение.
- `backend/recommendations_func.py` - текущая рекомендательная функция на признаках фильма.
- `backend/data_preprocessor.py` - сборка `data/processed/processed_metadata.csv` из исходных CSV.
- `data/raw/` - исходные CSV-датасеты, отслеживаются через Git LFS.
- `data/processed/` - локальные генерируемые артефакты, не коммитятся.
- `docs/` - проектная документация, дорожная карта, backlog, API-спецификация, решения и дневник.

## Локальный запуск API

```bash
python -m pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

Пример поиска фильмов:

```bash
curl "http://127.0.0.1:8000/movies/search?query=toy&limit=5"
```

Для `GET /movies/search` нужен локальный файл `data/processed/processed_metadata.csv`.

Пример запроса рекомендаций:

```bash
curl -X POST http://127.0.0.1:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"liked_movie_ids":["862","8844"],"include_adult":false,"limit":5}'
```

Для `POST /recommendations` нужен локальный файл `data/processed/processed_metadata.csv`.

## Конфигурация базы данных

PostgreSQL-подключение настраивается через переменную окружения `PROJECT_OCTOBER_DATABASE_URL`.
Значение не хранится в коде и не должно попадать в Git.

Пример формата без реальных секретов:

```bash
export PROJECT_OCTOBER_DATABASE_URL="postgresql+psycopg://<user>@<host>:<port>/<database>"
```

Alembic использует тот же URL:

```bash
alembic history
alembic upgrade head
```

Миграция `db003_user_history_schema` создает базовые таблицы пользовательской истории из DB-001:
`users`, `movie_catalog_entries`, `user_movie_history`, `user_movie_ratings`,
`user_preferences`, `api_clients`, `api_keys`, `user_events`.

`movie_catalog_entries` пока не синхронизируется с `processed_metadata.csv`; это отдельная задача DB-004.
Endpoints пользовательской истории и оценок также не реализованы и остаются для API-004.

## Проверка текущего рекомендательного прототипа

Если `data/processed/processed_metadata.csv` отсутствует, сначала пересоберите его:

```bash
python backend/data_preprocessor.py
```

Затем запустите пример рекомендаций:

```bash
python backend/recommendations_func.py
```

## Тесты

```bash
python -m pytest
```

## Правила хранения данных

Исходные CSV лежат в `data/raw/` и отслеживаются через Git LFS. Генерируемые файлы в `data/processed/`, секреты, локальные базы, файлы промптов Codex и ML-артефакты не коммитятся.
