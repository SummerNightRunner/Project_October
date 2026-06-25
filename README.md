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

## Локальный запуск backend MVP через Docker Compose

Docker Compose контур предназначен для локальной разработки и ручной проверки
backend MVP. Он поднимает:

- `db` - PostgreSQL 16 с локальной базой `project_october`;
- `backend` - FastAPI app на `http://127.0.0.1:8000`.

Compose использует только безопасные локальные значения окружения. Пароль
`project_october_local_password` не является production-секретом и нужен только
для локального контейнера. PostgreSQL data хранится в named volume
`project_october_postgres_data`. Для доступа с хоста PostgreSQL опубликован на
`127.0.0.1:5433`, а backend внутри Compose подключается к `db:5432` через
`PROJECT_OCTOBER_DATABASE_URL`.

Первый запуск начинается со сборки backend image:

```bash
docker compose build backend
```

Если `data/processed/processed_metadata.csv` отсутствует, сначала создайте его
из текущих raw CSV:

```bash
python backend/data_preprocessor.py
```

То же можно выполнить внутри backend image, если локальный Python не настроен:

```bash
docker compose run --rm --no-deps backend python backend/data_preprocessor.py
```

Запустите PostgreSQL и дождитесь healthy-состояния через Compose healthcheck:

```bash
docker compose up -d db
```

Примените Alembic migrations:

```bash
docker compose run --rm backend alembic upgrade head
```

Синхронизируйте `movie_catalog_entries` из обработанного каталога:

```bash
docker compose run --rm backend python -m backend.app.db.sync_movie_catalog
```

Запустите API:

```bash
docker compose up -d backend
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

Для повторного запуска после уже примененных миграций и синхронизации обычно
достаточно:

```bash
docker compose up -d db backend
```

Остановить контейнеры без удаления PostgreSQL volume:

```bash
docker compose down
```

Удалить локальную PostgreSQL data можно только осознанно:

```bash
docker compose down -v
```

DEV-001 не добавляет demo user/API key seed. Это остается отдельной задачей
`DEV-002`; до нее user-scoped endpoints требуют вручную созданные записи
`users`, `api_clients` и `api_keys` с корректными scopes.

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

`movie_catalog_entries` синхронизируется из обработанного каталога отдельной командой.
По умолчанию используется `data/processed/processed_metadata.csv`; если задана
`PROJECT_OCTOBER_PROCESSED_METADATA`, используется путь из этой переменной.

```bash
python -m backend.app.db.sync_movie_catalog
```

Для локальной проверки можно явно передать тестовый путь и тестовый DSN:

```bash
python -m backend.app.db.sync_movie_catalog \
  --metadata-path /path/to/processed_metadata.csv \
  --database-url "sqlite+pysqlite:////tmp/project_october_catalog_sync.sqlite" \
  --source-catalog-version "local-fixture"
```

Команда выполняет upsert по `catalog_movie_id`: добавляет новые фильмы и обновляет
`title_snapshot`, `release_date`, `source_catalog_version`, `updated_at` для фильмов,
которые уже есть в таблице. Фильмы, пропавшие из CSV, не удаляются.

## Пользовательская история, оценки и предпочтения

Endpoints пользовательской истории используют синхронизированную таблицу
`movie_catalog_entries`. Перед записью истории или оценки `movie_id` должен
существовать в этой таблице.

Эти endpoints требуют API key в заголовке `Authorization: Bearer <api_key>`.
Формат ключа: `oct_<prefix>_<secret>`. В базе хранятся только `key_prefix`
вида `oct_<prefix>` и `key_hash` полного ключа; полный секрет не хранится.
Для endpoints с `user_id` ключ должен принадлежать API client, у которого
`api_clients.owner_user_id` равен этому `user_id`. Если `owner_user_id` не
задан, user-scoped endpoints возвращают `403`.

Пример записи истории:

```bash
curl -X PUT http://127.0.0.1:8000/users/<user_uuid>/history/862 \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"status":"watched","watched_at":"2026-06-13T12:00:00Z","source":"manual"}'
```

Пример записи оценки:

```bash
curl -X PUT http://127.0.0.1:8000/users/<user_uuid>/ratings/862 \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"rating_value":8.5,"rated_at":"2026-06-13T12:05:00Z","source":"manual"}'
```

Пример чтения истории:

```bash
curl "http://127.0.0.1:8000/users/<user_uuid>/history?status=watched&limit=20" \
  -H "Authorization: Bearer <api_key>"
```

Пример записи ручного предпочтения:

```bash
curl -X PUT http://127.0.0.1:8000/users/<user_uuid>/preferences/genre/comedy \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"weight":2.25,"source":"manual","is_active":true}'
```

Пример чтения ручных предпочтений:

```bash
curl "http://127.0.0.1:8000/users/<user_uuid>/preferences?is_active=true&limit=20" \
  -H "Authorization: Bearer <api_key>"
```

Для MVP write-endpoints сохраняют идемпотентное создание активного пользователя
по переданному UUID, если такого пользователя еще нет, но доступ к этим
endpoints уже ограничен `api_clients.owner_user_id`. В нормальной PostgreSQL
схеме владелец API client должен ссылаться на существующего пользователя.

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
