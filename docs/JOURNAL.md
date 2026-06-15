# Дневник проекта

## 2026-06-15

- Создан дневной план на 2026-06-15 после проверки `docs/ROADMAP.md`, `docs/BACKLOG.md`, `docs/JOURNAL.md`, последней daily note и `git status --short`.
- Текущее состояние: `main` на merge `API-007`, рабочее дерево было чистым до документационных обновлений, Backend API MVP продвинут до owner-bound API-key доступа для истории и оценок.
- Первая рекомендуемая задача на день: `API-006`, endpoints ручных предпочтений пользователя.
- Дополнительные рекомендуемые задачи: `OCT-003`, `TEST-002`, `WEB-001`, `EXT-001`.
- Source-файлы не изменялись.
- Project HQ подтвердил `API-007` после merge в `main`: локальный запуск во временном venv `/tmp/project_october_api007_venv` прошел, 34 теста успешны, 1 предупреждение Starlette о deprecated `httpx` import.

## 2026-06-13

- Проведена очистка после неудачного/противоречивого результата задачи.
- Восстановлен нормальный проектный контур документации: `PROJECT_BRIEF`, `ROADMAP`, `ARCHITECTURE`, `API_SPEC`, `DECISIONS`, `PROCESS`, backlog и daily.
- Зафиксирована политика данных: исходные CSV находятся в `data/raw/` и отслеживаются через Git LFS, `data/processed/` игнорируется как генерируемый артефакт.
- Старый интерактивный CLI заменен на минимальный FastAPI app с `GET /health` в `backend/app/main.py`.
- Добавлен `requirements.txt` для локального запуска backend.
- Текущий рекомендательный прототип сохранен и продолжает жить отдельно от API-слоя.
- Добавлены дымовые тесты для `GET /health` и текущего примера рекомендаций.
- README дополнен командой запуска тестов.
- Project HQ подтвердил `TEST-001`: локальный запуск `python -m pytest` прошел, 2 теста успешны.
- Выполнен `API-002`: добавлен `POST /recommendations` с Pydantic-валидацией `liked_movie_ids`, `include_adult`, `limit`; endpoint вызывает текущую рекомендательную функцию и возвращает список рекомендаций.
- `docs/API_SPEC.md`, smoke-тесты и README обновлены под новый endpoint.
- Project HQ подтвердил `API-002` после merge в `main`: локальный запуск `python -m pytest` прошел, 5 тестов успешны.
- Выполнен `API-003`: добавлен `GET /movies/search` с поиском по локальному `processed_metadata.csv`, поддержкой `PROJECT_OCTOBER_PROCESSED_METADATA`, валидацией `query` и `limit`, стабильным ответом `items` и ошибкой `503` при недоступном каталоге.
- `docs/API_SPEC.md`, README, backlog, daily note и smoke-тесты обновлены под поиск фильмов.
- Проверка `python -m pytest` прошла: 9 тестов успешны.
- Выполнен `DB-001`: спроектирована PostgreSQL-схема пользовательской истории без изменения backend-кода и данных.
- Зафиксированы таблицы `users`, `movie_catalog_entries`, `user_movie_history`, `user_movie_ratings`, `user_preferences`, `api_clients`, `api_keys`, `user_events`.
- В `docs/API_SPEC.md` добавлены черновые будущие контракты для истории, оценок, предпочтений и API-key доступа.
- В backlog добавлены follow-up задачи `DB-002`, `DB-003`, `DB-004`, `API-004`, `API-005`.
- Project HQ подтвердил `DB-001` после merge в `main`: локальный запуск `python -m pytest` прошел, 9 тестов успешны.
- Выполнен `DB-002`: добавлены зависимости `SQLAlchemy`, `Alembic`, `psycopg`, DB-конфигурация через `PROJECT_OCTOBER_DATABASE_URL`, ленивые engine/session helpers и Alembic scaffold без пользовательских таблиц.
- README, архитектура, decisions, backlog и daily note обновлены под базовую DB-инфраструктуру.
- Project HQ подтвердил `DB-002` после merge в `main`: локальный запуск `python -m pytest` прошел, 11 тестов успешны.
- Выполнен `DB-003`: добавлены SQLAlchemy-модели и Alembic revision `db003_user_history_schema` для таблиц `users`, `movie_catalog_entries`, `user_movie_history`, `user_movie_ratings`, `user_preferences`, `api_clients`, `api_keys`, `user_events`.
- Миграция создает PostgreSQL extensions `pgcrypto` и `citext`, FK/unique/check constraints и индексы из DB-001, включая partial unique index для непустого `users.email`.
- Добавлены проверки, что `Base.metadata` содержит таблицы пользовательской истории, Alembic env импортирует эти metadata без `PROJECT_OCTOBER_DATABASE_URL`, а ревизия импортируется без подключения к базе.
- DB-004 и API-004 не реализованы: таблица `movie_catalog_entries` пока не синхронизируется с обработанным каталогом, endpoints пользовательской истории отсутствуют.
- Проверка `python -m pytest` прошла: 13 тестов успешны; `alembic history` видит `db003_user_history_schema`.
- Project HQ подтвердил `DB-003` после merge в `main`: локальный запуск `python -m pytest` прошел, 13 тестов успешны; `alembic history` показывает `db003_user_history_schema`.
- Выполнен `DB-004`: добавлен сервис синхронизации `movie_catalog_entries` из `processed_metadata.csv`, общий resolver `PROJECT_OCTOBER_PROCESSED_METADATA` и CLI entrypoint `python -m backend.app.db.sync_movie_catalog`.
- Синхронизация делает upsert по `catalog_movie_id`, обновляет `title_snapshot`, `release_date`, `source_catalog_version`, `updated_at`, сохраняет `created_at` и не удаляет фильмы, отсутствующие в новом CSV.
- Добавлены тесты чтения fixture CSV, повторного upsert без удаления устаревших строк и CLI entrypoint на временной SQLite-базе.
- Проверка `python -m pytest` прошла: 17 тестов успешны; отдельный CLI smoke на fixture CSV и временной SQLite-базе прошел без подключения к production DB.
- Project HQ подтвердил `DB-004` после merge в `main`: локальный запуск `python -m pytest` прошел, 17 тестов успешны.
- Выполнен `API-004`: добавлены `GET /users/{user_id}/history`, `PUT /users/{user_id}/history/{movie_id}` и `PUT /users/{user_id}/ratings/{movie_id}`.
- Endpoints используют `movie_id` как публичное имя `catalog_movie_id`, проверяют наличие фильма в `movie_catalog_entries`, возвращают `404` для неизвестного фильма и валидируют оценки в диапазоне `0..10` с одним знаком после запятой.
- Для MVP принято автосоздание активного пользователя на write-запросах истории и оценок; `GET` не создает пользователя и возвращает пустую историю, если записей нет.
- Добавлены endpoint-тесты на временной SQLite-базе через FastAPI dependency override, без подключения к production DB.
- Проверка `python -m pytest` прошла через временный `/tmp` alias на Python 3.12 Codex runtime: 24 теста успешны, 1 предупреждение Starlette о deprecated `httpx` import.
- Project HQ подтвердил `API-004` после merge в `main`: локальный запуск во временном venv прошел, 24 теста успешны, 1 предупреждение Starlette о deprecated `httpx` import.
- Выполнен `API-005`: добавлен API-key auth для endpoints пользовательской истории и оценок.
- Ключи имеют формат `oct_<prefix>_<secret>`; backend ищет запись по `key_prefix`, проверяет `key_hash` полного ключа через `hmac.compare_digest`, статус ключа, срок действия, статус API client и требуемый scope.
- Реализованы scopes `history:read`, `history:write`, `ratings:write`, `recommendations:read`; `GET /users/{user_id}/history`, `PUT /users/{user_id}/history/{movie_id}` и `PUT /users/{user_id}/ratings/{movie_id}` теперь требуют соответствующие scopes.
- `POST /recommendations` временно оставлен публичным до появления тарифов, лимитов и модели публичного API-доступа; решение зафиксировано в `docs/DECISIONS.md`.
- Добавлены auth-тесты на отсутствующий/неверный/revoked/expired ключ, недостаточный scope, успешный scope и обновление `last_used_at`.
- Проверка через временный venv: `python -m pytest` прошел, 31 тест успешен, 1 предупреждение Starlette о deprecated `httpx` import.
- Project HQ подтвердил `API-005` после merge в `main`: локальный запуск во временном venv прошел, 31 тест успешен, 1 предупреждение Starlette о deprecated `httpx` import.
- Выполнен `API-007`: user-scoped endpoints теперь проверяют, что
  `api_clients.owner_user_id` совпадает с path `user_id`.
- Client с `owner_user_id = NULL` получает `403` на endpoints пользовательской
  истории и оценок; service clients, роли и доступ к нескольким пользователям
  не реализованы.
- `api_keys.last_used_at` обновляется после успешной проверки ключа и scope,
  включая случай valid key + forbidden user.
- Добавлены тесты на доступ владельца к своему `user_id`, отказ для чужого
  `user_id`, отказ для client без владельца, сохранение `401` для неверного
  ключа и сохранение `403` для недостаточного scope.
- Проверка через временный venv `/tmp/project_october_api007_venv`:
  `python -m pytest` прошел, 34 теста успешны, 1 предупреждение Starlette о
  deprecated `httpx` import.
