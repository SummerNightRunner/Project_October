# Дневник проекта

## 2026-06-28

- Project Secretary создал дневной план на 2026-06-28 в 15:03 MSK после
  проверки `docs/ROADMAP.md`, `docs/BACKLOG.md`, `docs/JOURNAL.md`, последней
  daily note и `git status --short`.
- Текущее состояние: ветка `main`, последний commit
  `e6a35b2 docs: add daily plan for 2026-06-27`, рабочее дерево было чистым до
  документационных обновлений.
- Подтвержден фокус дня: `DEV-002` demo seed для локального MVP.
- Рекомендуемая последовательность: `DEV-002`, затем `API-008`, `OCT-003`,
  `TEST-002` и `WEB-002`.
- Source-файлы не изменялись; обновление ограничено проектной документацией.

## 2026-06-27

- Project Secretary создал дневной план на 2026-06-27 в 11:00 MSK после
  проверки `docs/ROADMAP.md`, `docs/BACKLOG.md`, `docs/JOURNAL.md`, последней
  daily note и `git status --short`.
- Текущее состояние: ветка `main`, последний commit
  `ba8e69d docs: add daily plan for 2026-06-26`, рабочее дерево было чистым до
  документационных обновлений.
- Подтвержден фокус дня: `DEV-002` demo seed для локального MVP.
- Рекомендуемая последовательность: `DEV-002`, затем `API-008`, `OCT-003`,
  `TEST-002` и `WEB-002`.
- Source-файлы не изменялись; обновление ограничено проектной документацией.

## 2026-06-26

- Project Secretary создал дневной план на 2026-06-26 в 11:11 MSK после
  проверки `docs/ROADMAP.md`, `docs/BACKLOG.md`, `docs/JOURNAL.md`, последней
  daily note и `git status --short`.
- Текущее состояние: ветка `main`, последний commit
  `b075a42 docs: confirm DEV-001 completion`, рабочее дерево было чистым до
  документационных обновлений.
- Подтвержден фокус дня: `DEV-002` demo seed для локального MVP.
- Рекомендуемая последовательность: `DEV-002`, затем `API-008`, `OCT-003`,
  `TEST-002` и `WEB-002`.
- Source-файлы не изменялись; обновление ограничено проектной документацией.

## 2026-06-25

- Выполнен `DEV-001`: добавлен `Dockerfile`, `.dockerignore` и
  `docker-compose.yml` для локального backend MVP.
- Compose поднимает PostgreSQL 16 и FastAPI backend; backend подключается к
  базе через `PROJECT_OCTOBER_DATABASE_URL`, а PostgreSQL data хранится в named
  volume `project_october_postgres_data`.
- Alembic migrations и синхронизация `movie_catalog_entries` оставлены явными
  one-off командами `docker compose run --rm backend alembic upgrade head` и
  `docker compose run --rm backend python -m backend.app.db.sync_movie_catalog`.
- README, архитектура, решения и backlog обновлены под локальный Docker Compose
  сценарий.
- Demo user/API key seed не реализован и остается задачей `DEV-002`.
- Проверка прошла: `python -m pytest`, `docker compose config`,
  `docker compose build backend`, `docker compose up -d db backend` и
  `GET /health` через Compose.
- Project HQ подтвердил `DEV-001` после merge в `main`: локально прошли
  `python -m pytest`, `docker compose config` и `docker compose build backend`.

## 2026-06-21

- Project Secretary создал дневной план на 2026-06-21 в 11:03 MSK после проверки `docs/ROADMAP.md`, `docs/BACKLOG.md`, `docs/JOURNAL.md`, последней daily note и `git status --short`.
- Текущее состояние: ветка `main`, последний commit `7706b4d docs: add daily plan for 2026-06-20`, рабочее дерево было чистым до документационных обновлений.
- Подтвержден фокус дня: `DEV-001` Docker Compose для локального backend MVP.
- Рекомендуемая последовательность: `DEV-001`, затем `DEV-002`, затем `OCT-003`, `TEST-002` и `WEB-002`.
- Source-файлы не изменялись; обновление ограничено проектной документацией.

## 2026-06-20

- Project Secretary создал дневной план на 2026-06-20 в 15:09 MSK после проверки `docs/ROADMAP.md`, `docs/BACKLOG.md`, `docs/JOURNAL.md`, последней daily note и `git status --short`.
- Текущее состояние: ветка `main`, последний commit `484eebe docs: record local mvp planning decisions`, рабочее дерево было чистым до документационных обновлений.
- Подтвержден фокус дня: `DEV-001` Docker Compose для локального backend MVP.
- Рекомендуемая последовательность: `DEV-001`, затем `DEV-002`, затем `WEB-002`/`WEB-003`; `OCT-003` и `TEST-002` остаются полезными сопутствующими задачами.
- Source-файлы не изменялись; обновление ограничено проектной документацией.

## 2026-06-19

- Project Secretary обновил дневной план на 2026-06-19 в 11:53 MSK после повторной проверки `docs/ROADMAP.md`, `docs/BACKLOG.md`, `docs/JOURNAL.md`, последней daily note и `git status --short`.
- Подтвержден текущий фокус дня: `DEV-001` Docker Compose для локального backend MVP; `DEV-002` demo seed оставлен следующей задачей после стабилизации запуска.
- Source-файлы не изменялись; обновление ограничено проектной документацией.
- Пользователь разрешил Project Secretary принять открытые решения самостоятельно: `DEV-001` остается первым фокусом, `DEV-002` следует после него, demo seed должен покрывать несколько ручных сценариев, а Docker Compose может использовать стандартные локальные PostgreSQL-значения без production-секретов.
- Создан дневной план на 2026-06-19 после проверки `docs/ROADMAP.md`, `docs/BACKLOG.md`, `docs/JOURNAL.md`, последней daily note и `git status --short`.
- Текущее состояние: ветка `main`, рабочее дерево было чистым до документационных обновлений, Фаза 1 почти закрыта после API-key auth и owner-bound доступа к user-scoped endpoints.
- Первая рекомендуемая задача на день: `API-006`, endpoints ручных предпочтений пользователя.
- Дополнительные рекомендуемые задачи: `OCT-003`, `TEST-002`, `WEB-001`, `EXT-001`.
- Source-файлы не изменялись.
- Принято решение по `WEB-001`: MVP сайта писать на React + Vite.
- Принято решение не расширять service clients в ближайшем MVP: user-scoped endpoints остаются owner-bound, clients без `owner_user_id` получают `403`.
- Project HQ добавил недостающую дорожку к локально проверяемому MVP: `DEV-001` Docker Compose для backend MVP, `DEV-002` demo seed, `WEB-002` frontend shell, `WEB-003` подключение frontend к backend и `API-008` user-based recommendations endpoint.
- Выполнен `API-006`: добавлены `GET /users/{user_id}/preferences` и
  `PUT /users/{user_id}/preferences/{preference_type}/{preference_key}`.
- Preferences endpoints используют существующий API-key auth, новые scopes
  `preferences:read` и `preferences:write`, а также owner-bound правило
  `api_clients.owner_user_id == user_id`.
- `GET` preferences поддерживает фильтры `is_active`, `preference_type` и
  `limit`; `PUT` создает или обновляет запись без дублей и автоматически
  создает активного пользователя по UUID после проверки доступа.
- Рекомендательная модель и frontend не менялись; `user_preferences` пока не
  участвуют в построении рекомендаций.
- Проверка `python -m pytest tests/test_user_history_api.py` прошла: 30 тестов
  успешны.
- Полная проверка `python -m pytest` прошла: 47 тестов успешны.
- Project HQ подтвердил `API-006` после merge в `main`: локальный запуск во
  временном venv `/tmp/project_october_api006_hq_venv` прошел, 47 тестов
  успешны, 1 предупреждение Starlette о deprecated `httpx` import.

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
