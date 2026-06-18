# Архитектура

## Текущее состояние

Проект очищен до базового состояния:

- точка входа FastAPI: `backend/app/main.py`.
- Рекомендательное ядро: `backend/recommendations_func.py`.
- Препроцессинг данных: `backend/data_preprocessor.py`.
- Исходные CSV: `data/raw/`.
- Генерируемые метаданные: `data/processed/`.

## Целевая архитектура

Для MVP сайта выбран React + Vite. Frontend должен работать как SPA поверх
FastAPI backend: поиск фильмов, история, оценки, ручные предпочтения и экран
рекомендаций. Next.js не используется в ближайшем MVP и остается вариантом на
будущее, если появятся SEO или server-side rendering требования.

```text
React/Vite frontend-приложение
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

## PostgreSQL-схема пользовательской истории

DB-001 фиксирует целевую схему хранения пользовательского состояния. DB-003 реализует этот контракт в SQLAlchemy-моделях и Alembic-миграции для PostgreSQL.

Основные правила:

- PostgreSQL хранит только состояние приложения: пользователей, историю, оценки, ручные предпочтения, API-доступ и события.
- Raw CSV и `data/processed/processed_metadata.csv` не становятся пользовательским состоянием.
- Пользовательские таблицы ссылаются на локальный каталог через `catalog_movie_id`.
- `catalog_movie_id` - строковое значение из колонки `id` в `data/processed/processed_metadata.csv`; оно соответствует текущему `movie_id` в API.
- Перед записью истории backend должен проверять, что `catalog_movie_id` существует в локальном обработанном каталоге или в синхронизированной таблице `movie_catalog_entries`.

### Таблицы

#### `users`

Аккаунт пользователя приложения. Пароли и внешние identity-провайдеры не проектируются в DB-001.

| Поле | Тип | Обязательное | Описание |
| --- | --- | --- | --- |
| `id` | `uuid` | да | Primary key, генерируется приложением или `gen_random_uuid()` |
| `email` | `citext` | нет | Уникальный email, появится при полноценной авторизации |
| `display_name` | `text` | нет | Имя для интерфейса |
| `status` | `text` | да | `active`, `disabled`, `deleted` |
| `created_at` | `timestamptz` | да | Время создания |
| `updated_at` | `timestamptz` | да | Время последнего изменения |

Ограничения и индексы:

- `primary key (id)`;
- `unique (email)` для непустых email;
- индекс `users_status_idx` по `status`.

#### `movie_catalog_entries`

Минимальная проекция локального каталога фильмов для ссылочной целостности пользовательских данных. Таблица не заменяет `data/processed/processed_metadata.csv` и не хранит признаки рекомендательной модели.

| Поле | Тип | Обязательное | Описание |
| --- | --- | --- | --- |
| `catalog_movie_id` | `text` | да | Primary key; значение `id` из `processed_metadata.csv` |
| `title_snapshot` | `text` | нет | Название на момент синхронизации, только для диагностики и админских экранов |
| `release_date` | `date` | нет | Дата релиза, если есть в обработанном каталоге |
| `source_catalog_version` | `text` | нет | Версия или идентификатор сборки обработанного каталога |
| `created_at` | `timestamptz` | да | Время первой синхронизации |
| `updated_at` | `timestamptz` | да | Время последней синхронизации |

Ограничения и индексы:

- `primary key (catalog_movie_id)`;
- индекс `movie_catalog_entries_title_idx` по `title_snapshot` опционален и нужен только для админских задач.

#### `user_movie_history`

Факты пользовательской истории по фильмам: просмотрено, запланировано, брошено. Для MVP основной статус - `watched`.

| Поле | Тип | Обязательное | Описание |
| --- | --- | --- | --- |
| `id` | `uuid` | да | Primary key |
| `user_id` | `uuid` | да | FK на `users.id` |
| `catalog_movie_id` | `text` | да | FK на `movie_catalog_entries.catalog_movie_id` |
| `status` | `text` | да | `watched`, `planned`, `dropped` |
| `watched_at` | `timestamptz` | нет | Когда пользователь посмотрел фильм; для импорта может быть неизвестно |
| `source` | `text` | да | `manual`, `csv_import`, `api`, `system` |
| `notes` | `text` | нет | Пользовательская заметка |
| `created_at` | `timestamptz` | да | Время создания записи |
| `updated_at` | `timestamptz` | да | Время последнего изменения |

Ограничения и индексы:

- `foreign key (user_id) references users(id) on delete cascade`;
- `foreign key (catalog_movie_id) references movie_catalog_entries(catalog_movie_id)`;
- `unique (user_id, catalog_movie_id)`, чтобы один фильм имел одну актуальную запись истории пользователя;
- индекс `user_movie_history_user_status_idx` по `(user_id, status)`;
- индекс `user_movie_history_user_watched_at_idx` по `(user_id, watched_at desc)`.

#### `user_movie_ratings`

Пользовательские оценки фильмов. Оценка отделена от истории, потому что фильм может быть оценен без точной даты просмотра, а история может быть записана без оценки.

| Поле | Тип | Обязательное | Описание |
| --- | --- | --- | --- |
| `id` | `uuid` | да | Primary key |
| `user_id` | `uuid` | да | FK на `users.id` |
| `catalog_movie_id` | `text` | да | FK на `movie_catalog_entries.catalog_movie_id` |
| `rating_value` | `numeric(3,1)` | да | Нормализованная оценка в шкале `0..10` |
| `rated_at` | `timestamptz` | нет | Когда пользователь поставил оценку |
| `source` | `text` | да | `manual`, `csv_import`, `api`, `system` |
| `created_at` | `timestamptz` | да | Время создания |
| `updated_at` | `timestamptz` | да | Время последнего изменения |

Ограничения и индексы:

- `check (rating_value >= 0 and rating_value <= 10)`;
- `foreign key (user_id) references users(id) on delete cascade`;
- `foreign key (catalog_movie_id) references movie_catalog_entries(catalog_movie_id)`;
- `unique (user_id, catalog_movie_id)`, чтобы хранить одну актуальную оценку пользователя на фильм;
- индекс `user_movie_ratings_user_rating_idx` по `(user_id, rating_value desc)`;
- индекс `user_movie_ratings_movie_idx` по `catalog_movie_id`.

#### `user_preferences`

Ручные предпочтения пользователя, которые дополняют историю и оценки.

| Поле | Тип | Обязательное | Описание |
| --- | --- | --- | --- |
| `id` | `uuid` | да | Primary key |
| `user_id` | `uuid` | да | FK на `users.id` |
| `preference_type` | `text` | да | `genre`, `keyword`, `person`, `language`, `adult_content`, `animation`, `free_text` |
| `preference_key` | `text` | да | Машиночитаемый ключ, например жанр или язык |
| `weight` | `numeric(4,2)` | да | Вес от `-10.00` до `10.00`; отрицательный вес означает нежелательное предпочтение |
| `source` | `text` | да | `manual`, `csv_import`, `api`, `system` |
| `is_active` | `boolean` | да | Включено ли предпочтение |
| `created_at` | `timestamptz` | да | Время создания |
| `updated_at` | `timestamptz` | да | Время последнего изменения |

Ограничения и индексы:

- `check (weight >= -10 and weight <= 10)`;
- `foreign key (user_id) references users(id) on delete cascade`;
- `unique (user_id, preference_type, preference_key)`;
- индекс `user_preferences_user_active_idx` по `(user_id, is_active)`;
- индекс `user_preferences_type_key_idx` по `(preference_type, preference_key)`.

#### `api_clients`

Сторонний проект или внутренний сервис, которому разрешен API-доступ. В MVP
user-scoped доступ используется только в owner-bound режиме: client без
`owner_user_id` не получает доступ к истории, оценкам и будущим ручным
предпочтениям пользователя.

| Поле | Тип | Обязательное | Описание |
| --- | --- | --- | --- |
| `id` | `uuid` | да | Primary key |
| `owner_user_id` | `uuid` | нет | FK на `users.id`, если ключ принадлежит конкретному пользователю или разработчику |
| `name` | `text` | да | Название интеграции |
| `contact_email` | `citext` | нет | Контакт владельца интеграции |
| `status` | `text` | да | `active`, `disabled`, `deleted` |
| `created_at` | `timestamptz` | да | Время создания |
| `updated_at` | `timestamptz` | да | Время последнего изменения |

Ограничения и индексы:

- `foreign key (owner_user_id) references users(id) on delete set null`;
- индекс `api_clients_owner_idx` по `owner_user_id`;
- индекс `api_clients_status_idx` по `status`.

#### `api_keys`

API-ключи для сторонних проектов. Секрет целиком не хранится в базе: сохраняются только `key_prefix` для поиска и `key_hash` для проверки.

| Поле | Тип | Обязательное | Описание |
| --- | --- | --- | --- |
| `id` | `uuid` | да | Primary key |
| `api_client_id` | `uuid` | да | FK на `api_clients.id` |
| `key_prefix` | `text` | да | Небольшой открытый префикс ключа для поиска и поддержки |
| `key_hash` | `text` | да | Хеш полного ключа |
| `scopes` | `text[]` | да | Разрешения, например `recommendations:read`, `history:write` |
| `status` | `text` | да | `active`, `revoked`, `expired` |
| `expires_at` | `timestamptz` | нет | Срок действия |
| `last_used_at` | `timestamptz` | нет | Последнее успешное использование |
| `created_at` | `timestamptz` | да | Время создания |
| `revoked_at` | `timestamptz` | нет | Время отзыва |

Ограничения и индексы:

- `foreign key (api_client_id) references api_clients(id) on delete cascade`;
- `unique (key_prefix)`;
- индекс `api_keys_client_status_idx` по `(api_client_id, status)`;
- индекс `api_keys_expires_at_idx` по `expires_at`.

#### `user_events`

Журнал действий для аудита, отладки импортов и будущей аналитики. Это не источник истины для текущей истории или оценок.

| Поле | Тип | Обязательное | Описание |
| --- | --- | --- | --- |
| `id` | `uuid` | да | Primary key |
| `user_id` | `uuid` | нет | FK на `users.id`, если событие связано с пользователем |
| `api_client_id` | `uuid` | нет | FK на `api_clients.id`, если событие пришло через API-клиента |
| `event_type` | `text` | да | Например `history.added`, `rating.updated`, `preferences.changed`, `api_key.used` |
| `catalog_movie_id` | `text` | нет | FK на `movie_catalog_entries.catalog_movie_id`, если событие связано с фильмом |
| `payload` | `jsonb` | да | Небольшой контекст события без секретов и приватных сырых импортов |
| `request_id` | `text` | нет | Идентификатор запроса для трассировки |
| `created_at` | `timestamptz` | да | Время события |

Ограничения и индексы:

- `foreign key (user_id) references users(id) on delete set null`;
- `foreign key (api_client_id) references api_clients(id) on delete set null`;
- `foreign key (catalog_movie_id) references movie_catalog_entries(catalog_movie_id)`;
- индекс `user_events_user_created_idx` по `(user_id, created_at desc)`;
- индекс `user_events_type_created_idx` по `(event_type, created_at desc)`;
- индекс `user_events_request_id_idx` по `request_id`.

### Связи

```text
users
  1 ├── n user_movie_history ── n:1 movie_catalog_entries
  1 ├── n user_movie_ratings ── n:1 movie_catalog_entries
  1 ├── n user_preferences
  1 ├── n api_clients
  1 └── n user_events

api_clients
  1 ├── n api_keys
  1 └── n user_events
```

### Связь с локальным каталогом фильмов

Текущий API принимает и возвращает `id` как строку из `processed_metadata.csv`. В базе этот идентификатор называется `catalog_movie_id`, чтобы не смешивать его с surrogate key таблиц.

Текущий план интеграции:

1. `GET /movies/search` продолжает искать по `data/processed/processed_metadata.csv`.
2. Перед записью пользовательской истории backend проверяет `catalog_movie_id` по локальному каталогу.
3. DB-004 синхронизирует минимальный список фильмов в `movie_catalog_entries`.
4. Рекомендательный сервис использует пользовательские `catalog_movie_id` как вход в текущую content-based модель.
5. Если позже появятся TMDB или другие внешние ID, они добавляются отдельной таблицей сопоставления или расширением `movie_catalog_entries`, без изменения пользовательских таблиц.

Синхронизация каталога запускается локально командой:

```bash
python -m backend.app.db.sync_movie_catalog
```

Команда читает `data/processed/processed_metadata.csv` или путь из
`PROJECT_OCTOBER_PROCESSED_METADATA`, затем делает upsert по `catalog_movie_id`.
Синхронизируются только минимальные поля FK-проекции: `catalog_movie_id` из CSV
`id`, `title_snapshot` из `original_title`/`title`/`name`, `release_date` при
наличии безопасно распарсенной даты, опциональный `source_catalog_version`,
`created_at` и `updated_at`. Строки, отсутствующие в новом CSV, не удаляются,
потому что они могут уже использоваться пользовательской историей.

## Backend-структура

Текущая ближайшая структура:

```text
backend/
  app/
    __init__.py
    catalog_paths.py
    db/
      base.py
      config.py
      movie_catalog_sync.py
      sync_movie_catalog.py
      session.py
    main.py
  recommendations_func.py
  data_preprocessor.py
  ratings_updater.py
  user_registration.py
```

DB-002 добавляет базовую инфраструктуру PostgreSQL:

- `backend/app/db/config.py` читает `PROJECT_OCTOBER_DATABASE_URL` и не содержит секретов или локальных DSN;
- `backend/app/db/base.py` содержит общий `Base` для будущих SQLAlchemy ORM-моделей;
- `backend/app/db/session.py` создает SQLAlchemy engine/session лениво, чтобы импорт FastAPI app не требовал доступной базы;
- `alembic/` содержит окружение миграций, которое использует metadata из `Base` и URL из переменной окружения.

DB-003 добавляет:

- `backend/app/db/models.py` с ORM-моделями пользовательской истории, оценок, предпочтений, API-доступа и событий;
- Alembic revision `db003_user_history_schema`, создающую таблицы DB-001;
- PostgreSQL extensions `pgcrypto` для `gen_random_uuid()` и `citext` для case-insensitive email-полей.

DB-004 добавляет:

- `backend/app/catalog_paths.py` с общим resolver пути к обработанному каталогу;
- `backend/app/db/movie_catalog_sync.py` с чтением CSV и upsert-синхронизацией `movie_catalog_entries`;
- `backend/app/db/sync_movie_catalog.py` как CLI/module entrypoint для локального запуска.

API-004 добавляет:

- `backend/app/user_history.py` с FastAPI router, Pydantic-схемами и минимальной
  сервисной логикой чтения/обновления `user_movie_history` и
  `user_movie_ratings`;
- проверку `movie_id` через `movie_catalog_entries` перед записью истории или
  оценки;
- автосоздание активного пользователя по переданному UUID на write-запросах до
  появления полноценной авторизации;
- SQLite-backed endpoint-тесты через FastAPI dependency override без обращения к
  production DB.

Существующие endpoints рекомендаций и поиска продолжают работать поверх
локального обработанного каталога. Пользовательская история и оценки уже пишутся
в DB-слой, но пока не используются текущей content-based моделью рекомендаций.

API-005 добавляет минимальный API-key auth для сторонних проектов:

- `backend/app/api_key_auth.py` содержит helper создания ключа
  `oct_<prefix>_<secret>`, вычисление `sha256:<hex>` hash полного ключа,
  проверку hash через `hmac.compare_digest` и FastAPI dependency для scopes;
- поиск ключа идет по `api_keys.key_prefix` в формате `oct_<prefix>`, полный
  ключ не хранится и не возвращается наружу;
- проверяются `api_keys.status`, `api_keys.expires_at`, `api_keys.scopes` и
  активный статус связанного `api_clients`;
- при успешной проверке обновляется `api_keys.last_used_at`;
- `GET /users/{user_id}/history` требует `history:read`;
- `PUT /users/{user_id}/history/{movie_id}` требует `history:write`;
- `PUT /users/{user_id}/ratings/{movie_id}` требует `ratings:write`;
- `POST /recommendations` временно остается публичным до появления тарифов,
  лимитов и полноценной модели публичного API-доступа.

API-007 связывает API key/client с допустимым пользователем:

- `ApiKeyPrincipal` включает `api_clients.owner_user_id`;
- user-scoped endpoints с `user_id` в path дополнительно проверяют, что
  `owner_user_id` равен запрошенному `user_id`;
- если `owner_user_id` не задан или не совпадает, endpoint возвращает `403` без
  раскрытия существования пользователя;
- `api_keys.last_used_at` обновляется после успешной проверки ключа и scope,
  даже если последующая owner-проверка возвращает `403`;
- service clients, роли, модераторы и доступ к нескольким пользователям не
  реализуются в MVP и требуют отдельного решения.

API-006 добавляет endpoints ручных предпочтений пользователя в тот же
owner-bound контур:

- `GET /users/{user_id}/preferences` читает `user_preferences`, поддерживает
  фильтры `is_active`, `preference_type` и `limit`, требует
  `preferences:read`;
- `PUT /users/{user_id}/preferences/{preference_type}/{preference_key}` создает
  или обновляет запись по уникальной паре пользователя, типа и ключа
  предпочтения, требует `preferences:write`;
- write-endpoint автоматически создает активного пользователя по UUID, как
  history/rating endpoints, а read-endpoint не создает пользователя;
- публичный API возвращает только `preference_type`, `preference_key`,
  `weight`, `source`, `is_active`, `created_at`, `updated_at`;
- предпочтения пока не подключены к content-based рекомендациям и не меняют
  рекомендательную модель.

Когда API начнет расти дальше, можно перейти к более явной пакетной структуре:

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
