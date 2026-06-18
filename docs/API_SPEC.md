# API-спецификация

## Статус

Черновик. Реализованы `GET /health`, `GET /movies/search`, базовый
`POST /recommendations`, `GET /users/{user_id}/history`,
`PUT /users/{user_id}/history/{movie_id}` и
`PUT /users/{user_id}/ratings/{movie_id}`,
`GET /users/{user_id}/preferences` и
`PUT /users/{user_id}/preferences/{preference_type}/{preference_key}`. Endpoints
пользовательской истории, оценок и ручных предпочтений защищены API-key auth.

## API-key auth

Защищенные публичные endpoints принимают ключ в заголовке:

```http
Authorization: Bearer <api_key>
```

Формат ключа MVP:

```text
oct_<prefix>_<secret>
```

В `api_keys.key_prefix` хранится открытая часть `oct_<prefix>`, по которой ключ
ищется в базе. В `api_keys.key_hash` хранится `sha256`-хеш полного ключа с
префиксом алгоритма, например `sha256:<hex>`. Полный API key и hash не
возвращаются в API responses.

Поддерживаемые scopes:

- `history:read`;
- `history:write`;
- `ratings:write`;
- `preferences:read`;
- `preferences:write`;
- `recommendations:read`.

Ошибки auth:

- `401`: заголовок отсутствует, имеет неверный формат, ключ не найден, hash не
  совпал, ключ отозван, срок действия истек или API client не активен;
- `403`: ключ валиден, но не содержит scope, нужный endpoint, или связанный
  API client не имеет доступа к запрошенному `user_id`.

Для user-scoped endpoints MVP использует `api_clients.owner_user_id` как
границу доступа. Если `owner_user_id` задан, ключи этого API client могут
работать только с тем же `user_id` в path. Если `owner_user_id` равен `NULL`,
доступ к user-scoped endpoints запрещен с `403`; service clients и роли не
реализованы. `api_keys.last_used_at` обновляется после успешной проверки ключа
и требуемого scope, включая случай, когда последующая проверка `owner_user_id`
возвращает `403`.

## `GET /health`

Проверяет, что API запускается.

Ответ:

```json
{
  "status": "ok"
}
```

## `POST /recommendations`

Рекомендации по списку понравившихся фильмов.

Временное решение MVP: endpoint пока остается публичным и не требует
`recommendations:read`, потому что для рекомендаций еще не введены публичные
API-тарифы, лимиты и договоренность о модели доступа.

Запрос:

```json
{
  "liked_movie_ids": ["862", "8844"],
  "include_adult": false,
  "limit": 20
}
```

Поля запроса:

- `liked_movie_ids`: непустой список локальных ID фильмов из обработанного каталога;
- `include_adult`: включать фильмы с adult-флагом, по умолчанию `false`;
- `limit`: число рекомендаций, от `1` до `100`, по умолчанию `20`.

Ответ `200`:

```json
{
  "items": [
    {
      "id": "863",
      "title": "Toy Story 2",
      "vote_average": 7.3,
      "site_user_rating": "No rating"
    }
  ]
}
```

Ошибки:

- `400`: список передан, но в нем нет ID, известных локальному каталогу;
- `422`: тело запроса не соответствует схеме, например `liked_movie_ids` пустой или `limit` вне допустимого диапазона;
- `503`: локальный файл `data/processed/processed_metadata.csv` недоступен и рекомендации нельзя построить.

## `GET /movies/search`

Поиск фильмов по названию в локальном обработанном каталоге.

Query-параметры:

- `query`: непустая строка поиска;
- `limit`: максимальное число результатов, от `1` до `100`, по умолчанию `20`.

По умолчанию endpoint читает `data/processed/processed_metadata.csv`. Если задана переменная окружения `PROJECT_OCTOBER_PROCESSED_METADATA`, используется путь из нее.

Пример:

```bash
curl "http://127.0.0.1:8000/movies/search?query=toy&limit=5"
```

Ответ `200`:

```json
{
  "items": [
    {
      "id": "862",
      "title": "Toy Story",
      "vote_average": 7.7
    }
  ]
}
```

Ошибки:

- `422`: `query` отсутствует или пустой после удаления пробелов, либо `limit` вне диапазона `1..100`;
- `503`: локальный файл `data/processed/processed_metadata.csv` недоступен или имеет неподдерживаемую схему.

## Пользовательская история, оценки и предпочтения

Статус: минимальная реализация API-004 с API-key auth из API-005 и
ограничением доступа по `api_clients.owner_user_id` из API-007. API-006
добавляет чтение и запись ручных предпочтений пользователя.

Endpoints используют `movie_id` как публичное API-имя локального идентификатора
фильма. В PostgreSQL этому полю соответствует `catalog_movie_id`, равный
строковому `id` из `data/processed/processed_metadata.csv`.

Перед записью истории или оценки backend проверяет, что `movie_id` существует в
`movie_catalog_entries`. Если фильма нет, endpoint возвращает `404`.

Для MVP write-endpoints истории, оценок и предпочтений автоматически создают
активного пользователя по переданному `user_id`, если такого пользователя еще
нет. `GET` для пользователя без записей возвращает пустой список и не создает
пользователя.

Все endpoints в этом разделе требуют `Authorization: Bearer <api_key>`.
API client должен иметь `owner_user_id`, равный path-параметру `user_id`;
client без `owner_user_id` получает `403` на этих endpoints.

### `GET /users/{user_id}/history`

Возвращает историю пользователя.

Требуемый scope: `history:read`.

Query-параметры:

- `status`: опциональный фильтр `watched`, `planned`, `dropped`;
- `limit`: число элементов от `1` до `100`, по умолчанию `20`.

Ответ `200`:

```json
{
  "items": [
    {
      "movie_id": "862",
      "status": "watched",
      "watched_at": "2026-06-13T12:00:00Z",
      "rating_value": 9.0,
      "source": "manual",
      "notes": "Пересмотреть позже"
    }
  ]
}
```

Ошибки:

- `401`: API key отсутствует, неверен, отозван, истек или связан с неактивным
  API client;
- `403`: API key не содержит `history:read` или API client не имеет доступа к
  этому `user_id`;
- `422`: `user_id` не является UUID, `status` не входит в допустимый набор или
  `limit` вне диапазона `1..100`.

### `PUT /users/{user_id}/history/{movie_id}`

Создает или обновляет запись истории пользователя по локальному фильму.

Требуемый scope: `history:write`.

Запрос:

```json
{
  "status": "watched",
  "watched_at": "2026-06-13T12:00:00Z",
  "source": "manual",
  "notes": "Пересмотреть позже"
}
```

Поля запроса:

- `status`: `watched`, `planned` или `dropped`;
- `watched_at`: опциональное время просмотра в ISO 8601;
- `source`: `manual`, `csv_import`, `api` или `system`, по умолчанию `manual`;
- `notes`: опциональная пользовательская заметка до 2000 символов.

Ответ `200`:

```json
{
  "movie_id": "862",
  "status": "watched",
  "watched_at": "2026-06-13T12:00:00Z",
  "rating_value": null,
  "source": "manual",
  "notes": "Пересмотреть позже"
}
```

Ошибки:

- `401`: API key отсутствует, неверен, отозван, истек или связан с неактивным
  API client;
- `403`: API key не содержит `history:write` или API client не имеет доступа к
  этому `user_id`;
- `404`: `movie_id` отсутствует в `movie_catalog_entries`;
- `422`: `user_id` не является UUID или тело запроса не соответствует схеме.

### `PUT /users/{user_id}/ratings/{movie_id}`

Создает или обновляет пользовательскую оценку фильма.

Требуемый scope: `ratings:write`.

Запрос:

```json
{
  "rating_value": 8.5,
  "rated_at": "2026-06-13T12:05:00Z",
  "source": "manual"
}
```

Поля запроса:

- `rating_value`: оценка от `0.0` до `10.0` с точностью до одного знака после
  запятой;
- `rated_at`: опциональное время оценки в ISO 8601;
- `source`: `manual`, `csv_import`, `api` или `system`, по умолчанию `manual`.

Ответ `200`:

```json
{
  "movie_id": "862",
  "rating_value": 8.5,
  "rated_at": "2026-06-13T12:05:00Z",
  "source": "manual"
}
```

Ошибки:

- `401`: API key отсутствует, неверен, отозван, истек или связан с неактивным
  API client;
- `403`: API key не содержит `ratings:write` или API client не имеет доступа к
  этому `user_id`;
- `404`: `movie_id` отсутствует в `movie_catalog_entries`;
- `422`: `rating_value` вне диапазона `0..10`, имеет больше одного знака после
  запятой, `user_id` не является UUID или тело запроса не соответствует схеме.

### `GET /users/{user_id}/preferences`

Возвращает ручные предпочтения пользователя.

Требуемый scope: `preferences:read`.

Query-параметры:

- `is_active`: опциональный boolean-фильтр;
- `preference_type`: опциональный фильтр `genre`, `keyword`, `person`,
  `language`, `adult_content`, `animation` или `free_text`;
- `limit`: число элементов от `1` до `100`, по умолчанию `20`.

Ответ `200`:

```json
{
  "items": [
    {
      "preference_type": "genre",
      "preference_key": "comedy",
      "weight": 2.25,
      "source": "manual",
      "is_active": true,
      "created_at": "2026-06-19T12:00:00Z",
      "updated_at": "2026-06-19T12:00:00Z"
    }
  ]
}
```

Ошибки:

- `401`: API key отсутствует, неверен, отозван, истек или связан с неактивным
  API client;
- `403`: API key не содержит `preferences:read` или API client не имеет доступа
  к этому `user_id`;
- `422`: `user_id` не является UUID, `preference_type` не входит в допустимый
  набор или `limit` вне диапазона `1..100`.

### `PUT /users/{user_id}/preferences/{preference_type}/{preference_key}`

Создает или обновляет ручное предпочтение пользователя.

Требуемый scope: `preferences:write`.

`preference_key` нормализуется через trim и не может быть пустым. Повторный
запрос с тем же `(user_id, preference_type, preference_key)` обновляет
существующую запись без создания дубля.

Запрос:

```json
{
  "weight": 2.25,
  "source": "manual",
  "is_active": true
}
```

Поля запроса:

- `preference_type`: path-параметр `genre`, `keyword`, `person`, `language`,
  `adult_content`, `animation` или `free_text`;
- `preference_key`: path-параметр, непустой после удаления пробелов;
- `weight`: вес от `-10.00` до `10.00`;
- `source`: `manual`, `csv_import`, `api` или `system`, по умолчанию `manual`;
- `is_active`: включено ли предпочтение, по умолчанию `true`.

Ответ `200`:

```json
{
  "preference_type": "genre",
  "preference_key": "comedy",
  "weight": 2.25,
  "source": "manual",
  "is_active": true,
  "created_at": "2026-06-19T12:00:00Z",
  "updated_at": "2026-06-19T12:00:00Z"
}
```

Ошибки:

- `401`: API key отсутствует, неверен, отозван, истек или связан с неактивным
  API client;
- `403`: API key не содержит `preferences:write` или API client не имеет
  доступа к этому `user_id`;
- `422`: `user_id` не является UUID, `preference_type` не входит в допустимый
  набор, `preference_key` пустой после trim, `weight` вне диапазона `-10..10`
  или тело запроса не соответствует схеме.

### API-доступ сторонних проектов

Публичные endpoints для сторонних проектов принимают API-ключ через
`Authorization: Bearer <api_key>`. Реализация проверяет `key_prefix`,
`key_hash`, `status`, `expires_at`, статус API client и `scopes`, а при успешной
проверке ключа и требуемого scope обновляет `last_used_at`.

Для endpoints с `user_id` в path дополнительно проверяется
`api_clients.owner_user_id`. Несовпадение владельца и `user_id`, а также
`owner_user_id = NULL`, возвращают `403` без раскрытия информации о
существовании пользователя.
