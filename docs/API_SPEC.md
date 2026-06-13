# API-спецификация

## Статус

Черновик. Реализованы `GET /health`, `GET /movies/search`, базовый
`POST /recommendations`, `GET /users/{user_id}/history`,
`PUT /users/{user_id}/history/{movie_id}` и
`PUT /users/{user_id}/ratings/{movie_id}`.

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

## Пользовательская история и оценки

Статус: минимальная реализация API-004.

Endpoints используют `movie_id` как публичное API-имя локального идентификатора
фильма. В PostgreSQL этому полю соответствует `catalog_movie_id`, равный
строковому `id` из `data/processed/processed_metadata.csv`.

Перед записью истории или оценки backend проверяет, что `movie_id` существует в
`movie_catalog_entries`. Если фильма нет, endpoint возвращает `404`.

Для MVP write-endpoints автоматически создают активного пользователя по
переданному `user_id`, если такого пользователя еще нет. `GET` для пользователя
без записей возвращает пустой список.

### `GET /users/{user_id}/history`

Возвращает историю пользователя.

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

- `422`: `user_id` не является UUID, `status` не входит в допустимый набор или
  `limit` вне диапазона `1..100`.

### `PUT /users/{user_id}/history/{movie_id}`

Создает или обновляет запись истории пользователя по локальному фильму.

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

- `404`: `movie_id` отсутствует в `movie_catalog_entries`;
- `422`: `user_id` не является UUID или тело запроса не соответствует схеме.

### `PUT /users/{user_id}/ratings/{movie_id}`

Создает или обновляет пользовательскую оценку фильма.

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

- `404`: `movie_id` отсутствует в `movie_catalog_entries`;
- `422`: `rating_value` вне диапазона `0..10`, имеет больше одного знака после
  запятой, `user_id` не является UUID или тело запроса не соответствует схеме.

## Будущие контракты пользовательских предпочтений

### `PUT /users/{user_id}/preferences/{preference_type}/{preference_key}`

Создает или обновляет ручное предпочтение пользователя.

Черновик запроса:

```json
{
  "weight": 2.0,
  "source": "manual",
  "is_active": true
}
```

### API-доступ сторонних проектов

Будущие публичные endpoints для сторонних проектов должны принимать API-ключ через `Authorization: Bearer <api_key>`. Реализация должна проверять `key_prefix`, `key_hash`, `status`, `expires_at` и `scopes`.
