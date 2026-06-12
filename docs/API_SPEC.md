# API-спецификация

## Статус

Черновик. Реализованы `GET /health`, `GET /movies/search` и базовый `POST /recommendations`.

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

## Будущие контракты пользовательской истории

Статус: проектный черновик DB-001, endpoints не реализованы.

Будущие endpoints должны использовать `movie_id` как публичное API-имя локального идентификатора фильма. В PostgreSQL этому полю соответствует `catalog_movie_id`, равный строковому `id` из `data/processed/processed_metadata.csv`.

### `GET /users/{user_id}/history`

Возвращает историю пользователя.

Query-параметры:

- `status`: опциональный фильтр `watched`, `planned`, `dropped`;
- `limit`: число элементов от `1` до `100`, по умолчанию `20`.

Черновик ответа `200`:

```json
{
  "items": [
    {
      "movie_id": "862",
      "status": "watched",
      "watched_at": "2026-06-13T12:00:00Z",
      "rating_value": 9.0,
      "source": "manual"
    }
  ]
}
```

### `PUT /users/{user_id}/history/{movie_id}`

Создает или обновляет запись истории пользователя по локальному фильму.

Черновик запроса:

```json
{
  "status": "watched",
  "watched_at": "2026-06-13T12:00:00Z",
  "source": "manual",
  "notes": "Пересмотреть позже"
}
```

### `PUT /users/{user_id}/ratings/{movie_id}`

Создает или обновляет пользовательскую оценку фильма.

Черновик запроса:

```json
{
  "rating_value": 8.5,
  "rated_at": "2026-06-13T12:05:00Z",
  "source": "manual"
}
```

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
