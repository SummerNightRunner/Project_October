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
