# API Spec

## Status

Черновик. Сейчас зафиксирован только минимальный health endpoint.

## `GET /health`

Проверка, что API запускается.

Response:

```json
{
  "status": "ok"
}
```

## Planned Endpoints

### `GET /movies/search`

Поиск фильмов по названию.

Планируемые query parameters:

- `query`: строка поиска;
- `limit`: максимальное число результатов.

### `POST /recommendations`

Рекомендации по списку понравившихся фильмов.

Планируемый request:

```json
{
  "liked_movie_ids": ["862", "8844"],
  "watched_movie_ids": [],
  "include_adult": false,
  "limit": 20
}
```

Планируемый response:

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
