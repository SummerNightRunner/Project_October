# API-спецификация

## Статус

Черновик. Сейчас реализован только минимальный endpoint проверки состояния.

## `GET /health`

Проверяет, что API запускается.

Ответ:

```json
{
  "status": "ok"
}
```

## Запланированные endpoint-ы

### `GET /movies/search`

Поиск фильмов по названию.

Планируемые query-параметры:

- `query`: строка поиска;
- `limit`: максимальное число результатов.

### `POST /recommendations`

Рекомендации по списку понравившихся фильмов.

Планируемый запрос:

```json
{
  "liked_movie_ids": ["862", "8844"],
  "watched_movie_ids": [],
  "include_adult": false,
  "limit": 20
}
```

Планируемый ответ:

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
