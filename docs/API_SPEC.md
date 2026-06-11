# API Specification Draft

## Status

Draft. This document defines the intended public and internal API shape. It should be updated whenever API behavior changes.

## Internal App API

### `GET /health`

Returns backend health.

Status: implemented.

Successful response: `200 OK`.

Response:

```json
{
  "status": "ok"
}
```

### `GET /movies/search`

Searches movies by title.

Query parameters:

- `query`: search text.
- `limit`: max number of results.

Response:

```json
{
  "items": [
    {
      "id": "862",
      "title": "Toy Story",
      "year": 1995,
      "vote_average": 7.7
    }
  ]
}
```

### `POST /recommendations`

Generates recommendations for the current app user or an anonymous input list.

Request:

```json
{
  "liked_movie_ids": ["862", "8844"],
  "disliked_movie_ids": [],
  "filters": {
    "include_adult": false,
    "genres": [],
    "year_from": null,
    "year_to": null
  },
  "limit": 20
}
```

Response:

```json
{
  "items": [
    {
      "id": "863",
      "title": "Toy Story 2",
      "vote_average": 7.3,
      "site_user_rating": "No rating",
      "score": 0.81,
      "reasons": ["same collection", "similar overview", "animation"]
    }
  ]
}
```

## User API

### `POST /auth/register`

Creates a user account.

### `POST /auth/login`

Returns an access token.

### `GET /users/me/history`

Returns current user's movie history.

### `POST /users/me/history`

Adds a movie event to current user's history.

Request:

```json
{
  "movie_id": "862",
  "event_type": "watched",
  "rating": 4.5,
  "watched_at": "2026-06-11T00:00:00Z"
}
```

## Public Developer API

### `POST /v1/recommendations`

Generates recommendations for third-party clients.

Authentication:

```text
Authorization: Bearer <api_key>
```

Request:

```json
{
  "liked_movie_ids": ["862", "8844"],
  "watched_movie_ids": ["862"],
  "filters": {
    "include_adult": false
  },
  "limit": 20
}
```

Response:

```json
{
  "items": [],
  "meta": {
    "model": "content_based_v1",
    "request_id": "req_example"
  }
}
```
