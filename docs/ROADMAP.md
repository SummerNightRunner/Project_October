# Roadmap

## Phase 0 - Clean Baseline

Статус: в работе.

- Зафиксировать структуру данных.
- Восстановить проектную документацию.
- Создать минимальный FastAPI skeleton.
- Добавить dependency file.
- Проверить текущий рекомендательный прототип.

## Phase 1 - Backend API MVP

Статус: запланировано.

- `GET /health`.
- `GET /movies/search`.
- `POST /recommendations`.
- Pydantic-схемы запросов и ответов.
- Документация API.

## Phase 2 - User History MVP

Статус: запланировано.

- PostgreSQL как целевое хранилище.
- Схема пользователей.
- История просмотра.
- Оценки и предпочтения.
- Исключение уже просмотренных фильмов из рекомендаций.

## Phase 3 - Website MVP

Статус: запланировано.

- Поиск фильмов.
- Управление историей просмотра.
- Экран рекомендаций.
- Подключение к backend API.

## Phase 4 - Public Developer API

Статус: позже.

- API keys.
- Версионирование `/v1`.
- Usage logs.
- Rate limits.
- Документация для внешних разработчиков.

## Phase 5 - Imports And Enrichment

Статус: позже.

- CSV import.
- TMDB metadata enrichment.
- Исследование Kinopoisk только после подтверждения стабильного и легального пути.
