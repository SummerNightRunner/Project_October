# Дневник проекта

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
