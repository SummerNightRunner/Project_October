# Project October

Project October - веб-приложение и API для рекомендаций фильмов.

Сейчас проект находится в состоянии очищенного Python-прототипа: есть рабочее рекомендательное ядро на CSV-датасетах, минимальная точка входа FastAPI и дымовые тесты.

## Структура

- `backend/` - Python-код прототипа и будущего API.
- `backend/app/main.py` - FastAPI-приложение.
- `backend/recommendations_func.py` - текущая рекомендательная функция на признаках фильма.
- `backend/data_preprocessor.py` - сборка `data/processed/processed_metadata.csv` из исходных CSV.
- `data/raw/` - исходные CSV-датасеты, отслеживаются через Git LFS.
- `data/processed/` - локальные генерируемые артефакты, не коммитятся.
- `docs/` - проектная документация, дорожная карта, backlog, API-спецификация, решения и дневник.

## Локальный запуск API

```bash
python -m pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

Пример поиска фильмов:

```bash
curl "http://127.0.0.1:8000/movies/search?query=toy&limit=5"
```

Для `GET /movies/search` нужен локальный файл `data/processed/processed_metadata.csv`.

Пример запроса рекомендаций:

```bash
curl -X POST http://127.0.0.1:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"liked_movie_ids":["862","8844"],"include_adult":false,"limit":5}'
```

Для `POST /recommendations` нужен локальный файл `data/processed/processed_metadata.csv`.

## Проверка текущего рекомендательного прототипа

Если `data/processed/processed_metadata.csv` отсутствует, сначала пересоберите его:

```bash
python backend/data_preprocessor.py
```

Затем запустите пример рекомендаций:

```bash
python backend/recommendations_func.py
```

## Тесты

```bash
python -m pytest
```

## Правила хранения данных

Исходные CSV лежат в `data/raw/` и отслеживаются через Git LFS. Генерируемые файлы в `data/processed/`, секреты, локальные базы, файлы промптов Codex и ML-артефакты не коммитятся.
