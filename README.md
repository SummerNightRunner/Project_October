# Project October

Project October - веб-приложение и API для рекомендаций фильмов.

Сейчас проект находится в состоянии очищенного Python-прототипа: есть рабочее рекомендательное ядро на CSV-датасетах и минимальный FastAPI entrypoint для дальнейшего API.

## Структура

- `backend/` - Python-код прототипа и будущего API.
- `backend/app/main.py` - FastAPI-приложение.
- `backend/recommendations_func.py` - текущая content-based рекомендационная функция.
- `backend/data_preprocessor.py` - сборка `data/processed/processed_metadata.csv` из raw CSV.
- `data/raw/` - исходные CSV-датасеты, tracked через Git LFS.
- `data/processed/` - локальные генерируемые артефакты, не коммитятся.
- `docs/` - проектная документация, roadmap, backlog, API spec, решения и дневник.

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

## Проверка текущего рекомендательного прототипа

Если `data/processed/processed_metadata.csv` отсутствует, сначала пересоберите его:

```bash
python backend/data_preprocessor.py
```

Затем запустите пример рекомендаций:

```bash
python backend/recommendations_func.py
```

## Правила данных

Исходные CSV лежат в `data/raw/` и отслеживаются через Git LFS. Генерируемые файлы в `data/processed/`, секреты, локальные базы, prompt-файлы Codex и ML-артефакты не коммитятся.
