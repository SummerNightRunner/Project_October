# Journal

## 2026-06-13

- Проведена очистка после неудачного/противоречивого task-результата.
- Восстановлен нормальный проектный контур документации: `PROJECT_BRIEF`, `ROADMAP`, `ARCHITECTURE`, `API_SPEC`, `DECISIONS`, `PROCESS`, backlog и daily.
- Зафиксирована data policy: исходные CSV находятся в `data/raw/` и отслеживаются через Git LFS, `data/processed/` игнорируется как generated output.
- Старый интерактивный CLI в `backend/app.py` заменен на минимальный FastAPI app с `GET /health`.
- Добавлен `requirements.txt` для локального запуска backend.
- Текущий recommendation prototype сохранен и продолжает жить отдельно от API-слоя.
- Добавлены smoke-тесты для `GET /health` и текущего recommendation example.
- README дополнен командой запуска тестов.
- Project HQ подтвердил `TEST-001`: локальный запуск `python -m pytest` прошел, 2 теста успешны.
- Следующий шаг: обернуть recommendation function в `POST /recommendations`.
