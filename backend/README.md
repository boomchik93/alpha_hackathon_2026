# backend

Серверная часть для фронтенда: публичный API, хранение пользовательских данных, запуск ML-слоя. Ветка разработки: `backend/dev`.

## Что нужно сделать

- API, которое отдаёт фронтенду компании, профиль, ленту, карточки влияния, ответы на уточняющие вопросы.
- Хранение того, что не живёт в файлах ML: пользователи, обратная связь, история просмотров.
- Прокси к ML-слою: вызывать его по HTTP или импортировать функции из `ml/api.py`.
- Обработка ошибок ML (мок, исчерпан дневной бюджет, превышено время ожидания) так, чтобы фронтенд получал понятный ответ.

## Что даёт ML-слой

ML-слой уже работает как сервис (`make serve` в корне, порт 8001, документация на `/docs`). Контракт:

| HTTP | Ответ |
|---|---|
| `GET /companies` | список компаний и групп |
| `GET /companies/{id}` | `CompanyProfile` |
| `GET /companies/{id}/feed` | `Feed`: материалы с оценкой и причинами |
| `GET /companies/{id}/insights/{item_id}` | `InsightCard` (на настоящей модели около 15 секунд при первом запросе) |
| `POST /cards/{card_id}/followup` | `FollowupAnswer` |
| `POST /cards/{card_id}/feedback` | сохранение оценки карточки |
| `GET /traces/{trace_id}` | трасса запуска агента |
| `GET /health` | режим мока, расход за день |

Модели данных: [`ml/schemas.py`](../ml/schemas.py). Описание и примеры вызовов: [`ml/README.md`](../ml/README.md), сценарий проверки в [`ml/OVERVIEW.md`](../ml/OVERVIEW.md).

Без ключа к моделям ML работает на заглушке (`LLM_MOCK=1`): структура ответов настоящая, тексты помечены `[MOCK]`. Этого достаточно, чтобы строить бэкенд.

## API для фронта (`backend/app`)

Стек: FastAPI + httpx, Python 3.13 (как у ML). ML-сервис вызывается по HTTP, пакет `ml` бэкенд не импортирует.
Фронт ходит только сюда; ML на :8001 наружу не выставляется.

```bash
make serve           # ML на :8001 (без ключа: LLM_MOCK=1 в .env)
make backend-serve   # бэкенд на :8000, документация на http://localhost:8000/docs
make backend-test    # тесты: настоящий ml.server в том же процессе, ml.api подменён, без сети и модели
```

На Windows без make: `uvicorn backend.app.main:app --port 8000` и `python -m pytest backend/tests` из корня репозитория.
Настройки (`BACKEND_*`) в [`.env.example`](../.env.example), значения по умолчанию в [`app/config.py`](app/config.py).

| HTTP | Ответ |
|---|---|
| `GET /api/health` | `{ok, ml_ok, llm_mock, spent_today_rub, budget_rub}`; при недоступном ML `ml_ok=false` и `ml_error` |
| `GET /api/companies` | список компаний и групп |
| `GET /api/companies/{id}` | `CompanyProfile` |
| `GET /api/companies/{id}/feed?tier=all\|main&limit=1..50&maybe_limit=0..50` | `Feed`; блок «может быть интересно» в `entries[].tier == "maybe"` |
| `GET /api/companies/{id}/insights/{item_id}?role=` | `200` + `InsightCard`, если готова; иначе `202` + задача (ниже) |
| `GET /api/jobs/{job_id}` | состояние генерации: `queued/running/done/failed`, `card` или `error` |
| `POST /api/cards/{card_id}/followup?role=` `{question}` | `FollowupAnswer`, вопрос до 500 символов |
| `POST /api/cards/{card_id}/feedback` `{rating, reason?, comment?, role?}` | запись оценки в ML |

`role`: `owner/ceo/cfo/treasurer/chief_accountant/other`. Пока нет авторизации, роль приходит параметром; потом её возьмём из сессии.
Трассы (`/traces`) наружу пока не выставлены: ML отдаёт их целиком, урезанный вид для пользователя будет на этапе 2.

### Карточка без зависания фронта

1. `GET /api/companies/{id}/insights/{item_id}`. Бэкенд ждёт до `BACKEND_INSIGHT_SYNC_WAIT_S` (2 с): карточка из кэша ML приходит сразу с `200`.
2. Если не успела, ответ `202` с заголовком `Retry-After: 1`:
   `{"job_id", "status": "queued|running", "poll_url": "/api/jobs/job_…", "created_at", "updated_at", "card": null, "error": null}`.
3. Фронт опрашивает `poll_url` раз в 1-2 с до `done` (карточка в `card`) или `failed` (ошибка в `error`, формат ниже).

Повторный запрос той же карточки (компания, материал, роль), пока она генерируется, возвращает ту же задачу и не тратит бюджет.
Одновременно идёт не больше `BACKEND_MAX_PARALLEL_GENERATIONS` генераций, очередь ограничена `BACKEND_MAX_QUEUED_JOBS`.
Задачи живут в памяти процесса: бэкенд запускается одним процессом, а после перезапуска карточка всё равно лежит в кэше ML.

### Ошибки

Любая ошибка: `{"code": "...", "message": "...", "retry_after": 123}` (`retry_after` только у 429, дублируется заголовком `Retry-After`).
`message` можно показывать пользователю.

| Статус | `code` | Когда |
|---|---|---|
| 404 | `not_found` | нет компании, материала, карточки, задачи или ручки |
| 422 | `validation_error` | неверные параметры или тело запроса |
| 429 | `budget_exceeded` | дневной бюджет моделей исчерпан, `retry_after` до сброса в 3:00 МСК; готовые карточки по-прежнему отдаются |
| 429 | `queue_full` | слишком много карточек в очереди |
| 502 | `model_failed` | модель не дала годный ответ, можно повторить |
| 502 | `ml_error` | ML ответил неожиданной ошибкой |
| 503 | `ml_unavailable` | ML-сервис недоступен |
| 504 | `ml_timeout` | ML не ответил вовремя |
| 500 | `internal_error` | ошибка бэкенда (подробности в логах) |

## Структура папки

```
backend/
  README.md          этот файл
  PLAN.md            план работ
  requirements.txt   зависимости API (версии как в requirements.lock)
  app/               API: main.py (сборка), routes.py (ручки), ml_client.py, jobs.py, errors.py, config.py
  tests/             тесты API на моке ML
```

## Договорённости

- Любое изменение формата ответа ML согласуйте с владельцем `ml/` и фронтендом.
- Ключи и пароли только в `.env`, в репозиторий не попадают.
