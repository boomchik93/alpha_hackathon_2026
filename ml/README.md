# ML-слой

Каркас по [ML_SPEC.md](docs/ML_SPEC.md) и [PLAN.md](docs/PLAN.md). Пока нет `LLM_BASE_URL`, все вызовы LLM идут в мок (`ml/llm/mock.py`): пайплайн работает целиком, но тексты карточек помечены `[MOCK]`.

## Запуск

```bash
make venv     # Python 3.13, зависимости
make ocr      # постраничный OCR PDF (tesseract) -> data/analytics_pages: запасной вариант и сверка
make vl-dry   # сколько страниц уйдёт в Qwen3-VL и сколько это стоит
make vl       # чтение страниц Qwen3-VL -> data/analytics_pages_vl (платно, один раз; ARGS="--only DOC:3 --max-rub 2")
make build    # профили, фрагменты аналитики, разобранные материалы -> var/
make test
make serve    # HTTP на :8001, документация /docs
```

Чтобы перейти на настоящую модель, задайте `LLM_BASE_URL` в `.env` (OpenAI-совместимый `/chat/completions`) и при необходимости имена моделей `LLM_MODEL_*` и цены `LLM_PRICES_RUB_PER_M`.

## Контракт для бэкенда

Модели в `ml/schemas.py`. Python-функции в `ml/api.py`, те же функции по HTTP в `ml/server.py`:

| HTTP | Функция | Ответ |
|---|---|---|
| `GET /companies` | `list_companies()` | краткий список 115 компаний и 298 групп |
| `GET /companies/{id}` | `build_profile(id)` | `CompanyProfile` |
| `GET /companies/{id}/feed?limit=20&maybe_limit=10&tier=all` | `get_ranked_feed(id, limit, tier, maybe_limit)` | `Feed` |
| `GET /companies/{id}/insights/{item_id}` | `get_insight(id, item_id)` | `InsightCard` |
| `GET /companies/{id}/insights/{item_id}/scenarios?role=&refresh=` | `get_scenarios(id, item_id, refresh, role)` | `ScenarioSet` |
| `POST /cards/{card_id}/followup` | `answer_followup(card_id, q)` | `FollowupAnswer` |
| `GET /traces/{trace_id}` | | трасса агента |
| `GET /health` | | режим мока, расход за день |

### Два уровня ленты

Лента состоит из основной части и блока «может быть интересно». Новые поля добавлены со значениями по умолчанию, старые не менялись:

- `FeedEntry.tier`: `"main"` (по умолчанию) или `"maybe"`. Записи `main` идут первыми, затем `maybe`; внутри уровня порядок по `score`.
- `Feed.maybe_count`: сколько записей `maybe` в ответе (0 при `tier=main`).
- `tier=all` (по умолчанию): основная лента (до `limit`, 20) и блок «может быть» (до `maybe_limit`, 10). `tier=main`: только основная.
- Уровень задаёт личный скор пары: от 0,40 или событие прямо в отрасли компании (ОКВЭД) — `main`; от 0,30 до 0,40 — `maybe`;
  ниже 0,30 материал не показывается. Пороги: `ranker.MAIN_SCORE` (`ML_FEED_MAIN_SCORE`) и `ranker.MIN_SCORE`.
- У `maybe` первая причина «слабая связь с профилем компании»: фронт может подписать блок и показать, почему материал здесь.
- Карточки предрасчёта (`make precompute`) считаются только для `main`.

### Сценарии (ветвление)

Для главы компании к карточке строятся три ветки развития события: благоприятная (`good`), базовая (`base`), неблагоприятная (`bad`) и разбор решений по ним. Модуль [ml/agent/scenarios.py](agent/scenarios.py), модели `ScenarioSet`, `ScenarioFork`, `ScenarioBranch`, `ScenarioAnalysis` в `ml/schemas.py`.

```
InsightCard (готовая, проверенная) -> scenario_fork -> scenario x3 (параллельно) -> scenario_analysis -> ScenarioSet
```

1. **Развилка** (`scenario_fork`): одна внешняя неопределённость и условие каждого исхода. Базовый исход стоит на прогнозе банка, если он есть во фрагментах (`fork.base_basis`).
2. **Ветки** (`scenario`, по вызову на ветку): условие (гипотеза), качественная вероятность `low|medium|high` с основанием, влияние на компанию через признаки профиля, риски и возможности, сигналы «мы идём по этой ветке», действия. Для каждой ветки отдельно ищутся фрагменты банка по её условию.
3. **Разбор** (`scenario_analysis`): решения из всех веток сведены в таблицу «решение x ветка» (`helps|neutral|hurts`). Тип: `robust` (не вредит ни в одной ветке, делать сейчас), `hedge` (страховка от `bad`), `conditional` (запускать по сигналу `trigger`). Плюс `key_uncertainty`, `recommendation`, `switch_signals`.

Проверки кода те же, что у карточки: цифры только из материала, фрагментов банка, профиля и развилки (вероятностей числом нет), ссылки только на известные источники, продукты банка, «лидер/крупнейшая», инъекции, id источников не в тексте. Каждый этап при замечаниях переписывается один раз; оставшиеся замечания в `branches[].verifier_issues` и `meta.verifier_issues`. Код следит за разбором: `robust` с `hurts` понижается до `conditional`, у `conditional` и `hedge` сигнал запуска берётся из сигналов веток.

- Роль: без `role` сценарии пишутся для роли карточки, а без неё для генерального директора (`ceo`).
- Карточка со слабой связью (`relevance` < 0,3): `NotApplicable`, HTTP 422.
- Кэш: `var/scenarios/` (`ML_RUNS_DIR`). Набор считается устаревшим, если сменились карточка, промпты или модель.
- Цена: 5 вызовов GLM, около 1 руб. за набор (замер 2026-10-25). Трасса `GET /traces/{trace_id}` показывает все этапы.

## Пайплайн

```
CSV -> profile/build_profile -> var/profiles.jsonl
PDF -> ingest/pdf_vl (текстовый слой | Qwen3-VL, проверки page_check, сверка с tesseract; запасной вариант ingest/pdf_ocr) -> ingest/analytics (чистка, фрагменты со страницей и датой) -> retrieval/index (BM25, e5 опционально)
фикстуры | RSS -> ingest/news_fetch -> safety.strip_injections -> ingest/parse_item (Gemma) -> ingest/dedup -> var/items.jsonl
профиль + материалы -> ranking/ranker (тема -> признак, свежесть) -> лента
agent/orchestrator: plan -> tools (белый список, только своя компания) -> interpret (GLM) -> verifier -> retry (до 2) -> InsightCard
```

Учёт расходов: `var/spend.jsonl`, остановка на 90% дневного лимита. Кэш LLM: `var/llm_cache/`. Трассы: `var/traces/`.

## Что дальше

- Подключить реальный API, довести `prompts/interpret.md` на 3 клиентах.
- Сервер e5 (`EMBEDDER_URL`) и замер BM25 / e5 / гибрид.
- Eval-набор и `eval/run_eval.py`.
- LLM-реранкинг топ-N (`prompts/relevance.md`).
