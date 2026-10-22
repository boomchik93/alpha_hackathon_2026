# ML-слой

Подробное описание: [OVERVIEW.md](OVERVIEW.md). Пока нет `LLM_BASE_URL`, все вызовы LLM идут в мок (`ml/llm/mock.py`): пайплайн работает целиком, но тексты карточек помечены `[MOCK]`.

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
| `GET /companies/{id}/feed` | `get_ranked_feed(id)` | `Feed` |
| `GET /companies/{id}/insights/{item_id}` | `get_insight(id, item_id)` | `InsightCard` |
| `POST /cards/{card_id}/followup` | `answer_followup(card_id, q)` | `FollowupAnswer` |
| `GET /traces/{trace_id}` | | трасса агента |
| `GET /health` | | режим мока, расход за день |

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
