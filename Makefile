PY := .venv/bin/python

.PHONY: venv ocr vl vl-dry build profiles flows chunks items items-importance bm25 test serve eval-ranking \
	real-pool real2-freeze real2-label real2-review real2-review-score real2-eval pair-model-cv pair-model pair-model-calibrate tier-check \
	backend-test backend-serve openapi collect \
	live-init live-up live-logs live-down demo-bundle demo-data

venv:
	python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Постраничный OCR сканов (один раз, ~5 мин).
ocr:
	$(PY) -m ml.ingest.pdf_ocr

# Чтение страниц vision-моделью (Qwen3-VL) с проверками и сверкой с tesseract. Платно: ~11-15 руб. на все 64 страницы.
# Сначала `make vl-dry`; запуск дописывает только новые страницы, ответы модели кэшируются.
# Примеры: make vl ARGS="--only ALFA-7M26:3 --max-rub 2"   |   make vl ARGS="--max-rub 20"
vl:
	$(PY) -m ml.ingest.pdf_vl $(ARGS)

vl-dry:
	$(PY) -m ml.ingest.pdf_vl --dry-run

profiles:
	$(PY) -m ml.profile.build_profile

# Смежность отраслей из платежей.
flows:
	$(PY) -m ml.profile.flows

chunks:
	$(PY) -m ml.ingest.analytics

# Разбор новостей Gemma из кэша (платно только при промахе кэша); важность материалов берётся из прежнего var/items.jsonl.
items:
	$(PY) -m ml.ingest.store

# То же плюс оценка важности материалов моделью (T11). ПЛАТНО: вызов Gemma на материал, ~0,5-1 руб. на корпус.
items-importance:
	$(PY) -m ml.ingest.store --importance

# Замороженная статистика BM25 ленты (var/bm25_background.json); пересчитывать после chunks и items.
bm25:
	$(PY) -m ml.ranking.bm25

build: profiles flows chunks items bm25

# Ранжирование без LLM: метрики, leave-company-out, bootstrap-интервалы.
eval-ranking:
	$(PY) -m ml.eval.run_eval --ranking-only

# Модель пары для ленты (ml/ranking/pair_model.py): выбор настроек перекрёстной проверкой по дням и обучение. Без LLM.
pair-model-cv:
	ML_AS_OF=2026-10-22 $(PY) -m ml.ranking.pair_model cv

pair-model:
	ML_AS_OF=2026-10-22 $(PY) -m ml.ranking.pair_model train

# Свежий набор реальных новостей для итоговой оценки. Порядок:
# 1) real-pool раз в день (RSS хранит записи несколько дней); 2) real2-freeze, когда в пуле 5-7 дней после START;
# 3) real2-label (ПЛАТНО, ~60-75 руб.); 4) real2-eval один раз, параметры после него не менять.
# Пример: make real2-freeze START=2026-10-23 ASOF=2026-10-29 && make real2-label ASOF=2026-10-29 && make real2-eval ASOF=2026-10-29
real-pool:
	$(PY) -m ml.eval.collect_real --accumulate

real2-freeze:
	$(PY) -m ml.eval.collect_real --name real2 --from-pool --start $(START) --as-of $(ASOF) --n 100 --holdout-days 0

real2-label:
	ML_AS_OF=$(ASOF) $(PY) -m ml.eval.label_llm --set real2 --batch 40 --max-rub 90

# Ручная разметка real2 (150 случайных пар) ДО просмотра результатов ленты: make real2-review ASOF=...,
# заполнить ml/eval/review_real2_1.md, затем make real2-review-score. Оценки лент в файл не попадают.
real2-review:
	ML_AS_OF=$(ASOF) $(PY) -m ml.eval.label_review_real make --set real2 --batch 1

real2-review-score:
	$(PY) -m ml.eval.label_review_real score --set real2 --batch 1 --apply

real2-eval:
	ML_AS_OF=$(ASOF) $(PY) -m ml.eval.final_eval
	ML_AS_OF=$(ASOF) $(PY) -m ml.eval.run_eval --ranking-only --set real2 --ranking-report ml/eval/ranking_report_real2.md

# Двухуровневая лента на данных разработки. Без LLM.
tier-check:
	ML_AS_OF=2026-10-22 $(PY) -m ml.eval.tier_check

pair-model-calibrate:
	ML_AS_OF=2026-10-22 $(PY) -m ml.ranking.pair_model calibrate

test:
	$(PY) -m pytest -q ml/tests

serve:
	.venv/bin/uvicorn ml.server:app --port 8001 --reload

# API бэкенда для фронта (backend/app) на :8000; ML-сервис должен быть поднят (`make serve`).
backend-test:
	$(PY) -m pytest -q backend/tests

backend-serve:
	.venv/bin/uvicorn backend.app.main:app --port 8000 --reload

# Схемы OpenAPI обоих сервисов в docs/openapi/ (без запуска серверов и без обращений к моделям).
openapi:
	mkdir -p docs/openapi
	$(PY) -c "import json; from backend.app.main import app; json.dump(app.openapi(), open('docs/openapi/backend.json', 'w'), ensure_ascii=False, indent=2)"
	$(PY) -c "import json; from ml.server import app; json.dump(app.openapi(), open('docs/openapi/ml.json', 'w'), ensure_ascii=False, indent=2)"

# Один проход сборщика новостей: var/news_collector/raw_items.jsonl, вход для `NEWS_SOURCE=collector make items`.
collect:
	$(PY) backend/news_collector/main.py run --once

# Боевой режим (docker-compose.live.yml): настоящие модели, живые новости, карточки для аккаунтов из backend/seed_users.json.
# ПЛАТНО: разбор новостей и карточки до LIVE_CARDS_BUDGET_RUB (по умолчанию 100 руб.) в день по общему журналу расходов.
LIVE := docker compose -f docker-compose.yml -f docker-compose.live.yml

# var_live/: профили, связи, фрагменты аналитики и история сборщика копируются из var/ один раз; дальше живут отдельно.
live-init:
	mkdir -p var_live var/llm_cache
	touch var/spend.jsonl
	for f in profiles.jsonl flows.parquet analytics_chunks.jsonl; do [ -e var_live/$$f ] || cp var/$$f var_live/; done
	for d in news_collector news_cache emb_cache; do [ ! -d var/$$d ] || [ -e var_live/$$d ] || cp -R var/$$d var_live/; done

live-up: live-init
	$(LIVE) up -d --build

live-logs:
	$(LIVE) logs -f --tail 100 pipeline

live-down:
	$(LIVE) down

# Демо-данные для запуска из чистого клона: data/ и var/ в репозиторий не входят (данные кейса и PDF аналитики банка).
# demo-bundle: собрать архив из рабочего var/ (только то, что читает сервис, без eval, кэша моделей и базы пользователей).
# demo-data: скачать и распаковать архив в корень репозитория; адрес архива в DEMO_DATA_URL (README, раздел «Запуск»).
DEMO_BUNDLE := dist/alpha26-demo-data.tar.gz
DEMO_FILES := var/profiles.jsonl var/analytics_chunks.jsonl var/items.jsonl var/bm25_background.json var/flows.parquet \
	var/feedback.jsonl var/cards var/scenarios var/traces data/analytics

demo-bundle:
	mkdir -p dist
	COPYFILE_DISABLE=1 tar -czf $(DEMO_BUNDLE) $(DEMO_FILES)
	@ls -lh $(DEMO_BUNDLE)

demo-data:
	@test -n "$(DEMO_DATA_URL)" || { echo "задайте DEMO_DATA_URL (адрес архива в README)"; exit 1; }
	@test ! -e var/profiles.jsonl || { echo "var/ уже собран, распаковка пропущена"; exit 0; }
	curl -fL "$(DEMO_DATA_URL)" | tar -xz
