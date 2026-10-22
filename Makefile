PY := .venv/bin/python

.PHONY: venv ocr vl vl-dry build profiles flows chunks items items-importance bm25 test serve eval-ranking \
	real-pool real2-freeze real2-label real2-review real2-eval pair-model-cv pair-model pair-model-calibrate tier-check

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

# Смежность отраслей из платежей (IMPROVEMENTS.md, B1).
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

# Свежий набор реальных новостей для итоговой оценки (STATUS_HANDOFF.md, шаг 3). Порядок:
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

# Двухуровневая лента на данных разработки (FEED_TIERS_PLAN.md, этапы 4 и 6). Без LLM.
tier-check:
	ML_AS_OF=2026-10-22 $(PY) -m ml.eval.tier_check

pair-model-calibrate:
	ML_AS_OF=2026-10-22 $(PY) -m ml.ranking.pair_model calibrate

test:
	$(PY) -m pytest -q ml/tests

serve:
	.venv/bin/uvicorn ml.server:app --port 8001 --reload
