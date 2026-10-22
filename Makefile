PY := .venv/bin/python

.PHONY: venv ocr vl vl-dry build profiles flows chunks items items-importance bm25 test serve eval-ranking

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

# То же плюс оценка важности материалов моделью. ПЛАТНО: вызов Gemma на материал, ~0,5-1 руб. на корпус.
items-importance:
	$(PY) -m ml.ingest.store --importance

# Замороженная статистика BM25 ленты (var/bm25_background.json); пересчитывать после chunks и items.
bm25:
	$(PY) -m ml.ranking.bm25

build: profiles flows chunks items bm25

# Ранжирование без LLM: метрики, leave-company-out, bootstrap-интервалы.
eval-ranking:
	$(PY) -m ml.eval.run_eval --ranking-only

test:
	$(PY) -m pytest -q ml/tests

serve:
	.venv/bin/uvicorn ml.server:app --port 8001 --reload
