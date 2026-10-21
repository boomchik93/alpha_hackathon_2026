# ML-сервис alpha-hack: HTTP API на :8001. Данные (data/) и рабочие файлы (var/) монтируются томами, см. docker-compose.yml.
FROM python:3.13-slim

# poppler-utils (pdftoppm) и tesseract нужны чтению PDF: ml/ingest/pdf_ocr.py и рендер страниц в ml/ingest/pdf_vl.py.
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock requirements-lemma.txt ./
RUN pip install --no-cache-dir -r requirements.lock
# Необязательная лемматизация для BM25 (ML_TOKENIZER=lemma): docker build --build-arg WITH_LEMMA=1 .
ARG WITH_LEMMA=0
RUN if [ "$WITH_LEMMA" = "1" ]; then pip install --no-cache-dir -r requirements-lemma.txt; fi

COPY ml ./ml
COPY Makefile ./
ENV PYTHONUNBUFFERED=1
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "ml.server:app", "--host", "0.0.0.0", "--port", "8001"]
