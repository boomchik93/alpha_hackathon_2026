# API бэкенда на :8000. Собирается из корня репозитория: docker compose build backend
# (или docker build -f backend/Dockerfile .). Пакет ml в образ не входит: бэкенд ходит в ML-сервис по HTTP.
# База (var/backend.db) и PDF аналитики (data/analytics) монтируются томами, см. docker-compose.yml.
FROM python:3.13-slim

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/app backend/app
COPY backend/seed_users.json backend/seed_users.json
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
