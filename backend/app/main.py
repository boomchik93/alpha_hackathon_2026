"""Бэкенд для фронта: `uvicorn backend.app.main:app --port 8000` из корня репозитория.

ML-сервис (`ml/server.py`, :8001) наружу не выставляется: фронт ходит только сюда.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app import errors
from backend.app.config import Settings
from backend.app.db import DB, load_seed
from backend.app.jobs import JobManager
from backend.app.ml_client import MLClient
from backend.app.routes import router

log = logging.getLogger("backend")


def create_app(settings: Settings | None = None, ml_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """ml_transport: для тестов, чтобы ходить в ML-приложение в том же процессе без сети."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.session_ttl = timedelta(hours=settings.session_ttl_h)
        app.state.db = DB(settings.db_path)
        if settings.seed_path:
            log.info("тестовые учётные записи: %s из %s", load_seed(app.state.db, Path(settings.seed_path)), settings.seed_path)
        app.state.group_members = {}
        app.state.ml = MLClient(settings.ml_url, settings.ml_timeout_s, transport=ml_transport)
        app.state.jobs = JobManager(settings.max_parallel_generations, settings.max_queued_jobs, settings.job_ttl_s)
        try:
            yield
        finally:
            await app.state.jobs.shutdown()
            await app.state.ml.aclose()
            app.state.db.close()

    app = FastAPI(title="alpha-hack backend", version="0.1", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "Authorization"], expose_headers=["Retry-After"])
    errors.install(app)
    app.include_router(router)
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
