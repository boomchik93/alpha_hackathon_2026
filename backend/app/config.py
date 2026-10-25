"""Настройки бэкенда из переменных окружения (`.env` в корне читает docker compose или shell)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _str(name: str, default: str) -> str:
    """Пустое значение равно отсутствию: compose подставляет пустую строку вместо незаданной переменной."""
    return os.getenv(name) or default


def _float(name: str, default: float) -> float:
    return float(os.getenv(name) or default)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name) or default)


@dataclass(frozen=True)
class Settings:
    ml_url: str = field(default_factory=lambda: _str("BACKEND_ML_URL", "http://localhost:8001").rstrip("/"))
    # Лента, профиль, обратная связь: ML отвечает из файлов, долгий ответ значит, что он завис.
    ml_timeout_s: float = field(default_factory=lambda: _float("BACKEND_ML_TIMEOUT_S", 15))
    # Генерация карточки на настоящей модели около 15 с, p95 около 31 с: берём с запасом.
    ml_insight_timeout_s: float = field(default_factory=lambda: _float("BACKEND_ML_INSIGHT_TIMEOUT_S", 120))
    # Уточняющий вопрос: до двух вызовов модели (ответ и повтор после проверки цифр).
    ml_followup_timeout_s: float = field(default_factory=lambda: _float("BACKEND_ML_FOLLOWUP_TIMEOUT_S", 60))
    # Сценарии: развилка, три ветки и разбор, пять вызовов модели подряд (ветки параллельно), поверх карточки.
    ml_scenarios_timeout_s: float = field(default_factory=lambda: _float("BACKEND_ML_SCENARIOS_TIMEOUT_S", 180))
    # Сколько ждать карточку в самом запросе: из кэша ML она приходит быстрее, тогда фронт сразу получает 200.
    insight_sync_wait_s: float = field(default_factory=lambda: _float("BACKEND_INSIGHT_SYNC_WAIT_S", 2))
    # Одновременных генераций карточек: каждая стоит денег из общего дневного бюджета.
    max_parallel_generations: int = field(default_factory=lambda: _int("BACKEND_MAX_PARALLEL_GENERATIONS", 2))
    max_queued_jobs: int = field(default_factory=lambda: _int("BACKEND_MAX_QUEUED_JOBS", 20))
    job_ttl_s: float = field(default_factory=lambda: _float("BACKEND_JOB_TTL_S", 3600))
    # var/ в .gitignore: пользователи, просмотры и журнал не попадают в репозиторий.
    db_path: str = field(default_factory=lambda: _str("BACKEND_DB_PATH", str(REPO_ROOT / "var" / "backend.db")))
    # Тестовые учётные записи: при старте добавляются или обновляются. Значение off отключает сид.
    seed_path: str = field(default_factory=lambda: _str("BACKEND_SEED_PATH", str(REPO_ROOT / "backend" / "seed_users.json")))
    session_ttl_h: float = field(default_factory=lambda: _float("BACKEND_SESSION_TTL_H", 24 * 7))
    # PDF аналитики банка для перехода к первоисточнику (ml/config.py, paths.analytics_pdf).
    analytics_dir: str = field(default_factory=lambda: _str("BACKEND_ANALYTICS_DIR", str(REPO_ROOT / "data" / "analytics")))
    # Лимиты частоты на пользователя за 10 минут: новые генерации карточек и уточняющие вопросы платные.
    # Карточка, которая уже генерируется для другого пользователя, лимит не тратит.
    limit_insights_per_10m: int = field(default_factory=lambda: _int("BACKEND_LIMIT_INSIGHTS_PER_10M", 20))
    limit_followups_per_10m: int = field(default_factory=lambda: _int("BACKEND_LIMIT_FOLLOWUPS_PER_10M", 10))
    # Набор сценариев стоит около 1 руб. (пять вызовов), заметно дороже карточки.
    limit_scenarios_per_10m: int = field(default_factory=lambda: _int("BACKEND_LIMIT_SCENARIOS_PER_10M", 10))
    # Неудачные входы за 5 минут: на логин (подбор пароля) и на адрес клиента (перебор логинов).
    limit_login_failures_per_login: int = field(default_factory=lambda: _int("BACKEND_LIMIT_LOGIN_FAILURES_PER_LOGIN", 5))
    limit_login_failures_per_ip: int = field(default_factory=lambda: _int("BACKEND_LIMIT_LOGIN_FAILURES_PER_IP", 20))
    cors_origins: list[str] = field(default_factory=lambda: [
        o.strip() for o in _str("BACKEND_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o.strip()
    ])
