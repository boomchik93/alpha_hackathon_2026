"""Настройки слоя ML. Всё, что зависит от окружения, берётся из `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Paths:
    data: Path = ROOT / "data"
    structured: Path = ROOT / "data" / "structured"
    analytics_pdf: Path = ROOT / "data" / "analytics"
    analytics_pages: Path = ROOT / "data" / "analytics_pages"
    # Страницы, прочитанные vision-моделью (ml/ingest/pdf_vl.py): page_NNN.md и page_NNN.json с флагами.
    analytics_pages_vl: Path = ROOT / "data" / "analytics_pages_vl"
    var: Path = ROOT / "var"
    fixtures: Path = ROOT / "ml" / "fixtures"
    prompts: Path = ROOT / "ml" / "prompts"

    @property
    def profiles(self) -> Path:
        return self.var / "profiles.jsonl"

    @property
    def chunks(self) -> Path:
        return self.var / "analytics_chunks.jsonl"

    @property
    def llm_cache(self) -> Path:
        return self.var / "llm_cache"

    @property
    def runs(self) -> Path:
        """Результаты запусков (карточки, трассы). Тесты подменяют через ML_RUNS_DIR."""
        return Path(_env("ML_RUNS_DIR")) if _env("ML_RUNS_DIR") else self.var

    @property
    def traces(self) -> Path:
        return self.runs / "traces"

    @property
    def cards(self) -> Path:
        return self.runs / "cards"

    @property
    def scenarios(self) -> Path:
        """Наборы сценариев по карточкам (ml/agent/scenarios.py)."""
        return self.runs / "scenarios"

    @property
    def feedback(self) -> Path:
        """Оценки карточек пользователями (ml/feedback.py)."""
        return self.runs / "feedback.jsonl"

    @property
    def spend_log(self) -> Path:
        return self.var / "spend.jsonl"


@dataclass(frozen=True)
class LLMSettings:
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    # Без base_url работаем на моке: код пайплайна от этого не меняется.
    mock: bool = field(default_factory=lambda: _env_bool("LLM_MOCK", not _env("LLM_BASE_URL")))
    timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "120")))
    daily_budget_rub: float = field(default_factory=lambda: float(_env("LLM_DAILY_BUDGET_RUB", "500")))
    # Доля бюджета, после которой клиент перестаёт делать платные вызовы.
    budget_stop_ratio: float = field(default_factory=lambda: float(_env("LLM_BUDGET_STOP_RATIO", "0.9")))


MSK = timezone(timedelta(hours=3))


def resolve_as_of(raw: str) -> date:
    """`today`: текущая дата по Москве (боевой режим, новости российские); иначе дата ISO."""
    return datetime.now(MSK).date() if raw == "today" else date.fromisoformat(raw)


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    llm: LLMSettings = field(default_factory=LLMSettings)
    big_revenue_rub: float = 15e9
    embedder_url: str = field(default_factory=lambda: _env("EMBEDDER_URL"))

    @property
    def as_of(self) -> date:
        """«Сейчас», от него считается свежесть материалов. Для демо и eval фиксированная дата (`ML_AS_OF`),
        в боевом режиме `ML_AS_OF=today`: читается при каждом обращении, сервер работает сутками."""
        return resolve_as_of(_env("ML_AS_OF", "2026-09-15"))


settings = Settings()
