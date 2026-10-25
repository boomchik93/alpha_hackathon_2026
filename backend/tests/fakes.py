"""Подмена `ml.api` для тестов бэкенда: данные по схемам `ml.schemas`, управляемая генерация карточки."""
import json
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from ml.errors import NotApplicable, NotFound
from ml.schemas import (
    SCENARIO_KINDS, BankProducts, CompanyProfile, DataQuality, Feed, FeedEntry, Geo, GroupInfo, Industry, InsightCard, Network,
    ScenarioSet, Size,
)

COMPANY = "U_1"          # компания alice и bob
OTHER = "U_2"            # чужая для всех обычных пользователей
GROUP = "group_9"        # группа пользователя grp
GROUP_MEMBER = "U_3"     # компания из этой группы
ITEM = "n_rate"

# Тестовые учётные записи: пароль "pw-<login>".
USERS = [
    {"login": "alice", "display_name": "Алиса", "role": "cfo", "access": "user", "companies": [COMPANY]},
    {"login": "bob", "display_name": "Боб", "role": "owner", "access": "user", "companies": [COMPANY]},
    {"login": "grp", "display_name": "Группа", "role": "ceo", "access": "user", "companies": [GROUP]},
    {"login": "exp", "display_name": "Эксперт", "role": "other", "access": "expert", "companies": []},
    {"login": "adm", "display_name": "Админ", "role": "other", "access": "admin", "companies": []},
]


def profile(company_id: str = COMPANY) -> CompanyProfile:
    if company_id not in (COMPANY, OTHER, GROUP, GROUP_MEMBER):
        raise NotFound(f"компания {company_id} не найдена")
    group = company_id == GROUP
    return CompanyProfile(company_id=company_id, level="group" if group else "company", eligible_big=True,
                          industry=Industry(okved_name="Оптовая торговля"), size=Size(revenue_rub=2.1e10),
                          geo=Geo(region="Москва"), bank_products=BankProducts(loan=True), group=GroupInfo(),
                          network=Network(), data_quality=DataQuality(), member_ids=[GROUP_MEMBER] if group else [])


def feed(company_id, limit, tier, maybe_limit) -> Feed:
    entry = lambda item_id, kind, score, day, **kw: FeedEntry(  # noqa: E731
        item_id=item_id, title=f"Материал {item_id}", published_at=date(2026, 10, day), source_kind=kind, score=score,
        reasons=[f"limit={limit} tier={tier}"], **kw)
    entries = [entry(ITEM, "news", 0.9, 23, topics=["key_rate"]),
               entry("bank_rate_2026_09_11", "bank_analytics", 0.7, 11, topics=["key_rate"], stale=True),
               entry("n_fx", "news", 0.6, 20, topics=["fx"])]
    if tier == "all":
        entries.append(entry("n_maybe", "news", 0.3, 22, topics=["logistics"], tier="maybe"))
    return Feed(company_id=profile(company_id).company_id, as_of=date(2026, 10, 24), entries=entries,
                maybe_count=int(tier == "all"))


def card(company_id: str, item_id: str, role=None) -> InsightCard:
    suffix = f"_{role.value}" if role else ""
    return InsightCard.model_validate({
        "relevance": {"score": 0.9, "reasons": ["есть кредитные продукты"]},
        "what_happened": {"text": "Банк России сохранил ключевую ставку на уровне 14%.", "facts": ["f1"]},
        "why_it_matters": {"text": "Кредиты дороже дольше.", "company_features_used": ["bank_products.loan"]},
        "consequences": {"risks": [{"text": "Долг не подешевеет.", "evidence": ["src_n_rate"]}], "opportunities": []},
        "bank_view": None,
        "ai_inference": {"text": "Пересмотреть график заимствований.", "source_ids": ["profile"]},
        "suggested_actions": [{"type": "action", "text": "Проверить условия рефинансирования.", "evidence": ["profile"]}],
        "card_id": f"ins_{company_id}_{item_id}{suffix}", "company_id": company_id, "item_ids": [item_id],
        "sources": [{"id": "src_n_rate", "kind": "news", "title": "ЦБ сохранил ставку", "published_at": date(2026, 10, 23)}],
        "trace_id": "tr_test", "role": role.value if role else None,
        "meta": {"model": "mock", "prompt_version": "interpret@test", "generated_at": datetime.now(timezone.utc)},
    })


TRACE = {
    "trace_id": "tr_test", "kind": "insight", "started_at": 0, "duration_s": 12.5, "cost_rub": 0.42, "retries": 1,
    "events": [
        {"t": 0, "event": "start", "company_id": COMPANY, "item_id": ITEM, "mock": True},
        {"t": 0.1, "event": "plan", "steps": [{"tool": "compute_exposure", "why": "найти признаки профиля"},
                                      {"tool": None, "why": "интерпретация, затем самопроверка"}]},
        {"t": 0.2, "event": "tool_call", "tool": "compute_exposure", "args": {"company_id": COMPANY}, "result": "dict", "ms": 12},
        {"t": 1.0, "event": "llm_call", "model": "glm", "prompt": "СЕКРЕТНЫЙ ПРОМПТ", "cost_rub": 0.2},
        {"t": 2.0, "event": "verify", "attempt": 0, "issues": ["цифра 15% не найдена"]},
        {"t": 2.1, "event": "replan", "trigger": "bank_miss", "why": "переформулировать запрос", "query": "ставка"},
        {"t": 3.0, "event": "verify", "attempt": 1, "issues": []},
    ],
}


WEAK_ITEM = "n_weak"     # карточка со слабой связью: сценарии не строятся
SCENARIOS_TRACE = {
    "trace_id": "tr_scn", "kind": "scenarios", "started_at": 0, "duration_s": 40.0, "cost_rub": 0.99, "retries": 1,
    "events": [
        {"t": 0, "event": "start", "card_id": "ins_x"},
        {"t": 1.0, "event": "llm_call", "model": "glm", "prompt": "СЕКРЕТНЫЙ ПРОМПТ", "cost_rub": 0.2},
        {"t": 3.0, "event": "scenario_fork", "attempt": 0, "issues": []},
        {"t": 9.0, "event": "scenario_branch", "kind": "good", "attempt": 0, "issues": []},
        {"t": 9.5, "event": "scenario_branch", "kind": "base", "attempt": 0, "issues": ["цифр ['37,5'] нет в источниках"]},
        {"t": 9.9, "event": "scenario_branch", "kind": "base", "attempt": 1, "issues": []},
        {"t": 10.0, "event": "scenario_branch", "kind": "bad", "attempt": 0, "issues": []},
        {"t": 20.0, "event": "scenario_analysis", "attempt": 0, "issues": []},
    ],
}


def scenario_set(company_id: str, item_id: str, role=None) -> ScenarioSet:
    """Набор сценариев по схеме `ml.schemas`: три ветки в порядке good, base, bad и разбор."""
    suffix = f"_{role.value}" if role else ""
    branch = lambda kind: {  # noqa: E731
        "kind": kind, "title": f"Ветка {kind}", "assumption": f"если исход {kind}", "likelihood": "medium",
        "likelihood_basis": "по материалу", "impact": "умеренное влияние", "company_features_used": ["bank_products.loan"],
        "risks": [{"text": "Долг подорожает.", "evidence": ["src_n_rate"]}], "opportunities": [],
        "signals": [f"сигнал {kind}"], "actions": [{"type": "action", "text": f"Действие {kind}", "evidence": ["profile"]}]}
    effects = {"good": "helps", "base": "helps", "bad": "neutral"}
    return ScenarioSet.model_validate({
        "set_id": f"scn_{company_id}_{item_id}{suffix}", "card_id": f"ins_{company_id}_{item_id}{suffix}", "company_id": company_id,
        "item_ids": [item_id], "role": role.value if role else "ceo",
        "fork": {"variable": "решение регулятора", "base_basis": "item_expectation", "evidence": ["src_n_rate"],
                 "outcomes": {"good": "если мягче", "base": "если по ожиданиям", "bad": "если жёстче"}},
        "branches": [branch(k) for k in SCENARIO_KINDS],
        "analysis": {"summary": "Ветки различаются нагрузкой.", "key_uncertainty": "решение регулятора",
                     "decisions": [{"text": "Проверить условия кредитов", "type": "robust", "effects": effects, "rationale": "полезно"}],
                     "recommendation": "Начать с проверки кредитов."},
        "sources": [{"id": "src_n_rate", "kind": "news", "title": "ЦБ сохранил ставку", "published_at": date(2026, 10, 23)}],
        "trace_id": SCENARIOS_TRACE["trace_id"],
        "meta": {"model": "mock", "prompt_version": "scenario@test", "generated_at": datetime.now(timezone.utc),
                 "card_generated_at": datetime.now(timezone.utc)},
    })


class FakeML:
    """Управляет поведением подменённых `ml.api.get_insight` и `ml.api.get_scenarios`."""

    def __init__(self):
        self.delay_s = 0.0
        self.error: Exception | None = None
        self.insight_calls = 0
        self.scenario_delay_s = 0.0
        self.scenario_error: Exception | None = None
        self.scenario_calls = 0
        self._lock = threading.Lock()

    def get_insight(self, company_id, item_id, refresh=False, role=None):
        with self._lock:
            self.insight_calls += 1
        time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        profile(company_id)
        if not item_id.startswith(("n_", "bank_")):
            raise NotFound(f"материал {item_id} не найден")
        return card(company_id, item_id, role)

    def get_scenarios(self, company_id, item_id, refresh=False, role=None):
        with self._lock:
            self.scenario_calls += 1
        time.sleep(self.scenario_delay_s)
        if self.scenario_error is not None:
            raise self.scenario_error
        profile(company_id)
        if item_id == WEAK_ITEM:
            raise NotApplicable("сценарии не строятся: связь материала с компанией слабая (relevance 0.10 < 0.3)")
        if not item_id.startswith(("n_", "bank_")):
            raise NotFound(f"материал {item_id} не найден")
        return scenario_set(company_id, item_id, role)


def ml_transport() -> httpx.ASGITransport:
    """Настоящий ml.server в том же процессе. Необработанное исключение ML превращается в ответ 500, как под uvicorn."""
    from ml import server as ml_server

    return httpx.ASGITransport(app=ml_server.app, raise_app_exceptions=False)


def make_client(transport: httpx.AsyncBaseTransport, tmp_path: Path, **overrides) -> TestClient:
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps([{**u, "password": f"pw-{u['login']}"} for u in USERS], ensure_ascii=False), encoding="utf-8")
    settings = Settings(**{"ml_url": "http://ml", "insight_sync_wait_s": 0.5, "cors_origins": ["http://localhost:5173"],
                           "db_path": str(tmp_path / "backend.db"), "seed_path": str(seed),
                           "analytics_dir": str(tmp_path / "analytics"), **overrides})
    return TestClient(create_app(settings, ml_transport=transport))


def login(client: TestClient, user: str) -> dict:
    """Заголовки с токеном пользователя."""
    resp = client.post("/api/auth/login", json={"login": user, "password": f"pw-{user}"})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}
