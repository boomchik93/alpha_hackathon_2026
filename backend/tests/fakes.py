"""Подмена `ml.api` для тестов бэкенда: данные по схемам `ml.schemas`, управляемая генерация карточки."""
import threading
import time
from datetime import date, datetime, timezone

import httpx
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from ml.errors import NotFound
from ml.schemas import (
    BankProducts, CompanyProfile, DataQuality, Geo, GroupInfo, Industry, InsightCard, Network, Size,
)

COMPANY = "U_1"
ITEM = "n_rate"


def profile(company_id: str = COMPANY) -> CompanyProfile:
    if company_id != COMPANY:
        raise NotFound(f"компания {company_id} не найдена")
    return CompanyProfile(company_id=COMPANY, eligible_big=True, industry=Industry(okved_name="Оптовая торговля"),
                          size=Size(revenue_rub=2.1e10), geo=Geo(region="Москва"), bank_products=BankProducts(loan=True),
                          group=GroupInfo(), network=Network(), data_quality=DataQuality())


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


class FakeML:
    """Управляет поведением подменённого `ml.api.get_insight`."""

    def __init__(self):
        self.delay_s = 0.0
        self.error: Exception | None = None
        self.insight_calls = 0
        self._lock = threading.Lock()

    def get_insight(self, company_id, item_id, refresh=False, role=None):
        with self._lock:
            self.insight_calls += 1
        time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        profile(company_id)
        if item_id != ITEM and not item_id.startswith("n_"):
            raise NotFound(f"материал {item_id} не найден")
        return card(company_id, item_id, role)


def make_client(transport: httpx.AsyncBaseTransport, **overrides) -> TestClient:
    settings = Settings(**{"ml_url": "http://ml", "insight_sync_wait_s": 0.5, "cors_origins": ["http://localhost:5173"],
                           **overrides})
    return TestClient(create_app(settings, ml_transport=transport))
