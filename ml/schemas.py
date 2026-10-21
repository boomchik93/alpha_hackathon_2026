"""Контракт данных между слоями: ML, бэкенд, фронт.

Все этапы пайплайна обмениваются этими моделями. Менять поля только по
согласованию с бэкендом и фронтом.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROFILE_SOURCE_ID = "profile"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# CompanyProfile
# ---------------------------------------------------------------------------


class Industry(_Model):
    okved_code: str | None = Field(None, description="нормализованный код, например 46.71")
    okved_raw: str | None = None
    okved_name: str | None = None
    okved_section: str | None = Field(None, description="первые две цифры кода")
    sector_tags: list[str] = []


class Size(_Model):
    revenue_rub: float | None = None
    revenue_band: str | None = None
    workers_band: str | None = Field(None, description="AMOUNT_WORKERS_FNS_GRADATION")
    age_band: str | None = None


class Geo(_Model):
    city: str | None = None
    region: str | None = None
    foreign_trade: bool | None = None
    fts_countries: list[str] = []


class BankProducts(_Model):
    rko: bool | None = None
    guarantee: bool | None = None
    loan: bool | None = None
    leasing: bool | None = None
    deposit: bool | None = None
    payroll: bool | None = None
    factoring: bool | None = None


class GroupInfo(_Model):
    group_id: str | None = None
    group_size: int | None = None
    group_revenue_rub: float | None = None
    group_clients: int | None = None


class Counterparty(_Model):
    id: str
    role: Literal["payer", "beneficiary"] = Field(description="роль контрагента по отношению к компании")
    amt_rub: float
    cnt: int
    known: bool = Field(description="есть ли контрагент в таблице компаний")
    okved_name: str | None = None


class Network(_Model):
    top_counterparties: list[Counterparty] = []
    counterparty_sectors: list[str] = []
    known_share: float | None = Field(None, description="доля оборота с известными контрагентами")
    note: str | None = None


class DataQuality(_Model):
    missing_fields: list[str] = []
    conflicts: list[str] = []
    notes: list[str] = []


class MarketPosition(_Model):
    """Положение компании среди компаний выборки банка с тем же префиксом ОКВЭД (по выручке). Это не доля рынка РФ."""

    okved_prefix: str = Field(description="префикс ОКВЭД сравнения, например 41.2")
    digits: int = Field(description="сколько цифр кода в префиксе: 4, 3 или 2")
    peers: int = Field(description="сколько субъектов в сравнении, включая саму компанию или группу")
    rank: int = Field(description="место по выручке, 1 = крупнейшая")
    percentile: float = Field(ge=0, le=1, description="доля остальных субъектов сравнения с меньшей выручкой")
    revenue_share: float | None = Field(None, description="выручка / суммарная выручка сравнения, доля от 0 до 1")
    note: str


class CompanyProfile(_Model):
    company_id: str
    level: Literal["company", "group"] = "company"
    org_type: str | None = None
    ownership: str | None = None
    client_type: str | None = Field(None, description="CL_TYPE: CLIENT / PROSPECT")
    stage: str | None = Field(None, description="ECO_MARK_LABEL")
    eligible_big: bool = Field(description="выручка компании или группы > 15 млрд руб.")
    industry: Industry
    size: Size
    geo: Geo
    bank_products: BankProducts
    group: GroupInfo
    network: Network
    data_quality: DataQuality
    member_ids: list[str] = Field([], description="для level=group: компании группы")
    market_position: MarketPosition | None = Field(None, description="положение в отрасли среди компаний выборки банка")

    def prompt_view(self) -> dict:
        """Профиль для промпта: без пустых полей и служебных идентификаторов."""
        return self.model_dump(exclude_none=True, exclude={"member_ids"})


# ---------------------------------------------------------------------------
# ParsedItem
# ---------------------------------------------------------------------------


class SourceKind(str, Enum):
    news = "news"
    bank_analytics = "bank_analytics"
    synthetic_test = "synthetic_test"


class ItemSource(_Model):
    kind: SourceKind
    title: str
    publisher: str | None = None
    url: str | None = None
    published_at: date
    doc_id: str | None = None
    page: int | None = None


class Fact(_Model):
    id: str
    text: str
    span: str | None = Field(None, description="цитата из исходного текста")
    confidence: Literal["confirmed", "assumption"]


class ParsedItem(_Model):
    item_id: str
    source: ItemSource
    title: str
    summary: str | None = None
    event_type: str = Field(description="см. ml.topics.EVENT_TYPES")
    topics: list[str] = Field([], description="ключи из ml.topics.TOPICS")
    sectors: list[str] = []
    okved_hints: list[str] = Field([], description="префиксы ОКВЭД, которых касается событие")
    regions: list[str] = []
    facts: list[Fact] = []
    freshness: float = Field(1.0, ge=0, le=1, description="1 = свежий, 0 = устаревший")
    importance: float = Field(0.5, ge=0, le=1, description="общая важность для экономики, не зависит от компании; 0,5 = не оценена")
    noisy: bool = Field(False, description="исходный текст был зашумлён (OCR, сбои кодировки) и почищен при разборе")
    dedup_cluster: str | None = None


# ---------------------------------------------------------------------------
# InsightCard
# ---------------------------------------------------------------------------


class Relevance(_Model):
    score: float = Field(ge=0, le=1)
    reasons: list[str]


class WhatHappened(_Model):
    text: str
    facts: list[str] = Field([], description="id фактов из ParsedItem")


class WhyItMatters(_Model):
    text: str
    company_features_used: list[str] = Field(min_length=1, description="пути полей профиля, например bank_products.loan")


class Claim(_Model):
    text: str
    horizon: str | None = None
    evidence: list[str] = []
    is_model_inference: bool = False

    @model_validator(mode="after")
    def _evidence_or_flag(self) -> "Claim":
        if not self.evidence and not self.is_model_inference:
            raise ValueError("утверждение без evidence должно быть помечено is_model_inference=true")
        return self


class Consequences(_Model):
    risks: list[Claim] = []
    opportunities: list[Claim] = []


class BankQuote(_Model):
    source_id: str
    text: str = Field(description="дословная цитата из фрагмента аналитики банка")


class BankView(_Model):
    # Модель возвращает цитаты; text собирает код из проверенных цитат.
    text: str = ""
    source_ids: list[str] = Field(min_length=1)
    quotes: list[BankQuote] = []


class AIInference(_Model):
    text: str
    source_ids: list[str] = []
    label: Literal["Вывод модели"] = "Вывод модели"


class SuggestedAction(Claim):
    type: Literal["action", "question_for_analysis"]


class SourceRef(_Model):
    id: str
    kind: SourceKind
    title: str
    url: str | None = None
    published_at: date | None = None
    doc_id: str | None = None
    page: int | None = None
    quote: str | None = None
    stale: bool = Field(False, description="материал старый, может быть неактуален")


class ImpactChannel(_Model):
    """Канал влияния события на компанию (стадия 1 генерации). Ключи: ml.topics.IMPACT_CHANNELS."""

    channel: str
    strength: Literal["strong", "weak", "none"]
    direction: Literal["negative", "positive", "mixed", "unclear"] = "unclear"
    mechanism: str = ""
    profile_features: list[str] = []
    evidence: list[str] = []


class CardMeta(_Model):
    model: str
    prompt_version: str
    generated_at: datetime
    retries: int = 0
    cost_rub: float = 0.0
    verifier_issues: list[str] = []


class InsightDraft(_Model):
    """То, что возвращает LLM. Источники и служебные поля добавляет код."""

    relevance: Relevance
    what_happened: WhatHappened
    why_it_matters: WhyItMatters
    consequences: Consequences
    bank_view: BankView | None
    ai_inference: AIInference
    suggested_actions: list[SuggestedAction] = Field(min_length=1)
    limitations: list[str] = []

    def referenced_source_ids(self) -> set[str]:
        ids: set[str] = set()
        for claim in [*self.consequences.risks, *self.consequences.opportunities, *self.suggested_actions]:
            ids.update(claim.evidence)
        if self.bank_view:
            ids.update(self.bank_view.source_ids)
            ids.update(q.source_id for q in self.bank_view.quotes)
        ids.update(self.ai_inference.source_ids)
        return ids


class InsightCard(InsightDraft):
    card_id: str
    company_id: str
    item_ids: list[str] = Field(min_length=1)
    sources: list[SourceRef]
    trace_id: str
    meta: CardMeta
    role: str | None = Field(None, description="роль пользователя, для которой написан акцент карточки (ml.roles.Role)")
    channels: list[ImpactChannel] = Field([], description="проверенные каналы влияния, если включена двухстадийная генерация")

    @model_validator(mode="after")
    def _sources_resolve(self) -> "InsightCard":
        known = {s.id for s in self.sources} | {PROFILE_SOURCE_ID}
        missing = self.referenced_source_ids() - known
        if missing:
            raise ValueError(f"ссылки на неизвестные источники: {sorted(missing)}")
        if self.bank_view:
            if not self.bank_view.text.strip():
                raise ValueError("bank_view.text пустой")
            kinds = {s.id: s.kind for s in self.sources}
            wrong = [i for i in self.bank_view.source_ids if kinds.get(i) != SourceKind.bank_analytics]
            if wrong:
                raise ValueError(f"bank_view ссылается не на аналитику банка: {wrong}")
        return self


# ---------------------------------------------------------------------------
# Лента и уточняющие вопросы
# ---------------------------------------------------------------------------


class FeedEntry(_Model):
    item_id: str
    title: str
    published_at: date
    source_kind: SourceKind
    score: float = Field(ge=0, le=1)
    reasons: list[str]
    topics: list[str] = []
    stale: bool = False


class Feed(_Model):
    company_id: str
    as_of: date
    entries: list[FeedEntry]
    filtered_out: int = Field(0, description="сколько материалов отсеяно как нерелевантные")


class FollowupAnswer(_Model):
    card_id: str
    question: str
    answer: str
    source_ids: list[str] = []
    sources: list[SourceRef] = []
    is_model_inference: bool = True
    verifier_issues: list[str] = Field([], description="замечания проверки цифр ответа, оставшиеся после повтора")
    trace_id: str
