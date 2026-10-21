"""Темы аналитики и карта «тема -> признак компании».

Используется эвристическим ранжированием, разбором материалов без LLM и eval.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from ml.schemas import CompanyProfile

EVENT_TYPES = [
    "monetary_policy", "inflation", "fuel_prices", "consumption", "labor_market",
    "fx_commodities", "macro_growth", "debt_market", "industry", "regulation", "company_news", "other",
]


@dataclass(frozen=True)
class Topic:
    key: str
    title: str
    keywords: tuple[str, ...]
    # Префиксы ОКВЭД, для которых тема важна напрямую.
    okved_prefixes: tuple[str, ...] = ()
    # Признаки профиля, которые делают тему важной: путь поля -> причина.
    features: dict[str, str] = field(default_factory=dict)
    # Тема касается всех крупных компаний хотя бы слабо.
    broad: bool = False


TOPICS: dict[str, Topic] = {t.key: t for t in [
    Topic(
        "key_rate", "Ключевая ставка и стоимость денег",
        ("ключев", "ставк", "денежно-кредит", "дкп", "совет директоров", "смягчени", "ужесточени"),
        features={
            "bank_products.loan": "есть кредитные продукты: ставка влияет на стоимость долга",
            "bank_products.leasing": "есть лизинг: ставка влияет на платежи",
            "bank_products.deposit": "есть депозиты: ставка влияет на доходность ликвидности",
        },
        broad=True,
    ),
    Topic(
        "bonds", "Долговой рынок, ОФЗ",
        ("офз", "облигац", "доходност", "кривая", "долгов", "купон", "размещени"),
        features={
            "bank_products.loan": "заёмное финансирование: доходности ОФЗ задают уровень ставок",
            "bank_products.deposit": "свободная ликвидность: альтернатива депозиту",
        },
    ),
    Topic(
        "inflation", "Инфляция",
        ("инфляц", "рост цен", "цены", "ценов", "потребительских цен", "дефлятор"),
        okved_prefixes=("47", "46.3", "46.4", "46.9", "10", "11"),
        broad=True,
    ),
    Topic(
        "fuel", "Топливо и бензин",
        ("бензин", "топлив", "дизел", "нефтепродукт", "азс", "биржев"),
        okved_prefixes=("46.71", "49.4", "49.3", "52.2", "47.3", "19"),
    ),
    Topic(
        "consumption", "Потребление и розница",
        ("потреблен", "розничн", "оборот розн", "спрос", "автокредит", "продаж авто", "потребкредит"),
        okved_prefixes=("47", "46.4", "46.3", "46.9", "45", "56", "55"),
    ),
    Topic(
        "labor", "Рынок труда и зарплаты",
        ("безработиц", "зарплат", "заработн", "рынок труда", "дефицит кадр", "занятост"),
        features={"size.workers_band": "трудоёмкий бизнес: зарплаты влияют на издержки"},
        broad=True,
    ),
    Topic(
        "fx_gold", "Валюта, золото, внешние рынки",
        ("золот", "доллар", "курс рубл", "фрс", "валют", "юан"),
        features={"geo.foreign_trade": "внешняя торговля: курс влияет на выручку и закупки"},
    ),
    Topic(
        "gdp_industry", "ВВП, промышленность, импорт",
        ("ввп", "промышленн", "импорт", "экспорт", "инвестиц", "строительств", "деловая активност",
         "застройщ", "новострой", "девелоп", "ипотек", "проектного финансир"),
        okved_prefixes=("41", "42", "43", "68", "23", "24", "25", "28", "29", "46.73"),
        features={"geo.foreign_trade": "импорт/экспорт: внешнеторговые потоки"},
        broad=True,
    ),
    Topic(
        "steel", "Металлургия и сталь",
        ("сталь", "сталелит", "металлург", "прокат", "металлопрокат", "чугун", "жрс"),
        okved_prefixes=("24", "46.72", "46.73", "41", "42", "25"),
    ),
]}


def okved_matches(code: str | None, prefixes: tuple[str, ...]) -> bool:
    """Иерархия ОКВЭД по цифрам: префикс `49.4` покрывает `49.41`, `47` покрывает `47.71`."""
    if not code:
        return False
    digits = code.replace(".", "")
    return any(digits.startswith(p.replace(".", "")) for p in prefixes)


@lru_cache(maxsize=None)
def _keyword_pattern(keywords: tuple[str, ...]) -> re.Pattern:
    """Ключ совпадает с началом слова: «ставк» не находится в «поставки», «доставка», «выставка»."""
    return re.compile(r"(?<![а-яёa-z])(?:" + "|".join(re.escape(k) for k in keywords) + ")", re.I)


def detect_topics(text: str, min_hits: int = 2) -> list[str]:
    """Темы по ключевым словам. Грубо, но без LLM и детерминированно."""
    found = []
    for t in TOPICS.values():
        hits = len(_keyword_pattern(t.keywords).findall(text))
        if hits >= min_hits:
            found.append((hits, t.key))
    return [k for _, k in sorted(found, reverse=True)]


def _feature_on(profile: CompanyProfile, path: str) -> bool:
    obj = profile
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    if path == "size.workers_band":
        return obj in ("101-250", "Более 251")
    return bool(obj)


INDUSTRY_WEIGHT = 0.6
FEATURE_WEIGHT = 0.45
BROAD_WEIGHT = 0.15


def topic_reasons(profile: CompanyProfile, topic_key: str) -> list[tuple[float, str, str]]:
    """Причины, почему тема важна компании: (вес, причина, путь признака)."""
    t = TOPICS[topic_key]
    out: list[tuple[float, str, str]] = []
    if okved_matches(profile.industry.okved_code, t.okved_prefixes):
        out.append((INDUSTRY_WEIGHT, f"отрасль компании ({profile.industry.okved_name}) напрямую в теме «{t.title}»", "industry.okved_code"))
    for path, reason in t.features.items():
        if _feature_on(profile, path):
            out.append((FEATURE_WEIGHT, reason, path))
    if t.broad and not out:
        out.append((BROAD_WEIGHT, f"тема «{t.title}» касается всех крупных компаний", "eligible_big"))
    return out


# Смежность отраслей: событие в отрасли-источнике косвенно затрагивает отрасли-получатели.
# Направление важно: (источник события, затронутые отрасли, механизм связи).
ADJACENCY: list[tuple[tuple[str, ...], tuple[str, ...], str]] = [
    (("24",), ("46.72",), "металлургия — поставщик металлотрейдеров: цены и выпуск металла прямо задают их закупки"),
    (("41", "42", "43"), ("23.5", "23.6", "24", "25.1", "46.72", "46.73"),
     "стройка — основной потребитель цемента, бетона, металла и стройматериалов"),
    (("24", "46.72"), ("41", "42", "43", "25", "46.73"),
     "металл — существенная статья затрат стройки и производителей металлоизделий"),
    (("19.2", "46.71", "47.3"), ("49.3", "49.4", "52.29", "50", "51"),
     "топливо — ключевая статья затрат перевозчиков"),
    (("47.73",), ("46.46",), "продажи аптек определяют закупки фармдистрибьюторов"),
    # Розница без топлива (47.3): топливная розница связана с перевозками, а не с оптом потребтоваров.
    (("47.1", "47.2", "47.4", "47.5", "47.6", "47.7", "47.8", "47.9"), ("46.3", "46.4", "46.9"),
     "спрос розницы определяет закупки оптовиков потребительских товаров"),
]


def adjacency_reason(company_okved: str | None, hints: tuple[str, ...]) -> str | None:
    """Механизм связи по ручной карте, если событие в отрасли, смежной с отраслью компании (и не в самой её отрасли)."""
    if not company_okved or not hints or okved_matches(company_okved, hints):
        return None
    for sources, targets, why in ADJACENCY:
        if any(okved_matches(h, sources) for h in hints) and okved_matches(company_okved, targets):
            return why
    return None


# Источник смежности: "flows" только платежи, "rules" только ручная карта,
# "flows+rules" платежи, а ручная карта как резерв для пар, по которым платежей мало.
ADJACENCY_SOURCE = os.getenv("ML_ADJACENCY", "flows+rules")


def adjacency_link(company_okved: str | None, hints: tuple[str, ...], source: str | None = None) -> tuple[str, str] | None:
    """(механизм, источник: payments | expert_rule) для события в смежной отрасли."""
    from ml.profile import flows

    source = source or ADJACENCY_SOURCE
    if not company_okved or not hints or okved_matches(company_okved, hints):
        return None
    if source in ("flows", "flows+rules"):
        links = [l for l in (flows.adjacency(company_okved, h) for h in hints) if l]
        if links:
            return max(links, key=lambda l: l.weight).mechanism, "payments"
        if source == "flows" or all(flows.covered(company_okved, h) for h in hints):
            return None
    why = adjacency_reason(company_okved, hints)
    return (why, "expert_rule") if why else None


# Каналы влияния события на компанию: общая таксономия экономических
# механизмов, не темы конкретного банка и не отрасли eval-набора. Признаки профиля — подсказка,
# через какие поля обычно проходит канал; проверка кода требует только, чтобы поле было заполнено.
@dataclass(frozen=True)
class ImpactChannelDef:
    key: str
    title: str
    meaning: str
    features: tuple[str, ...]


IMPACT_CHANNELS: dict[str, ImpactChannelDef] = {c.key: c for c in [
    ImpactChannelDef("cost_of_money", "Стоимость денег", "ставки по кредитам, лизингу, депозитам",
                     ("bank_products.loan", "bank_products.leasing", "bank_products.deposit")),
    ImpactChannelDef("credit_availability", "Доступность финансирования", "готовность банков и рынка давать деньги, требования к заёмщикам",
                     ("bank_products.loan", "bank_products.leasing", "bank_products.guarantee", "bank_products.factoring")),
    ImpactChannelDef("liquidity", "Ликвидность", "доходность свободных денег, оборачиваемость расчётов",
                     ("bank_products.deposit", "bank_products.rko", "bank_products.factoring")),
    ImpactChannelDef("input_costs", "Затраты на ресурсы", "цены сырья, материалов, топлива, услуг, которые компания покупает",
                     ("industry.okved_code", "industry.okved_name", "network.counterparty_sectors")),
    ImpactChannelDef("labor_costs", "Затраты на персонал", "зарплаты, дефицит кадров",
                     ("size.workers_band", "bank_products.payroll", "industry.okved_name")),
    ImpactChannelDef("demand", "Спрос", "спрос покупателей компании, объём продаж её отрасли",
                     ("industry.okved_code", "industry.okved_name", "network.counterparty_sectors")),
    ImpactChannelDef("commodity_prices", "Цены товаров отрасли", "цены того, чем торгует или что производит отрасль компании",
                     ("industry.okved_code", "industry.okved_name")),
    ImpactChannelDef("fx", "Валютный курс", "курс для выручки и закупок во внешней торговле",
                     ("geo.foreign_trade", "geo.fts_countries")),
    ImpactChannelDef("supply_chain", "Цепочка поставок", "сбои и изменения у поставщиков и покупателей",
                     ("network.counterparty_sectors", "network.top_counterparties", "industry.okved_name")),
    ImpactChannelDef("regulation", "Регулирование", "пошлины, налоги, требования к отрасли",
                     ("industry.okved_code", "industry.okved_name", "geo.foreign_trade")),
]}
