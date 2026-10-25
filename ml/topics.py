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


# Подтемы широкой темы для сопоставления материала с фрагментами банка: «Валюта, золото, внешние рынки» объединяет курс
# рубля и золото, и обзор рынка золота считался позицией банка по новости о курсе (карточки real, interpret@v12).
# Подтема определяется по ключам в тексте; тема разбора («fx_gold») и лента не меняются: ключи разбора и признаки модели
# ленты остаются прежними. Подтема fx — курс рубля: «доллар» в обзорах золота это индекс доллара США, а «юань» там —
# интернационализация юаня через золото, к курсу рубля не относится.
FACETS: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "fx_gold": {
        "fx": ("Валютный курс рубля", ("рубл", "валютн", "курс доллар", "курсы доллар", "курс евро", "курс валют")),
        "gold": ("Золото и драгметаллы", ("золот", "драгмет", "etf")),
    },
}


def _facet_hits(topic: str, text: str) -> dict[str, int]:
    """Подтемы, которые преобладают в тексте: не меньше половины ключей самой частой (обзор золота с упоминанием юаня — золото)."""
    hits = {k: len(_keyword_pattern(kw).findall(text)) for k, (_, kw) in FACETS.get(topic, {}).items()}
    top = max(hits.values(), default=0)
    return {k: n for k, n in hits.items() if n and n * 2 >= top}


def topic_facets(topics: list[str], text: str) -> set[str]:
    """Темы с подтемами («fx_gold:fx», «fx_gold:gold»); тема без найденной подтемы остаётся целой («fx_gold»)."""
    out: set[str] = set()
    for t in topics:
        out |= {f"{t}:{k}" for k in _facet_hits(t, text)} or {t}
    return out


def facets_overlap(a: set[str], b: set[str]) -> bool:
    """Общая тема: совпадение подтем; целая тема совпадает с любой своей подтемой."""
    if a & b:
        return True
    whole_a, whole_b = {x for x in a if ":" not in x}, {x for x in b if ":" not in x}
    return any(x.split(":")[0] in whole_b for x in a) or any(x.split(":")[0] in whole_a for x in b)


def facet_titles(topics: list[str], text: str) -> list[str]:
    """Названия тем для поисковых запросов к аналитике банка: подтема вместо широкой темы, если она определена."""
    out = []
    for t in topics:
        found = [FACETS[t][k][0] for k in _facet_hits(t, text)]
        out += found or ([TOPICS[t].title] if t in TOPICS else [])
    return out


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
# Откуда тема даёт причину «отрасль компании напрямую в теме» (ручная проверка real, analysis.md): "all" — по okved_prefixes
# всех тем; "narrow" — кроме широких тем (broad=True: ставка, инфляция, труд, ВВП), где список отраслей не следует из текста
# новости; "none" — ни у одной темы: отраслевая связь только по okved_hints разбора самой новости и смежности.
TOPIC_OKVED = os.getenv("ML_TOPIC_OKVED", "narrow")
FEATURE_WEIGHT = 0.45
BROAD_WEIGHT = 0.15


def topic_reasons(profile: CompanyProfile, topic_key: str) -> list[tuple[float, str, str]]:
    """Причины, почему тема важна компании: (вес, причина, путь признака)."""
    t = TOPICS[topic_key]
    out: list[tuple[float, str, str]] = []
    by_topic = TOPIC_OKVED == "all" or (TOPIC_OKVED == "narrow" and not t.broad)
    if by_topic and okved_matches(profile.industry.okved_code, t.okved_prefixes):
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


# Признаки профиля, через которые канал касается компании напрямую (флаги продуктов, ВЭД, численность).
# Отраслевые каналы (затраты, спрос, цены товаров, цепочка поставок, регулирование) идут через okved_hints материала
# и смежность по платежам в ленте, а не через этот словарь. РКО и зарплатный проект есть почти у всех: связи не доказывают.
CHANNEL_HOOKS = {
    "bank_products.loan": "есть кредитные продукты",
    "bank_products.leasing": "есть лизинг",
    "bank_products.deposit": "есть депозиты",
    "bank_products.guarantee": "есть банковские гарантии",
    "bank_products.factoring": "есть факторинг",
    "geo.foreign_trade": "есть внешняя торговля",
    "size.workers_band": "трудоёмкий бизнес",
}


def channel_reasons(profile: CompanyProfile, channels: list[str], scope: str | None) -> list[tuple[float, str, str]]:
    """Причины по каналам влияния материала (оценка модели, ml/ingest/importance.py) вместо ручной карты TOPICS:
    (вес, причина, путь признака). Без зацепок и при scope=economy: слабая широкая причина."""
    out: list[tuple[float, str, str]] = []
    for key in channels:
        ch = IMPACT_CHANNELS.get(key)
        if ch is None:
            continue
        for path in ch.features:
            if path in CHANNEL_HOOKS and _feature_on(profile, path):
                out.append((FEATURE_WEIGHT, f"{ch.title}: {CHANNEL_HOOKS[path]}", path))
    if not out and scope == "economy":
        out.append((BROAD_WEIGHT, "событие касается всей экономики (оценка модели)", "eligible_big"))
    return out
