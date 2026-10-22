"""Лента под компанию: гибрид эвристики «тема -> признак» и BM25 по профилю, со свежестью.

LLM в ранжировании не используется: дорогие вызовы только на интерпретацию
выбранной пары «компания + материал».

Скор = ALPHA * эвристика + (1 - ALPHA) * BM25, умноженный на множитель свежести.
Эвристика объясняет, почему материал важен (причины идут в ленту и в промпт),
BM25 по тексту профиля ловит связи, которых нет в карте тем.

Два правила по масштабу события (scope из ml/ingest/importance.py; разбор ошибок на реальных новостях в STATUS_HANDOFF.md):
- отраслевая новость или событие компании (scope sector/company) без связи с отраслью компании (ОКВЭД, смежность,
  отраслевая тема) не получает причин по темам: широкая тема от разбора («gdp_industry») иначе уводит её всем.
  Аналитика банка под правило не попадает: это обзоры рынков для клиентов банка, а не отраслевые события;
- событие всей экономики (scope economy) с важностью от ECONOMY_MIN_IMPORTANCE проходит отсев у всех компаний:
  прогноз инфляции или ставки нужен и компании без продуктов банка. Кроме валютных событий (каналы влияния только
  курс и ликвидность): по инструкции разметки курс касается компании через внешнюю торговлю;
- событие прямо в отрасли компании (okved_hints материала совпали с ОКВЭД) проходит отсев всегда
  (IN_SECTOR_KEEPS; v2: 26 из 26 таких пар релевантны, ручная разметка: 13 из 13).
Поверх модели пары действует только правило про отрасль компании; пол для макрособытий входит в эвристику,
а модель может его перевесить (ручная разметка v2 не считает инфляцию и ОФЗ релевантными компании без продуктов банка).

Если есть модель пары (`ml/ranking/pair_model.json`, обучение `python -m ml.ranking.pair_model train`), личный скор
и порядок дают модель и эвристика вместе: логистическая регрессия по признакам пары (масштаб события, совпадение
отрасли и смежность, причины по темам и каналам, BM25, важность) с весами по разметке, смешанная с эвристикой
(веса смеси подобраны перекрёстной проверкой). Причины в ленте из эвристики. ML_RANK_MODEL=0 возвращает эвристику.
Аналитика банка старше STALE_AFTER_DAYS в ленту не попадает: она остаётся
в поиске для RAG, но с датой и пометкой «устарело».

Два уровня ленты (FEED_TIERS_PLAN.md): основная лента `main` (личный скор от MAIN_SCORE или событие в отрасли компании) и
блок «может быть интересно» `maybe` (личный скор от MIN_SCORE до MAIN_SCORE); ниже MIN_SCORE материал отсеивается.
Порядок: сначала main, затем maybe, внутри уровня по порядку ленты.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from ml.config import settings
from ml.ingest.analytics import STALE_AFTER_DAYS, is_stale
from ml.ingest.dedup import representatives
from ml.ranking import rerank
from ml.ranking.bm25 import load_background
from ml.profile.build_profile import normalize_okved
from ml.retrieval.index import tokenize
from ml.schemas import CompanyProfile, Feed, FeedEntry, ParsedItem, SourceKind
from ml import topics
from ml.topics import TOPICS, okved_matches, topic_reasons

# Подобраны на eval-наборе (ml/eval/analysis.md); набор маленький, значения ориентировочные.
# 0,7 и порог 0,3: на реальных новостях (ml/eval/real_news.jsonl) меньше ложных срабатываний на dev и отложенной части.
ALPHA = 0.7
MIN_SCORE = 0.3
# Порог основной ленты по личному скору (FEED_TIERS_PLAN.md; объявлен до проверки на real2, ml/eval/PROTOCOL.md).
# Материалы между MIN_SCORE и MAIN_SCORE идут в блок «может быть интересно». Поднимать MIN_SCORE вместо этого нельзя:
# в режиме с моделью личный скор нормируется на MIN_SCORE (blend_scores).
MAIN_SCORE = float(os.getenv("ML_FEED_MAIN_SCORE", "0.40"))
# Событие всей экономики, прошедшее отсев только по полу ECONOMY_MIN_IMPORTANCE (личный скор ниже MAIN_SCORE): "maybe"
# или "main". Ручная разметка real и ручной срез v2 такие пары чаще считают ненужными.
MACRO_FLOOR_TIER = os.getenv("ML_FEED_MACRO_TIER", "maybe")
MAIN_LIMIT, MAYBE_LIMIT = 20, 10
WEAK_REASON = "слабая связь с профилем компании"
# Вес широкой темы (ставка, инфляция, ВВП, рынок труда) для компаний без «зацепок»:
# нет продуктов банка и внешней торговли. Иначе их лента почти пустая.
BROAD_WEIGHT_NO_HOOKS = 0.35
BM25_REASON_MIN = 0.5
# Событие про конкретные отрасли (okved_hints), но не про отрасль компании: тематические связи слабее.
OFF_SECTOR_DAMP = 0.5
# Событие в смежной отрасли (ml.topics.ADJACENCY): косвенная, но прямая по механизму связь.
ADJACENCY_WEIGHT = 0.35
# Приор общей важности материала (ParsedItem.importance, ml/ingest/importance.py). Отсев идёт по личному скору,
# порядок ленты по смеси: BETA * importance + (1 - BETA) * личный скор. 0: порядок только по личному скору.
# ml/eval/analysis.md, «Реальные новости»: на реальных новостях nDCG@5 растёт до BETA 0,5-0,65, на ручной разметке v2
# выше 0,35 падает; 0,35 не хуже прежнего ни на одном срезе.
BETA = 0.35
# Откуда причины «материал -> признак профиля» (GAIN_PLAN.md, «новость -> отрасль»): "topics" ручная карта ml.topics.TOPICS,
# "channels" каналы влияния, которые модель оценивает по тексту материала, "both" обе.
SIGNALS = os.getenv("ML_RANK_SIGNALS", "topics")
# Материал вне экономики по оценке модели (scope="none"): личный скор умножается на этот множитель. 1: не влияет, 0: исключить.
NONE_SCOPE_DAMP = 0.0
# Событие всей экономики с важностью не ниже порога проходит отсев (личный скор не ниже MIN_SCORE). None: правило выключено.
ECONOMY_MIN_IMPORTANCE: float | None = 0.5
# Отраслевое событие без связи с отраслью компании: причины по темам не считаются (False: как раньше).
NARROW_NEEDS_SECTOR = True
# Событие прямо в отрасли компании (совпадение ОКВЭД с okved_hints) проходит отсев.
IN_SECTOR_KEEPS = True
# Пол для макро только у компаний с зацепкой в профиле (продукты банка, ВЭД): МРОТ и региональную инфляцию человек
# не считает релевантными компании без зацепок (ручная проверка real и ручной срез v2).
ECONOMY_FLOOR_NEEDS_HOOK = os.getenv("ML_ECONOMY_FLOOR_HOOK", "0") == "1"
IMPORTANT_AT = 0.7
IMPORTANT_REASON = "событие крупного масштаба (оценка модели)"
ECONOMY_REASON = "событие касается всей экономики (оценка модели)"
MODEL_PATH = Path(__file__).with_name("pair_model.json")
USE_MODEL = os.getenv("ML_RANK_MODEL", "1") != "0"


@dataclass
class Scored:
    item: ParsedItem
    score: float  # порядок ленты: смесь личного скора и важности материала
    reasons: list[tuple[float, str, str]]
    heuristic: float = 0.0
    bm25: float = 0.0
    personal: float = 0.0  # личный скор компании: по нему отсев (MIN_SCORE) и уровень ленты (MAIN_SCORE)
    tier: str = ""  # "main" | "maybe" (заполняет rank), "" у материала, не прошедшего отсев


def profile_query(p: CompanyProfile) -> str:
    """Текстовый запрос из признаков профиля (не из сырых полей CSV)."""
    parts = [p.industry.okved_name or "", " ".join(p.industry.sector_tags)]
    prod = {"loan": "кредит ставка", "leasing": "лизинг", "deposit": "депозит доходность", "guarantee": "гарантии"}
    parts += [v for k, v in prod.items() if getattr(p.bank_products, k)]
    if p.geo.foreign_trade:
        parts.append("импорт экспорт курс валюта " + " ".join(p.geo.fts_countries))
    parts += p.network.counterparty_sectors[:3]
    return " ".join(parts)


def has_hooks(p: CompanyProfile) -> bool:
    bp = p.bank_products
    return bool(bp.loan or bp.leasing or bp.deposit or bp.guarantee or bp.factoring or p.geo.foreign_trade)


def _item_text(item: ParsedItem) -> str:
    facts = " ".join(f.text for f in item.facts)
    return f"{item.title} {item.summary or ''} {facts}"


def signal_reasons(p: CompanyProfile, item: ParsedItem) -> list[tuple[float, str, str]]:
    """Причины «материал -> признак профиля»: по ручной карте тем (topics), по каналам влияния от модели (channels) или обе."""
    out: list[tuple[float, str, str]] = []
    if SIGNALS in ("topics", "both"):
        for t in item.topics:
            out += topic_reasons(p, t)
    if SIGNALS in ("channels", "both"):
        out += topics.channel_reasons(p, item.channels, item.scope)
    return sorted(out, key=lambda r: -r[0])


def heuristic_reasons(p: CompanyProfile, item: ParsedItem) -> list[tuple[float, str, str]]:
    broad_w = topics.BROAD_WEIGHT if has_hooks(p) else BROAD_WEIGHT_NO_HOOKS
    reasons: list[tuple[float, str, str]] = []
    seen_paths: set[str] = set()
    for w, why, path in signal_reasons(p, item):
        if path in seen_paths:
            continue
        seen_paths.add(path)
        reasons.append((broad_w if path == "eligible_big" else w, why, path))
    # Подсказки от LLM бывают в формате «41.20»; приводим к тому же виду, что коды в профиле («41.2»).
    hints = tuple(h for h in (normalize_okved(x) for x in item.okved_hints) if h)
    in_sector = okved_matches(p.industry.okved_code, hints) if hints else None
    adjacent = topics.adjacency_link(p.industry.okved_code, hints) if in_sector is False else None
    if adjacent:
        why, src = adjacent
        note = "" if src == "payments" else " (экспертное правило)"
        reasons.append((ADJACENCY_WEIGHT, f"событие в смежной отрасли: {why}{note}", "industry.okved_code"))
    elif in_sector is False:
        reasons = [(w * OFF_SECTOR_DAMP, why, path) for w, why, path in reasons]
    if in_sector and "industry.okved_code" not in seen_paths:
        reasons.append((0.5, "событие касается отрасли компании напрямую", "industry.okved_code"))
    return sorted(reasons, key=lambda r: -r[0])


def _freshness_factor(item: ParsedItem) -> float:
    return 0.6 + 0.4 * item.freshness


SCOPES = ("economy", "multi_sector", "sector", "company")


def pair_features(p: CompanyProfile, item: ParsedItem, bm25: float, h: float) -> dict[str, float]:
    """Признаки пары для модели (ml/ranking/pair_model.py). Только то, что есть в рабочем пути: профиль и сигналы материала."""
    hints = tuple(x for x in (normalize_okved(x) for x in item.okved_hints) if x)
    in_sector = okved_matches(p.industry.okved_code, hints) if hints else None
    adjacent = bool(topics.adjacency_link(p.industry.okved_code, hints)) if in_sector is False else False
    t_paths = [r[2] for t in item.topics for r in topic_reasons(p, t)]
    c_paths = [r[2] for r in topics.channel_reasons(p, item.channels, item.scope) if r[2] != "eligible_big"]
    scope = item.scope or "sector"
    econ, narrow = scope == "economy", scope in ("sector", "company")
    topic_industry = "industry.okved_code" in t_paths
    f = {f"scope_{s}": float(scope == s) for s in SCOPES}
    f.update({
        "importance": item.importance,
        "in_sector": float(bool(in_sector)),
        "adjacent": float(adjacent),
        "off_sector": float(in_sector is False and not adjacent),
        "topic_industry": float(topic_industry),
        "topic_features": min(2.0, sum(x not in ("industry.okved_code", "eligible_big") for x in t_paths)),
        "topic_broad": float("eligible_big" in t_paths),
        "channel_hooks": min(2.0, float(len(c_paths))),
        "fx_trade": float("fx" in item.channels and bool(p.geo.foreign_trade)),
        "bm25": bm25,
        "heuristic": h,
        "freshness": item.freshness,
        "hooks": float(has_hooks(p)),
        "econ_x_hooks": float(econ and has_hooks(p)),
        "econ_x_channel": float(econ and bool(c_paths)),
        "narrow_x_match": float(narrow and bool(in_sector or adjacent or topic_industry)),
        "narrow_x_nomatch": float(narrow and not (in_sector or adjacent or topic_industry)),
    })
    return f


@lru_cache(maxsize=1)
def load_model() -> dict | None:
    """Веса модели пары или None (нет файла или ML_RANK_MODEL=0): тогда лента по эвристике."""
    if not USE_MODEL or not MODEL_PATH.exists():
        return None
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


_override: list[dict | None] = []


def set_model(model: dict | None) -> None:
    """Подменить модель в процессе (eval и обучение): None — эвристика. Вызов без аргумента снимает подмену."""
    _override[:] = [model]


def clear_model() -> None:
    _override.clear()


def active_model() -> dict | None:
    return _override[0] if _override else load_model()


def _logit(w: dict[str, float], f: dict[str, float], mean: dict[str, float], std: dict[str, float]) -> float:
    z = w.get("bias", 0.0) + sum(w[k] * (f[k] - mean[k]) / std[k] for k in w if k != "bias")
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def model_scores(model: dict, f: dict[str, float]) -> tuple[float, float]:
    """(P(оценка >= 1), P(оценка = 2)) по модели пары."""
    return (_logit(model["w_rel"], f, model["mean"], model["std"]), _logit(model["w_top"], f, model["mean"], model["std"]))


FX_ONLY_CHANNELS = {"fx", "liquidity"}


def narrow_unlinked(item: ParsedItem, reasons: list[tuple[float, str, str]]) -> bool:
    """Отраслевая новость без связи с отраслью компании (правило NARROW_NEEDS_SECTOR)."""
    return (NARROW_NEEDS_SECTOR and item.scope in ("sector", "company") and item.source.kind != SourceKind.bank_analytics
            and not any(r[2] == "industry.okved_code" for r in reasons))


def economy_floor(p: CompanyProfile, item: ParsedItem) -> bool:
    """Событие всей экономики проходит отсев у компании (правило ECONOMY_MIN_IMPORTANCE)."""
    if ECONOMY_MIN_IMPORTANCE is None or item.scope != "economy" or item.importance < ECONOMY_MIN_IMPORTANCE:
        return False
    if ECONOMY_FLOOR_NEEDS_HOOK and not has_hooks(p):
        return False
    fx_only = bool(item.channels) and set(item.channels) <= FX_ONLY_CHANNELS
    return not fx_only or bool(p.geo.foreign_trade)


def in_sector(p: CompanyProfile, item: ParsedItem) -> bool:
    hints = tuple(x for x in (normalize_okved(x) for x in item.okved_hints) if x)
    return bool(hints) and okved_matches(p.industry.okved_code, hints)


def hard_keep(p: CompanyProfile, item: ParsedItem) -> bool:
    """Материал проходит отсев независимо от скора: макрособытие или событие в отрасли компании."""
    if item.scope == "none":
        return False
    return economy_floor(p, item) or (IN_SECTOR_KEEPS and in_sector(p, item))


def rule_personal(p: CompanyProfile, item: ParsedItem, reasons: list[tuple[float, str, str]], h: float, bm25: float) -> float:
    """Личный скор эвристики с правилами по масштабу события (без модели)."""
    if narrow_unlinked(item, reasons):
        h = 0.0
    damp = NONE_SCOPE_DAMP if item.scope == "none" else 1.0
    personal = min((ALPHA * h + (1 - ALPHA) * bm25) * _freshness_factor(item) * damp, 1.0)
    if hard_keep(p, item):
        personal = max(personal, MIN_SCORE)
    return personal


def blend_scores(model: dict, p_rel: float, p_top: float, rule: float, importance: float) -> tuple[float, float]:
    """(личный скор, порядок) смеси модели и эвристики. Личный скор в шкале эвристики: проходит отсев при >= MIN_SCORE,
    то есть когда взвешенная сумма «P(релевантно) / порог модели» и «скор эвристики / MIN_SCORE» не меньше 1."""
    w, beta, mix = model.get("w_model", 1.0), model.get("beta", 0.0), model.get("order_mix", 1.0)
    vote = w * p_rel / model["threshold"] + (1 - w) * rule / MIN_SCORE
    personal = min(1.0, MIN_SCORE * vote)
    order = mix * (p_rel + p_top) / 2 + (1 - mix) * rule
    return personal, beta * importance + (1 - beta) * order


def score_item(p: CompanyProfile, item: ParsedItem, bm25: float = 0.0) -> Scored:
    reasons = heuristic_reasons(p, item)
    h = min(1.0, sum(r[0] for r in reasons))
    rule = rule_personal(p, item, reasons, h, bm25)
    if narrow_unlinked(item, reasons):
        reasons = []  # темы не считаются: и причинами в ленте их не показываем
    if bm25 >= BM25_REASON_MIN:
        reasons.append((0.0, "текст материала близок к описанию бизнеса компании", "industry.okved_name"))
    if item.scope == "economy" and (economy_floor(p, item) or not reasons):
        reasons.append((0.0, ECONOMY_REASON, "eligible_big"))
    model = active_model()
    if model is not None and item.scope != "none":
        p_rel, p_top = model_scores(model, pair_features(p, item, bm25, h))
        personal, blended = blend_scores(model, p_rel, p_top, rule, item.importance)
        if IN_SECTOR_KEEPS and in_sector(p, item):
            personal = max(personal, MIN_SCORE)
        return Scored(item, round(blended, 3), reasons, round(h, 3), round(bm25, 3), round(personal, 3))
    beta = model.get("beta", BETA) if model is not None else BETA
    personal = round(rule, 3)
    blended = round(beta * item.importance + (1 - beta) * personal, 3)
    return Scored(item, blended, reasons, round(h, 3), round(bm25, 3), personal)


def min_score() -> float:
    """Порог отсева по личному скору (у смеси с моделью личный скор в той же шкале)."""
    return MIN_SCORE


def feed_pool(items: list[ParsedItem]) -> tuple[list[ParsedItem], int]:
    """Пул ленты: без дубликатов и без устаревшей аналитики банка. Возвращает пул и число отброшенных."""
    reps = representatives(items)
    pool = [i for i in reps if not (i.source.kind == SourceKind.bank_analytics and is_stale(i.source.published_at))]
    return pool, len(items) - len(pool)


def bm25_scores(p: CompanyProfile, pool: list[ParsedItem]) -> list[float]:
    q = tokenize(profile_query(p))
    frozen = load_background()
    if frozen is not None:
        # Замороженные idf и avgdl: скор материала не зависит от состава пула (ml/ranking/bm25.py).
        return [frozen.score(q, tokenize(_item_text(i))) for i in pool]
    raw = BM25Okapi([tokenize(_item_text(i)) for i in pool]).get_scores(q)
    top = max(raw) if len(raw) else 0.0
    return [float(raw[k] / top) if top > 0 else 0.0 for k in range(len(pool))]


def score_all(p: CompanyProfile, items: list[ParsedItem]) -> tuple[list[Scored], int]:
    """Все материалы пула со скором, по убыванию. Второе значение: сколько отброшено до скоринга."""
    pool, dropped = feed_pool(items)
    if not pool:
        return [], dropped
    scored = [score_item(p, i, b) for i, b in zip(pool, bm25_scores(p, pool))]
    return sorted(scored, key=lambda s: -s.score), dropped


def tier_of(p: CompanyProfile, s: Scored, main_score: float | None = None, macro_tier: str | None = None) -> str | None:
    """Уровень ленты: "main", "maybe" или None (отсев). Событие в отрасли компании всегда main."""
    main_score = MAIN_SCORE if main_score is None else main_score
    macro_tier = macro_tier or MACRO_FLOOR_TIER
    if s.personal < min_score():
        return None
    if s.personal >= main_score or (IN_SECTOR_KEEPS and in_sector(p, s.item)):
        return "main"
    return "main" if macro_tier == "main" and economy_floor(p, s.item) else "maybe"


def rank(p: CompanyProfile, items: list[ParsedItem], limit: int = MAIN_LIMIT, maybe_limit: int = MAYBE_LIMIT) -> tuple[list[Scored], int]:
    """Сначала до `limit` материалов основной ленты, затем до `maybe_limit` блока «может быть»; у каждого заполнен `tier`."""
    scored, dropped = score_all(p, items)
    if rerank.enabled():  # эксперимент T13: оценка модели для топ-N и материалов у порога
        scored = rerank.rerank(p, scored, MIN_SCORE, BETA)
    main, maybe = [], []
    for s in scored:
        s.tier = tier_of(p, s) or ""
        {"main": main, "maybe": maybe}.get(s.tier, []).append(s)
    return main[:limit] + maybe[:maybe_limit], dropped + len(scored) - len(main) - len(maybe)


def feed(p: CompanyProfile, items: list[ParsedItem], limit: int = MAIN_LIMIT, maybe_limit: int = MAYBE_LIMIT, tier: str = "all") -> Feed:
    """tier="main": только основная лента; "all": основная и блок «может быть интересно»."""
    kept, dropped = rank(p, items, limit, maybe_limit if tier == "all" else 0)
    entries = [
        FeedEntry(
            item_id=s.item.item_id, title=s.item.title, published_at=s.item.source.published_at,
            source_kind=s.item.source.kind, score=s.score, tier=s.tier,
            reasons=([WEAK_REASON] if s.tier == "maybe" else []) + [r[1] for r in s.reasons][:3]
            + ([IMPORTANT_REASON] if s.item.importance >= IMPORTANT_AT else []),
            topics=s.item.topics, stale=is_stale(s.item.source.published_at),
        )
        for s in kept
    ]
    return Feed(company_id=p.company_id, as_of=settings.as_of, entries=entries, filtered_out=dropped,
                maybe_count=sum(e.tier == "maybe" for e in entries))


def topic_titles(keys: list[str]) -> list[str]:
    return [TOPICS[k].title for k in keys if k in TOPICS]


__all__ = ["ALPHA", "MIN_SCORE", "MAIN_SCORE", "tier_of", "STALE_AFTER_DAYS", "feed", "rank", "score_all", "score_item", "heuristic_reasons", "min_score"]
