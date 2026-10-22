"""Лента под компанию: гибрид эвристики «тема -> признак» и BM25 по профилю, со свежестью.

LLM в ранжировании не используется: дорогие вызовы только на интерпретацию
выбранной пары «компания + материал».

Скор = ALPHA * эвристика + (1 - ALPHA) * BM25, умноженный на множитель свежести.
Эвристика объясняет, почему материал важен (причины идут в ленту и в промпт),
BM25 по тексту профиля ловит связи, которых нет в карте тем.
Аналитика банка старше STALE_AFTER_DAYS в ленту не попадает: она остаётся
в поиске для RAG, но с датой и пометкой «устарело».
"""

from __future__ import annotations

from dataclasses import dataclass

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

# Подобраны на eval-наборе; набор маленький, значения ориентировочные.
ALPHA = 0.6
MIN_SCORE = 0.25
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
# Подобрано на dev: 0,2 лучше на dev, test и 9 компаниях с ручной разметкой;
# больший вес растит test (LLM-разметка), но роняет ручную разметку.
BETA = 0.2
IMPORTANT_AT = 0.7
IMPORTANT_REASON = "событие крупного масштаба (оценка модели)"


@dataclass
class Scored:
    item: ParsedItem
    score: float  # порядок ленты: смесь личного скора и важности материала
    reasons: list[tuple[float, str, str]]
    heuristic: float = 0.0
    bm25: float = 0.0
    personal: float = 0.0  # личный скор компании: по нему отсев (MIN_SCORE)


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


def heuristic_reasons(p: CompanyProfile, item: ParsedItem) -> list[tuple[float, str, str]]:
    broad_w = topics.BROAD_WEIGHT if has_hooks(p) else BROAD_WEIGHT_NO_HOOKS
    reasons: list[tuple[float, str, str]] = []
    seen_paths: set[str] = set()
    for t in item.topics:
        for w, why, path in topic_reasons(p, t):
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


def score_item(p: CompanyProfile, item: ParsedItem, bm25: float = 0.0) -> Scored:
    reasons = heuristic_reasons(p, item)
    h = min(1.0, sum(r[0] for r in reasons))
    if bm25 >= BM25_REASON_MIN:
        reasons.append((0.0, "текст материала близок к описанию бизнеса компании", "industry.okved_name"))
    personal = round(min((ALPHA * h + (1 - ALPHA) * bm25) * _freshness_factor(item), 1.0), 3)
    blended = round(BETA * item.importance + (1 - BETA) * personal, 3)
    return Scored(item, blended, reasons, round(h, 3), round(bm25, 3), personal)


def feed_pool(items: list[ParsedItem]) -> tuple[list[ParsedItem], int]:
    """Пул ленты: без дубликатов и без устаревшей аналитики банка. Возвращает пул и число отброшенных."""
    reps = representatives(items)
    pool = [i for i in reps if not (i.source.kind == SourceKind.bank_analytics and is_stale(i.source.published_at))]
    return pool, len(items) - len(pool)


def score_all(p: CompanyProfile, items: list[ParsedItem]) -> tuple[list[Scored], int]:
    """Все материалы пула со скором, по убыванию. Второе значение: сколько отброшено до скоринга."""
    pool, dropped = feed_pool(items)
    if not pool:
        return [], dropped
    q = tokenize(profile_query(p))
    frozen = load_background()
    if frozen is not None:
        # Замороженные idf и avgdl: скор материала не зависит от состава пула (ml/ranking/bm25.py).
        scored = [score_item(p, i, frozen.score(q, tokenize(_item_text(i)))) for i in pool]
    else:
        raw = BM25Okapi([tokenize(_item_text(i)) for i in pool]).get_scores(q)
        top = max(raw) if len(raw) else 0.0
        scored = [score_item(p, i, float(raw[k] / top) if top > 0 else 0.0) for k, i in enumerate(pool)]
    return sorted(scored, key=lambda s: -s.score), dropped


def rank(p: CompanyProfile, items: list[ParsedItem], limit: int = 20) -> tuple[list[Scored], int]:
    scored, dropped = score_all(p, items)
    if rerank.enabled():  # эксперимент: оценка модели для топ-N и материалов у порога
        scored = rerank.rerank(p, scored, MIN_SCORE, BETA)
    kept = [s for s in scored if s.personal >= MIN_SCORE]
    return kept[:limit], dropped + len(scored) - len(kept)


def feed(p: CompanyProfile, items: list[ParsedItem], limit: int = 20) -> Feed:
    kept, dropped = rank(p, items, limit)
    entries = [
        FeedEntry(
            item_id=s.item.item_id, title=s.item.title, published_at=s.item.source.published_at,
            source_kind=s.item.source.kind, score=s.score,
            reasons=[r[1] for r in s.reasons][:3] + ([IMPORTANT_REASON] if s.item.importance >= IMPORTANT_AT else []),
            topics=s.item.topics, stale=is_stale(s.item.source.published_at),
        )
        for s in kept
    ]
    return Feed(company_id=p.company_id, as_of=settings.as_of, entries=entries, filtered_out=dropped)


def topic_titles(keys: list[str]) -> list[str]:
    return [TOPICS[k].title for k in keys if k in TOPICS]


__all__ = ["ALPHA", "MIN_SCORE", "STALE_AFTER_DAYS", "feed", "rank", "score_all", "score_item", "heuristic_reasons"]
