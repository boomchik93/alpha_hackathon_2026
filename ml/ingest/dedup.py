"""Схлопывание дубликатов: близкие по словам материалы в пределах нескольких дней."""

from __future__ import annotations

from ml.retrieval.index import tokenize
from ml.schemas import ParsedItem

MAX_DAYS_APART = 1
THRESHOLD = 0.4


def _sig(item: ParsedItem) -> set[str]:
    facts = " ".join(f.text for f in item.facts)
    return set(tokenize(f"{item.title} {item.summary or ''} {facts}"))


def overlap(a: set[str], b: set[str]) -> float:
    """Коэффициент перекрытия: короткая заметка целиком внутри длинной даёт 1."""
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def assign_clusters(items: list[ParsedItem]) -> list[ParsedItem]:
    """Проставляет dedup_cluster. Представитель кластера: материал с большим числом фактов."""
    sigs = [_sig(i) for i in items]
    cluster = list(range(len(items)))
    for i in range(len(items)):
        for j in range(i):
            if cluster[j] != j:
                continue
            days = abs((items[i].source.published_at - items[j].source.published_at).days)
            if items[i].event_type == items[j].event_type and days <= MAX_DAYS_APART and overlap(sigs[i], sigs[j]) >= THRESHOLD:
                cluster[i] = j
                break
    return [it.model_copy(update={"dedup_cluster": f"c_{items[cluster[k]].item_id}"}) for k, it in enumerate(items)]


def representatives(items: list[ParsedItem]) -> list[ParsedItem]:
    best: dict[str, ParsedItem] = {}
    for it in items:
        key = it.dedup_cluster or it.item_id
        if key not in best or len(it.facts) > len(best[key].facts):
            best[key] = it
    return list(best.values())
