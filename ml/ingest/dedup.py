"""Схлопывание дубликатов: близкие по словам материалы в пределах нескольких дней."""

from __future__ import annotations

import re

from ml.retrieval.index import _WORD, STOP, _morph, tokenize
from ml.schemas import ParsedItem, SourceKind

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


# Одно событие разными материалами («ЦБ сохранил ставку», «ЦБ рассматривал только сохранение», «реакция рынка на решение
# по ставке»): тексты перекрываются слабее дубликатов (0,27-0,49 по словам), а по словам всего текста склеиваются и разные
# события одной темы. Поэтому нужны тот же тип события и главная тема, не дальше дня, не меньше двух общих слов заголовка
# и ещё доля от EVENT_THRESHOLD: общих слов от короткого заголовка или перекрытие текстов (`overlap`). Второе ловит
# «ЦБ сопроводил сообщение о ключевой ставке изображением кота»: в заголовке 2 общих слова из 7. 0,3 выбран на свежих
# новостях 23-24.10: при 0,25 по заголовку склеиваются «золото» с «курсом доллара»; на real (16-22.10) и v2 ложных склеек
# при 0,3 нет (ml/eval/analysis.md).
EVENT_THRESHOLD = 0.3
EVENT_MIN_SHARED = 2
EVENT_MAX_DAYS_APART = 1
_NAMES = re.compile(r"\bЦБ( РФ)?\b|\bБанк[а-я]* России\b|\bЦентробанк[а-я]*\b", re.I)
# Слова, которые есть почти в любом заголовке об экономике и событие не различают.
_GENERIC = {"россия", "росси", "рф", "банк", "год", "году", "млрд", "млн", "руб", "рубля"}


def _norm(word: str) -> str:
    """Лемма (pymorphy3, если установлен), иначе первые 5 букв: «ставку», «ставке», «ставки» дают одно слово."""
    morph = _morph()
    return morph.parse(word)[0].normal_form if morph is not None and re.match(r"[а-яё]", word) else word[:5]


def title_terms(item: ParsedItem) -> set[str]:
    words = (m.group().lower() for m in _WORD.finditer(_NAMES.sub(" ", item.title)))
    return {n for w in words if w not in STOP and len(w) > 1 and (n := _norm(w)) not in _GENERIC}


def same_event(a: ParsedItem, b: ParsedItem, ta: set[str], tb: set[str]) -> bool:
    if a.event_type != b.event_type or a.event_type == "other" or not a.topics or not b.topics or a.topics[0] != b.topics[0]:
        return False
    if abs((a.source.published_at - b.source.published_at).days) > EVENT_MAX_DAYS_APART:
        return False
    shared = len(ta & tb)
    if shared < EVENT_MIN_SHARED:
        return False
    return shared / min(len(ta), len(tb)) >= EVENT_THRESHOLD or overlap(_sig(a), _sig(b)) >= EVENT_THRESHOLD


def event_groups(items: list[ParsedItem]) -> dict[str, str]:
    """item_id новости -> item_id первой новости её события в порядке `items` (аналитика банка не группируется).
    Новость присоединяется к событию, если совпадает хотя бы с одним его материалом."""
    news = [i for i in items if i.source.kind != SourceKind.bank_analytics]
    terms = {i.item_id: title_terms(i) for i in news}
    groups: list[list[ParsedItem]] = []
    out: dict[str, str] = {}
    for it in news:
        g = next((g for g in groups if any(same_event(m, it, terms[m.item_id], terms[it.item_id]) for m in g)), None)
        if g is None:
            groups.append(g := [])
        g.append(it)
        out[it.item_id] = g[0].item_id
    return out
