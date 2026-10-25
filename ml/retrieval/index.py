"""Поиск по фрагментам аналитики банка и новостям: BM25, при наличии e5 гибрид (RRF)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Okapi

from ml.ingest.analytics import Chunk, load_chunks
from ml.retrieval.embedder import Embedder, cosine

_WORD = re.compile(r"[a-zа-яё0-9]+", re.I)
STOP = set("и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между".split())


# "lemma": лемматизация pymorphy3, на eval ленты хуже stem6;
# "stem6": грубый стемминг по первым 6 символам. Без pymorphy3 откатываемся на stem6.
TOKENIZER = os.getenv("ML_TOKENIZER", "stem6")


@lru_cache(maxsize=1)
def _morph():
    try:
        import pymorphy3
    except ImportError:
        return None
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def _lemma(word: str) -> str:
    m = _morph()
    if m is None or not re.match(r"[а-яё]", word):
        return word[:6] if m is None else word
    return m.parse(word)[0].normal_form


def tokenize(text: str) -> list[str]:
    words = [w for w in (m.group().lower() for m in _WORD.finditer(text)) if w not in STOP and len(w) > 1]
    if TOKENIZER == "lemma" and _morph() is not None:
        return [_lemma(w) for w in words]
    # Грубый стемминг: первые 6 символов слова. Для русского даёт заметный выигрыш над точным совпадением.
    return [w[:6] for w in words]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float
    dense: float | None = None


_CAPTION = re.compile(r"Рис\.\s*\d|Источник(?:и)?\s*:|Примечание\s*:|Табл(?:ица|\.)\s*\d|©|Все права защищены|Генеральная лицензия", re.I)
_CAPTION_WORD = re.compile(r"[А-Яа-яЁёA-Za-z]{3,}")


def caption_factor(text: str) -> float:
    """Множитель скора для фрагментов из подписей к графикам и таблицам («Рис. 1: …», «Источники: …»).
    После чтения страниц моделью (T9) такие куски стали отдельными короткими фрагментами и обходили в BM25 текст
    с позицией банка: 1 / (1 + маркеров подписей на 20 слов)."""
    words = len(_CAPTION_WORD.findall(text))
    return 1 / (1 + len(_CAPTION.findall(text)) / max(1.0, words / 20))


class AnalyticsIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text + " " + c.title) for c in chunks]) if chunks else None
        self.embedder = embedder if embedder and embedder.enabled else None
        self._vecs = self.embedder.embed([c.text for c in chunks], "passage") if self.embedder and chunks else None
        self._quality = [caption_factor(c.text) for c in chunks]

    def search(self, query: str, k: int = 5, include_stale: bool = True, doc_ids: set[str] | None = None) -> list[Hit]:
        if not self.bm25:
            return []
        raw = self.bm25.get_scores(tokenize(query))
        top = max(raw) if len(raw) else 0.0
        bm = [s / top if top > 0 else 0.0 for s in raw]
        dense = None
        if self._vecs is not None:
            q = self.embedder.embed([query], "query")[0]
            dense = [cosine(q, v) for v in self._vecs]
        hits = []
        for i, c in enumerate(self.chunks):
            if doc_ids and c.doc_id not in doc_ids:
                continue
            if c.stale and not include_stale:
                continue
            hits.append((i, c))
        if dense is not None:
            # Reciprocal rank fusion двух списков.
            rb = {i: r for r, i in enumerate(sorted((i for i, _ in hits), key=lambda i: -bm[i]))}
            rd = {i: r for r, i in enumerate(sorted((i for i, _ in hits), key=lambda i: -dense[i]))}
            base = {i: 1 / (60 + rb[i]) + 1 / (60 + rd[i]) for i, _ in hits}
            norm = max(base.values(), default=1.0)
            base = {i: v / norm for i, v in base.items()}
        else:
            base = {i: bm[i] for i, _ in hits}
        # Свежесть понижает, но не обнуляет: устаревшее находится, но ниже.
        scored = [Hit(c, base[i] * (0.4 + 0.6 * c.freshness) * self._quality[i], bm[i], dense[i] if dense else None) for i, c in hits]
        scored = [h for h in scored if h.bm25 > 0 or (h.dense or 0) > 0]
        return sorted(scored, key=lambda h: -h.score)[:k]


@lru_cache(maxsize=1)
def get_index() -> AnalyticsIndex:
    return AnalyticsIndex(load_chunks(), Embedder())
