"""BM25 лента с замороженной статистикой.

Скор материала не должен зависеть от того, что ещё лежит в пуле: idf и средняя длина документа считаются один раз
на фоновом корпусе (фрагменты аналитики банка и материалы на момент сборки) и сохраняются в `var/bm25_background.json`.
Нормировка насыщением `s / (s + K)`, где K — медиана положительных сырых скоров запросов-профилей на фоновом корпусе:
K считается при сборке по всем сохранённым профилям, по метрикам не подбирается.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from functools import lru_cache

from ml.config import settings
from ml.fileio import write_atomic

K1 = 1.5
B = 0.75
PATH_NAME = "bm25_background.json"


class FrozenBM25:
    def __init__(self, idf: dict[str, float], idf_unseen: float, avgdl: float, k: float):
        self.idf, self.idf_unseen, self.avgdl, self.k = idf, idf_unseen, avgdl, k

    @classmethod
    def fit(cls, docs: list[list[str]], queries: list[list[str]]) -> "FrozenBM25":
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}
        model = cls(idf, math.log((n + 0.5) / 0.5 + 1), sum(map(len, docs)) / n if n else 1.0, 1.0)
        raw = [s for q in queries for d in docs if (s := model.raw(q, d)) > 0]
        model.k = statistics.median(raw) if raw else 1.0
        return model

    def raw(self, query: list[str], doc: list[str]) -> float:
        tf = Counter(doc)
        norm = K1 * (1 - B + B * len(doc) / self.avgdl)
        return sum(self.idf.get(t, self.idf_unseen) * tf[t] * (K1 + 1) / (tf[t] + norm) for t in set(query) if tf[t])

    def score(self, query: list[str], doc: list[str]) -> float:
        """Нормированный скор в [0, 1): не зависит от других документов пула."""
        s = self.raw(query, doc)
        return s / (s + self.k) if s > 0 else 0.0

    def to_json(self) -> dict:
        return {"idf": self.idf, "idf_unseen": self.idf_unseen, "avgdl": self.avgdl, "k": self.k}

    @classmethod
    def from_json(cls, d: dict) -> "FrozenBM25":
        return cls(d["idf"], d["idf_unseen"], d["avgdl"], d["k"])


def background_path():
    return settings.paths.var / PATH_NAME


@lru_cache(maxsize=1)
def load_background() -> FrozenBM25 | None:
    """Замороженная модель из `var/bm25_background.json`; None, если её нет (тогда лента нормирует по пулу, как раньше)."""
    p = background_path()
    return FrozenBM25.from_json(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None


def build_background(save: bool = True) -> FrozenBM25:
    """Фоновый корпус: фрагменты банка и материалы из `var/items.jsonl`; запросы: все сохранённые профили (`var/profiles.jsonl`),
    а не компании eval: рабочий путь от eval-набора не зависит."""
    from ml.ingest.analytics import load_chunks
    from ml.ingest.store import load_items
    from ml.profile.build_profile import list_profiles
    from ml.ranking.ranker import _item_text, profile_query
    from ml.retrieval.index import tokenize

    docs = [tokenize(f"{c.title} {c.text}") for c in load_chunks()] + [tokenize(_item_text(i)) for i in load_items().values()]
    queries = [tokenize(profile_query(p)) for p in list_profiles()]
    model = FrozenBM25.fit([d for d in docs if d], queries)
    if save:
        write_atomic(background_path(), json.dumps(model.to_json(), ensure_ascii=False))
        load_background.cache_clear()
    return model


if __name__ == "__main__":
    m = build_background()
    print(f"bm25 фон: терминов {len(m.idf)}, avgdl {m.avgdl:.1f}, K {m.k:.3f} -> {background_path()}")
