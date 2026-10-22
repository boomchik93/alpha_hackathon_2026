"""Клиент локального сервера эмбеддингов e5 (OpenAI-совместимый `/embeddings`).

Без `EMBEDDER_URL` поиск работает только на BM25. Эмбеддинги документов
кэшируются на диск.
"""

from __future__ import annotations

import math

import httpx

from ml.config import settings
from ml.llm.cache import DiskCache, make_key

MODEL = "intfloat/multilingual-e5-base"


class Embedder:
    def __init__(self, url: str | None = None):
        self.url = (url if url is not None else settings.embedder_url).rstrip("/")
        self.cache = DiskCache(settings.paths.var / "emb_cache")

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        r = httpx.post(f"{self.url}/embeddings", json={"model": MODEL, "input": texts}, timeout=60)
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        """kind: `query` или `passage`, префикс e5 обязателен."""
        prefixed = [f"{kind}: {t}" for t in texts]
        keys = [make_key(MODEL, t) for t in prefixed]
        out: list[list[float] | None] = [None] * len(texts)
        todo = []
        for i, k in enumerate(keys):
            hit = self.cache.get(k)
            if hit is not None:
                out[i] = hit["v"]
            else:
                todo.append(i)
        for start in range(0, len(todo), 32):
            batch = todo[start : start + 32]
            for i, v in zip(batch, self._embed([prefixed[i] for i in batch])):
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                v = [x / n for x in v]
                self.cache.set(keys[i], {"v": v})
                out[i] = v
        return out  # type: ignore[return-value]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
