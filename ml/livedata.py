"""Горячая перезагрузка данных в долгоживущем процессе (`ml/server.py` в боевом режиме).

Материалы, фон BM25 и фрагменты аналитики читаются один раз и держатся в памяти (`lru_cache`).
Конвейер `ml/live.py` обновляет файлы на ходу; сервис сверяет время изменения файлов перед запросом
и сбрасывает кэш только того, что изменилось. Первая проверка запоминает состояние и ничего не сбрасывает.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from ml.config import settings
from ml.ingest import analytics, store
from ml.ranking import bm25
from ml.retrieval import index

# имя -> (путь к файлу, сбросы кэшей, которые от него зависят)
WATCH: dict[str, tuple[Callable[[], Path], list[Callable[[], None]]]] = {
    "items": (lambda: store.ITEMS_PATH, [store.load_items.cache_clear]),
    "bm25": (bm25.background_path, [bm25.load_background.cache_clear]),
    "chunks": (lambda: settings.paths.chunks, [analytics.load_chunks.cache_clear, index.get_index.cache_clear]),
}

_seen: dict[str, int | None] = {}
_lock = threading.Lock()


def _mtime(p: Path) -> int | None:
    try:
        return p.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def refresh() -> list[str]:
    """Сбрасывает кэши изменившихся файлов; возвращает их имена."""
    changed = []
    with _lock:
        for name, (path, clears) in WATCH.items():
            m = _mtime(path())
            if name in _seen and _seen[name] != m:
                for clear in clears:
                    clear()
                changed.append(name)
            _seen[name] = m
    return changed
