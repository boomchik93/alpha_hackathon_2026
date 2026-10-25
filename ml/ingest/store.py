"""Хранилище разобранных материалов: новости + документы банка.

`python -m ml.ingest.store` собирает `var/items.jsonl`. Разбор делается один раз,
дальше лента и карточки читают готовые ParsedItem. Важность материалов (`--importance`) оценивается
моделью только по явному флагу: это платно.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from ml.config import settings
from ml.errors import NotFound
from ml.fileio import write_atomic
from ml.ingest.analytics import doc_items
from ml.ingest.dedup import assign_clusters
from ml.ingest.news_fetch import fetch_news
from ml.ingest.importance import with_importance
from ml.ingest.parse_item import parse_item
from ml.llm.client import LLMError, get_client
from ml.schemas import ParsedItem

ITEMS_PATH = settings.paths.var / "items.jsonl"


def build_items(importance: bool = False, skip_failed: bool = False, workers: int = 1) -> list[ParsedItem]:
    """importance=True: важность каждого материала оценивает модель (платно, вызов Gemma на материал, кэшируется).
    Иначе важность берётся из прежнего `var/items.jsonl`, новые материалы получают нейтральные 0,5: пересборка бесплатна.
    skip_failed=True (боевой конвейер): новость, которую модель не разобрала, пропускается, а не роняет всю сборку;
    превышение бюджета (BudgetExceeded, не LLMError) пропуском не считается и останавливает сборку.
    workers: сколько новостей разбирать параллельно (первый проход по свежему потоку — сотни вызовов)."""
    prev = {i.item_id: i.importance for i in load_items().values()}

    def keep(it: ParsedItem) -> ParsedItem:
        return it.model_copy(update={"importance": prev.get(it.item_id, it.importance)})

    def one(r) -> ParsedItem | None:
        try:
            it = parse_item(r)
        except LLMError as e:
            if not skip_failed:
                raise
            print(f"пропущена {r.id}: {type(e).__name__}: {str(e)[:150]}", file=sys.stderr)
            return None
        return with_importance(it, r.text) if importance else keep(it)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        news = [it for it in ex.map(one, fetch_news()) if it is not None]
    docs = doc_items(get_client()) if importance else [keep(d) for d in doc_items()]
    items = [*docs, *assign_clusters(news)]
    write_atomic(ITEMS_PATH, "".join(i.model_dump_json() + "\n" for i in items))
    load_items.cache_clear()
    return items


@lru_cache(maxsize=1)
def load_items() -> dict[str, ParsedItem]:
    if not ITEMS_PATH.exists():
        return {}
    items = [ParsedItem.model_validate_json(l) for l in ITEMS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {i.item_id: i for i in items}


def add_items(items: list[ParsedItem]) -> None:
    """Добавляет материалы в память процесса, не трогая var/items.jsonl (eval на holdout)."""
    load_items().update({i.item_id: i for i in items})


def get_item(item_id: str) -> ParsedItem:
    items = load_items()
    if item_id not in items:
        raise NotFound(f"материал {item_id} не найден")
    return items[item_id]


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--importance", action="store_true", help="оценить важность материалов моделью (ПЛАТНО, Gemma на материал)")
    args = ap.parse_args(argv)
    items = build_items(importance=args.importance)
    print(f"items={len(items)} importance={'модель' if args.importance else 'из прежнего файла'} -> {ITEMS_PATH}", file=sys.stderr)
    from ml.ranking.bm25 import build_background

    build_background()  # idf и avgdl ленты замораживаются на этом корпусе


if __name__ == "__main__":
    main()
