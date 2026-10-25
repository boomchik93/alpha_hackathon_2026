"""Сборка var/items.jsonl: оценка важности только по явному флагу (она платная)."""

import pytest

from ml.ingest import store
from ml.ingest.news_fetch import RawItem


def _raw() -> list[RawItem]:
    return [RawItem("n_t1", "Банк России сохранил ключевую ставку", "Совет директоров сохранил ставку на уровне 14%.", "2026-09-11")]


def test_build_items_without_flag_keeps_previous_importance(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ITEMS_PATH", tmp_path / "items.jsonl")
    monkeypatch.setattr(store, "fetch_news", _raw)
    monkeypatch.setattr(store, "doc_items", lambda *a, **k: [])

    def paid(*a, **k):
        raise AssertionError("без --importance модель важности вызываться не должна")

    monkeypatch.setattr(store, "with_importance", paid)
    store.load_items.cache_clear()
    first = store.build_items()
    assert first[0].importance == 0.5  # новый материал: нейтральная важность

    (tmp_path / "items.jsonl").write_text(first[0].model_copy(update={"importance": 0.9}).model_dump_json() + "\n", encoding="utf-8")
    store.load_items.cache_clear()
    again = store.build_items()
    assert again[0].importance == 0.9  # прежняя оценка сохранилась
    store.load_items.cache_clear()


def test_build_items_with_flag_scores_importance(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ITEMS_PATH", tmp_path / "items.jsonl")
    monkeypatch.setattr(store, "fetch_news", _raw)
    monkeypatch.setattr(store, "doc_items", lambda *a, **k: [])
    monkeypatch.setattr(store, "with_importance", lambda item, text, client=None: item.model_copy(update={"importance": 0.95}))
    store.load_items.cache_clear()
    assert store.build_items(importance=True)[0].importance == 0.95
    store.load_items.cache_clear()


def test_fetch_news_collector_reads_raw_items(tmp_path, monkeypatch):
    import json

    from ml.ingest import news_fetch

    f = tmp_path / "raw_items.jsonl"
    rec = {"id": "n_c1", "title": "Заголовок", "text": "Текст", "published_at": "2026-10-24", "publisher": "РБК",
           "url": "https://x/1", "kind": "news", "fetched_at": "2026-10-25T00:00:00+00:00"}
    f.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setenv("NEWS_SOURCE", "collector")
    monkeypatch.setenv("NEWS_COLLECTOR_FILE", str(f))
    assert [i.id for i in news_fetch.fetch_news()] == ["n_c1"]

    monkeypatch.setenv("NEWS_COLLECTOR_FILE", str(tmp_path / "нет.jsonl"))
    import pytest

    with pytest.raises(FileNotFoundError):
        news_fetch.fetch_news()


def test_build_items_skip_failed_drops_only_unparsed_news(tmp_path, monkeypatch):
    from ml.llm.client import LLMError

    monkeypatch.setattr(store, "ITEMS_PATH", tmp_path / "items.jsonl")
    bad = RawItem("n_bad", "Битая новость", "текст", "2026-09-11")
    monkeypatch.setattr(store, "fetch_news", lambda: [*_raw(), bad])
    monkeypatch.setattr(store, "doc_items", lambda *a, **k: [])
    real_parse = store.parse_item

    def parse(raw, *a, **k):
        if raw.id == "n_bad":
            raise LLMError("модель вернула не JSON")
        return real_parse(raw, *a, **k)

    monkeypatch.setattr(store, "parse_item", parse)
    store.load_items.cache_clear()
    with pytest.raises(LLMError):  # по умолчанию сбой разбора, как и раньше, останавливает сборку
        store.build_items()
    items = store.build_items(skip_failed=True, workers=2)
    assert [i.item_id for i in items] == ["n_t1"]
    assert not list(tmp_path.glob(".*.tmp"))  # запись атомарная: временный файл не остаётся
    store.load_items.cache_clear()
