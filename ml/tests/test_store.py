"""Сборка var/items.jsonl: оценка важности только по явному флагу (она платная)."""

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
