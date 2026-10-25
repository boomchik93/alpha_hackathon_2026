"""Боевой режим (ml/live.py): пары из сида, порядок и потолок расходов карточек, дата «сегодня», горячая перезагрузка."""

import json
import os
from datetime import date
from types import SimpleNamespace

import pytest

from ml import config, live, livedata
from ml.ingest import store
from ml.roles import Role
from ml.schemas import Feed, FeedEntry


def test_seed_pairs_take_only_users_and_drop_other_role(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps([
        {"login": "a", "role": "cfo", "access": "user", "companies": ["U_1", "group_9"]},
        {"login": "b", "role": "cfo", "access": "user", "companies": ["U_1"]},   # повтор пары
        {"login": "c", "role": "other", "access": "user", "companies": ["U_2"]},
        {"login": "e", "role": "other", "access": "expert", "companies": []},
    ]), encoding="utf-8")
    assert live.seed_pairs(seed) == [("U_1", Role.cfo), ("group_9", Role.cfo), ("U_2", None)]


def _entry(item_id: str, score: float) -> FeedEntry:
    return FeedEntry(item_id=item_id, title=item_id, published_at=date(2026, 10, 24), source_kind="news", score=score,
                     reasons=[], topics=[], tier="main")


class Ledger:
    def __init__(self, spent: float = 0.0):
        self.spent = spent

    def spent_today(self) -> float:
        return self.spent


def _fake_api(monkeypatch, feeds: dict, cached: set, ledger: Ledger, cost: float):
    from ml import api

    made = []

    def get_insight(cid, item_id, role=None):
        made.append((cid, item_id, role))
        ledger.spent += cost

    monkeypatch.setattr(api, "get_ranked_feed", lambda cid, **k: Feed(company_id=cid, as_of=date(2026, 10, 25), entries=feeds[cid]))
    monkeypatch.setattr(api, "cached_insight", lambda cid, item_id, role=None: object() if (cid, item_id) in cached else None)
    monkeypatch.setattr(api, "get_insight", get_insight)
    monkeypatch.setattr(live, "SpendLedger", lambda: ledger)
    monkeypatch.setattr(live, "CARD_WORKERS", 1)
    return made


def test_cards_skip_ready_and_go_by_score(monkeypatch):
    ledger = Ledger()
    feeds = {"U_1": [_entry("n_low", 0.4), _entry("n_ready", 0.9)], "U_2": [_entry("n_top", 0.95)]}
    made = _fake_api(monkeypatch, feeds, {("U_1", "n_ready")}, ledger, cost=0.2)
    res = live.cards([("U_1", Role.cfo), ("U_2", None)], budget_rub=10)
    assert made == [("U_2", "n_top", None), ("U_1", "n_low", Role.cfo)]
    assert res == {"ready": 1, "todo": 2, "made": 2, "failed": 0, "skipped_budget": 0}


def test_cards_stop_at_daily_budget(monkeypatch):
    ledger = Ledger(spent=0.5)
    feeds = {"U_1": [_entry(f"n_{i}", 0.9 - i / 10) for i in range(5)]}
    made = _fake_api(monkeypatch, feeds, set(), ledger, cost=0.2)
    res = live.cards([("U_1", Role.cfo)], budget_rub=0.85)
    assert [m[1] for m in made] == ["n_0", "n_1"]  # 0,5 -> 0,7 -> 0,9: потолок 0,85 пройден, дальше стоп
    assert res["made"] == 2 and res["skipped_budget"] == 3


def test_as_of_today_is_moscow_date(monkeypatch):
    monkeypatch.setenv("ML_AS_OF", "2026-09-15")
    assert config.settings.as_of == date(2026, 9, 15)
    monkeypatch.setenv("ML_AS_OF", "today")
    assert config.settings.as_of == config.resolve_as_of("today")
    monkeypatch.delenv("ML_AS_OF")
    assert config.settings.as_of == date(2026, 9, 15)  # по умолчанию дата демо, как раньше


def test_live_refuses_mock_llm_and_other_news_sources(monkeypatch):
    monkeypatch.setattr(live, "once", lambda **k: pytest.fail("проход не должен начаться"))
    monkeypatch.setenv("NEWS_SOURCE", "fixtures")
    with pytest.raises(SystemExit, match="NEWS_SOURCE=collector"):
        live.main(["once"])
    monkeypatch.setenv("NEWS_SOURCE", "collector")
    monkeypatch.setattr(live, "settings", SimpleNamespace(llm=SimpleNamespace(mock=True)))
    with pytest.raises(SystemExit, match="только с настоящими моделями"):
        live.main(["once"])


def test_refresh_clears_cache_only_after_file_changes(tmp_path, monkeypatch):
    items = tmp_path / "items.jsonl"
    items.write_text("", encoding="utf-8")
    cleared = []
    monkeypatch.setattr(livedata, "WATCH", {"items": (lambda: items, [lambda: cleared.append("items")])})
    monkeypatch.setattr(livedata, "_seen", {})
    assert livedata.refresh() == []          # первая проверка только запоминает состояние
    assert livedata.refresh() == [] and cleared == []
    st = items.stat()
    os.utime(items, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert livedata.refresh() == ["items"] and cleared == ["items"]
    assert livedata.refresh() == []


def test_watch_covers_items_file():
    path, clears = livedata.WATCH["items"]
    assert path() == store.ITEMS_PATH and store.load_items.cache_clear in clears
