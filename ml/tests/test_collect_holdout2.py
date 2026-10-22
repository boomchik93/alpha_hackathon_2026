"""Сбор holdout2 на сохранённом RSS-ответе (REVIEW_PLAN.md, T3): даты, дедупликация, стратификация, lock-файл."""

import hashlib
import json
from pathlib import Path

from ml.eval import collect_holdout2 as ch

XML = (Path(__file__).parent / "fixtures" / "rss_sample.xml").read_bytes()


def test_records_without_date_dropped_not_today():
    items = ch.parse_xml(XML, "u")
    assert items and all(i.published_at.startswith("2026-10") for i in items)
    assert not any("без даты" in i.title for i in items)


def test_collect_filters_dedups_and_keeps_offtopic_share():
    items, stats = ch.collect({"u": XML}, n=12, seed=1, fetched_at="2026-10-21T00:00:00")
    titles = [i.title for i in items]
    assert not any("Короткая" in t for t in titles)
    # из пары «оригинал + повтор» остаётся одна запись
    assert sum(t.startswith("Банк России и ставки: сообщение 1") for t in titles) <= 1
    assert stats["after_dedup"] < stats["long_enough"]
    assert all(i.id.startswith("h2_") and i.fetched_at for i in items)
    off = [i for i in items if ch.is_offtopic(i)]
    assert len(off) == round(12 * ch.OFFTOPIC_SHARE) == 1
    assert len(items) == 12 and len({i.id for i in items}) == 12
    assert len(stats["strata"]) >= 3


def test_selection_is_deterministic_and_stratified():
    a, _ = ch.collect({"u": XML}, n=9, seed=7, fetched_at="x")
    b, _ = ch.collect({"u": XML}, n=9, seed=7, fetched_at="x")
    assert [i.id for i in a] == [i.id for i in b]
    assert len({ch.stratum(i) for i in a if not ch.is_offtopic(i)}) >= 3


def test_write_frozen_lock_has_hash_and_sources(tmp_path):
    items, stats = ch.collect({"u": XML}, n=6, seed=1, fetched_at="x")
    news, lock = tmp_path / "n.jsonl", tmp_path / "n.lock"
    got = ch.write_frozen(items, stats, ["u"], 1, "2026-10-21", news, lock)
    saved = json.loads(lock.read_text(encoding="utf-8"))
    assert saved == got and saved["sha256"] == hashlib.sha256(news.read_bytes()).hexdigest()
    assert saved["as_of"] == saved["collected_at"] == "2026-10-21" and saved["sources"] == ["u"] and saved["n"] == 6
    assert len(news.read_text(encoding="utf-8").splitlines()) == 6
