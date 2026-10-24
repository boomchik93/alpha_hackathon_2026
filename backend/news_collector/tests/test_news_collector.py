"""Тесты сборщика без сети: теги, дедупликация, формат выгрузки RawItem."""
import json
from dataclasses import fields
from datetime import datetime, timezone

import pytest

import config
import taxonomy as tx
from collector import article_id, canonical_url, is_ru_or_en
from dedup import cluster_new_articles, tokenize
from export import build_raw_item, export, raw_item_id
from store import Store


def _article(url, title, source="ТАСС", summary="", text=None, source_type="media", scope="russia", lang="ru"):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tags = tx.rule_tags(title, summary)
    return {
        "id": article_id(url), "url": canonical_url(url), "source_id": source, "source_name": source,
        "source_type": source_type, "lang": lang, "scope": scope, "title": title, "summary": summary,
        "text": text, "text_status": "ok" if text else "skipped", "published_at": now, "fetched_at": now,
        "relevant": 1, "rule_tags": json.dumps(tags), "tokens": json.dumps(sorted(tokenize(title, summary))),
    }


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_PATH", tmp_path / "news.json")
    monkeypatch.setattr(config, "RAW_ITEMS_PATH", tmp_path / "raw_items.jsonl")
    monkeypatch.setattr(config, "TAXONOMY_PATH", tmp_path / "taxonomy.json")
    s = Store(tmp_path / "news.db")
    yield s
    s.close()


def test_rule_tags_ru_and_en():
    ru = tx.rule_tags("Банк России сохранил ключевую ставку 14%, кредиты для бизнеса дороже")
    assert "key_rate" in ru["topics"] and "borrowers" in ru["affects"]
    en = tx.rule_tags("EU adopts new sanctions package against Russia, China trade at risk")
    assert "sanctions" in en["topics"] and {"EU", "CN"} <= set(en["countries"])


def test_rule_tags_no_false_positives():
    assert tx.rule_tags("Игра престолов: bankruptcy of important football club")["topics"] == ["bankruptcy"]
    assert not tx.is_business_relevant(tx.rule_tags("Шафранов стал тренером хоккейного клуба Металлург"))


def test_language_filter_and_url_canonical():
    assert is_ru_or_en("Ключевая ставка") and is_ru_or_en("Key rate")
    assert not is_ru_or_en("سعر الفائدة الرئيسي")
    assert canonical_url("https://Site.ru/a/?utm_source=x&id=5#frag") == "https://site.ru/a?id=5"


def test_clustering_merges_same_event_only(store):
    store.insert_article(_article("https://a.ru/1", "Трамп заявил, что не время вводить новые санкции против России", "Интерфакс"))
    store.insert_article(_article("https://b.ru/2", "Трамп объяснил отказ вводить новые санкции против России", "RTVI"))
    store.insert_article(_article("https://c.ru/3", "Минпромторг предложил утилизационный сбор на буровое оборудование", "ПРАЙМ"))
    stats = cluster_new_articles(store)
    assert stats == {"joined": 1, "created": 2}


def test_raw_item_matches_ml_rawitem_contract(store):
    from ml.ingest.news_fetch import RawItem

    arts = [_article("https://a.ru/1?utm_x=1", "ЦБ сохранил ставку", "Банк России", summary="Решение принято", text="Полный текст решения.", source_type="primary")]
    raw = build_raw_item(arts, "2026-10-23T00:00:00+00:00")
    assert set(raw) == {f.name for f in fields(RawItem)}
    assert raw["kind"] == "news" and len(raw["published_at"]) == 10 and raw["text"] == "Полный текст решения."
    assert raw["id"] == raw_item_id(arts[0]["url"], arts[0]["title"])
    RawItem(**raw)


def test_export_writes_loadable_files(store):
    store.insert_article(_article("https://a.ru/1", "Банк России сохранил ключевую ставку", "Банк России", summary="Ставка 14%", source_type="primary"))
    cluster_new_articles(store)
    result = export(store, days=2)
    assert result["events"] == 1
    news = json.loads(config.OUTPUT_PATH.read_text(encoding="utf-8"))
    assert news["items"][0]["verification"] == "confirmed" and news["items"][0]["tagging"]["method"] == "rules"
    lines = config.RAW_ITEMS_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["id"] == news["items"][0]["item_id"]


def test_core_sources_are_subset_and_have_no_aggregators():
    core, everything = config.select_sources("core"), config.select_sources("all")
    assert core and len(core) < len(everything)
    assert all(s["type"] != "aggregator" and s["scope"] == "russia" for s in core)
