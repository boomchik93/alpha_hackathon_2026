"""Выгрузки сборщика: raw_items.jsonl (формат RawItem для ML-слоя), news.json (события с тегами), taxonomy.json."""
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
import taxonomy as tx
from store import Store

SCHEMA_VERSION = "2.0"


def _atomic_write(path: Path, text: str) -> None:
    """Пишем во временный файл и подменяем: читатель никогда не увидит половину файла."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _ordered(arts):
    """Первоисточники, статьи с текстом и прямые ссылки (не редиректы Google News) идут первыми."""
    return sorted(arts, key=lambda a: (a["source_type"] != "primary", a["text"] is None,
                                       "news.google.com" in a["url"], not a["summary"], a["published_at"]))


def _merge_rule_tags(arts) -> dict:
    """Теги по правилам для всего события: заголовки, аннотации и начало текста всех статей."""
    merged = {k: [] for k in ("topics", "okved_sections", "regions", "countries", "affects")}
    for a in arts:
        for k, values in tx.rule_tags(a["title"], a["summary"], (a["text"] or "")[:3000]).items():
            merged[k] += [v for v in values if v not in merged[k]]
    return merged


def _verification(arts) -> str:
    distinct = {a["source_name"] for a in arts}
    return "confirmed" if any(a["source_type"] == "primary" for a in arts) or len(distinct) >= 2 else "single_source"


def raw_item_id(url: str | None, title: str) -> str:
    """Тот же id, что у ml.ingest.news_fetch.items_from_rss: n_ + sha1(ссылка или заголовок)[:12]."""
    return "n_" + hashlib.sha1((url or title).encode()).hexdigest()[:12]


def build_raw_item(arts, fetched_at: str) -> dict:
    """Одно событие -> одна запись RawItem: ведущая статья, дата публикации без времени."""
    lead = _ordered(arts)[0]
    return {
        "id": raw_item_id(lead["url"], lead["title"]),
        "title": lead["title"],
        "text": lead["text"] or lead["summary"] or lead["title"],
        "published_at": min(a["published_at"] for a in arts)[:10],
        "publisher": lead["source_name"],
        "url": lead["url"],
        "kind": "news",
        "fetched_at": fetched_at,
    }


def build_item(event, arts) -> dict:
    arts = _ordered(arts)
    lead = arts[0]
    rules = _merge_rule_tags(arts)
    return {
        "event_id": event["id"],
        "item_id": raw_item_id(lead["url"], lead["title"]),  # связь с raw_items.jsonl
        "title": lead["title"],
        "summary": lead["summary"] or (lead["text"] or "")[:300],
        "language": lead["lang"],
        "scope": "international" if all(a["scope"] == "international" for a in arts) else "russia",
        "published_at": min(a["published_at"] for a in arts),
        "updated_at": max(a["published_at"] for a in arts),
        "sources": [{"name": a["source_name"], "type": a["source_type"], "url": a["url"],
                     "title": a["title"], "published_at": a["published_at"]} for a in arts],
        "verification": _verification(arts),
        "tags": rules,
        "tagging": {"method": "rules", "taxonomy_version": tx.TAXONOMY_VERSION},
    }


def export(store: Store, days: int = 2) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    items, raw_lines = [], []
    for ev in store.events_for_export(since):
        arts = store.event_articles(ev["id"])
        items.append(build_item(ev, arts))
        raw_lines.append(json.dumps(build_raw_item(arts, fetched_at), ensure_ascii=False))
    items.sort(key=lambda i: (len(i["sources"]), i["updated_at"]), reverse=True)  # чем больше изданий, тем выше
    payload = {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": tx.TAXONOMY_VERSION,
        "generated_at": fetched_at,
        "window_days": days,
        "count": len(items),
        "items": items,
    }
    _atomic_write(config.OUTPUT_PATH, json.dumps(payload, ensure_ascii=False, indent=2))
    _atomic_write(config.RAW_ITEMS_PATH, "\n".join(raw_lines) + ("\n" if raw_lines else ""))
    _atomic_write(config.TAXONOMY_PATH, json.dumps(tx.catalog(), ensure_ascii=False, indent=2))
    return {"events": len(items), "raw_items": str(config.RAW_ITEMS_PATH)}
