"""Загрузка RSS-лент и полного текста статей."""
import hashlib
import html
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import requests
import trafilatura

import config
from dedup import tokenize
from store import Store, now_iso
from taxonomy import is_business_relevant, rule_tags

log = logging.getLogger("collector")

_TRACKING = re.compile(r"^(utm_|from$|ysclid$|fbclid$|gclid$|yclid$|ref$|source$)", re.I)
_TAGS = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")


def clean_html(raw: str | None) -> str:
    if not raw:
        return ""
    return _SPACES.sub(" ", html.unescape(_TAGS.sub(" ", raw))).strip()


def is_ru_or_en(text: str) -> bool:
    """Google News подмешивает издания на арабском, хинди и т.п. Модель и теги рассчитаны на RU/EN."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    ok = sum(1 for ch in letters if ch.isascii() or "а" <= ch.lower() <= "я" or ch.lower() == "ё")
    return ok / len(letters) >= 0.7


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not _TRACKING.match(k)])
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def article_id(url: str) -> str:
    return hashlib.sha1(canonical_url(url).encode("utf-8")).hexdigest()[:16]


def _entry_time(entry) -> datetime:
    for key in ("published_parsed", "updated_parsed"):
        value = entry.get(key)
        if value:
            return datetime(*value[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def fetch_feed(src: dict) -> tuple[dict, list[dict], str | None]:
    """Возвращает (источник, записи, ошибка). Сбой одного источника не должен ронять сбор."""
    try:
        resp = requests.get(src["url"], headers=config.HTTP_HEADERS, timeout=config.HTTP_TIMEOUT)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception as exc:  # noqa: BLE001 - сеть ненадёжна, логируем и идём дальше
        return src, [], f"{type(exc).__name__}: {str(exc)[:100]}"

    items = []
    for e in feed.entries:
        link, title = e.get("link"), clean_html(e.get("title"))
        if not link or not title or not is_ru_or_en(title):
            continue
        source_name = src["name"]
        if src["type"] == "aggregator":
            # Google News: реальное издание лежит в entry.source, в заголовке оно дописано через " - "
            publisher = (e.get("source") or {}).get("title")
            if publisher:
                source_name = publisher
                suffix = f" - {publisher}"
                if title.endswith(suffix):
                    title = title[: -len(suffix)].strip()
        summary = clean_html(e.get("summary") or e.get("description"))
        if src["type"] == "aggregator" and summary.startswith(title[:30]):
            summary = ""  # в Google News аннотация часто просто повторяет заголовок
        items.append({
            "url": canonical_url(link), "title": title, "summary": summary[:1000],
            "source_name": source_name, "published": _entry_time(e),
        })
    return src, items, None


def fetch_fulltext(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=config.HTTP_HEADERS, timeout=config.HTTP_TIMEOUT)
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False, favor_precision=True)
    except Exception as exc:  # noqa: BLE001
        log.debug("fulltext failed %s: %s", url, exc)
        return None
    return text[: config.FULLTEXT_CHARS] if text else None


def collect(store: Store, sources: list[dict], prefilter: bool = True, workers: int = 8) -> dict:
    """Один проход: забрать ленты, отфильтровать, сохранить новое, докачать тексты."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.MAX_AGE_DAYS)
    stats = {"feeds_ok": 0, "feeds_failed": 0, "new": 0, "new_relevant": 0, "fulltext_ok": 0}
    fulltext_sources = {s["id"] for s in sources if s["fulltext"]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fetch_feed, sources))

    for src, items, error in results:
        if error:
            stats["feeds_failed"] += 1
            log.warning("источник %s недоступен: %s", src["id"], error)
            continue
        stats["feeds_ok"] += 1
        for it in items:
            if it["published"] < cutoff:
                continue
            aid = article_id(it["url"])
            if store.has_article(aid):
                continue
            tags = rule_tags(it["title"], it["summary"])
            relevant = src["always_relevant"] or not prefilter or is_business_relevant(tags)
            store.insert_article({
                "id": aid, "url": it["url"], "source_id": src["id"], "source_name": it["source_name"],
                "source_type": "media" if src["type"] == "aggregator" else src["type"],
                "lang": src["lang"], "scope": src["scope"], "title": it["title"], "summary": it["summary"],
                "text": None,
                # NULL = «ещё надо скачать»; для источников без полного текста сразу skipped
                "text_status": None if (relevant and src["id"] in fulltext_sources) else "skipped",
                "published_at": it["published"].isoformat(timespec="seconds"),
                "fetched_at": now_iso(), "relevant": int(relevant),
                "rule_tags": json.dumps(tags, ensure_ascii=False),
                "tokens": json.dumps(sorted(tokenize(it["title"], it["summary"]))),
            })
            stats["new"] += 1
            stats["new_relevant"] += int(relevant)
    store.commit()

    todo = store.articles_needing_text()
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            texts = list(pool.map(lambda r: fetch_fulltext(r["url"]), todo))
        for row, text in zip(todo, texts):
            store.set_article_text(row["id"], text)
            stats["fulltext_ok"] += int(bool(text))
        store.commit()
    return stats
