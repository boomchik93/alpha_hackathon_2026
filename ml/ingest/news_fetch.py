"""Источник новостей: фикстуры (детерминированная демо и eval) или RSS с кэшем на диск.

Режим задаётся `NEWS_SOURCE=fixtures|rss` в `.env`. По умолчанию фикстуры.
Список RSS-лент `NEWS_RSS_URLS` через запятую; разрешённость источников
нужно подтвердить у организаторов.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date
from email.utils import parsedate_to_datetime

import httpx

from ml.config import settings


@dataclass
class RawItem:
    id: str
    title: str
    text: str
    published_at: str
    publisher: str | None = None
    url: str | None = None
    kind: str = "news"
    fetched_at: str | None = None


def load_fixtures() -> list[RawItem]:
    p = settings.paths.fixtures / "news.jsonl"
    return [RawItem(**json.loads(l)) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def items_from_rss(root: ET.Element, url: str, require_date: bool = False) -> list[RawItem]:
    """Записи RSS-ленты. Без даты публикации: сегодняшняя дата, а при require_date запись пропускается."""
    publisher = (root.findtext("channel/title") or url).strip()
    items = []
    for it in root.iter("item"):
        link = (it.findtext("link") or "").strip()
        title = _strip_html(it.findtext("title") or "")
        try:
            pub = parsedate_to_datetime(it.findtext("pubDate") or "").date()
        except (TypeError, ValueError):
            if require_date:
                continue
            pub = date.today()
        iid = "n_" + hashlib.sha1((link or title).encode()).hexdigest()[:12]
        items.append(RawItem(iid, title, _strip_html(it.findtext("description") or ""), pub.isoformat(), publisher, link or None))
    return items


def fetch_rss(url: str) -> list[RawItem]:
    cache = settings.paths.var / "news_cache" / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "alpha-hack-news-bot/0.1"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:
        # Сеть недоступна: работаем по кэшу.
        if cache.exists():
            return [RawItem(**d) for d in json.loads(cache.read_text(encoding="utf-8"))]
        return []
    items = items_from_rss(root, url)
    cache.write_text(json.dumps([asdict(i) for i in items], ensure_ascii=False), encoding="utf-8")
    return items


def fetch_news() -> list[RawItem]:
    if os.getenv("NEWS_SOURCE", "fixtures") == "rss":
        urls = [u.strip() for u in os.getenv("NEWS_RSS_URLS", "").split(",") if u.strip()]
        return [i for u in urls for i in fetch_rss(u)]
    return load_fixtures()
