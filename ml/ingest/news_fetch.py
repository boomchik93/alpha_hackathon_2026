"""Источник новостей: фикстуры (детерминированная демо и eval) или RSS с кэшем на диск.

Режим задаётся `NEWS_SOURCE=fixtures|rss` в `.env`. По умолчанию фикстуры.
Список RSS-лент `NEWS_RSS_URLS` через запятую; разрешённость источников
нужно подтвердить у организаторов (PLAN.md, раздел 7).
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


# Полный текст статьи: в RSS многих изданий description пустой или в одну фразу, а короткий текст даёт модели мало фактов.
_DROP_BLOCKS = re.compile(r"<(script|style|noscript|svg|figure|figcaption|aside|nav|footer|header|form)\b[^>]*>.*?</\1>", re.S | re.I)
_PARA = re.compile(r"<p\b[^>]*>(.*?)</p>", re.S | re.I)
_PAGE_JUNK = re.compile(r"cookie|подпис(ывайтесь|аться|ку)|читайте (также|нас)|telegram|реклама|все права защищены|©|"
                        r"при перепечатке|сообщить об опечатке|наш канал|в соцсетях", re.I)
ARTICLE_MAX_CHARS = 3000
MIN_PARA_CHARS = 60


def article_text(html: str, max_chars: int = ARTICLE_MAX_CHARS) -> str:
    """Текст статьи из HTML: абзацы <p> без служебных блоков, подписей и призывов подписаться."""
    import html as html_mod

    body = _DROP_BLOCKS.sub(" ", html)
    paras = [html_mod.unescape(_strip_html(p)) for p in _PARA.findall(body)]
    paras = [p for p in paras if len(p) >= MIN_PARA_CHARS and not _PAGE_JUNK.search(p)]
    out = " ".join(dict.fromkeys(paras))  # повторы абзацев (мобильная и полная версии) убираются
    return out[:max_chars].rsplit(" ", 1)[0] if len(out) > max_chars else out


def fetch_article_text(url: str) -> str:
    """Текст статьи по ссылке, с кэшем на диск. Пустая строка, если страница недоступна."""
    cache = settings.paths.var / "news_cache" / "pages" / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    try:
        r = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (alpha-hack-news-bot/0.1)"})
        r.raise_for_status()
    except Exception:
        return ""
    text = article_text(r.text)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def with_full_text(item: RawItem, min_chars: int = 300) -> RawItem:
    """Если описание из RSS короче min_chars, подставляет текст статьи (когда он длиннее описания)."""
    if len(item.text) >= min_chars or not item.url:
        return item
    full = fetch_article_text(item.url)
    return RawItem(**{**item.__dict__, "text": full}) if len(full) > len(item.text) else item


def fetch_news() -> list[RawItem]:
    if os.getenv("NEWS_SOURCE", "fixtures") == "rss":
        urls = [u.strip() for u in os.getenv("NEWS_RSS_URLS", "").split(",") if u.strip()]
        items = [i for u in urls for i in fetch_rss(u)]
        # NEWS_FULLTEXT=1: короткие описания RSS дополняются текстом статьи (по запросу к странице на каждую новость).
        return [with_full_text(i) for i in items] if os.getenv("NEWS_FULLTEXT", "0") == "1" else items
    return load_fixtures()
