"""Сбор holdout2: настоящий отложенный набор новостей из RSS.

    python -m ml.eval.collect_holdout2 --dry-run            # посмотреть счётчики, ничего не писать
    python -m ml.eval.collect_holdout2 --n 50               # собрать и заморозить (пишет holdout2_news.jsonl и holdout2.lock)

Ничего не разбирается LLM и не размечается. После заморозки (есть holdout2.lock) содержимое набора не читается и
не используется для настройки до итоговой оценки (PROTOCOL.md); повторный сбор поверх lock-файла запрещён без --force.
Источники: --url (можно несколько) или `NEWS_RSS_URLS`; разрешённость источников подтверждает пользователь.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

from ml.eval.run_eval import EVAL_DIR
from ml.ingest.dedup import MAX_DAYS_APART, THRESHOLD, overlap
from ml.ingest.news_fetch import RawItem, items_from_rss
from ml.retrieval.index import tokenize
from ml.topics import detect_topics

NEWS_PATH = EVAL_DIR / "holdout2_news.jsonl"
LOCK_PATH = EVAL_DIR / "holdout2.lock"
SEED = 20261021
MIN_TEXT = 200
OFFTOPIC_SHARE = 0.12  # доля нерелевантных экономике материалов: проверка отсева
OTHER = "прочее"

# Нерелевантное экономике: спорт, культура, происшествия.
OFFTOPIC = re.compile(
    r"(?<![а-яёa-z])(?:футбол|хоккей|олимпиад|чемпионат|матч|турнир|спортсмен|теннис|биатлон|"
    r"кино|фильм|сериал|актёр|актер|театр|концерт|фестиваль|выставк|музе[йя]|певиц|певец|"
    r"пожар|дтп|авари[яи]|убийств|задержан|погиб|взрыв|землетрясен|наводнен|ураган)", re.I)


def parse_xml(content: bytes, url: str) -> list[RawItem]:
    """Записи ленты без даты публикации отбрасываются: дату не подставляем."""
    return items_from_rss(ET.fromstring(content), url, require_date=True)


def is_offtopic(item: RawItem) -> bool:
    return bool(OFFTOPIC.search(f"{item.title} {item.text}"))


def dedup_raw(items: list[RawItem]) -> list[RawItem]:
    """Дубликаты до разбора: то же правило, что ml.ingest.dedup (перекрытие слов, не дальше суток), остаётся более длинный текст."""
    kept: list[tuple[RawItem, set[str]]] = []
    for it in sorted(items, key=lambda x: (x.published_at, x.id)):
        sig = set(tokenize(f"{it.title} {it.text}"))
        for n, (other, osig) in enumerate(kept):
            days = abs((date.fromisoformat(it.published_at) - date.fromisoformat(other.published_at)).days)
            if days <= MAX_DAYS_APART and overlap(sig, osig) >= THRESHOLD:
                if len(it.text) > len(other.text):
                    kept[n] = (it, sig)
                break
        else:
            kept.append((it, sig))
    return [it for it, _ in kept]


def stratum(item: RawItem) -> str:
    topics = detect_topics(f"{item.title}. {item.text}", min_hits=1)
    return topics[0] if topics else OTHER


def select(items: list[RawItem], n: int, seed: int = SEED) -> tuple[list[RawItem], dict[str, int]]:
    """n материалов: ~12% нерелевантных экономике, остальные по кругу по темам (detect_topics, плюс «прочее»)."""
    rnd = random.Random(seed)
    off = [i for i in items if is_offtopic(i)]
    on = [i for i in items if not is_offtopic(i)]
    n_off = min(len(off), round(n * OFFTOPIC_SHARE))
    strata: dict[str, list[RawItem]] = {}
    for it in sorted(on, key=lambda x: x.id):
        strata.setdefault(stratum(it), []).append(it)
    for v in strata.values():
        rnd.shuffle(v)
    picked: list[RawItem] = []
    keys = sorted(strata)
    while len(picked) < n - n_off and any(strata.values()):
        for k in keys:
            if strata[k] and len(picked) < n - n_off:
                picked.append(strata[k].pop())
    picked += rnd.sample(sorted(off, key=lambda x: x.id), n_off)
    counts: dict[str, int] = {}
    for it in picked:
        k = "нерелевантное" if is_offtopic(it) else stratum(it)
        counts[k] = counts.get(k, 0) + 1
    return sorted(picked, key=lambda x: (x.published_at, x.id)), counts


def collect(contents: dict[str, bytes], n: int, seed: int, fetched_at: str) -> tuple[list[RawItem], dict]:
    """contents: url -> тело RSS-ответа. Возвращает (материалы holdout2, статистика отбора)."""
    raw = [i for url, body in contents.items() for i in parse_xml(body, url)]
    long_enough = [i for i in raw if len(i.text) >= MIN_TEXT]
    uniq = dedup_raw(long_enough)
    chosen, counts = select(uniq, n, seed)
    out = [RawItem(id="h2_" + hashlib.sha1((i.url or i.title).encode()).hexdigest()[:12], title=i.title, text=i.text,
                   published_at=i.published_at, publisher=i.publisher, url=i.url, kind="news", fetched_at=fetched_at)
           for i in chosen]
    stats = {"records_with_date": len(raw), "long_enough": len(long_enough), "after_dedup": len(uniq),
             "selected": len(out), "strata": counts}
    return out, stats


def write_frozen(items: list[RawItem], stats: dict, sources: list[str], seed: int, as_of: str,
                 news_path: Path = NEWS_PATH, lock_path: Path = LOCK_PATH) -> dict:
    lines = [json.dumps(i.__dict__, ensure_ascii=False) for i in items]
    news_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    lock = {"file": news_path.name, "sha256": hashlib.sha256(news_path.read_bytes()).hexdigest(), "n": len(items),
            "collected_at": as_of, "as_of": as_of, "seed": seed, "sources": sources, **stats}
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return lock


def fetch_all(urls: list[str]) -> dict[str, bytes]:
    import httpx

    out = {}
    for u in urls:
        r = httpx.get(u, timeout=30, follow_redirects=True, headers={"User-Agent": "alpha-hack-news-bot/0.1"})
        r.raise_for_status()
        out[u] = r.content
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", help="RSS-лента (можно несколько); по умолчанию NEWS_RSS_URLS")
    ap.add_argument("--xml", action="append", default=[], help="сохранённый RSS-ответ вместо сети (для проверки)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--as-of", default=date.today().isoformat(), help="дата сбора = as_of для оценки")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="перезаписать замороженный набор (делает его использованным)")
    args = ap.parse_args(argv)

    if LOCK_PATH.exists() and not args.force and not args.dry_run:
        sys.exit(f"{LOCK_PATH.name} уже есть: holdout2 заморожен. Повторный сбор только с --force.")
    urls = args.url or [u.strip() for u in os.getenv("NEWS_RSS_URLS", "").split(",") if u.strip()]
    contents = {f"file:{p}": Path(p).read_bytes() for p in args.xml}
    if urls:
        contents.update(fetch_all(urls))
    if not contents:
        sys.exit("нет источников: задайте --url, NEWS_RSS_URLS или --xml")
    items, stats = collect(contents, args.n, args.seed, datetime.now().isoformat(timespec="seconds"))
    print(json.dumps(stats, ensure_ascii=False), file=sys.stderr)
    if args.dry_run:
        return
    lock = write_frozen(items, stats, sorted(contents), args.seed, args.as_of)
    print(f"записано {lock['n']} материалов, sha256 {lock['sha256'][:12]}", file=sys.stderr)


if __name__ == "__main__":
    main()
