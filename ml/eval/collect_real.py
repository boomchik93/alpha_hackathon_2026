"""Набор реальных новостей за последние дни (GAIN_PLAN.md, фаза D): RSS + полный текст статьи, заморозка, деление по дате.

    python -m ml.eval.collect_real --as-of 2026-10-22 --days 7 --n 80 --dry-run
    python -m ml.eval.collect_real --as-of 2026-10-22 --days 7 --n 80

Отбор: только записи с датой публикации в окне [as_of - days + 1; as_of]; рубрики, не относящиеся к экономике,
отбрасываются по категории RSS (где она есть); около 10% нерелевантных экономике материалов остаются для проверки
отсева. Если описание в RSS короче MIN_TEXT, берётся текст статьи со страницы (`ml.ingest.news_fetch.fetch_article_text`).
Источники чередуются по кругу, внутри источника темы тоже (`detect_topics`), чтобы набор не состоял из одной ленты.

Деление по дате: последние `--holdout-days` дней окна идут в split="holdout" (не открывать до итоговой оценки),
остальные в split="dev" (на них можно настраивать ленту). Пишет `ml/eval/<name>_news.jsonl` и `ml/eval/<name>_news.lock`
(`--name`, по умолчанию real). Ничего не разбирает LLM и не размечает.

Свежий набор для итоговой оценки (STATUS_HANDOFF.md, T19): RSS хранит записи несколько дней, поэтому записи копятся
в пул ежедневно, а набор замораживается из пула, когда наберётся нужное число дней:

    python -m ml.eval.collect_real --accumulate                     # раз в день: новые записи лент в real_pool.jsonl
    python -m ml.eval.collect_real --name real2 --from-pool --start 2026-10-23 --as-of 2026-10-29 --n 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

from ml.eval.collect_holdout2 import dedup_raw, is_offtopic, stratum
from ml.eval.run_eval import EVAL_DIR
from ml.ingest.news_fetch import RawItem, items_from_rss, with_full_text

POOL_PATH = EVAL_DIR / "real_pool.jsonl"
SEED = 20261022
MIN_TEXT = 300
MIN_TEXT_NO_PAGE = 150  # издания, у которых текст страницы не извлекается: достаточно описания из RSS
OFFTOPIC_SHARE = 0.1
HOLDOUT_SHARE = 0.4  # доля набора из последних holdout_days дней
OFFTOPIC = "нерелевантное"

# Экономические ленты, открытые без авторизации. Категории: что брать из общих лент (пусто = всё).
SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "lenta": ("https://lenta.ru/rss/news/economics", ()),
    "vedomosti": ("https://www.vedomosti.ru/rss/rubric/economics", ()),
    "cbr": ("https://www.cbr.ru/rss/eventrss", ()),
    "kommersant": ("https://www.kommersant.ru/RSS/news.xml", ("Экономика", "Индустрии", "Медиа и Телеком", "Новости")),
    "interfax": ("https://www.interfax.ru/rss.asp?r=business", ()),
    "tass": ("https://tass.ru/rss/v2.xml?sections=ekonomika", ()),
}


def set_paths(name: str = "real") -> tuple[Path, Path]:
    """Файлы замороженного набора: real -> real_news.jsonl и real_news.lock, real2 -> real2_news.jsonl и real2_news.lock."""
    return EVAL_DIR / f"{name}_news.jsonl", EVAL_DIR / f"{name}_news.lock"


def labels_path(name: str = "real") -> Path:
    return EVAL_DIR / f"labels_{name}.jsonl"


NEWS_PATH, LOCK_PATH = set_paths()


def raw_id(item: RawItem) -> str:
    return "r_" + hashlib.sha1((item.url or item.title).encode()).hexdigest()[:12]


def parse_feed(body: bytes, url: str, categories: tuple[str, ...] = ()) -> list[RawItem]:
    """Записи ленты с датой; при `categories` только записи этих рубрик."""
    root = ET.fromstring(body)
    items = items_from_rss(root, url, require_date=True)
    if not categories:
        return items
    keep = {(it.findtext("link") or "").strip() for it in root.iter("item") if (it.findtext("category") or "").strip() in categories}
    return [i for i in items if i.url in keep]


def in_window(item: RawItem, as_of: date, days: int) -> bool:
    d = date.fromisoformat(item.published_at)
    return as_of - timedelta(days=days - 1) <= d <= as_of


def order_balanced(items: list[RawItem], seed: int) -> list[RawItem]:
    """Порядок отбора: источники по кругу, внутри источника темы по кругу, нерелевантное в конце каждого круга источника."""
    rnd = random.Random(seed)
    by_src: dict[str, dict[str, list[RawItem]]] = {}
    for it in sorted(items, key=lambda x: x.id):
        k = OFFTOPIC if is_offtopic(it) else stratum(it)
        by_src.setdefault(it.publisher or "", {}).setdefault(k, []).append(it)
    queues = {}
    for src, strata in by_src.items():
        for v in strata.values():
            rnd.shuffle(v)
        keys = sorted(k for k in strata if k != OFFTOPIC) + ([OFFTOPIC] if OFFTOPIC in strata else [])
        q = []
        while any(strata.get(k) for k in keys):
            q += [strata[k].pop() for k in keys if strata.get(k)]
        queues[src] = q
    out = []
    while any(queues.values()):
        for src in sorted(queues):
            if queues[src]:
                out.append(queues[src].pop(0))
    return out


def pick(ordered: list[RawItem], n: int, offtopic_share: float = OFFTOPIC_SHARE) -> list[RawItem]:
    off = [i for i in ordered if is_offtopic(i)][: round(n * offtopic_share)]
    on = [i for i in ordered if not is_offtopic(i)][: n - len(off)]  # нерелевантных мало: добираем релевантными
    return on + off


def collect(contents: dict[str, tuple[bytes, tuple[str, ...]]], as_of: date, days: int, n: int, holdout_days: int, seed: int,
            fetched_at: str, full_text=with_full_text) -> tuple[list[RawItem], dict]:
    """contents: url -> (тело RSS, рубрики). Возвращает (материалы со split, статистика)."""
    raw = [i for url, (body, cats) in contents.items() for i in parse_feed(body, url, cats)]
    return collect_raw(raw, as_of, days, n, holdout_days, seed, fetched_at, full_text)


def collect_raw(raw: list[RawItem], as_of: date, days: int, n: int, holdout_days: int, seed: int,
                fetched_at: str, full_text=with_full_text) -> tuple[list[RawItem], dict]:
    """Отбор из записей лент: окно дат, дедупликация, квоты dev/holdout, полный текст."""
    win = [i for i in raw if in_window(i, as_of, days)]
    uniq = dedup_raw(win)
    boundary = as_of - timedelta(days=holdout_days - 1)
    # Квоты по частям: иначе свежие дни (больше лент с длинными описаниями) забирают почти весь набор.
    n_hold = round(n * HOLDOUT_SHARE) if holdout_days > 0 else 0  # 0: весь набор одной частью (итоговый real2)
    chosen, texted = [], []
    for part, quota in (("dev", n - n_hold), ("holdout", n_hold)):
        items = [i for i in uniq if (date.fromisoformat(i.published_at) >= boundary) == (part == "holdout")]
        got = []
        # Полный текст запрашивается только для кандидатов с запасом, а не для всей ленты.
        for it in order_balanced(items, seed):
            if len(got) >= int(quota * 1.6):
                break
            full = full_text(it, MIN_TEXT)
            if len(full.text) >= MIN_TEXT or (len(full.text) >= MIN_TEXT_NO_PAGE and full.text == it.text):
                got.append(full)
        keep = {i.id for i in dedup_raw(got)}  # дубликаты после подстановки полного текста
        chosen += pick([i for i in got if i.id in keep], quota)
        texted += got
    out = [RawItem(id=raw_id(i), title=i.title, text=i.text, published_at=i.published_at, publisher=i.publisher, url=i.url,
                   kind="news", fetched_at=i.fetched_at or fetched_at)
           for i in chosen]
    split = {i.id: ("holdout" if date.fromisoformat(i.published_at) >= boundary else "dev") for i in out}
    counts: dict[str, dict[str, int]] = {}
    for i in out:
        c = counts.setdefault(split[i.id], {})
        k = OFFTOPIC if is_offtopic(i) else stratum(i)
        c[k] = c.get(k, 0) + 1
    by_src: dict[str, int] = {}
    for i in out:
        by_src[i.publisher or ""] = by_src.get(i.publisher or "", 0) + 1
    stats = {"records_with_date": len(raw), "in_window": len(win), "after_dedup": len(uniq), "with_text": len(texted),
             "selected": len(out), "dev": sum(v == "dev" for v in split.values()), "holdout": sum(v == "holdout" for v in split.values()),
             "holdout_from": boundary.isoformat(), "by_source": by_src, "strata": counts}
    return out, {**stats, "split": split}


def write_frozen(items: list[RawItem], stats: dict, sources: list[str], seed: int, as_of: str, days: int,
                 news_path: Path = NEWS_PATH, lock_path: Path = LOCK_PATH) -> dict:
    split = stats["split"]
    lines = [json.dumps({**i.__dict__, "split": split[i.id]}, ensure_ascii=False) for i in items]
    news_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    lock = {"file": news_path.name, "sha256": hashlib.sha256(news_path.read_bytes()).hexdigest(), "n": len(items),
            "as_of": as_of, "window_days": days, "seed": seed, "sources": sources,
            **{k: v for k, v in stats.items() if k != "split"}}
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return lock


def load_real(split: str | None = None, name: str = "real") -> list[RawItem]:
    """Замороженный набор. split: dev, holdout или None (все)."""
    rows = [json.loads(l) for l in set_paths(name)[0].read_text(encoding="utf-8").splitlines() if l.strip()]
    return [RawItem(**{k: v for k, v in r.items() if k != "split"}) for r in rows if split is None or r["split"] == split]


def real_split(name: str = "real") -> dict[str, str]:
    return {json.loads(l)["id"]: json.loads(l)["split"] for l in set_paths(name)[0].read_text(encoding="utf-8").splitlines() if l.strip()}


def load_pool(pool_path: Path = POOL_PATH) -> list[RawItem]:
    if not pool_path.exists():
        return []
    return [RawItem(**json.loads(l)) for l in pool_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def accumulate(contents: dict[str, tuple[bytes, tuple[str, ...]]], fetched_at: str, pool_path: Path = POOL_PATH,
               full_text=with_full_text) -> dict:
    """Дописывает в пул записи лент, которых в нём ещё нет (id по ссылке), с полным текстом статьи. Старые записи не меняются."""
    have = {i.id for i in load_pool(pool_path)}
    raw = [i for url, (body, cats) in contents.items() for i in parse_feed(body, url, cats)]
    new = []
    for it in raw:
        rid = raw_id(it)
        if rid in have:
            continue
        have.add(rid)
        full = full_text(it, MIN_TEXT)
        new.append(RawItem(id=rid, title=it.title, text=full.text, published_at=it.published_at, publisher=it.publisher,
                           url=it.url, kind="news", fetched_at=fetched_at))
    with pool_path.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(i.__dict__, ensure_ascii=False) + "\n" for i in new)
    days = sorted({i.published_at for i in load_pool(pool_path)})
    return {"records_with_date": len(raw), "new": len(new), "pool": len(have), "days": f"{days[0]}..{days[-1]}" if days else ""}


def collect_from_pool(pool: list[RawItem], start: date, as_of: date, n: int, holdout_days: int, seed: int,
                      fetched_at: str) -> tuple[list[RawItem], dict]:
    """Набор из пула за [start; as_of]: тот же отбор, что у сбора из лент; полный текст уже в пуле."""
    days = (as_of - start).days + 1
    return collect_raw([i for i in pool if date.fromisoformat(i.published_at) >= start], as_of, days, n, holdout_days, seed,
                       fetched_at, full_text=lambda it, _n: it)


def main(argv: list[str] | None = None) -> None:
    import httpx

    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=date.today().isoformat())
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--holdout-days", type=int, default=3)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="перезаписать замороженный набор")
    ap.add_argument("--name", default="real", help="имя набора: файлы ml/eval/<name>_news.jsonl и <name>_news.lock")
    ap.add_argument("--accumulate", action="store_true", help="только дописать новые записи лент в пул real_pool.jsonl")
    ap.add_argument("--from-pool", action="store_true", help="заморозить набор из пула, а не из текущих лент")
    ap.add_argument("--start", help="--from-pool: первый день набора (новости раньше не берутся)")
    args = ap.parse_args(argv)
    news_path, lock_path = set_paths(args.name)
    if lock_path.exists() and not args.force and not args.dry_run and not args.accumulate:
        sys.exit(f"{lock_path.name} уже есть: набор заморожен. Повторный сбор только с --force.")
    as_of = date.fromisoformat(args.as_of)
    now = datetime.now().isoformat(timespec="seconds")
    if args.from_pool:
        if not args.start:
            sys.exit("--from-pool требует --start")
        items, stats = collect_from_pool(load_pool(), date.fromisoformat(args.start), as_of, args.n, args.holdout_days, args.seed, now)
        print(json.dumps({k: v for k, v in stats.items() if k != "split"}, ensure_ascii=False, indent=1), file=sys.stderr)
        if not args.dry_run:
            lock = write_frozen(items, stats, ["pool:" + POOL_PATH.name], args.seed, args.as_of, (as_of - date.fromisoformat(args.start)).days + 1,
                                news_path, lock_path)
            print(f"записано {lock['n']} материалов (dev {lock['dev']}, holdout {lock['holdout']}), sha256 {lock['sha256'][:12]}", file=sys.stderr)
        return
    contents = {}
    for name, (url, cats) in SOURCES.items():
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (alpha-hack-news-bot/0.1)"})
            r.raise_for_status()
            ET.fromstring(r.content)
            contents[url] = (r.content, cats)
        except Exception as e:  # noqa: BLE001  недоступная лента не должна ронять сбор
            print(f"пропуск {name}: {type(e).__name__}", file=sys.stderr)
    if args.accumulate:
        print(json.dumps(accumulate(contents, now), ensure_ascii=False), file=sys.stderr)
        return
    items, stats = collect(contents, as_of, args.days, args.n, args.holdout_days, args.seed, now)
    print(json.dumps({k: v for k, v in stats.items() if k != "split"}, ensure_ascii=False, indent=1), file=sys.stderr)
    if args.dry_run:
        return
    lock = write_frozen(items, stats, sorted(contents), args.seed, args.as_of, args.days, news_path, lock_path)
    print(f"записано {lock['n']} материалов (dev {lock['dev']}, holdout {lock['holdout']}), sha256 {lock['sha256'][:12]}", file=sys.stderr)


if __name__ == "__main__":
    main()
