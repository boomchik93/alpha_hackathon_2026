"""Боевой режим: новости без остановки, разбор и карточки для тестовых аккаунтов.

    python -m ml.live loop      # бесконечный цикл: проход раз в LIVE_INTERVAL_S секунд, каждый в отдельном процессе
    python -m ml.live once      # один проход
    python -m ml.live once --no-cards

Проход:
1. сборщик новостей (`backend/news_collector`) дописывает свежие RSS и выгружает события за `LIVE_COLLECT_DAYS` дней;
2. фрагменты аналитики банка пересобираются раз в день: от даты зависит пометка «устарело» (бесплатно);
3. `var/items.jsonl`: документы банка и новости сборщика, разбор и важность через Gemma (платно только при промахе кэша);
4. фон BM25 ленты;
5. карточки для основной ленты каждой пары «компания + роль» из `backend/seed_users.json`, самые релевантные первыми.
   Генерация останавливается, когда расход за день по журналу достигает `LIVE_CARDS_BUDGET_RUB`; остальные карточки
   делаются по клику пользователя (до общего порога клиента `LLM_BUDGET_STOP_RATIO`).

Нужны `NEWS_SOURCE=collector` и `ML_AS_OF=today` (docker-compose.yml, сервис pipeline). Работающий ML-сервис подхватывает
новые файлы сам (`ml/livedata.py`). Каждый проход идёт в новом процессе: дата «сегодня» и кэши в памяти всегда свежие.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from ml.config import ROOT, settings
from ml.llm.client import BudgetExceeded, SpendLedger
from ml.roles import Role

INTERVAL_S = int(os.getenv("LIVE_INTERVAL_S") or 900)
COLLECT_DAYS = int(os.getenv("LIVE_COLLECT_DAYS") or 3)
CARDS_BUDGET_RUB = float(os.getenv("LIVE_CARDS_BUDGET_RUB") or 100)
PARSE_WORKERS = int(os.getenv("LIVE_PARSE_WORKERS") or 6)
CARD_WORKERS = int(os.getenv("LIVE_CARD_WORKERS") or 4)
FEED_LIMIT = 50  # верхняя граница основной ленты на пару: больше в ленту и так не попадает
SEED_PATH = ROOT / "backend" / "seed_users.json"
STATE_PATH = settings.paths.var / "live_state.json"


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} live: {msg}", file=sys.stderr, flush=True)


def seed_pairs(path=SEED_PATH) -> list[tuple[str, Role | None]]:
    """Пары «компания + роль» обычных пользователей из сида; роль other карточку не меняет (ml/roles.py), для неё None."""
    pairs = []
    for u in json.loads(path.read_text(encoding="utf-8")):
        if u.get("access", "user") != "user":
            continue
        role = None if u["role"] == "other" else Role(u["role"])
        pairs += [(c, role) for c in u.get("companies", []) if (c, role) not in pairs]
    return pairs


def _state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def collect() -> None:
    cmd = [sys.executable, str(ROOT / "backend" / "news_collector" / "main.py"), "run", "--once", "--days", str(COLLECT_DAYS)]
    res = subprocess.run(cmd, cwd=ROOT, timeout=1800)
    if res.returncode != 0:
        log(f"сборщик завершился с кодом {res.returncode}: работаем на прежней выгрузке")


def rebuild() -> None:
    from ml.fileio import write_atomic
    from ml.ingest import analytics
    from ml.ingest.store import build_items
    from ml.ranking.bm25 import build_background

    state, today = _state(), settings.as_of.isoformat()
    if state.get("chunks_day") != today:
        analytics.main()
        analytics.load_chunks.cache_clear()
        state["chunks_day"] = today
        write_atomic(STATE_PATH, json.dumps(state))
    items = build_items(importance=True, skip_failed=True, workers=PARSE_WORKERS)
    news = sum(1 for i in items if i.source.kind.value != "bank_analytics")
    log(f"материалов {len(items)}, из них новостей {news}")
    build_background()


def cards(pairs: list[tuple[str, Role | None]], budget_rub: float = CARDS_BUDGET_RUB) -> dict:
    """Карточки основной ленты пар, по убыванию оценки; стоп, когда расход за день по журналу дошёл до budget_rub."""
    from ml import api

    todo, ready = [], 0
    for cid, role in pairs:
        for e in api.get_ranked_feed(cid, limit=FEED_LIMIT, maybe_limit=0, tier="main").entries:
            if api.cached_insight(cid, e.item_id, role) is not None:
                ready += 1
            else:
                todo.append((e.score, cid, e.item_id, role))
    todo.sort(key=lambda t: -t[0])
    ledger = SpendLedger()
    res = {"ready": ready, "todo": len(todo), "made": 0, "failed": 0, "skipped_budget": 0}
    lock = threading.Lock()

    def count(key: str) -> None:
        with lock:
            res[key] += 1

    def one(t) -> None:
        _, cid, item_id, role = t
        if ledger.spent_today() >= budget_rub:
            count("skipped_budget")
            return
        try:
            api.get_insight(cid, item_id, role=role)
            count("made")
        except BudgetExceeded:
            count("skipped_budget")
        except Exception as e:  # noqa: BLE001 - одна карточка не должна останавливать проход
            count("failed")
            log(f"карточка {cid} x {item_id} ({role.value if role else '-'}): {type(e).__name__}: {str(e)[:150]}")

    with ThreadPoolExecutor(max_workers=CARD_WORKERS) as ex:
        list(ex.map(one, todo))
    return res


def once(with_cards: bool = True) -> None:
    t0 = time.time()
    collect()
    try:
        rebuild()
    except BudgetExceeded as e:
        log(f"сборка остановлена по бюджету: {e}")
        return
    if with_cards:
        log(f"карточки: {cards(seed_pairs())}")
    log(f"проход за {time.time() - t0:.0f} с, расход за день {SpendLedger().spent_today():.2f} руб.")


def loop() -> None:
    """Каждый проход в отдельном процессе: падение прохода не останавливает цикл."""
    while True:
        res = subprocess.run([sys.executable, "-m", "ml.live", "once"], cwd=ROOT)
        if res.returncode != 0:
            log(f"проход завершился с кодом {res.returncode}")
        time.sleep(INTERVAL_S)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["loop", "once"])
    ap.add_argument("--no-cards", action="store_true", help="once: без генерации карточек")
    args = ap.parse_args(argv)
    if os.getenv("NEWS_SOURCE") != "collector":
        sys.exit("нужен NEWS_SOURCE=collector: боевой режим берёт новости только у сборщика")
    if settings.llm.mock:
        # Карточки мока сохранились бы рядом с настоящими: боевой режим без модели не запускаем.
        sys.exit("LLM_MOCK включён или не задан LLM_BASE_URL: боевой режим работает только с настоящими моделями")
    if args.command == "loop":
        loop()
    else:
        once(with_cards=not args.no_cards)


if __name__ == "__main__":
    main()
