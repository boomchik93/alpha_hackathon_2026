"""Предрасчёт карточек для демо: лента -> топ-N -> карточка, валидные сохранённые пропускаются.

    python -m ml.precompute --dry-run                         # число карточек и оценка стоимости, вызовов нет
    python -m ml.precompute --companies demo --top 3 --max-rub 15 --workers 6
    python -m ml.precompute --companies U_1595,group_134463_sb --role cfo
    python -m ml.precompute --companies all                   # все профили: 413 x 3 карточки ~150-200 руб.

`demo` — 30 крупнейших по выручке профилей. Каждый запуск останавливается, как только расход прогона превысил --max-rub.
Платно: запускать только по решению команды, бюджет общий на всех (500 руб. в день).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from ml import api
from ml.llm.client import BudgetExceeded, SpendLedger
from ml.roles import Role

DEMO_COMPANIES = 30
COST_PER_CARD_RUB = 0.16  # среднее по прогону dev (0,14-0,16 руб.), без судьи


def resolve_companies(spec: str) -> list[str]:
    profiles = api.list_companies()
    if spec == "all":
        return [p.company_id for p in profiles]
    if spec == "demo":
        return [p.company_id for p in sorted(profiles, key=lambda p: -(p.size.revenue_rub or 0))[:DEMO_COMPANIES]]
    return [x.strip() for x in spec.split(",") if x.strip()]


def plan(companies: list[str], top: int, role: Role | None) -> tuple[list[tuple[str, str]], int]:
    """(пары «компания, материал» без свежей сохранённой карточки; сколько карточек уже есть)."""
    todo, cached = [], 0
    for cid in companies:
        for e in api.get_ranked_feed(cid, limit=top).entries[:top]:
            if api.cached_insight(cid, e.item_id, role) is not None:
                cached += 1
            else:
                todo.append((cid, e.item_id))
    return todo, cached


def run(todo: list[tuple[str, str]], max_rub: float, workers: int, role: Role | None) -> dict:
    ledger = SpendLedger()
    start = ledger.spent_today()
    stop = {"budget": False}
    res = {"made": 0, "failed": [], "skipped_budget": 0}

    def one(pair: tuple[str, str]) -> None:
        if stop["budget"] or ledger.spent_today() - start > max_rub:
            stop["budget"] = True
            res["skipped_budget"] += 1
            return
        try:
            api.get_insight(pair[0], pair[1], role=role)
            res["made"] += 1
        except BudgetExceeded:
            stop["budget"] = True
            res["skipped_budget"] += 1
        except Exception as e:  # noqa: BLE001  ошибка одной карточки не должна останавливать предрасчёт
            res["failed"].append(f"{pair[0]} x {pair[1]}: {type(e).__name__}: {str(e)[:150]}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    res["spent_rub"] = ledger.spent_today() - start
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", default="demo", help="demo (30 крупнейших), all или список id через запятую")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--max-rub", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--role", choices=[r.value for r in Role])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    role = Role(args.role) if args.role else None

    companies = resolve_companies(args.companies)
    todo, cached = plan(companies, args.top, role)
    est = len(todo) * COST_PER_CARD_RUB
    print(f"компаний {len(companies)}, карточек к созданию {len(todo)}, уже готово {cached}, оценка {est:.1f} руб. "
          f"(~{COST_PER_CARD_RUB} руб. за карточку), лимит прогона {args.max_rub} руб.", file=sys.stderr)
    if args.dry_run:
        return
    if est > args.max_rub:
        print(f"предупреждение: оценка выше --max-rub, прогон остановится по лимиту, часть карточек останется без предрасчёта", file=sys.stderr)
    t0 = time.time()
    res = run(todo, args.max_rub, args.workers, role)
    print(f"готово за {time.time() - t0:.0f} с: создано {res['made']}, ошибок {len(res['failed'])}, "
          f"пропущено по лимиту {res['skipped_budget']}, расход {res['spent_rub']:.2f} руб.", file=sys.stderr)
    for f in res["failed"][:10]:
        print("  ", f, file=sys.stderr)


if __name__ == "__main__":
    main()
