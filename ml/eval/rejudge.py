"""Переоценка сохранённых карточек текущей версией судьи, без перегенерации карточек.

    python -m ml.eval.rejudge var/eval/<run> --split dev --pairs-from var/eval/<other> --dry-run
    python -m ml.eval.rejudge var/eval/<run> --split dev --pairs-from var/eval/<other> --max-rub 15

Нужна, когда меняется судья: `ml.eval.compare` честно сравнивает только прогоны, оценённые одной версией судьи.
Пишет `var/eval/<run>_rejudge_<версия>/results.jsonl` (строки исходного прогона с новой оценкой) и ссылку `cards`
на карточки исходного прогона. `--pairs-from`: только пары, которые есть и в другом прогоне (для парного сравнения).
Платно: один вызов судьи на карточку (~0,13-0,24 руб.).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ml.eval.run_eval import Budget, card_hits, judge_card, load_eval_items
from ml.llm.client import BudgetExceeded, LLMClient, SpendLedger
from ml.prompts import prompt_version
from ml.schemas import InsightCard

COST_PER_CARD_RUB = 0.24  # судья DeepSeek в прогоне 20261021_124003


def _ok_rows(run: Path, split: str | None) -> list[dict]:
    rows = [json.loads(l) for l in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["status"] == "ok" and (split is None or r.get("split") == split)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", choices=("dev", "test"))
    ap.add_argument("--pairs-from", help="оставить только пары, которые есть (status ok) в этом прогоне")
    ap.add_argument("--max-rub", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", choices=("v2", "real"), default="v2", help="real: реальные новости (запуск с ML_AS_OF из real_news.lock)")
    ap.add_argument("--tag", default="", help="суффикс папки результата: новая переоценка того же прогона в другом сеансе судьи")
    args = ap.parse_args(argv)

    run = Path(args.run)
    rows = _ok_rows(run, args.split)
    if args.pairs_from:
        other = {(r["company"], r["item"]) for r in _ok_rows(Path(args.pairs_from), args.split)}
        rows = [r for r in rows if (r["company"], r["item"]) in other]
    version = prompt_version("eval_judge").split("@")[-1]
    print(f"карточек к переоценке {len(rows)}, судья eval_judge@{version}, оценка {len(rows) * COST_PER_CARD_RUB:.1f} руб., "
          f"лимит {args.max_rub} руб.", file=sys.stderr)
    if args.dry_run:
        return

    from ml.profile.build_profile import get_profile
    from ml.ingest.store import get_item

    if args.set == "real":
        from ml.eval.run_eval import check_real_as_of, load_real_items

        check_real_as_of()
        load_real_items(cache_only=True)  # реальные новости в память процесса
    else:
        load_eval_items(cache_only=True)  # holdout-новости в память процесса
    client = LLMClient(use_cache=False)
    budget = Budget(start=SpendLedger().spent_today(), max_rub=args.max_rub)

    dst = run.parent / f"{run.name}_rejudge_{version}{'_' + args.tag if args.tag else ''}"
    # Дозапуск: уже переоценённые карточки (без ошибки) не оцениваются повторно.
    done_before = {}
    if (dst / "results.jsonl").exists():
        for l in (dst / "results.jsonl").read_text(encoding="utf-8").splitlines():
            x = json.loads(l)
            if x.get("judge") and "error" not in x["judge"]:
                done_before[(x["company"], x["item"])] = x

    def one(r: dict) -> dict:
        if (r["company"], r["item"]) in done_before:
            return done_before[(r["company"], r["item"])]
        out = {**r, "judge_version": version}
        if budget.exceeded():
            return {**out, "judge": {"error": "skipped_budget"}}
        card = InsightCard.model_validate(json.loads((run / "cards" / f"{r['card_id']}.json").read_text(encoding="utf-8")))
        try:
            v, cost = judge_card(client, card, get_profile(r["company"]), get_item(r["item"]), card_hits(card))
        except BudgetExceeded as e:
            budget.stopped = True
            return {**out, "judge": {"error": f"BudgetExceeded: {e}"}}
        except Exception as e:  # noqa: BLE001  ошибка одной карточки не должна ронять переоценку
            return {**out, "judge": {"error": f"{type(e).__name__}: {e}"[:300]}}
        return {**out, "judge": v, "judge_cost_rub": cost}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(one, rows))
    dst.mkdir(exist_ok=True)
    (dst / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    if not (dst / "cards").exists():
        (dst / "cards").symlink_to((run / "cards").resolve())
    done = sum("error" not in r["judge"] for r in results)
    print(f"переоценено {done} из {len(results)}, расход {SpendLedger().spent_today() - budget.start:.2f} руб. -> {dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
