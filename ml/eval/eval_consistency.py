"""Целевой набор для сверки противоречий (REVIEW_PLAN.md, T15): precision / recall статуса `contradicts`.

    python -m ml.eval.eval_consistency --mock      # без расходов, проверка скрипта
    python -m ml.eval.eval_consistency             # реальная модель (Gemma), ~10 вызовов, ~0,5 руб.

Набор `ml/eval/contradictions.jsonl` синтетический (написан командой): 10 пар «материал + фрагмент банка»,
5 расхождений, 3 согласованных, 2 несвязанных.
"""

from __future__ import annotations

import argparse
import json

from ml.agent.consistency import check
from ml.eval.run_eval import EVAL_DIR
from ml.llm.client import LLMClient, SpendLedger

STATUSES = ("contradicts", "consistent", "unrelated")


def load() -> list[dict]:
    return [json.loads(l) for l in (EVAL_DIR / "contradictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def score(expected: list[str], got: list[str]) -> dict:
    """precision / recall по `contradicts`, точность по трём статусам и матрица ошибок."""
    tp = sum(e == g == "contradicts" for e, g in zip(expected, got))
    pred, real = sum(g == "contradicts" for g in got), sum(e == "contradicts" for e in expected)
    conf = {e: {g: sum(x == e and y == g for x, y in zip(expected, got)) for g in STATUSES} for e in STATUSES}
    return {"n": len(expected), "precision": tp / pred if pred else None, "recall": tp / real if real else None,
            "accuracy": sum(e == g for e, g in zip(expected, got)) / len(expected), "confusion": conf}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args(argv)
    client = LLMClient(mock=args.mock, use_cache=not args.mock)
    ledger = SpendLedger()
    day0 = ledger.spent_today()
    rows = load()
    got = [check(client, r["title"], r["facts"], r["chunks"]).status for r in rows]
    res = score([r["expected"] for r in rows], got)
    for r, g in zip(rows, got):
        print(f"{r['id']} ожидалось {r['expected']:<12} получено {g:<12} {'OK' if r['expected'] == g else 'ОШИБКА'}")
    print(json.dumps(res, ensure_ascii=False))
    print(f"режим: {'мок' if args.mock else 'реальная модель'}; расход {ledger.spent_today() - day0:.3f} руб.")


if __name__ == "__main__":
    main()
