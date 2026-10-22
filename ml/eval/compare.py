"""Парное сравнение двух прогонов eval на общих парах «компания x новость».

    python -m ml.eval.compare var/eval/<run_a> var/eval/<run_b> [--split dev]

Для каждой метрики судьи считается средняя разница b − a по одним и тем же парам и
95% интервал bootstrap по парам. Парное сравнение убирает разброс между компаниями
и новостями, поэтому различает меньшие эффекты, чем сравнение средних.
Разные версии судьи в двух прогонах сравнивать нельзя: разница будет в судье, а не в карточках.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BOOTSTRAP_N = 2000


def _load(run: str) -> dict[tuple[str, str], dict]:
    rows = [json.loads(l) for l in (Path(run) / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(r["company"], r["item"]): r for r in rows if r["status"] == "ok" and r.get("judge") and "error" not in r["judge"]}


METRICS = {
    "factual": lambda r: r["judge"]["factual"],
    "factual_2": lambda r: float(r["judge"]["factual"] == 2),
    "with_unsupported": lambda r: float(bool(r["judge"]["unsupported"])),
    "n_unsupported": lambda r: len(r["judge"]["unsupported"]),
    "bank_ai_mixed": lambda r: float(r["judge"]["bank_ai_mixed"]),
    "personalized": lambda r: r["judge"]["personalized"],
    "actionable": lambda r: r["judge"]["actionable"],
    "relevance": lambda r: r["relevance"] or 0.0,
    "cost_rub": lambda r: r["cost_rub"],
    "latency_s": lambda r: r["latency_s"],
    "no_retry": lambda r: float(r["retries"] == 0),
}


def compare(run_a: str, run_b: str, split: str | None = None, seed: int = 7) -> dict:
    a, b = _load(run_a), _load(run_b)
    keys = sorted(k for k in a.keys() & b.keys() if not split or a[k].get("split") == split)
    rnd = random.Random(seed)
    out = {"pairs": len(keys)}
    for name, f in METRICS.items():
        d = [f(b[k]) - f(a[k]) for k in keys]
        if not d:
            continue
        boots = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(BOOTSTRAP_N))
        out[name] = {"a": sum(f(a[k]) for k in keys) / len(keys), "b": sum(f(b[k]) for k in keys) / len(keys),
                     "diff": sum(d) / len(d), "ci": (boots[int(0.025 * BOOTSTRAP_N)], boots[int(0.975 * BOOTSTRAP_N) - 1])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--split", choices=("dev", "test"))
    args = ap.parse_args()
    res = compare(args.run_a, args.run_b, args.split)
    print(f"Общих пар: {res['pairs']}\n")
    print("| Метрика | A | B | B − A, 95% интервал |\n|---|---|---|---|")
    for name in METRICS:
        if name in res:
            m = res[name]
            print(f"| {name} | {m['a']:.3f} | {m['b']:.3f} | {m['diff']:+.3f} [{m['ci'][0]:+.3f}; {m['ci'][1]:+.3f}] |")


if __name__ == "__main__":
    main()
