"""Разметка релевантности двумя LLM (IMPROVEMENTS.md, E3).

    python -m ml.eval.label_llm             # все компании companies_v2.json x материалы eval
    python -m ml.eval.label_llm --mock

Каждую пару «компания x материал» независимо размечают две модели разных
семейств по одной инструкции (`prompts/label_relevance.md`), один вызов на
компанию. Результат `ml/eval/labels.jsonl`:
- source="human": ручная разметка `labels.py` (исходные 9 компаний), модели для сверки;
- source="llm_agreed": модели совпали;
- source="llm_disputed": разошлись, grade = меньшая оценка, пара идёт на ручную проверку
  (`ml/eval/review_labels.md`); после проверки поменять source на "human".

Метрики ленты считаются отдельно по human и по human + llm_agreed.
Согласие: доля совпадений и κ Коэна (взвешенная) модель-модель и модель-человек.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from ml.config import settings

EVAL_DIR = Path(__file__).parent
LABELS_OUT = EVAL_DIR / "labels.jsonl"
REVIEW_OUT = EVAL_DIR / "review_labels.md"
TASKS = ("label_a", "label_b")


class Label(BaseModel):
    item_id: str
    grade: int
    reason: str = ""


class Labels(BaseModel):
    labels: list[Label] = []


def load_v2() -> list[dict]:
    return json.loads((EVAL_DIR / "companies_v2.json").read_text(encoding="utf-8"))


def materials(items: list) -> list[dict]:
    from ml.ingest.analytics import is_stale

    return [{"item_id": i.item_id, "kind": i.source.kind.value, "date": i.source.published_at.isoformat(),
             "stale": is_stale(i.source.published_at), "title": i.title, "summary": (i.summary or "")[:500],
             "facts": [f.text for f in i.facts[:4]]} for i in items]


def label_company(client, task: str, profile, mats: list[dict], batch: int = 0) -> dict[str, Label]:
    """Оценки модели для всех материалов; batch > 0: материалы частями (длинный список не помещается в ответ модели)."""
    if batch and len(mats) > batch:
        out: dict[str, Label] = {}
        for k in range(0, len(mats), batch):
            out.update(label_company(client, task, profile, mats[k:k + batch]))
        return out
    from ml.prompts import render

    p = render("label_relevance", as_of=settings.as_of.isoformat(), profile=profile.prompt_view(), materials=mats)
    res, _ = client.complete_json(task, p.system, p.user, Labels, mock_context={"materials": mats}, prompt_version=p.version)
    ids = {m["item_id"] for m in mats}
    return {l.item_id: Label(item_id=l.item_id, grade=min(2, max(0, l.grade)), reason=l.reason) for l in res.labels if l.item_id in ids}


def cohen_kappa(a: list[int], b: list[int], weighted: bool = True) -> float | None:
    """κ Коэна для шкалы 0/1/2; weighted: квадратичные веса (ошибка 0↔2 хуже 0↔1)."""
    if not a:
        return None
    k = 3
    w = [[((i - j) ** 2) / ((k - 1) ** 2) if weighted else float(i != j) for j in range(k)] for i in range(k)]
    n = len(a)
    obs = sum(w[x][y] for x, y in zip(a, b)) / n
    pa = [a.count(i) / n for i in range(k)]
    pb = [b.count(i) / n for i in range(k)]
    exp = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return 1 - obs / exp if exp else None


def agreement(a: list[int], b: list[int]) -> dict:
    return {"n": len(a), "exact": sum(x == y for x, y in zip(a, b)) / len(a) if a else None,
            "kappa_w": cohen_kappa(a, b), "kappa": cohen_kappa(a, b, weighted=False)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--set", choices=("v2", "real", "real2"), default="v2",
                    help="real, real2: реальные новости ml/eval/<set>_news.jsonl (разбор платный при промахе кэша), запуск с ML_AS_OF из <set>_news.lock")
    ap.add_argument("--batch", type=int, default=0, help="материалов на один вызов (0: все сразу, как для v2)")
    ap.add_argument("--max-rub", type=float, default=0.0, help="остановить новые вызовы, когда расход прогона превысит лимит (0: без лимита)")
    ap.add_argument("--force", action="store_true", help="real: перезаписать готовую разметку")
    args = ap.parse_args(argv)
    if args.set.startswith("real"):
        return main_real(args)

    from ml.eval.labels import DUPLICATES, LABELS
    from ml.eval.run_eval import load_eval_items
    from ml.ingest.dedup import representatives
    from ml.llm.client import LLMClient, SpendLedger
    from ml.llm.router import model_name, route
    from ml.profile.build_profile import get_profile

    ledger = SpendLedger()
    day0 = ledger.spent_today()
    items, _, _ = load_eval_items(mock=args.mock, cache_only=not args.mock)
    pool = representatives(items)
    mats = materials(pool)
    # Дубликаты из пула выпадают; их оценка берётся у представителя кластера.
    rep_of = {i.item_id: next((p.item_id for p in pool if p.dedup_cluster and p.dedup_cluster == i.dedup_cluster), i.item_id)
              for i in items}
    client = LLMClient(mock=args.mock, use_cache=True)
    companies = load_v2()

    def run(c: dict) -> tuple[str, dict]:
        prof = get_profile(c["id"])
        return c["id"], {t: label_company(client, t, prof, mats) for t in TASKS}

    with ThreadPoolExecutor(args.workers) as ex:
        got = dict(ex.map(run, companies))

    rows, mm, human_a, human_b, human_g = [], ([], []), [], [], []
    for c in companies:
        res = got[c["id"]]
        for it in items:
            rid = rep_of[it.item_id]
            la, lb = res["label_a"].get(rid), res["label_b"].get(rid)
            if la is None or lb is None:
                continue
            mm[0].append(la.grade)
            mm[1].append(lb.grade)
            row = {"company": c["id"], "key": c["key"], "split": c["split"], "item": it.item_id,
                   "grade_a": la.grade, "grade_b": lb.grade, "reason_a": la.reason, "reason_b": lb.reason}
            if c["labels"] == "human":
                g = LABELS.get(it.item_id, {}).get(c["key"], 0)
                human_a.append(la.grade)
                human_b.append(lb.grade)
                human_g.append(g)
                row.update(source="human", grade=g)
            elif la.grade == lb.grade:
                row.update(source="llm_agreed", grade=la.grade)
            else:
                row.update(source="llm_disputed", grade=min(la.grade, lb.grade))
            rows.append(row)

    with LABELS_OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dup_items = {b for _, b in DUPLICATES}
    disputed = [r for r in rows if r["source"] == "llm_disputed" and r["item"] not in dup_items]
    L = ["# Разметка на ручную проверку", "",
         "Пары, где модели разошлись. Поставьте свою оценку в колонку «Ваша» (0/1/2) и перенесите в `labels.jsonl` "
         "с source=\"human\". Шкала: `ml/prompts/label_relevance.md`.", "",
         "| Компания | Отрасль | Материал | A | B | Причина A | Причина B | Ваша |", "|---|---|---|---|---|---|---|---|"]
    okved = {c["id"]: (c["okved"] or "")[:45] for c in companies}
    for r in sorted(disputed, key=lambda r: (r["company"], r["item"])):
        L.append(f"| {r['company']} | {okved[r['company']]} | {r['item']} | {r['grade_a']} | {r['grade_b']} | "
                 f"{r['reason_a'][:90]} | {r['reason_b'][:90]} | |")
    REVIEW_OUT.write_text("\n".join(L) + "\n", encoding="utf-8")

    summary = {
        "models": {t: model_name(route(t).family) for t in TASKS},
        "pairs": len(rows),
        "by_source": {s: sum(r["source"] == s for r in rows) for s in ("human", "llm_agreed", "llm_disputed")},
        "model_vs_model": agreement(*mm),
        "model_a_vs_human": agreement(human_a, human_g),
        "model_b_vs_human": agreement(human_b, human_g),
        "agreed_vs_human": agreement([a for a, b in zip(human_a, human_b) if a == b],
                                     [g for a, b, g in zip(human_a, human_b, human_g) if a == b]),
        "cost_rub": ledger.spent_today() - day0,
    }
    (EVAL_DIR / "labels_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1), file=sys.stderr)
    print(f"-> {LABELS_OUT}, на проверку {len(disputed)} пар -> {REVIEW_OUT}", file=sys.stderr)


def main_real(args) -> None:
    """Разметка реальных новостей: только две модели (ручной разметки нет), спорные пары в метрики не входят."""
    from ml.eval.run_eval import check_real_as_of, load_real_items
    from ml.llm.client import LLMClient, SpendLedger
    from ml.llm.router import model_name, route
    from ml.profile.build_profile import get_profile

    check_real_as_of(args.set)
    out = EVAL_DIR / f"labels_{args.set}.jsonl"
    if out.exists() and not args.force:
        sys.exit(f"{out.name} уже есть: разметка не перезаписывается без --force")
    ledger = SpendLedger()
    day0 = ledger.spent_today()
    items = load_real_items(mock=args.mock, name=args.set)  # платно при промахе кэша: разбор и сигналы Gemma
    parse_rub = ledger.spent_today() - day0
    from ml.ingest.dedup import representatives

    pool = representatives(items)
    mats = materials(pool)
    rep_of = {i.item_id: next((p.item_id for p in pool if p.dedup_cluster and p.dedup_cluster == i.dedup_cluster), i.item_id)
              for i in items}
    client = LLMClient(mock=args.mock, use_cache=True)
    companies = load_v2()
    print(f"разбор {parse_rub:.2f} руб.; материалов {len(mats)}, компаний {len(companies)}, пар {len(mats) * len(companies)}",
          file=sys.stderr)

    def run(c: dict) -> tuple[str, dict | None]:
        if args.max_rub and ledger.spent_today() - day0 > args.max_rub:
            return c["id"], None
        prof = get_profile(c["id"])
        return c["id"], {t: label_company(client, t, prof, mats, args.batch) for t in TASKS}

    with ThreadPoolExecutor(args.workers) as ex:
        got = dict(ex.map(run, companies))
    rows, mm = [], ([], [])
    for c in companies:
        res = got.get(c["id"])
        if res is None:
            continue
        for it in items:
            rid = rep_of[it.item_id]
            la, lb = res["label_a"].get(rid), res["label_b"].get(rid)
            if la is None or lb is None:
                continue
            mm[0].append(la.grade)
            mm[1].append(lb.grade)
            row = {"company": c["id"], "key": c["key"], "split": c["split"], "item": it.item_id,
                   "grade_a": la.grade, "grade_b": lb.grade, "reason_a": la.reason, "reason_b": lb.reason,
                   "source": "llm_agreed" if la.grade == lb.grade else "llm_disputed", "grade": min(la.grade, lb.grade)}
            rows.append(row)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {"models": {t: model_name(route(t).family) for t in TASKS}, "as_of": settings.as_of.isoformat(), "pairs": len(rows),
               "companies_labelled": sum(v is not None for v in got.values()),
               "by_source": {s: sum(r["source"] == s for r in rows) for s in ("llm_agreed", "llm_disputed")},
               "model_vs_model": agreement(*mm), "parse_rub": parse_rub, "cost_rub": ledger.spent_today() - day0}
    (EVAL_DIR / f"labels_{args.set}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
