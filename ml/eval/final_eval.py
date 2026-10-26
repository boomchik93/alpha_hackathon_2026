"""Итоговая оценка ленты на свежем наборе.

    ML_AS_OF=<as_of из real2_news.lock> python -m ml.eval.final_eval            # -> ml/eval/final_real2.md
    ML_AS_OF=2026-10-22 python -m ml.eval.final_eval --name real --out <файл>   # проверка кода на старом наборе (в выборке обучения)

Сравниваются три варианта ленты на одних и тех же парах, парный bootstrap по компаниям:
- «прежняя»: эвристика до шага 2 (без правил по масштабу события, без модели);
- «правила»: эвристика с правилами (ranker.ECONOMY_MIN_IMPORTANCE, NARROW_NEEDS_SECTOR, IN_SECTOR_KEEPS), без модели;
- «модель+правила»: рабочая лента с `ml/ranking/pair_model.json`.
Набор real2 замораживается до оценки и в обучение модели не входит. Запускать один раз, после разметки
(`python -m ml.eval.label_llm --set real2`); параметры после просмотра результата не менять (иначе нужен real3).
LLM не вызывается: разбор и сигналы материалов из кэша разметки.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from ml.config import settings
from ml.eval.run_eval import EVAL_DIR, check_real_as_of, load_gold_real, load_real_items
from ml.ranking import pair_model as pm
from ml.ranking import ranker

VARIANTS = ("прежняя", "правила", "модель+правила")


def scores_for(variant: str, companies: list[dict], items: list) -> dict[tuple[str, str], tuple[float, float]]:
    from ml.profile.build_profile import get_profile

    from ml import topics

    old = ranker.ECONOMY_MIN_IMPORTANCE, ranker.NARROW_NEEDS_SECTOR, ranker.IN_SECTOR_KEEPS, topics.TOPIC_OKVED
    if variant == "прежняя":
        ranker.ECONOMY_MIN_IMPORTANCE, ranker.NARROW_NEEDS_SECTOR, ranker.IN_SECTOR_KEEPS, topics.TOPIC_OKVED = None, False, False, "all"
    ranker.set_model(ranker.load_model() if variant == "модель+правила" else None)
    try:
        out = {}
        for c in companies:
            ss, _ = ranker.score_all(get_profile(c["id"]), items)
            out.update({(c["id"], s.item.item_id): (s.personal, s.score) for s in ss})
        return out
    finally:
        ranker.clear_model()
        ranker.ECONOMY_MIN_IMPORTANCE, ranker.NARROW_NEEDS_SECTOR, ranker.IN_SECTOR_KEEPS, topics.TOPIC_OKVED = old


def run(name: str) -> dict:
    from ml.eval.label_llm import load_v2

    if ranker.load_model() is None:
        sys.exit("нет ml/ranking/pair_model.json (или ML_RANK_MODEL=0): итоговая оценка рабочей ленты невозможна")
    check_real_as_of(name)
    companies = load_v2()
    items = load_real_items(cache_only=True, name=name)
    gold = load_gold_real(name=name)
    ranker.set_model(None)
    try:
        rows = pm.build_rows(companies, items, gold, name)
    finally:
        ranker.clear_model()
    scores = {v: scores_for(v, companies, items) for v in VARIANTS}
    res = {v: pm.metrics(rows, scores[v], ranker.MIN_SCORE) for v in VARIANTS}
    for v in VARIANTS:  # уровни ленты: у «прежней» нет правил (событие в отрасли, пол макро), у остальных они есть
        res[v]["tiers"] = pm.tier_metrics(pm.without_rule_flags(rows) if v == "прежняя" else rows, scores[v])
    paired = {f"{b} против {a}": pm.paired_delta(res[a], res[b])
              for a, b in (("прежняя", "модель+правила"), ("прежняя", "правила"), ("правила", "модель+правила"))}
    human = human_section(name, rows, scores)
    return {"set": name, "as_of": settings.as_of.isoformat(), "items": len({r.item for r in rows}),
            "pairs": sum(r.grade is not None for r in rows), "metrics": res, "paired": paired, "human": human,
            "model": {k: ranker.load_model().get(k) for k in ("trained_on", "w_model", "order_mix", "beta", "l2", "threshold")}}


def human_section(name: str, rows: list, scores: dict) -> dict | None:
    """Уровни ленты против ручной разметки (source=human в labels_<name>.jsonl), если она есть."""
    from ml.eval.label_review_real import human_pairs

    pairs = human_pairs(name)
    if not pairs:
        return None
    sub = [dataclasses.replace(r, grade=pairs[(r.company, r.item)]) for r in rows if (r.company, r.item) in pairs]
    return {"pairs": len(sub), "metrics": {v: pm.tier_metrics(pm.without_rule_flags(sub) if v == "прежняя" else sub, scores[v])
                                           for v in VARIANTS}}


def tier_lines(t: dict) -> str:
    m, b = t["main"], t["both"]
    return (f"{m['precision']:.2f} | {m['recall']:.2f} | {m['f05']:.3f} | {m['precision5']:.2f} | {b['recall']:.2f} | "
            f"{t['maybe_share_of_rel']:.2f} | {t['n_main']} / {t['n_maybe']}")


TIER_HEAD = ["| Вариант | precision осн. | recall осн. | F0.5 осн. | precision@5 осн. | recall обоих | доля нужного в «может быть» | пар осн. / «может быть» |",
             "|---|---|---|---|---|---|---|---|"]


def report(r: dict) -> list[str]:
    L = [f"# Итоговая оценка ленты: набор {r['set']}", "",
         f"as_of {r['as_of']}; материалов {r['items']}, пар с эталоном {r['pairs']} (40 компаний, разметка: согласие двух LLM, "
         f"спорные пары исключены). Модель: {json.dumps(r['model'], ensure_ascii=False)}.", "",
         "| Вариант | precision отсева | recall отсева | F1 отсева | nDCG@5 | precision@5 | AUC взв. |", "|---|---|---|---|---|---|---|"]
    for v, m in r["metrics"].items():
        L += [f"| {v} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.3f} | {m['ndcg5']:.3f} | {m['precision5']:.2f} | {m['auc']:.3f} |"]
    L += ["", "Парный bootstrap по компаниям (2000 выборок), разница и 95% интервал:", "", "| Сравнение | ΔF1 отсева | ΔnDCG@5 |", "|---|---|---|"]
    for k, d in r["paired"].items():
        L += [f"| {k} | {d['f1']['delta']:+.3f} [{d['f1']['ci95'][0]:+.3f}; {d['f1']['ci95'][1]:+.3f}] | "
              f"{d['ndcg5']['delta']:+.3f} [{d['ndcg5']['ci95'][0]:+.3f}; {d['ndcg5']['ci95'][1]:+.3f}] |"]
    L += ["", f"## Уровни ленты (основная от {ranker.MAIN_SCORE}, «может быть» от {ranker.MIN_SCORE}), эталон: разметка набора", "", *TIER_HEAD]
    L += [f"| {v} | {tier_lines(m['tiers'])} |" for v, m in r["metrics"].items()]
    if r.get("human"):
        L += ["", f"## Уровни ленты против человека ({r['human']['pairs']} пар с ручной оценкой)", "", *TIER_HEAD]
        L += [f"| {v} | {tier_lines(t)} |" for v, t in r["human"]["metrics"].items()]
    return L + [""]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="real2")
    ap.add_argument("--out", help="файл отчёта (по умолчанию ml/eval/final_<name>.md)")
    ap.add_argument("--without-human", action="store_true",
                    help="запустить без ручной разметки набора (для real2 это нарушает PROTOCOL.md: сначала ручная разметка)")
    args = ap.parse_args(argv)
    # PROTOCOL.md: на итоговом наборе ручная разметка делается до просмотра результатов ленты, иначе она перестаёт быть слепой.
    from ml.eval.label_review_real import human_pairs

    if args.name != "real" and not human_pairs(args.name) and not args.without_human:
        sys.exit(f"нет ручной разметки {args.name} (labels_{args.name}.jsonl, source=human): сначала make real2-review и "
                 "make real2-review-score, потом итоговая оценка (PROTOCOL.md). Обойти: --without-human")
    r = run(args.name)
    for m in r["metrics"].values():
        m.pop("per_company", None)
    out = Path(args.out) if args.out else EVAL_DIR / f"final_{args.name}.md"
    out.write_text("\n".join(report(r)), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("\n".join(report(r)))


if __name__ == "__main__":
    main()
