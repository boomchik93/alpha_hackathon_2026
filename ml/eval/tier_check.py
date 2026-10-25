"""Проверка порогов двухуровневой ленты на данных разработки. Без LLM.

    ML_AS_OF=2026-10-22 python -m ml.eval.tier_check          # -> ml/eval/tier_check.md + tier_check.json

Внефолдовые предсказания (настройки модели из pair_model_cv.json): real по дням, v2 по материалам (real в обучении).
Для порогов основной ленты 0,35 / 0,40 / 0,45 и двух вариантов размещения макро по полу: метрики уровней на парах real, на 210 ручных
парах real, на v2 и на 9 компаниях v2 с ручной разметкой. Для 9 компаний ещё сравнение основной ленты с прежней лентой
(один уровень от 0,30, эвристика до правил) в пределах допусков `pair_model.TOL_*`.
"""

from __future__ import annotations

import json
import sys

from ml.eval.label_review_real import human_pairs
from ml.eval.label_llm import EVAL_DIR
from ml.ranking import pair_model as pm
from ml.ranking import ranker

MAIN_GRID = (0.35, 0.40, 0.45)
MACRO_GRID = ("maybe", "main")


def tiered_order(rows: list[pm.Row], scored: dict, main: float, macro: str) -> dict:
    """Порядок «сначала основная, затем блок, затем отсев»: для nDCG@5 двухуровневой ленты."""
    shift = {"main": 2.0, "maybe": 0.0, None: -2.0}
    return {(r.company, r.item): (scored[(r.company, r.item)][0],
                                  scored[(r.company, r.item)][1] + shift[pm.tier_of_row(r, scored[(r.company, r.item)][0], main, ranker.MIN_SCORE, macro)])
            for r in rows if r.grade is not None}


def run() -> dict:
    sets = pm.load_sets()
    chosen = json.loads(ranker.MODEL_PATH.with_name("pair_model_cv.json").read_text(encoding="utf-8"))["chosen"]
    cfg = {k: chosen[k] for k in pm.PARAMS}
    real_s, _ = pm.out_of_fold_scores(sets["real"], list(sets["v2"]) if "v2" in chosen["train"] else [], **cfg)
    v2_s, human9_s = pm.out_of_fold_scores(sets["v2"], sets["real"], **cfg, also=sets["human"])
    hp = human_pairs()
    real_human = [r for r in sets["real"] if (r.company, r.item) in hp]
    before = {"human9": pm.heuristic_metrics(sets["human"], sets["v2_items"], sets["companies"], rules=False),
              "real": pm.heuristic_metrics(sets["real"], sets["real_items"], sets["companies"], rules=False)}
    out = {"config": chosen, "pairs": {"real": sum(r.grade is not None for r in sets["real"]), "real_human": len(real_human),
                                       "v2": sum(r.grade is not None for r in sets["v2"]), "human9": len(sets["human"])},
           "before": {k: {m: v[m] for m in ("precision", "recall", "f1", "ndcg5")} | {"tiers": v["tiers"]} for k, v in before.items()},
           "grid": []}
    data = (("real", sets["real"], real_s), ("real_human", real_human, real_s), ("v2", sets["v2"], v2_s), ("human9", sets["human"], human9_s))
    for macro in MACRO_GRID:
        for main in MAIN_GRID:
            row = {"main": main, "macro": macro}
            for name, rows, sc in data:
                t = pm.tier_metrics(rows, sc, main, ranker.MIN_SCORE, macro)
                m = pm.metrics(rows, tiered_order(rows, sc, main, macro), ranker.MIN_SCORE)
                row[name] = {**t, "ndcg5_tiered": m["ndcg5"]}
            out["grid"].append(row)
    return out


def report(r: dict) -> list[str]:
    b9 = r["before"]["human9"]
    L = ["# Проверка порогов двухуровневой ленты на данных разработки", "",
         f"Внефолдовые предсказания, настройки модели {json.dumps(r['config'], ensure_ascii=False)}. Пар с эталоном: {r['pairs']}. "
         "«Основная» — precision / recall / F0.5 / precision@5 основной ленты, «обе» — recall основной ленты вместе с блоком «может быть», "
         "«в maybe» — доля нужных пар в блоке «может быть». Пара считается нужной при оценке >= 1.", ""]
    for name, title in (("real_human", "210 ручных пар real (эталон — человек)"), ("real", "все пары real (эталон — разметка LLM, часть пар ручная)"),
                        ("v2", "v2 (синтетика, эталон — две LLM и человек)"), ("human9", "9 компаний v2 с ручной разметкой")):
        L += [f"## {title}", "", "| Порог | макро по полу | основная P | R | F0.5 | P@5 | обе R | в maybe | пар осн. / maybe | nDCG@5 (осн. первой) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for g in r["grid"]:
            t, m = g[name], g[name]["main"]
            L += [f"| {g['main']:.2f} | {g['macro']} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f05']:.3f} | {m['precision5']:.2f} | "
                  f"{t['both']['recall']:.2f} | {t['maybe_share_of_rel']:.2f} | {t['n_main']} / {t['n_maybe']} | {t['ndcg5_tiered']:.3f} |"]
        L += [""]
    f05 = 1.25 * b9["precision"] * b9["recall"] / (0.25 * b9["precision"] + b9["recall"])
    L += ["## 9 компаний v2: двухуровневая лента против прежней (один уровень от 0,30, эвристика до правил)", "",
          f"Прежняя: precision {b9['precision']:.2f}, recall {b9['recall']:.2f}, F1 {b9['f1']:.3f}, F0.5 {f05:.3f}, nDCG@5 {b9['ndcg5']:.3f}. "
          f"Допуски выбора модели: F1 {pm.TOL_HUMAN_F1}, nDCG@5 {pm.TOL_HUMAN_NDCG}. Основная лента нарочно жертвует полнотой ради точности, "
          "поэтому ΔF1 основной уровня смотрится вместе с ΔP основной и ΔR обоих уровней (критерий плана: точность основной и полнота обоих).", "",
          "| Порог | макро | F1 основной | ΔF1 | ΔF0.5 | ΔP основной | ΔR обоих | nDCG@5 | ΔnDCG@5 | F1 в допуске | ΔnDCG в допуске |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in r["grid"]:
        h = g["human9"]
        m = h["main"]
        d1, d2 = m["f1"] - b9["f1"], h["ndcg5_tiered"] - b9["ndcg5"]
        L += [f"| {g['main']:.2f} | {g['macro']} | {m['f1']:.3f} | {d1:+.3f} | {m['f05'] - f05:+.3f} | {m['precision'] - b9['precision']:+.2f} | "
              f"{h['both']['recall'] - b9['recall']:+.2f} | {h['ndcg5_tiered']:.3f} | {d2:+.3f} | {'да' if d1 >= -pm.TOL_HUMAN_F1 else 'нет'} | "
              f"{'да' if d2 >= -pm.TOL_HUMAN_NDCG else 'нет'} |"]
    return L + [""]


def main() -> None:
    r = run()
    (EVAL_DIR / "tier_check.md").write_text("\n".join(report(r)), encoding="utf-8")
    (EVAL_DIR / "tier_check.json").write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("\n".join(report(r)))


if __name__ == "__main__":
    main()
