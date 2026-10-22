"""Модель пары «компания × материал» для ленты (STATUS_HANDOFF.md, шаг 2 плана).

    ML_AS_OF=2026-10-22 python -m ml.ranking.pair_model cv       # выбор настроек: перекрёстная проверка по дням новостей
    ML_AS_OF=2026-10-22 python -m ml.ranking.pair_model train    # обучить на выбранных настройках -> ml/ranking/pair_model.json
    ML_AS_OF=2026-10-22 python -m ml.ranking.pair_model calibrate  # калибровка скор -> вероятность по ручным парам (вложенная CV)

Зачем. Ручные веса эвристики не различают два главных типа ошибок на реальных новостях (ml/eval/analysis.md,
«Реальные новости», и разбор ошибок в STATUS_HANDOFF.md): отраслевое событие с широкой темой («gdp_industry») уходит
всем компаниям, а макрособытие (scope="economy": инфляция, ставка) отсекается у компаний без продуктов банка.
Нужные для этого сигналы уже есть (масштаб события, совпадение ОКВЭД, причины по темам и каналам), не хватает весов.

Что. Две логистические регрессии с L2 по признакам `ml.ranking.ranker.pair_features`: P(оценка >= 1) для отсева
и P(оценка = 2) для порядка. Без внешних библиотек (Ньютон по numpy), веса и нормировка в JSON. Материалы вне экономики
(scope="none") в обучение не входят: их исключает правило ленты.

Честность оценки. Отложенная часть real (holdout, 20-22.10) уже использовалась при выборе режима причин
(analysis.md), поэтому весь набор real (7 дней) считается данными для разработки. Настройки (L2, BETA, порядок,
состав обучающих данных) выбираются перекрёстной проверкой «без одного дня» по 7 дням real: модель не видела ни
одной новости дня, на котором её проверяют. Ограничения (не хуже прежней ленты больше чем на допуск): синтетика v2
перекрёстно по материалам и её 9 компаний с ручной разметкой (единственный эталон от человека; внефолдовые
предсказания тех же фолдов). Итоговая цифра только на свежем наборе real2 (PROTOCOL.md, T19), он в обучение не входит.

Почему не dev -> holdout. Модель, обученная на dev, на holdout хуже эвристики: holdout держится на 3-5 материалах,
и разметка почти одинаковых новостей расходится между днями («Банк России снизил курсы доллара и евро» в dev
релевантна 15 компаниям из 35, «ЦБ резко понизил курс доллара» в holdout 32 из 32). Одно деление по дате
слишком шумное для выбора настроек, 7 фолдов по дням устойчивее.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import date

import numpy as np

from ml.ranking import ranker

DEFAULT = {"l2": 30.0, "beta": 0.5, "w_model": 0.5, "order_mix": 0.5, "train": "v2+real"}
# w_model: вес модели в отсеве (остальное эвристика с правилами), order_mix: вес модели в порядке (ranker.blend_scores).
GRID = {"train": (("real",), ("v2", "real")), "l2": (3.0, 30.0), "beta": (0.35, 0.5), "w_model": (0.3, 0.5, 0.7, 1.0),
        "order_mix": (0.0, 0.5, 1.0)}
# Допуски ограничений выбора: v2 (LLM + человек) и 9 компаний v2 с ручной разметкой против прежней ленты.
# Допуск по nDCG@5 ручного среза 0,05: на 9 компаниях 95% интервал разницы ±0,10, и при 0,04 выбор переключался из-за
# промаха 0,004 на шуме (после добавления ручной разметки real, analysis.md), теряя 0,07 nDCG на реальных новостях.
TOL_V2, TOL_HUMAN_F1, TOL_HUMAN_NDCG = 0.01, 0.02, 0.05
PARAMS = ("l2", "beta", "w_model", "order_mix")


@dataclass
class Row:
    company: str
    item: str
    group: str  # день новости (real) или материал (v2): единица перекрёстной проверки
    source: str  # v2 | real
    f: dict[str, float]
    grade: int | None  # None: пары нет в эталоне (в метрики и обучение не входит)
    importance: float
    none_scope: bool
    split: str | None = None  # dev/test компании v2 или dev/holdout новости real
    heur: float = 0.0  # личный скор эвристики (ranker без модели)
    keep: bool = False  # событие в отрасли компании (ranker.IN_SECTOR_KEEPS): проходит отсев и при модели
    floor: bool = False  # макрособытие, проходящее отсев по полу ranker.ECONOMY_MIN_IMPORTANCE (для уровней ленты)


def build_rows(companies: list[dict], items: list, gold_fn, source: str, splits: dict[str, str] | None = None) -> list[Row]:
    """Все пары пула ленты с признаками; эвристика, BM25 и пул те же, что в ranker.score_all."""
    from ml.profile.build_profile import get_profile

    pool, _ = ranker.feed_pool(items)
    rows = []
    for c in companies:
        p = get_profile(c["id"])
        for it, b in zip(pool, ranker.bm25_scores(p, pool)):
            h = min(1.0, sum(r[0] for r in ranker.heuristic_reasons(p, it)))
            group = it.source.published_at.isoformat() if source.startswith("real") else it.item_id
            heur = ranker.score_item(p, it, b).personal
            rows.append(Row(c["id"], it.item_id, group, source, ranker.pair_features(p, it, b, h), gold_fn(c, it.item_id),
                            it.importance, it.scope == "none", (splits or {}).get(it.item_id, c.get("split")), heur,
                            ranker.IN_SECTOR_KEEPS and ranker.in_sector(p, it), ranker.economy_floor(p, it)))
    return rows


def _matrix(rows: list[Row], names: list[str]) -> np.ndarray:
    return np.array([[r.f[k] for k in names] for r in rows], dtype=float)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float, iters: int = 40) -> np.ndarray:
    """Логистическая регрессия с L2 (свободный член не штрафуется), метод Ньютона. X уже нормирован."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    R = np.eye(Xb.shape[1]) * l2
    R[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + R @ w
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def fit(rows: list[Row], l2: float, beta: float, w_model: float = 1.0, order_mix: float = 1.0) -> dict:
    train = [r for r in rows if r.grade is not None and not r.none_scope]
    names = list(train[0].f)
    X = _matrix(train, names)
    mean, std = X.mean(0), X.std(0)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    y = np.array([r.grade for r in train])
    model = {"features": names, "mean": dict(zip(names, mean.tolist())), "std": dict(zip(names, std.tolist())), "l2": l2, "beta": beta,
             "w_model": w_model, "order_mix": order_mix}
    for key, target in (("w_rel", y >= 1), ("w_top", y >= 2)):
        w = fit_logistic(Z, target.astype(float), l2)
        model[key] = {"bias": float(w[0]), **{k: float(v) for k, v in zip(names, w[1:])}}
    # Порог самой модели (по P(релевантно)); отсев смеси идёт по ranker.MIN_SCORE в шкале эвристики.
    model["threshold"] = best_threshold([ranker.model_scores(model, r.f)[0] for r in train], [r.grade >= 1 for r in train])
    model["n_train"] = len(train)
    return model


def best_threshold(scores: list[float], rel: list[bool]) -> float:
    """Порог P(релевантно), максимизирующий F1 отсева на обучающих парах."""
    best, thr = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        tp = sum(s >= t and r for s, r in zip(scores, rel))
        fp = sum(s >= t and not r for s, r in zip(scores, rel))
        fn = sum(s < t and r for s, r in zip(scores, rel))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best + 1e-9:
            best, thr = f1, round(float(t), 2)
    return thr


def row_scores(model: dict, r: Row) -> tuple[float, float]:
    """(личный скор, порядок) как в ranker.score_item; отсев по ranker.MIN_SCORE."""
    if r.none_scope:
        return 0.0, model["beta"] * r.importance
    p_rel, p_top = ranker.model_scores(model, r.f)
    personal, order = ranker.blend_scores(model, p_rel, p_top, r.heur, r.importance)
    return (max(personal, ranker.MIN_SCORE) if r.keep else personal), order


def metrics(rows: list[Row], scored: dict[tuple[str, str], tuple[float, float]], thr: dict[str, float] | float) -> dict:
    """Метрики ленты на парах с эталоном: F1 отсева (по всем парам), nDCG@5 (среднее по компаниям), AUC по материалам.
    thr: порог отсева, общий или по группе (у каждого фолда свой)."""
    from ml.eval.run_eval import _filter_metrics, _order_metrics, material_auc

    gold: dict[str, dict[str, int]] = {}
    kept: dict[str, set[str]] = {}
    orders: dict[str, list[str]] = {}
    personal: dict[str, dict[str, float]] = {}
    for r in rows:
        if r.grade is None:
            continue
        s, o = scored[(r.company, r.item)]
        t = thr[r.group] if isinstance(thr, dict) else thr
        gold.setdefault(r.company, {})[r.item] = r.grade
        personal.setdefault(r.company, {})[r.item] = s
        if s >= t:
            kept.setdefault(r.company, set()).add(r.item)
        orders.setdefault(r.company, []).append((o, r.item))
    cids = sorted(gold)
    orders2 = {c: [i for _, i in sorted(orders[c], key=lambda x: -x[0])] for c in cids}
    kept = {c: kept.get(c, set()) for c in cids}
    per = {}
    for c in cids:
        rel = {i for i, x in gold[c].items() if x >= 1}
        per[c] = [len(kept[c] & rel), len(kept[c] - rel), len(rel - kept[c]), _order_metrics(orders2, gold, [c])["ndcg5"]]
    return {**_filter_metrics(kept, gold, cids), **_order_metrics(orders2, gold, cids),
            "auc": material_auc(personal, gold, cids)["weighted"], "companies": len(cids), "per_company": per}


def tier_of_row(r: Row, s: float, main: float, maybe: float, macro_tier: str) -> str | None:
    """Уровень пары как в ranker.tier_of, по готовому личному скору."""
    if s < maybe:
        return None
    if s >= main or r.keep or (macro_tier == "main" and r.floor):
        return "main"
    return "maybe"


def tier_metrics(rows: list[Row], scored: dict[tuple[str, str], tuple[float, float]], main: float | dict[str, float] = ranker.MAIN_SCORE,
                 maybe: float | dict[str, float] = ranker.MIN_SCORE, macro_tier: str = ranker.MACRO_FLOOR_TIER) -> dict:
    """Метрики двух уровней ленты на парах с эталоном (FEED_TIERS_PLAN.md, этап 2), пары всех компаний суммируются.
    main: precision, recall, F1, F0.5 основной ленты и precision@5 (первые 5 по порядку ленты у компании, суммарно);
    both: precision и recall основной ленты вместе с блоком «может быть»; maybe_share_of_rel: доля нужных пар,
    лежащих в блоке «может быть» (не прячем ли туда важное); maybe_precision: доля нужных среди пар блока.
    main, maybe: пороги, общие или по группе (у каждого фолда свои, как в metrics)."""
    rel_total = tp_m = fp_m = tp_b = fp_b = hits5 = shown5 = 0
    by_company: dict[str, list[tuple[float, bool]]] = {}
    for r in rows:
        if r.grade is None:
            continue
        s, o = scored[(r.company, r.item)]
        rel = r.grade >= 1
        rel_total += rel
        tier = tier_of_row(r, s, main[r.group] if isinstance(main, dict) else main,
                           maybe[r.group] if isinstance(maybe, dict) else maybe, macro_tier)
        if tier is None:
            continue
        tp_b, fp_b = tp_b + rel, fp_b + (not rel)
        if tier == "main":
            tp_m, fp_m = tp_m + rel, fp_m + (not rel)
            by_company.setdefault(r.company, []).append((o, rel))
    for lst in by_company.values():
        top = sorted(lst, key=lambda x: -x[0])[:5]
        hits5, shown5 = hits5 + sum(x[1] for x in top), shown5 + len(top)
    div = lambda a, b: a / b if b else 0.0
    p, rc = div(tp_m, tp_m + fp_m), div(tp_m, rel_total)
    return {"n_main": tp_m + fp_m, "n_maybe": tp_b + fp_b - tp_m - fp_m, "n_rel": rel_total,
            "main": {"precision": p, "recall": rc, "f1": div(2 * p * rc, p + rc), "f05": div(1.25 * p * rc, 0.25 * p + rc),
                     "precision5": div(hits5, shown5)},
            "both": {"precision": div(tp_b, tp_b + fp_b), "recall": div(tp_b, rel_total)},
            "maybe_share_of_rel": div(tp_b - tp_m, rel_total), "maybe_precision": div(tp_b - tp_m, tp_b + fp_b - tp_m - fp_m)}


def without_rule_flags(rows: list[Row]) -> list[Row]:
    """Пары без признаков правил (событие в отрасли, пол макро): для ленты до правил по масштабу события."""
    import dataclasses

    return [dataclasses.replace(r, keep=False, floor=False) for r in rows]


def paired_delta(a: dict, b: dict, n: int = 2000, seed: int = 13) -> dict:
    """Разница b - a по F1 отсева и nDCG@5 с 95% интервалом парного bootstrap по компаниям (как ml.eval.compare)."""
    import random

    cids = sorted(set(a["per_company"]) & set(b["per_company"]))

    def stat(m: dict, cs: list[str]) -> tuple[float, float]:
        tp, fp, fn = (sum(m["per_company"][c][k] for c in cs) for k in range(3))
        return (2 * tp / (2 * tp + fp + fn) if tp else 0.0), statistics.mean(m["per_company"][c][3] for c in cs)

    rnd = random.Random(seed)
    d = [tuple(y - x for x, y in zip(stat(a, cs), stat(b, cs))) for cs in ([rnd.choice(cids) for _ in cids] for _ in range(n))]
    pt = tuple(y - x for x, y in zip(stat(a, cids), stat(b, cids)))
    q = lambda v, k: sorted(v)[int(k * (len(v) - 1))]
    return {name: {"delta": pt[k], "ci95": (q([x[k] for x in d], 0.025), q([x[k] for x in d], 0.975))}
            for k, name in enumerate(("f1", "ndcg5"))}


def out_of_fold_scores(target: list[Row], extra: list[Row], l2: float, beta: float, w_model: float = 1.0, order_mix: float = 1.0,
                       also: list[Row] | None = None) -> tuple[dict, dict]:
    """Внефолдовые (личный скор, порядок) пар target и also: без одной группы target (день новости или порция материалов),
    extra всегда в обучении."""
    groups = sorted({r.group for r in target})
    folds = groups if len(groups) <= 8 else [groups[k::5] for k in range(5)]
    scored: dict[tuple[str, str], tuple[float, float]] = {}
    also_scored: dict[tuple[str, str], tuple[float, float]] = {}
    for fold in folds:
        held = set(fold) if isinstance(fold, list) else {fold}
        m = fit(extra + [r for r in target if r.group not in held], l2, beta, w_model, order_mix)
        for r in target:
            if r.group in held:
                scored[(r.company, r.item)] = row_scores(m, r)
        for r in also or []:  # другой эталон тех же пар (ручная разметка v2): те же внефолдовые модели
            if r.group in held:
                also_scored[(r.company, r.item)] = row_scores(m, r)
    return scored, also_scored


def cross_val(target: list[Row], extra: list[Row], l2: float, beta: float, w_model: float = 1.0, order_mix: float = 1.0,
              by_split: bool = False, also: list[Row] | None = None) -> dict:
    """Метрики по внефолдовым предсказаниям (out_of_fold_scores); tiers: метрики двух уровней ленты.
    by_split: ещё метрики отдельно по частям target (dev/holdout) из тех же внефолдовых предсказаний."""
    scored, also_scored = out_of_fold_scores(target, extra, l2, beta, w_model, order_mix, also)
    out = {**metrics(target, scored, ranker.MIN_SCORE), "tiers": tier_metrics(target, scored)}
    if also:
        out["also"] = {**metrics(also, also_scored, ranker.MIN_SCORE), "tiers": tier_metrics(also, also_scored)}
    if by_split:
        out["by_split"] = {s: {**metrics([r for r in target if r.split == s], scored, ranker.MIN_SCORE),
                               "tiers": tier_metrics([r for r in target if r.split == s], scored)}
                           for s in sorted({r.split for r in target})}
    return out


def heuristic_metrics(rows: list[Row], items: list, companies: list[dict], rules: bool = True) -> dict:
    """Эвристика на тех же парах (для сравнения в одном формате); rules=False: без правил по масштабу (как до шага 2)."""
    from ml.profile.build_profile import get_profile

    from ml import topics

    old = ranker.ECONOMY_MIN_IMPORTANCE, ranker.NARROW_NEEDS_SECTOR, ranker.IN_SECTOR_KEEPS, topics.TOPIC_OKVED
    if not rules:  # лента до шага 2: без правил и с отраслями у всех тем
        ranker.ECONOMY_MIN_IMPORTANCE, ranker.NARROW_NEEDS_SECTOR, ranker.IN_SECTOR_KEEPS, topics.TOPIC_OKVED = None, False, False, "all"
    ranker.set_model(None)
    try:
        scored = {}
        for c in companies:
            ss, _ = ranker.score_all(get_profile(c["id"]), items)
            scored.update({(c["id"], s.item.item_id): (s.personal, s.score) for s in ss})
    finally:
        ranker.clear_model()
        ranker.ECONOMY_MIN_IMPORTANCE, ranker.NARROW_NEEDS_SECTOR, ranker.IN_SECTOR_KEEPS, topics.TOPIC_OKVED = old
    return {**metrics(rows, scored, ranker.MIN_SCORE), "tiers": tier_metrics(rows if rules else without_rule_flags(rows), scored)}


# Калибровка (FEED_TIERS_PLAN.md, этап 6): модель учится на машинной разметке, она мягче человека (на случайных парах
# «релевантно» 15% против 8%), поэтому скор нужно перевести в вероятность, что человек сочтёт материал нужным.
# Нужных ручных пар мало (28 из 210), поэтому функция из двух параметров (Platt): P = sigmoid(a * скор + b).
PROB_MAIN, PROB_MAYBE = 0.5, 0.25  # пороги в вероятностях: основная лента и блок «может быть»
CAL_L2 = 0.1


def fit_platt(scores: list[float], rel: list[bool], l2: float = CAL_L2) -> tuple[float, float]:
    """(a, b) логистической калибровки скор -> вероятность нужного по ручной разметке."""
    X = np.array(scores, dtype=float)[:, None]
    w = fit_logistic(X, np.array(rel, dtype=float), l2)
    return float(w[1]), float(w[0])


def prob_thresholds(a: float, b: float, p_main: float = PROB_MAIN, p_maybe: float = PROB_MAYBE) -> tuple[float, float]:
    """Пороги по личному скору, соответствующие вероятностям p_main и p_maybe: (основная, «может быть»)."""
    import math

    inv = lambda p: (math.log(p / (1 - p)) - b) / a
    return inv(p_main), inv(p_maybe)


def human_rows(sets: dict, name: str = "real") -> list[Row]:
    """Пары real с ручной оценкой (grade уже ручной: load_gold_real берёт последнюю строку пары)."""
    from ml.eval.label_review_real import human_pairs

    hp = human_pairs(name)
    return [r for r in sets[name] if (r.company, r.item) in hp]


def nested_calibration(sets: dict, cfg: dict, main: float = ranker.MAIN_SCORE, maybe: float = ranker.MIN_SCORE) -> dict:
    """Фиксированные пороги против порогов из калибровки на ручных парах, вложенно по дням: калибровка фолда учится на
    внефолдовых скорах ручных пар других дней и применяется к ручным парам этого дня."""
    train = tuple(cfg["train"].split("+"))
    scored, _ = out_of_fold_scores(sets["real"], list(sets["v2"]) if "v2" in train else [],
                                   *(cfg[k] for k in PARAMS))
    hr = human_rows(sets)
    days = sorted({r.group for r in hr})
    thr_main, thr_maybe, params = {}, {}, {}
    for day in days:
        rest = [r for r in hr if r.group != day]
        a, b = fit_platt([scored[(r.company, r.item)][0] for r in rest], [r.grade >= 1 for r in rest])
        params[day] = (a, b)
        thr_main[day], thr_maybe[day] = prob_thresholds(a, b)
    fixed = tier_metrics(hr, scored, main, maybe)
    cal = tier_metrics(hr, scored, thr_main, thr_maybe)
    a, b = fit_platt([scored[(r.company, r.item)][0] for r in hr], [r.grade >= 1 for r in hr])
    return {"pairs": len(hr), "relevant": sum(r.grade >= 1 for r in hr), "fixed": fixed, "calibrated": cal,
            "final": {"a": a, "b": b, "thresholds": prob_thresholds(a, b)},
            "fold_thresholds": {d: (round(thr_main[d], 3), round(thr_maybe[d], 3)) for d in days}}


def save_calibration(sets: dict, cfg: dict) -> dict:
    """Дописать калибровку по ручным парам (внефолдовые скоры) в pair_model.json; веса модели не меняются."""
    train = tuple(cfg["train"].split("+"))
    scored, _ = out_of_fold_scores(sets["real"], list(sets["v2"]) if "v2" in train else [], *(cfg[k] for k in PARAMS))
    hr = human_rows(sets)
    a, b = fit_platt([scored[(r.company, r.item)][0] for r in hr], [r.grade >= 1 for r in hr])
    model = json.loads(ranker.MODEL_PATH.read_text(encoding="utf-8"))
    model["calibration"] = {"a": a, "b": b, "n": len(hr), "positives": sum(r.grade >= 1 for r in hr), "l2": CAL_L2,
                            "p_main": PROB_MAIN, "p_maybe": PROB_MAYBE, "date": date.today().isoformat(),
                            "labels": "ручные пары real (labels_real.jsonl, source=human), скоры внефолдовые по дням"}
    ranker.MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ranker.load_model.cache_clear()
    return model["calibration"]


def load_sets(cache_only: bool = True) -> dict:
    """Наборы с разметкой: v2 (40 компаний, синтетика), real dev и real holdout (реальные новости)."""
    from ml.eval.collect_real import real_split
    from ml.eval.label_llm import load_v2
    from ml.eval.run_eval import check_real_as_of, load_eval_items, load_gold_real, load_gold_v2, load_real_items

    check_real_as_of()
    companies = load_v2()
    v2_items = load_eval_items(cache_only=cache_only)[0]
    real_items = load_real_items(cache_only=cache_only)
    split = real_split()
    ranker.set_model(None)  # признаки от модели не зависят, но пул и BM25 считаем без подмены
    try:
        v2 = build_rows(companies, v2_items, load_gold_v2(), "v2")
        real = build_rows(companies, real_items, load_gold_real(), "real", split)
    finally:
        ranker.clear_model()
    human_gold = load_gold_v2(("human",))
    human = [Row(**{**r.__dict__, "grade": g}) for r in v2 if (g := human_gold({"id": r.company}, r.item)) is not None]
    return {"companies": companies, "v2_items": v2_items, "real_items": real_items, "v2": v2, "real": real, "human": human,
            "real_dev": [r for r in real if r.split == "dev"], "real_holdout": [r for r in real if r.split == "holdout"]}


def run_cv(sets: dict) -> list[dict]:
    """Сетка настроек: CV на real (7 дней, без одного дня) и на v2 (по материалам; real всегда в обучении)."""
    import itertools

    res = []
    for train, *vals in itertools.product(GRID["train"], *(GRID[k] for k in PARAMS)):
        cfg = dict(zip(PARAMS, vals))
        real = cross_val(sets["real"], list(sets["v2"]) if "v2" in train else [], **cfg, by_split=True)
        v2 = cross_val(sets["v2"], sets["real"], **cfg, also=sets["human"])
        human = v2.pop("also")
        res.append({"train": "+".join(train), **cfg, "real_cv": real, "v2_cv": v2, "human_cv": human})
        for m in (real, v2, human, *real["by_split"].values()):
            m["_per_company"] = m.pop("per_company")
        bs = real["by_split"]
        print(f"{'+'.join(train):8s} {' '.join(f'{k}={v}' for k, v in cfg.items())}: real F1 {real['f1']:.3f} nDCG {real['ndcg5']:.3f} "
              f"AUC {real['auc']:.3f} (dev F1 {bs['dev']['f1']:.3f}, holdout F1 {bs['holdout']['f1']:.3f}) | "
              f"v2 F1 {v2['f1']:.3f} nDCG {v2['ndcg5']:.3f} | human9 F1 {human['f1']:.3f} nDCG {human['ndcg5']:.3f}", file=sys.stderr)
    return res


def admissible(r: dict, base_v2: dict, base_human: dict) -> bool:
    return (r["v2_cv"]["f1"] >= base_v2["f1"] - TOL_V2 and r["v2_cv"]["ndcg5"] >= base_v2["ndcg5"] - TOL_V2
            and r["human_cv"]["f1"] >= base_human["f1"] - TOL_HUMAN_F1 and r["human_cv"]["ndcg5"] >= base_human["ndcg5"] - TOL_HUMAN_NDCG)


def choose(res: list[dict], base_v2: dict, base_human: dict) -> dict:
    """Лучшее по real CV (среднее F1 отсева и nDCG@5) среди допустимых по v2 и ручной разметке (admissible)."""
    ok = [r for r in res if admissible(r, base_v2, base_human)]
    return max(ok or res, key=lambda r: round((r["real_cv"]["f1"] + r["real_cv"]["ndcg5"]) / 2, 4))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("cv", "train", "calibrate"))
    ap.add_argument("--l2", type=float)
    ap.add_argument("--beta", type=float)
    ap.add_argument("--w-model", type=float)
    ap.add_argument("--order-mix", type=float)
    ap.add_argument("--train", help="v2+real или real")
    args = ap.parse_args(argv)
    sets = load_sets()
    out = ranker.MODEL_PATH.with_name("pair_model_cv.json")
    if args.cmd == "cv":
        part_items = {part: [i for i in sets["real_items"] if i.item_id in {r.item for r in sets[f"real_{part}"]}] for part in ("dev", "holdout")}
        base = {"real_before": heuristic_metrics(sets["real"], sets["real_items"], sets["companies"], rules=False),
                "v2_before": heuristic_metrics(sets["v2"], sets["v2_items"], sets["companies"], rules=False),
                "real": heuristic_metrics(sets["real"], sets["real_items"], sets["companies"]),
                "real_dev": heuristic_metrics(sets["real_dev"], part_items["dev"], sets["companies"]),
                "real_holdout": heuristic_metrics(sets["real_holdout"], part_items["holdout"], sets["companies"]),
                "v2": heuristic_metrics(sets["v2"], sets["v2_items"], sets["companies"]),
                "human_before": heuristic_metrics(sets["human"], sets["v2_items"], sets["companies"], rules=False),
                "human": heuristic_metrics(sets["human"], sets["v2_items"], sets["companies"])}
        res = run_cv(sets)
        best = choose(res, base["v2_before"], base["human_before"])
        for m in (best["real_cv"], best["v2_cv"], best["human_cv"], *best["real_cv"]["by_split"].values()):
            m["per_company"] = m["_per_company"]
        deltas = {}
        for name, before, after in (("real: эвристика с правилами против прежней", base["real_before"], base["real"]),
                                    ("real: модель+правила (CV) против прежней", base["real_before"], best["real_cv"]),
                                    ("real: модель+правила (CV) против эвристики с правилами", base["real"], best["real_cv"]),
                                    ("v2: эвристика с правилами против прежней", base["v2_before"], base["v2"]),
                                    ("v2: модель+правила (CV) против прежней", base["v2_before"], best["v2_cv"]),
                                    ("v2, ручная разметка 9 компаний: правила против прежней", base["human_before"], base["human"]),
                                    ("v2, ручная разметка 9 компаний: модель+правила (CV) против прежней", base["human_before"], best["human_cv"])):
            deltas[name] = paired_delta(before, after)
        for r in res:
            for m in (r["real_cv"], r["v2_cv"], r["human_cv"], *r["real_cv"]["by_split"].values()):
                m.pop("_per_company", None)
                m.pop("per_company", None) if r is not best else None
        summary = {"heuristic": base, "chosen": {k: best[k] for k in ("train", *PARAMS)},
                   "chosen_cv": {"real": best["real_cv"], "v2": best["v2_cv"], "human": best["human_cv"]}, "paired": deltas,
                   "admissible": sum(admissible(r, base["v2_before"], base["human_before"]) for r in res), "grid": res,
                   "date": date.today().isoformat()}
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: summary[k] for k in ("chosen", "paired")}, ensure_ascii=False, indent=1))
        return
    cfg = {**DEFAULT, **(json.loads(out.read_text(encoding="utf-8"))["chosen"] if out.exists() else {})}
    if args.cmd == "calibrate":
        r = nested_calibration(sets, cfg)
        c = save_calibration(sets, cfg)
        print(json.dumps({"nested": r, "saved": c}, ensure_ascii=False, indent=1, default=list))
        return
    for k, a in (("l2", args.l2), ("beta", args.beta), ("w_model", args.w_model), ("order_mix", args.order_mix)):
        if a is not None:
            cfg[k] = a
    l2, beta = cfg["l2"], cfg["beta"]
    train = tuple((args.train or cfg["train"]).split("+"))
    model = fit((list(sets["v2"]) if "v2" in train else []) + sets["real"], l2, beta, cfg["w_model"], cfg["order_mix"])
    model["trained_on"] = {"sets": [*(["v2"] if "v2" in train else []), "real (dev+holdout)"],
                           "date": date.today().isoformat(), "labels": "v2: labels.jsonl; real: labels_real.jsonl (llm_agreed + human)"}
    ranker.MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ranker.load_model.cache_clear()
    save_calibration(sets, cfg)  # калибровка не должна пропадать при переобучении
    top = sorted(((abs(v), k, v) for k, v in model["w_rel"].items() if k != "bias"), reverse=True)[:8]
    print(f"{ranker.MODEL_PATH.name}: пар {model['n_train']}, порог {model['threshold']}, beta {beta}, l2 {l2}; "
          f"главные веса P(rel): {', '.join(f'{k} {v:+.2f}' for _, k, v in top)}", file=sys.stderr)


if __name__ == "__main__":
    main()
