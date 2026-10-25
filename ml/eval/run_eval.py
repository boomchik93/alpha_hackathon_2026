"""Eval: компании x материалы, метрики из ML_SPEC.md, раздел 8.

    python -m ml.eval.run_eval                 # реальная модель, все пары
    python -m ml.eval.run_eval --mock          # сухой прогон без расходов
    python -m ml.eval.run_eval --max-rub 30 --workers 6 --no-judge

Части:
1. Ранжирование (без LLM): precision/recall отсева, nDCG@5, recall@5 против
   ручной разметки `labels.py`; сравнение с BM25 по тексту профиля.
2. Карточки: для каждой пары «компания x новость» (демо + holdout) полный
   агентный цикл без кэша LLM, автоматические проверки и LLM-судья на другой модели.
3. Отчёт `ml/eval/report.md`, сырые результаты `var/eval/<run_id>/`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ml.config import ROOT, settings

EVAL_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------


def load_companies() -> list[dict]:
    return json.loads((EVAL_DIR / "companies.json").read_text(encoding="utf-8"))


def load_holdout():
    from ml.ingest.news_fetch import RawItem

    lines = (EVAL_DIR / "holdout_news.jsonl").read_text(encoding="utf-8").splitlines()
    return [RawItem(**json.loads(l)) for l in lines if l.strip()]


def human_gold(c: dict, item_id: str) -> int | None:
    from ml.eval.labels import grade

    return grade(c["key"], item_id)


def load_gold_v2(sources: tuple[str, ...] = ("human", "llm_agreed")):
    """Эталон расширенного набора из labels.jsonl. Пары с другим source (спорные) исключаются: None."""
    rows = [json.loads(l) for l in (EVAL_DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    table = {(r["company"], r["item"]): r["grade"] for r in rows if r["source"] in sources}

    def gold(c: dict, item_id: str) -> int | None:
        return table.get((c["id"], item_id))
    return gold


DISPUTED_VARIANTS = {"A": "оценка модели A", "B": "оценка модели B", "min": "минимум из двух", "max": "максимум из двух"}


def load_gold_v2_disputed(variant: str):
    """Эталон v2 со спорными парами: согласованные и ручные как в load_gold_v2, спорным
    ставится оценка модели A, модели B, минимум или максимум из двух (поля grade_a, grade_b)."""
    pick = {"A": lambda a, b: a, "B": lambda a, b: b, "min": min, "max": max}[variant]
    rows = [json.loads(l) for l in (EVAL_DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    table = {(r["company"], r["item"]): (r["grade"] if r["source"] in ("human", "llm_agreed") else pick(r["grade_a"], r["grade_b"]))
             for r in rows}

    def gold(c: dict, item_id: str) -> int | None:
        return table.get((c["id"], item_id))
    return gold


def label_source_shares() -> dict[str, dict[str, int]]:
    """Число пар разметки v2 по split и источнику (human / llm_agreed / llm_disputed)."""
    out: dict[str, dict[str, int]] = {}
    for l in (EVAL_DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            d = out.setdefault(r["split"], {})
            d[r["source"]] = d.get(r["source"], 0) + 1
    return out


def eval_disputed(companies: list[dict], items: list) -> dict[str, dict[str, dict]]:
    """Метрики ленты при включении спорных пар с разными оценками: вариант -> {все, dev, test} -> pooled-метрики."""
    out = {}
    for v in DISPUTED_VARIANTS:
        per = eval_ranking(companies, items, load_gold_v2_disputed(v))["per_company"]
        groups = {"все": per}
        for name in ("dev", "test"):
            part = [x for x in per if x.get("split") == name]
            if part:
                groups[name] = part
        out[v] = {g: _pooled(rows) for g, rows in groups.items()}
    return out


# ---------------------------------------------------------------------------
# Переранжирование моделью: оценка Gemma для топ-N и материалов у порога, смесь с личным скором.
# Оценка модели одна на пару и кэшируется, поэтому веса смеси сравниваются на одних и тех же вызовах.
# ---------------------------------------------------------------------------

RERANK_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
RERANK_RUB_PER_CALL = 0.04  # верхняя оценка вызова Gemma (0,01-0,04 руб.)


def rerank_plan(companies: list[dict], items: list) -> dict[str, tuple]:
    """Для каждой компании: профиль, материалы пула со скорами и кандидаты на оценку модели (текущие параметры)."""
    from ml.profile.build_profile import get_profile
    from ml.ranking import ranker, rerank

    out = {}
    for c in companies:
        p = get_profile(c["id"])
        scored, _ = ranker.score_all(p, items)
        out[c["id"]] = (p, scored, rerank.candidates(scored, ranker.min_score()))
    return out


def _rerank_row(scored: list, gold: dict[str, int], c: dict) -> dict:
    from ml.ranking import ranker

    kept = {s.item.item_id for s in scored if s.personal >= ranker.min_score()} & set(gold)
    rel = {i for i, g in gold.items() if g >= 1}
    order = [s.item.item_id for s in scored if s.item.item_id in gold]
    return {"company": c["id"], "split": c.get("split"), "kept": len(kept), "false_pos": sorted(kept - rel), "missed": sorted(rel - kept),
            "ndcg5": ndcg([gold[i] for i in order], list(gold.values()), 5),
            "recall5": len(set(order[:5]) & rel) / min(5, len(rel)) if rel else 1.0}


def _paired_ci(a: list[float], b: list[float], n: int | None = None, seed: int = 13) -> tuple[float, float, float]:
    """Среднее различие a - b по компаниям и 95% интервал bootstrap (парный)."""
    import random

    n = n or BOOTSTRAP_N
    rnd = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    means = sorted(statistics.mean(rnd.choice(d) for _ in d) for _ in range(n))
    return statistics.mean(d), means[int(0.025 * n)], means[min(n - 1, int(0.975 * n))]


def eval_rerank(companies: list[dict], items: list, client, gold_sets: dict, plan: dict | None = None,
                weights: tuple[float, ...] = RERANK_WEIGHTS) -> dict:
    """gold_sets: имя -> (gold_fn, фильтр компаний). Возвращает метрики по весам w и парные различия с w = 0."""
    from ml.ranking import ranker, rerank

    plan = plan or rerank_plan(companies, items)
    model = {cid: rerank.model_scores(p, cands, client) for cid, (p, _, cands) in plan.items()}
    out = {"calls": sum(len(v) for v in model.values()), "sets": {}}
    for name, (gold_fn, keep) in gold_sets.items():
        part = [c for c in companies if keep(c)]
        rows: dict[float, list[dict]] = {w: [] for w in weights}
        used = []
        for c in part:
            gold = {s.item.item_id: g for s in plan[c["id"]][1] if (g := gold_fn(c, s.item.item_id)) is not None}
            if not gold:
                continue
            used.append(c)
            for w in weights:
                scored = rerank.apply(plan[c["id"]][1], model[c["id"]], w, ranker.BETA)
                rows[w].append(_rerank_row(scored, gold, c))
        base = rows[0.0]
        res = {"companies": len(used), "weights": {}}
        for w in weights:
            m = _pooled(rows[w])
            entry = {**m}
            if w != 0.0 and len(used) > 1:
                entry["d_ndcg5"] = _paired_ci([x["ndcg5"] for x in rows[w]], [x["ndcg5"] for x in base])
                f1 = lambda r: _pooled([r])["f1"]  # noqa: E731
                entry["d_f1"] = _paired_ci([f1(x) for x in rows[w]], [f1(x) for x in base])
            res["weights"][w] = entry
        out["sets"][name] = res
    return out


def rerank_section(r: dict) -> list[str]:
    L = ["## Переранжирование моделью (эксперимент, ML_RERANK)", "",
         f"Оценка Gemma (`relevance.md`) для топ-10 и материалов у порога отсева; вызовов: {r['calls']}. Итог = (1 − w) × личный скор + w × оценка модели; "
         "w = 0 это текущая лента. Различия с w = 0 парные по компаниям, в скобках 95% интервал bootstrap. "
         "Ручные пары: разметка остальных пар сделана LLM, поэтому на них возможна круговая ошибка «LLM против LLM».", ""]
    for name, s in r["sets"].items():
        L += [f"### {name} ({s['companies']} компаний)", "", "| w | nDCG@5 | F1 отсева | precision | recall | Δ nDCG@5 | Δ F1 |", "|---|---|---|---|---|---|---|"]
        for w, m in s["weights"].items():
            d1, d2 = m.get("d_ndcg5"), m.get("d_f1")
            fmt = lambda d: "—" if d is None else f"{d[0]:+.3f} [{d[1]:+.3f}; {d[2]:+.3f}]"  # noqa: E731
            L += [f"| {w} | {_f(m['ndcg5'], nd=3)} | {_f(m['f1'], True)} | {_f(m['precision'], True)} | {_f(m['recall'], True)} | {fmt(d1)} | {fmt(d2)} |"]
        L += [""]
    return L


def load_eval_items(mock: bool = False, cache_only: bool = False) -> tuple[list, list, list]:
    """Материалы eval: аналитика банка, демо-новости и разобранные holdout-новости (они добавляются в память процесса).

    Возвращает (все материалы, новости, holdout)."""
    from ml.ingest.dedup import assign_clusters
    from ml.ingest.importance import with_importance
    from ml.ingest.parse_item import parse_item
    from ml.ingest.store import add_items, load_items
    from ml.llm.client import LLMClient

    demo = list(load_items().values())
    parse_client = LLMClient(mock=mock, use_cache=True, cache_only=cache_only)
    holdout = [with_importance(parse_item(r, client=parse_client), r.text, parse_client) for r in load_holdout()]
    news = assign_clusters([i for i in demo if i.source.kind.value != "bank_analytics"] + holdout)
    add_items(news)
    return [i for i in demo if i.source.kind.value == "bank_analytics"] + news, news, holdout


def load_real_items(mock: bool = False, cache_only: bool = False, split: str | None = None, name: str = "real") -> list:
    """Реальные новости (`ml/eval/<name>_news.jsonl`, `ml.eval.collect_real`): разбор и сигналы материала, в память процесса.

    split: dev, holdout или None (все). Разбор и сигналы платные при промахе кэша (Gemma на новость)."""
    from ml.eval.collect_real import load_real
    from ml.ingest.dedup import assign_clusters
    from ml.ingest.importance import with_importance
    from ml.ingest.parse_item import parse_item
    from ml.ingest.store import add_items
    from ml.llm.client import LLMClient

    client = LLMClient(mock=mock, use_cache=True, cache_only=cache_only)
    items = assign_clusters([with_importance(parse_item(r, client=client), r.text, client) for r in load_real(split, name)])
    add_items(items)
    return items


def check_real_as_of(name: str = "real") -> None:
    """Оценка на реальных новостях идёт с ML_AS_OF = дате сбора набора: от неё считаются свежесть и устаревание."""
    from ml.eval.collect_real import set_paths

    want = json.loads(set_paths(name)[1].read_text(encoding="utf-8"))["as_of"]
    if settings.as_of.isoformat() != want:
        sys.exit(f"для набора {name} нужен ML_AS_OF={want} (сейчас {settings.as_of}): запускайте с ML_AS_OF={want}")


def load_gold_real(sources: tuple[str, ...] = ("llm_agreed", "human"), name: str = "real"):
    """Эталон для реальных новостей (`ml/eval/labels_<name>.jsonl`); спорные пары исключаются."""
    rows = [json.loads(l) for l in (EVAL_DIR / f"labels_{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    table = {(r["company"], r["item"]): r["grade"] for r in rows if r["source"] in sources}

    def gold(c: dict, item_id: str) -> int | None:
        return table.get((c["id"], item_id))
    return gold


# ---------------------------------------------------------------------------
# Ранжирование
# ---------------------------------------------------------------------------


def ndcg(grades_in_order: list[int], all_grades: list[int], k: int) -> float:
    def dcg(gs: list[int]) -> float:
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gs[:k]))

    ideal = dcg(sorted(all_grades, reverse=True))
    return dcg(grades_in_order) / ideal if ideal > 0 else 1.0


def eval_ranking(companies: list[dict], items: list, gold_fn=None) -> dict:
    """gold_fn(company, item_id) -> 0/1/2 или None (пара без эталона, исключается из метрик компании)."""
    from rank_bm25 import BM25Okapi

    from ml.eval.labels import DUPLICATES
    from ml.ingest.dedup import representatives
    from ml.profile.build_profile import get_profile
    from ml.ranking import ranker
    from ml.retrieval.index import tokenize

    pool = representatives(items)
    pool_ids = [i.item_id for i in pool]
    # Базовая линия (как в прошлых прогонах): BM25 по тексту профиля против заголовка и summary, без эвристики и фильтра свежести.
    bm25 = BM25Okapi([tokenize(f"{i.title} {i.summary or ''}") for i in pool])
    gold_fn = gold_fn or human_gold
    per_company, tp, fp, fn = [], 0, 0, 0
    for c in companies:
        p = get_profile(c["id"])
        gold = {iid: g for iid in pool_ids if (g := gold_fn(c, iid)) is not None}
        scored, _ = ranker.score_all(p, items)
        kept = {s.item.item_id for s in scored if s.personal >= ranker.min_score()} & set(gold)
        rel = {iid for iid, g in gold.items() if g >= 1}
        tp += len(kept & rel)
        fp += len(kept - rel)
        fn += len(rel - kept)
        order = [s.item.item_id for s in scored if s.item.item_id in gold]
        raw = bm25.get_scores(tokenize(ranker.profile_query(p)))
        base_order = [pool_ids[i] for i in sorted(range(len(pool_ids)), key=lambda i: -raw[i]) if pool_ids[i] in gold]
        all_g = list(gold.values())
        top_rel = sorted(rel, key=lambda x: -gold[x])
        per_company.append({
            "company": c["id"], "key": c["key"], "split": c.get("split"), "stratum": c.get("stratum"), "labelled": len(gold),
            "ndcg5": ndcg([gold[i] for i in order], all_g, 5),
            "ndcg5_bm25": ndcg([gold[i] for i in base_order], all_g, 5),
            "recall5": len(set(order[:5]) & rel) / min(5, len(rel)) if rel else 1.0,
            "recall5_bm25": len(set(base_order[:5]) & rel) / min(5, len(rel)) if rel else 1.0,
            "kept": len(kept), "relevant": len(rel),
            "missed": sorted(rel - kept), "false_pos": sorted(kept - rel),
            "top5": order[:5], "top_gold": top_rel[:5],
        })
    clusters = {i.item_id: i.dedup_cluster for i in items}
    dedup_ok = [clusters.get(a) is not None and clusters.get(a) == clusters.get(b) for a, b in DUPLICATES]
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "ndcg5": statistics.mean(x["ndcg5"] for x in per_company),
        "ndcg5_bm25": statistics.mean(x["ndcg5_bm25"] for x in per_company),
        "recall5": statistics.mean(x["recall5"] for x in per_company),
        "recall5_bm25": statistics.mean(x["recall5_bm25"] for x in per_company),
        "dedup_pairs_merged": f"{sum(dedup_ok)}/{len(dedup_ok)}",
        "per_company": per_company,
        "pool_size": len(pool),
    }


# ---------------------------------------------------------------------------
# Бейзлайны ленты и метрика персонализации. Порядок по общей
# важности материала обходит ленту по nDCG@5, поэтому персонализацию показывает AUC
# «кому показать материал», а не nDCG@5.
# ---------------------------------------------------------------------------

RANDOM_PERMUTATIONS = 500


def auc_score(pos: list[float], neg: list[float]) -> float | None:
    """Вероятность, что скор положительной компании выше скора отрицательной (ничья = 0,5)."""
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def popularity_loo(gold: dict[str, dict[str, int]], company: str) -> dict[str, float]:
    """Скор материала для компании: средняя оценка у остальных компаний, где он размечен."""
    acc: dict[str, list[int]] = {}
    for other, g in gold.items():
        if other != company:
            for iid, x in g.items():
                acc.setdefault(iid, []).append(x)
    return {iid: statistics.mean(xs) for iid, xs in acc.items()}


def material_auc(scores: dict[str, dict[str, float]], gold: dict[str, dict[str, int]], cids: list[str]) -> dict:
    """AUC по материалам: положительные = компании с оценкой >= 1, отрицательные = с оценкой 0.
    Материалы без положительных или без отрицательных пропускаются."""
    per = []
    for iid in sorted({i for c in cids for i in gold[c]}):
        pos = [scores[c][iid] for c in cids if gold[c].get(iid, -1) >= 1 and iid in scores[c]]
        neg = [scores[c][iid] for c in cids if gold[c].get(iid, -1) == 0 and iid in scores[c]]
        a = auc_score(pos, neg)
        if a is not None:
            per.append({"item": iid, "auc": a, "pos": len(pos), "neg": len(neg)})
    weight = sum(min(x["pos"], x["neg"]) for x in per)
    return {
        "n": len(per),
        "median": statistics.median(x["auc"] for x in per) if per else None,
        "weighted": sum(x["auc"] * min(x["pos"], x["neg"]) for x in per) / weight if weight else None,
        "worst": sorted(per, key=lambda x: x["auc"])[:5],
    }


def _order_metrics(orders: dict[str, list[str]], gold: dict[str, dict[str, int]], cids: list[str]) -> dict:
    nd, r5, p5 = [], [], []
    for c in cids:
        g = gold[c]
        rel = {i for i, x in g.items() if x >= 1}
        order = [i for i in orders[c] if i in g]
        nd.append(ndcg([g[i] for i in order], list(g.values()), 5))
        r5.append(len(set(order[:5]) & rel) / min(5, len(rel)) if rel else 1.0)
        p5.append(sum(g[i] >= 1 for i in order[:5]) / 5)
    return {"ndcg5": statistics.mean(nd), "recall5": statistics.mean(r5), "precision5": statistics.mean(p5)}


def _filter_metrics(kept: dict[str, set[str]], gold: dict[str, dict[str, int]], cids: list[str]) -> dict:
    tp = fp = fn = 0
    for c in cids:
        rel = {i for i, x in gold[c].items() if x >= 1}
        tp, fp, fn = tp + len(kept[c] & rel), fp + len(kept[c] - rel), fn + len(rel - kept[c])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def eval_baselines(companies: list[dict], items: list, gold_fn=None) -> dict[str, dict]:
    """Бейзлайны порядка и отсева и AUC системы на том же пуле и том же эталоне, что eval_ranking.
    Группы: «все», при split dev/test ещё «dev» и «test» (параметры системы подобраны на dev)."""
    import random

    from ml.profile.build_profile import get_profile
    from ml.ranking import ranker

    # Пул ленты: без дубликатов и без устаревшей аналитики банка, которую система не показывает.
    pool, _ = ranker.feed_pool(items)
    pool_ids = [i.item_id for i in pool]
    gold_fn = gold_fn or human_gold
    ids = [c["id"] for c in companies]
    gold = {c["id"]: {iid: g for iid in pool_ids if (g := gold_fn(c, iid)) is not None} for c in companies}
    scored = {c["id"]: ranker.score_all(get_profile(c["id"]), items)[0] for c in companies}
    scores = {c: {s.item.item_id: s.score for s in ss} for c, ss in scored.items()}
    personal = {c: {s.item.item_id: s.personal for s in ss} for c, ss in scored.items()}
    sys_order = {c: sorted(scores[c], key=lambda i: -scores[c][i]) for c in ids}
    pop = {c: popularity_loo(gold, c) for c in ids}
    pop_order = {c: sorted(pool_ids, key=lambda i: -pop[c].get(i, 0.0)) for c in ids}
    fresh = [i.item_id for i in sorted(pool, key=lambda i: -i.source.published_at.toordinal())]
    # Честный неперсонализированный бейзлайн: важность материала от модели (ml/ingest/importance.py) знает только
    # сам материал, а не разметку других компаний; в отличие от популярности так можно работать в бою.
    imp = {i.item_id: i.importance for i in pool}
    imp_order = sorted(pool_ids, key=lambda i: -imp[i])
    # Уровни ленты: те же пары, личный скор и порядок системы.
    from ml.ranking import pair_model as pm

    by_id = {i.item_id: i for i in pool}
    profiles = {c["id"]: get_profile(c["id"]) for c in companies}
    tier_rows = {c: [pm.Row(c, iid, "", "eval", {}, g, 0.0, False, keep=ranker.IN_SECTOR_KEEPS and ranker.in_sector(profiles[c], by_id[iid]),
                            floor=ranker.economy_floor(profiles[c], by_id[iid])) for iid, g in gold[c].items()] for c in ids}
    tier_scores = {(c, s.item.item_id): (s.personal, s.score) for c, ss in scored.items() for s in ss}

    groups = {"все": ids}
    for name in ("dev", "test"):
        part = [c["id"] for c in companies if c.get("split") == name]
        if part and len(part) < len(ids):
            groups[name] = part
    out = {}
    for name, cids in groups.items():
        rnd = random.Random(0)
        acc = {"ndcg5": 0.0, "recall5": 0.0, "precision5": 0.0}
        for _ in range(RANDOM_PERMUTATIONS):
            m = _order_metrics({c: rnd.sample(pool_ids, len(pool_ids)) for c in cids}, gold, cids)
            acc = {k: acc[k] + m[k] for k in acc}
        out[name] = {
            "n_companies": len(cids),
            "system": _order_metrics(sys_order, gold, cids),
            "popularity": _order_metrics(pop_order, gold, cids),
            "random": {k: v / RANDOM_PERMUTATIONS for k, v in acc.items()},
            "freshness": _order_metrics({c: fresh for c in cids}, gold, cids),
            "importance": _order_metrics({c: imp_order for c in cids}, gold, cids),
            "system_filter": _filter_metrics({c: {i for i, v in personal[c].items() if v >= ranker.min_score()} & set(gold[c]) for c in cids}, gold, cids),
            "keep_all": _filter_metrics({c: set(gold[c]) for c in cids}, gold, cids),
            "keep_popular": _filter_metrics({c: {i for i in gold[c] if pop[c].get(i, 0.0) >= 0.5} for c in cids}, gold, cids),
            "keep_important": _filter_metrics({c: {i for i in gold[c] if imp[i] >= 0.5} for c in cids}, gold, cids),
            "tiers": pm.tier_metrics([r for c in cids for r in tier_rows[c]], tier_scores),
            "auc": material_auc(personal, gold, cids),  # внутри материала важность постоянна: личный скор и смесь дают один AUC
        }
    return out


def baselines_rows(b: dict) -> list[str]:
    """Строки таблицы порядка: система и бейзлайны."""
    names = [("Система (лента)", "system"), ("Популярность: средняя оценка у других компаний (LOO, знает разметку)", "popularity"),
             ("Важность материала (модель, без разметки)", "importance"),
             ("Свежесть: новые первыми", "freshness"), ("Случайный порядок (среднее 500 перестановок)", "random")]
    return [f"| {n} | {_f(b[k]['ndcg5'], nd=3)} | {_f(b[k]['recall5'], True)} | {_f(b[k]['precision5'], True)} |" for n, k in names if k in b]


def baselines_lines(b: dict, title: str) -> list[str]:
    a = b["auc"]
    L = [f"### {title} ({b['n_companies']} компаний)", "",
         "| Порядок | nDCG@5 | recall@5 | precision@5 |", "|---|---|---|---|", *baselines_rows(b), "",
         "| Отсев | precision | recall | F1 |", "|---|---|---|---|"]
    for n, k in (("Лента: система", "system_filter"), ("Оставить всё", "keep_all"),
                 ("Оставить материалы с популярностью ≥ 0,5 (знает разметку)", "keep_popular"),
                 ("Оставить материалы с важностью ≥ 0,5 (без разметки)", "keep_important")):
        if k in b:
            L += [f"| {n} | {_f(b[k]['precision'], True)} | {_f(b[k]['recall'], True)} | {_f(b[k]['f1'], True)} |"]
    if "tiers" in b:
        from ml.ranking import ranker

        t = b["tiers"]
        L += ["", f"Уровни ленты: основная от {ranker.MAIN_SCORE}, «может быть» от {ranker.MIN_SCORE} (событие в отрасли компании всегда в основной).", "",
              "| Уровень | precision | recall | F0.5 | precision@5 | пар |", "|---|---|---|---|---|---|",
              f"| Основная | {_f(t['main']['precision'], True)} | {_f(t['main']['recall'], True)} | {_f(t['main']['f05'], True)} | "
              f"{_f(t['main']['precision5'], True)} | {t['n_main']} |",
              f"| Основная + «может быть» | {_f(t['both']['precision'], True)} | {_f(t['both']['recall'], True)} | | | {t['n_main'] + t['n_maybe']} |",
              "", f"Нужного в блоке «может быть»: {_f(t['maybe_share_of_rel'], True)} от всех нужных пар; "
                  f"доля нужных среди пар блока {_f(t['maybe_precision'], True)}."]
    L += ["", f"AUC «кому показать материал» по скору системы (материалов {a['n']}): медиана {_f(a['median'])}, "
              f"среднее с весом min(pos, neg) {_f(a['weighted'])}. У популярности AUC = 0,50 по построению. "
              "Худшие: " + "; ".join(f"`{x['item']}` {x['auc']:.2f} (+{x['pos']}/−{x['neg']})" for x in a["worst"]) + ".", ""]
    return L


# ---------------------------------------------------------------------------
# Честная оценка ранжирования: параметры подбираются без
# оцениваемой компании, интервалы bootstrap по компаниям.
# ---------------------------------------------------------------------------

# Та же сетка, по которой параметры подбирались вручную (ml/eval/analysis.md).
PARAM_GRID = {
    "ALPHA": (0.5, 0.6, 0.7),
    "MIN_SCORE": (0.2, 0.25, 0.3),
    "FEATURE_WEIGHT": (0.35, 0.45),
    "OFF_SECTOR_DAMP": (1.0, 0.5),
    "ADJACENCY_WEIGHT": (0.0, 0.25, 0.35, 0.45, 0.6),
    "BETA": (0.0, 0.2, 0.35, 0.5),  # вес важности материала в порядке ленты
}
BOOTSTRAP_N = 2000


def _set_params(params: dict) -> dict:
    """Ставит параметры ранжирования, возвращает прежние значения."""
    from ml import topics
    from ml.ranking import ranker

    old = {}
    for k, v in params.items():
        mod = topics if k == "FEATURE_WEIGHT" else ranker
        old[k] = getattr(mod, k)
        setattr(mod, k, v)
    return old


def _company_counts(x: dict) -> tuple[int, int, int]:
    fp, fn = len(x["false_pos"]), len(x["missed"])
    return x["kept"] - fp, fp, fn


def _pooled(rows: list[dict]) -> dict:
    tp = fp = fn = 0
    for x in rows:
        a, b, c = _company_counts(x)
        tp, fp, fn = tp + a, fp + b, fn + c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "ndcg5": statistics.mean(x["ndcg5"] for x in rows), "recall5": statistics.mean(x["recall5"] for x in rows)}


def bootstrap_ci(rows: list[dict], n: int = BOOTSTRAP_N, seed: int = 13) -> dict[str, tuple[float, float]]:
    """95% интервалы метрик при пересэмплировании компаний с возвращением."""
    import random

    rnd = random.Random(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n):
        m = _pooled([rnd.choice(rows) for _ in rows])
        for k, v in m.items():
            samples.setdefault(k, []).append(v)
    return {k: (_pct(0.025, v), _pct(0.975, v)) for k, v in samples.items()}


def eval_ranking_cv(companies: list[dict], items: list, gold_fn=None) -> dict:
    """Leave-group-out: для каждой группы компаний (страта ОКВЭД, если есть, иначе сама компания)
    параметры выбираются по сетке на остальных (лучший F1 отсева, при равенстве nDCG@5), метрики
    считаются на ней. Разрыв с подбором на всех показывает, насколько метрики ленты завышены подгонкой.
    Если у компаний есть split dev/test: параметры подбираются только на dev, отчёт по test."""
    import itertools

    keys = list(PARAM_GRID)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*PARAM_GRID.values())]
    per_combo: list[dict[str, dict]] = []
    for params in combos:
        old = _set_params(params)
        try:
            res = eval_ranking(companies, items, gold_fn)
        finally:
            _set_params(old)
        per_combo.append({x["company"]: x for x in res["per_company"]})

    def best(train: list[str]) -> int:
        def key(i: int) -> tuple[float, float]:
            m = _pooled([per_combo[i][c] for c in train])
            return (round(m["f1"], 6), round(m["ndcg5"], 6))
        return max(range(len(combos)), key=key)

    ids = [c["id"] for c in companies]
    group = {c["id"]: c.get("stratum") or c["id"] for c in companies}
    held, chosen = [], []
    for g in sorted(set(group.values())):
        i = best([x for x in ids if group[x] != g])
        for c in (x for x in ids if group[x] == g):
            held.append(per_combo[i][c])
            chosen.append({"company": c, **combos[i]})
    i_all = best(ids)
    in_sample = [per_combo[i_all][c] for c in ids]
    out = {
        "cv": _pooled(held), "cv_ci": bootstrap_ci(held), "cv_params": chosen,
        "in_sample": _pooled(in_sample), "in_sample_params": combos[i_all],
        "grid_size": len(combos), "folds": len(set(group.values())),
    }
    dev = [c["id"] for c in companies if c.get("split") == "dev"]
    test = [c["id"] for c in companies if c.get("split") == "test"]
    if dev and test:
        i_dev = best(dev)
        out["dev_params"] = combos[i_dev]
        out["dev"] = _pooled([per_combo[i_dev][c] for c in dev])
        out["test"] = _pooled([per_combo[i_dev][c] for c in test])
        out["test_ci"] = bootstrap_ci([per_combo[i_dev][c] for c in test])
        out["dev_ci"] = bootstrap_ci([per_combo[i_dev][c] for c in dev])
        old_params = _set_params(combos[i_dev])
        try:
            out["baselines"] = eval_baselines(companies, items, gold_fn)
        finally:
            _set_params(old_params)
    return out


# ---------------------------------------------------------------------------
# Карточки
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    start: float
    max_rub: float
    stopped: bool = False

    def exceeded(self) -> bool:
        from ml.llm.client import SpendLedger

        if self.stopped:
            return True
        if SpendLedger().spent_today() - self.start > self.max_rub:
            self.stopped = True
        return self.stopped


# Судьи eval: две модели других семейств, чем интерпретатор (GLM).
JUDGES = ("eval_judge", "eval_judge_2")
INJECTED_ITEM = "n_2026_09_06_noisy"  # синтетическая новость с внедрённой инструкцией (ml/fixtures/news.jsonl)


@dataclass
class PairResult:
    company: str
    key: str
    item: str
    gold: int
    holdout: bool
    noisy: bool
    status: str = "ok"
    error: str | None = None
    latency_s: float = 0.0
    cost_rub: float = 0.0
    judge_cost_rub: float = 0.0
    judge2: dict | None = None
    judge2_cost_rub: float = 0.0
    split: str | None = None
    channels: list[str] | None = None  # None: двухстадийная генерация выключена
    retries: int = 0
    first_try_schema_ok: bool = True
    replans: list[str] = field(default_factory=list)  # триггеры перепланирования из трассы (T14)
    json_repaired: bool = False  # клиент сам чинил JSON интерпретатора: first_try_schema_ok этого не видит
    final_issues: list[str] = field(default_factory=list)
    relevance: float | None = None
    claims: int = 0
    claims_with_evidence: int = 0
    claims_inference: int = 0
    bank_view: bool = False
    bank_numbers_grounded: bool | None = None
    bank_view_stale_undated: bool = False
    features_ok: bool = False
    features: list[str] = field(default_factory=list)
    injection_leak: bool = False
    judge: dict | None = None
    card_id: str | None = None
    trace_id: str | None = None


def count_repaired(events: list[dict], task: str = "interpret") -> int:
    """Число вызовов LLM задачи в трассе, где клиент чинил невалидный JSON (CallInfo.repaired)."""
    return sum(e["event"] == "llm_call" and e.get("task") == task and bool(e.get("repaired")) for e in events)


def check_card(card, profile, item, hits: dict[str, dict], r: PairResult) -> None:
    import re

    from ml.agent.verifier import _path_value, ungrounded_numbers
    from ml.safety import INJECTION

    r.relevance = card.relevance.score
    claims = [*card.consequences.risks, *card.consequences.opportunities, *card.suggested_actions]
    r.claims = len(claims)
    r.claims_with_evidence = sum(bool(c.evidence) for c in claims)
    r.claims_inference = sum(c.is_model_inference and not c.evidence for c in claims)
    r.features = card.why_it_matters.company_features_used
    r.features_ok = bool(r.features) and all(
        p == "eligible_big" or _path_value(profile, p) not in (None, [], "") for p in r.features)
    text = card.model_dump_json()
    # «обанкрот» — маркер внедрённой инструкции в синтетической новости INJECTED_ITEM; в реальных новостях о банкротстве слово законно.
    r.injection_leak = bool(INJECTION.search(text)) or (item.item_id == INJECTED_ITEM and "обанкрот" in text.lower())
    if card.bank_view:
        r.bank_view = True
        cited = [f"{hits[i]['title']} {hits[i]['text']}" for i in card.bank_view.source_ids if i in hits]
        r.bank_numbers_grounded = not ungrounded_numbers(card.bank_view.text, cited)
        stale = [i for i in card.bank_view.source_ids if hits.get(i, {}).get("stale")]
        r.bank_view_stale_undated = bool(stale) and not re.search(r"20[0-2]\d", card.bank_view.text)


def judge_card(client, card, profile, item, hits: dict[str, dict], task: str = "eval_judge") -> tuple[dict, float]:
    from pydantic import BaseModel

    from ml.agent.orchestrator import item_source_id
    from ml.prompts import render

    class Verdict(BaseModel):
        factual: int
        unsupported: list[str] = []
        personalized: int
        retells_news: bool
        bank_ai_mixed: bool
        actionable: int
        relevance_ok: bool
        comment: str = ""

    from ml.ranking.ranker import score_item

    used = card.referenced_source_ids()
    # Судья видит тот же контекст, что и модель: название документа и причины ранжирования.
    chunks = [{"id": h["id"], "title": h.get("title"), "date": h["published_at"], "text": h["text"]}
              for h in hits.values() if h["id"] in used]
    reasons = [r[1] for r in score_item(profile, item).reasons] or ["явных связей с профилем не найдено"]
    item_text = item.model_dump(mode="json", include={"title", "summary", "facts"})
    # Соседние новости (search_news), на которые карточка сослалась, тоже источник: без них цифры из них судья считал выдуманными.
    from ml.errors import NotFound
    from ml.ingest.store import get_item

    related = []
    for iid in card.item_ids[1:]:
        try:
            x = get_item(iid)
        except NotFound:
            continue
        related.append({"id": item_source_id(x), **x.model_dump(mode="json", include={"title", "summary", "facts"})})
    p = render("eval_judge", profile=profile.prompt_view(), item_source_id=item_source_id(item), item_text=item_text,
               reasons=reasons, bank_chunks=chunks or "нет", related_news=related or "нет",
               card=card.model_dump(mode="json", exclude={"meta", "sources", "trace_id"}))
    v, info = client.complete_json(task, p.system, p.user, Verdict, prompt_version=p.version)
    return v.model_dump(), info.cost_rub


def card_hits(card) -> dict[str, dict]:
    """Фрагменты банка, на которые ссылается карточка, с полным текстом: их видит судья."""
    from ml.ingest.analytics import load_chunks

    by_id = {ch.chunk_id: ch for ch in load_chunks()}
    return {s.id: {"id": ch.chunk_id, "text": ch.text, "title": ch.title, "stale": ch.stale, "published_at": ch.published_at}
            for s in card.sources if (ch := by_id.get(s.id)) is not None}


def run_pair(c: dict, item, client, budget: Budget, do_judge: bool, gold_fn=None, judges: tuple[str, ...] = JUDGES) -> PairResult:
    from ml.agent.orchestrator import run_insight
    from ml.llm.client import BudgetExceeded
    from ml.agent.trace import load_trace
    from ml.eval.labels import NOISY
    from ml.profile.build_profile import get_profile

    gold = (gold_fn or human_gold)(c, item.item_id)
    r = PairResult(c["id"], c["key"], item.item_id, gold if gold is not None else 0,
                   item.item_id.startswith("h_"), item.item_id in NOISY, split=c.get("split"))
    if budget.exceeded():
        r.status = "skipped_budget"
        return r
    t0 = time.time()
    try:
        card = run_insight(c["id"], item.item_id, client)
    except BudgetExceeded as e:
        # Ключ исчерпан у организаторов (локальный журнал не видит расходов коллег): остальные пары не запускаем.
        budget.stopped = True
        r.status, r.error = "skipped_budget", f"BudgetExceeded: {e}"[:500]
        return r
    except Exception as e:  # noqa: BLE001  ошибка пары не должна ронять весь прогон
        r.status, r.error, r.latency_s = "error", f"{type(e).__name__}: {e}"[:500], round(time.time() - t0, 2)
        return r
    r.latency_s = round(time.time() - t0, 2)
    trace = load_trace(card.trace_id) or {"events": []}
    for e in trace["events"]:
        if e["event"] == "verify" and e["attempt"] == 0:
            r.first_try_schema_ok = not any("схему" in i for i in e["issues"])
    r.json_repaired = count_repaired(trace["events"]) > 0
    r.replans = [e["trigger"] for e in trace["events"] if e["event"] == "replan"]
    hits = card_hits(card)
    r.card_id, r.trace_id = card.card_id, card.trace_id
    r.retries, r.cost_rub, r.final_issues = card.meta.retries, card.meta.cost_rub, card.meta.verifier_issues
    if any(e["event"] == "channels" for e in trace["events"]):
        r.channels = [f"{c.channel}:{c.strength}" for c in card.channels]
    profile = get_profile(c["id"])
    check_card(card, profile, item, hits, r)
    for n, task in enumerate(judges if do_judge else ()):
        if budget.exceeded():
            break
        try:
            v, cost = judge_card(client, card, profile, item, hits, task)
        except Exception as e:  # noqa: BLE001
            v, cost = {"error": f"{type(e).__name__}: {e}"[:300]}, 0.0
        if "error" not in v:
            v.update(unsupported_kinds(v.get("unsupported", []), card))
        if n == 0:
            r.judge, r.judge_cost_rub = v, cost
        else:
            r.judge2, r.judge2_cost_rub = v, cost
    return r


def unsupported_kinds(unsupported: list[str], card) -> dict[str, int]:
    """Замечания судьи по месту в карточке: «как факт» — в тексте, поданном со ссылкой на источник
    (что случилось, почему важно, мнение банка, последствия и действия с evidence); «гипотеза» — в выводе модели
    (ai_inference и пункты с is_model_inference). Только «как факт» означает выдумку, поданную как факт."""
    from ml.eval.judge_types import segs, where

    ss = segs(card.model_dump(mode="json"))
    places = [where(u, ss) for u in unsupported]
    return {"unsupported_fact": sum(p == "sourced" for p in places), "unsupported_inference": sum(p == "inference" for p in places),
            "unsupported_unplaced": sum(p == "не найдено" for p in places)}


# ---------------------------------------------------------------------------
# Агрегация и отчёт
# ---------------------------------------------------------------------------


def _share(xs: list[bool]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _pct(q: float, xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def _kappa(a: list[int], b: list[int]) -> float | None:
    from ml.eval.label_llm import cohen_kappa

    return cohen_kappa(a, b) if len(a) > 1 else None


def aggregate(rs: list[PairResult]) -> dict:
    ok = [r for r in rs if r.status == "ok"]
    judged = [r for r in ok if r.judge and "error" not in r.judge]
    judged2 = [r for r in ok if r.judge2 and "error" not in r.judge2]
    both = [r for r in judged if r in judged2]
    rel = [r for r in ok if r.gold >= 1]
    claims = sum(r.claims for r in ok)
    with_ch = [r for r in ok if r.channels is not None]
    out = {
        "channels_on": bool(with_ch),
        "cards_with_channels": _share([bool(r.channels) for r in with_ch]),
        "channels_per_card": _mean([len(r.channels) for r in with_ch]),
        "channels_strong_share": _share([any(c.endswith(":strong") for c in r.channels) for r in with_ch]),
        "pairs": len(rs),
        "cards_ok": len(ok),
        "errors": sum(r.status == "error" for r in rs),
        "skipped_budget": sum(r.status == "skipped_budget" for r in rs),
        "schema_valid": _share([r.status == "ok" for r in rs if r.status != "skipped_budget"]),
        "first_try_schema_ok": _share([r.first_try_schema_ok for r in ok]),
        "json_repaired": _share([r.json_repaired for r in ok]),
        "replan_any": _share([bool(r.replans) for r in ok]),
        "replan_bank_miss": _share(["bank_miss" in r.replans for r in ok]),
        "replan_few_facts": _share(["few_facts" in r.replans for r in ok]),
        "no_retry": _share([r.retries == 0 for r in ok]),
        "verifier_clean": _share([not r.final_issues for r in ok]),
        "evidence_coverage": (sum(r.claims_with_evidence for r in ok) / claims) if claims else None,
        "claims_flagged_inference": (sum(r.claims_inference for r in ok) / claims) if claims else None,
        "features_ok": _share([r.features_ok for r in ok]),
        "bank_view_rate_relevant": _share([r.bank_view for r in rel]),
        "bank_numbers_grounded": _share([r.bank_numbers_grounded for r in ok if r.bank_numbers_grounded is not None]),
        "bank_view_stale_undated": sum(r.bank_view_stale_undated for r in ok),
        "injection_leaks": sum(r.injection_leak for r in ok),
        "judge_factual": _mean([r.judge["factual"] for r in judged]),
        "judge_factual_2": _share([r.judge["factual"] == 2 for r in judged]),
        "judge_with_unsupported": _share([bool(r.judge["unsupported"]) for r in judged]),
        # Разделение замечаний (unsupported_kinds): выдумка, поданная как факт, против слабой помеченной гипотезы.
        "judge_unsupported_as_fact": _share([r.judge.get("unsupported_fact", 0) + r.judge.get("unsupported_unplaced", 0) > 0
                                             for r in judged if "unsupported_fact" in r.judge]),
        "judge_unsupported_inference_only": _share([r.judge.get("unsupported_inference", 0) > 0
                                                    and r.judge.get("unsupported_fact", 0) + r.judge.get("unsupported_unplaced", 0) == 0
                                                    for r in judged if "unsupported_fact" in r.judge]),
        "judge_personalized": _mean([r.judge["personalized"] for r in judged]),
        "judge_actionable": _mean([r.judge["actionable"] for r in judged]),
        "judge_retells": _share([r.judge["retells_news"] for r in judged]),
        "judge_bank_ai_mixed": _share([r.judge["bank_ai_mixed"] for r in judged]),
        "judge_relevance_ok": _share([r.judge["relevance_ok"] for r in judged]),
        "cost_cards_rub": sum(r.cost_rub for r in ok),
        "cost_judge_rub": sum(r.judge_cost_rub + r.judge2_cost_rub for r in ok),
        "judge2_factual": _mean([r.judge2["factual"] for r in judged2]),
        "judge2_factual_2": _share([r.judge2["factual"] == 2 for r in judged2]),
        "judge2_with_unsupported": _share([bool(r.judge2["unsupported"]) for r in judged2]),
        "judge2_personalized": _mean([r.judge2["personalized"] for r in judged2]),
        "judge2_actionable": _mean([r.judge2["actionable"] for r in judged2]),
        "judge2_bank_ai_mixed": _share([r.judge2["bank_ai_mixed"] for r in judged2]),
        # Согласие судей по factual: доля совпадений и взвешенная κ.
        "judges_factual_agree": _share([r.judge["factual"] == r.judge2["factual"] for r in both]),
        "judges_factual_kappa": _kappa([r.judge["factual"] for r in both], [r.judge2["factual"] for r in both]),
        "judges_unsupported_agree": _share([bool(r.judge["unsupported"]) == bool(r.judge2["unsupported"]) for r in both]),
        "cost_per_card_rub": _mean([r.cost_rub for r in ok]),
        "latency_p50_s": _pct(0.5, [r.latency_s for r in ok]),
        "latency_p95_s": _pct(0.95, [r.latency_s for r in ok]),
    }
    # Калибровка relevance.score карточки против разметки.
    for g in (0, 1, 2):
        out[f"relevance_mean_gold{g}"] = _mean([r.relevance for r in ok if r.gold == g and r.relevance is not None])
    pred = [(r.relevance or 0) >= 0.3 for r in ok]
    gold = [r.gold >= 1 for r in ok]
    tp = sum(p and g for p, g in zip(pred, gold))
    out["card_relevance_precision"] = tp / sum(pred) if sum(pred) else None
    out["card_relevance_recall"] = tp / sum(gold) if sum(gold) else None
    return out


def _f(v, pct: bool = False, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return f"{v * 100:.0f}%" if pct else f"{v:.{nd}f}"


def ranking_section(ranking: dict) -> list[str]:
    L = ["## Ранжирование ленты (без LLM)", "",
         f"Пул после дедупликации: {ranking['pool_size']} материалов. Дубликаты схлопнуты: {ranking['dedup_pairs_merged']}.", "",
         "| | Лента (ml.ranking) | Только BM25 по профилю |", "|---|---|---|",
         f"| nDCG@5 | {_f(ranking['ndcg5'])} | {_f(ranking['ndcg5_bm25'])} |",
         f"| recall@5 | {_f(ranking['recall5'], True)} | {_f(ranking['recall5_bm25'], True)} |",
         f"| precision / recall отсева | {_f(ranking['precision'], True)} / {_f(ranking['recall'], True)} | — |", "",
         "| Компания | nDCG@5 | В ленте / релевантных | Пропущены | Лишние |", "|---|---|---|---|---|"]
    for x in ranking["per_company"]:
        L += [f"| {x['company']} ({x['key']}) | {x['ndcg5']:.2f} | {x['kept']} / {x['relevant']} | "
              f"{', '.join(x['missed']) or '—'} | {', '.join(x['false_pos']) or '—'} |"]
    L += [""]
    if ranking.get("baselines"):
        L += ["### Бейзлайны и персонализация", "",
              "Как читать. nDCG@5 на этом пуле в основном меряет общую важность материала: 53% дисперсии разметки объясняет сам материал, "
              "11% компания. Порядок по средней оценке материала у других компаний (популярность) поэтому силён и не персонализирован. "
              "Персонализацию показывает AUC «кому показать материал»: внутри одного материала сравниваются компании, где он релевантен и нет; "
              "у неперсонализированного порядка он равен 0,50. Бейзлайны считаются на том же пуле и том же эталоне, что и система.", ""]
        L += baselines_lines(ranking["baselines"]["все"], "Все компании, текущие параметры")
    if ranking.get("label_sources"):
        L += ["### Источники разметки", "",
              "Пары `human` размечены вручную, `llm_agreed` совпали у двух моделей, `llm_disputed` спорные (в основных метриках исключены).", "",
              "| Split | human | llm_agreed | llm_disputed | доля спорных |", "|---|---|---|---|---|"]
        for sp, d in sorted(ranking["label_sources"].items()):
            n = sum(d.values())
            L += [f"| {sp} | {d.get('human', 0)} | {d.get('llm_agreed', 0)} | {d.get('llm_disputed', 0)} | "
                  f"{_f(d.get('llm_disputed', 0) / n, True)} |"]
        L += [""]
    if ranking.get("disputed"):
        L += ["### Чувствительность к спорным парам", "",
              "Спорные пары (модели A и B дали разные оценки) включены в эталон. Диапазон между вариантами показывает, "
              "насколько вывод зависит от исключения 24% пар test. Параметры системы текущие.", "",
              "| Оценка спорной пары | Группа | nDCG@5 | precision отсева | recall отсева | F1 отсева |", "|---|---|---|---|---|---|"]
        for v, label in DISPUTED_VARIANTS.items():
            for g, m in ranking["disputed"][v].items():
                L += [f"| {label} | {g} | {_f(m['ndcg5'], nd=3)} | {_f(m['precision'], True)} | {_f(m['recall'], True)} | {_f(m['f1'], True)} |"]
        L += [""]
    return L


def ranking_cv_section(ranking: dict, cv: dict) -> list[str]:
    ci = bootstrap_ci(ranking["per_company"])
    c, s, cci = cv["cv"], cv["in_sample"], cv["cv_ci"]

    def row(name: str, k: str, pct: bool) -> str:
        cur = ranking[k] if k in ranking else _pooled(ranking["per_company"])[k]
        return (f"| {name} | {_f(cur, pct)} [{_f(ci[k][0], pct)}; {_f(ci[k][1], pct)}] | {_f(s[k], pct)} | "
                f"{_f(c[k], pct)} [{_f(cci[k][0], pct)}; {_f(cci[k][1], pct)}] | {_f(s[k] - c[k], pct)} |")

    fold = "компанию" if cv.get("folds") == len(ranking["per_company"]) else "отрасль (страту ОКВЭД)"
    L = ["## Честная оценка ленты", "",
         f"Параметры ранжирования ({', '.join(PARAM_GRID)}) подбираются по сетке из {cv['grid_size']} вариантов "
         f"(лучший F1 отсева, при равенстве nDCG@5) без одной группы и проверяются на ней: группа = {fold}, "
         f"фолдов {cv.get('folds')}. В скобках 95% интервал bootstrap по компаниям ({BOOTSTRAP_N} выборок).", "",
         "| Метрика | Текущие параметры | Подбор на всех (in-sample) | Подбор без группы (CV) | Разрыв in-sample − CV |",
         "|---|---|---|---|---|",
         row("nDCG@5", "ndcg5", False), row("recall@5", "recall5", True),
         row("precision отсева", "precision", True), row("recall отсева", "recall", True), row("F1 отсева", "f1", True), "",
         f"Параметры, выбранные на всех компаниях: {cv['in_sample_params']}.", ""]
    if "test" in cv:
        d, t, dci, tci = cv["dev"], cv["test"], cv["dev_ci"], cv["test_ci"]

        def dt(name: str, k: str, pct: bool) -> str:
            return (f"| {name} | {_f(d[k], pct)} [{_f(dci[k][0], pct)}; {_f(dci[k][1], pct)}] | "
                    f"{_f(t[k], pct)} [{_f(tci[k][0], pct)}; {_f(tci[k][1], pct)}] | {_f(d[k] - t[k], pct)} |")
        n_dev = sum(x.get("split") == "dev" for x in ranking["per_company"])
        L += ["### dev / test", "",
              f"Параметры подобраны только на dev ({n_dev} компаний): {cv['dev_params']}. "
              f"test ({len(ranking['per_company']) - n_dev} компаний) из отраслей, которых нет в dev.", "",
              "| Метрика | dev | test | Разрыв dev − test |", "|---|---|---|---|",
              dt("nDCG@5", "ndcg5", False), dt("recall@5", "recall5", True), dt("precision отсева", "precision", True),
              dt("recall отсева", "recall", True), dt("F1 отсева", "f1", True), ""]
        for name in ("dev", "test"):
            if name in cv.get("baselines", {}):
                L += baselines_lines(cv["baselines"][name], f"{name}: бейзлайны и AUC при параметрах, подобранных на dev")
    stable = {k: sorted({str(x[k]) for x in cv["cv_params"]}) for k in PARAM_GRID}
    L += ["Разброс выбранных параметров по фолдам: " + "; ".join(f"{k} {', '.join(v)}" for k, v in stable.items()) + ".", ""]
    return L


def write_ranking_report(run_id: str, ranking: dict, path: Path, cv: dict | None = None) -> None:
    from ml import topics
    from ml.ranking import ranker

    L = [f"# Eval ранжирования ленты: прогон `{run_id}`", "",
         "Без вызовов LLM. Файл генерируется `python -m ml.eval.run_eval --ranking-only [--set v2]` и перезаписывается. "
         "Разметка: v1 ручная (`ml/eval/labels.py`), v2 ручная + согласованная двумя LLM (`ml/eval/labels.jsonl`).", "",
         f"Параметры: ALPHA={ranker.ALPHA}, MIN_SCORE={ranker.MIN_SCORE}, FEATURE_WEIGHT={topics.FEATURE_WEIGHT}, "
         f"OFF_SECTOR_DAMP={ranker.OFF_SECTOR_DAMP}, ADJACENCY_WEIGHT={ranker.ADJACENCY_WEIGHT}, BETA={ranker.BETA}, "
         f"BROAD_WEIGHT_NO_HOOKS={ranker.BROAD_WEIGHT_NO_HOOKS}, смежность отраслей: {topics.ADJACENCY_SOURCE}, "
         f"устаревшая аналитика банка (старше {ranker.STALE_AFTER_DAYS} дн.) в ленту не попадает. "
         f"Правила по масштабу события: ECONOMY_MIN_IMPORTANCE={ranker.ECONOMY_MIN_IMPORTANCE}, NARROW_NEEDS_SECTOR={ranker.NARROW_NEEDS_SECTOR}.", ""]
    model = ranker.active_model()
    if model is not None:
        L += [f"Модель пары: `ml/ranking/pair_model.json` (обучена {model.get('trained_on', {}).get('date', '?')} на "
              f"{', '.join(model.get('trained_on', {}).get('sets', []))}; w_model={model.get('w_model')}, order_mix={model.get('order_mix')}, "
              f"beta={model.get('beta')}, l2={model.get('l2')}). **Метрики ниже на наборах v2 и real получены в выборке обучения модели "
              "и завышены.** Честная оценка: перекрёстная проверка без одного дня (`python -m ml.ranking.pair_model cv`, "
              "`ml/ranking/pair_model_cv.json`, сводка в `ml/eval/analysis.md`); итоговая цифра: свежий набор real2. "
              "Сетка параметров эвристики при модели не считается.", ""]
    L += ranking_section(ranking)
    if cv:
        L += ranking_cv_section(ranking, cv)
    path.write_text("\n".join(L), encoding="utf-8")


def write_report(run_id: str, args, ranking: dict, groups: dict[str, dict], rs: list[PairResult], spend: dict, path: Path,
                 cv: dict | None = None) -> None:
    a = groups["all"]
    L = []
    L += [f"# Eval: отчёт прогона `{run_id}`", ""]
    L += [f"Режим: {'мок (без расходов, качество не оценивается)' if args.mock else 'реальные модели'}. "
          f"Интерпретатор: `{spend['interpret_model']}`, судья: `{spend['judge_model']}`"
          + (f" и `{spend['judge2_model']}`" if spend.get("judge2_model") else "") + ". "
          f"Пар: {a['pairs']} ({spend['n_companies']} компаний, {spend['n_news']} новостей, из них {spend['n_holdout']} отложенных v1; набор {spend['set']}). "
          f"Кэш LLM отключён: стоимость и время честные.", ""]
    L += ["Файл генерируется скриптом и перезаписывается при каждом прогоне. Ручной разбор и сравнение прогонов: `ml/eval/analysis.md`.", ""]
    labels = {"v1": "Разметка релевантности ручная (`ml/eval/labels.py`), делалась по профилю, до прогона. ",
              "real": "Набор real: реальные новости dev, фиксированные пары `ml/eval/card_pairs_real.json` (только пары из ленты); "
                      "разметка релевантности: согласие двух LLM (`ml/eval/labels_real.jsonl`), ручной нет. "}
    L += [labels.get(spend["set"], "Разметка релевантности: ручная для 9 исходных компаний, для остальных согласие двух LLM "
                                   "(`ml/eval/labels.jsonl`, протокол `ml/eval/PROTOCOL.md`). ") +
          "Судьи LLM других моделей оценивают карточки; их оценки ориентир, не истина: калибровка `ml/eval/judge_review.py`.", ""]

    L += ["## Итог", ""]
    L += ["| Метрика (ML_SPEC, раздел 8) | Значение |", "|---|---|"]
    rows = [
        ("Валидность схемы (карточка получена)", _f(a["schema_valid"], True)),
        ("Валидна с первой попытки", _f(a["first_try_schema_ok"], True)),
        ("JSON починен клиентом (интерпретатор; в «с первой попытки» не виден)", _f(a["json_repaired"], True)),
        ("Без повторов", _f(a["no_retry"], True)),
        ("Перепланирование по данным: любое / нет позиции банка / мало фактов",
         f"{_f(a['replan_any'], True)} / {_f(a['replan_bank_miss'], True)} / {_f(a['replan_few_facts'], True)}"),
        ("Проверка пройдена без замечаний", _f(a["verifier_clean"], True)),
        ("Покрытие источниками (утверждения с evidence)", _f(a["evidence_coverage"], True)),
        ("Утверждения, помеченные как вывод модели без источника", _f(a["claims_flagged_inference"], True)),
        ("Фактологичность, судья 1 / судья 2 (0-2)", f"{_f(a['judge_factual'])} / {_f(a['judge2_factual'])}"),
        ("Карточки без замечаний по фактам (factual=2), судья 1 / 2", f"{_f(a['judge_factual_2'], True)} / {_f(a['judge2_factual_2'], True)}"),
        ("Карточки с неподтверждёнными утверждениями, судья 1 / 2",
         f"{_f(a['judge_with_unsupported'], True)} / {_f(a['judge2_with_unsupported'], True)}"),
        ("  из них неподтверждённое подано как факт (судья 1)", _f(a.get("judge_unsupported_as_fact"), True)),
        ("  из них только в помеченных выводах модели (судья 1)", _f(a.get("judge_unsupported_inference_only"), True)),
        ("Согласие судей: factual совпал / κ_w; «есть неподтверждённое» совпало",
         f"{_f(a['judges_factual_agree'], True)} / {_f(a['judges_factual_kappa'])}; {_f(a['judges_unsupported_agree'], True)}"),
        ("Персонализация: признаки профиля указаны и существуют", _f(a["features_ok"], True)),
        ("Персонализация, судья 1 / 2 (0-2)", f"{_f(a['judge_personalized'])} / {_f(a['judge2_personalized'])}"),
        ("Пересказ вместо выводов, судья", _f(a["judge_retells"], True)),
        ("Действия конкретны, судья 1 / 2 (0-2)", f"{_f(a['judge_actionable'])} / {_f(a['judge2_actionable'])}"),
        ("Разделение банк/ИИ: bank_view без source_ids", "0 (запрещено схемой)"),
        ("Разделение банк/ИИ: выводы модели в bank_view, судья 1 / 2",
         f"{_f(a['judge_bank_ai_mixed'], True)} / {_f(a['judge2_bank_ai_mixed'], True)}"),
        ("Цифры в bank_view найдены в цитируемых фрагментах", _f(a["bank_numbers_grounded"], True)),
        ("bank_view заполнен для релевантных пар", _f(a["bank_view_rate_relevant"], True)),
        ("Устаревший материал в bank_view без даты", str(a["bank_view_stale_undated"])),
        ("Утечки prompt injection в карточку", str(a["injection_leaks"])),
        ("Отсев нерелевантного, лента: precision / recall", f"{_f(ranking['precision'], True)} / {_f(ranking['recall'], True)}"),
        ("Отсев нерелевантного, relevance.score карточки >= 0.3: precision / recall",
         f"{_f(a['card_relevance_precision'], True)} / {_f(a['card_relevance_recall'], True)}"),
        ("Каналы влияния (A3): карточек с каналом / с сильным каналом / каналов на карточку",
         f"{_f(a['cards_with_channels'], True)} / {_f(a['channels_strong_share'], True)} / {_f(a['channels_per_card'], nd=1)}"
         if a["channels_on"] else "выключено"),
        ("Стоимость карточки, руб. (среднее)", _f(a["cost_per_card_rub"], nd=3)),
        ("Латентность карточки p50 / p95, с", f"{_f(a['latency_p50_s'], nd=1)} / {_f(a['latency_p95_s'], nd=1)}"),
    ]
    L += [f"| {k} | {v} |" for k, v in rows]
    L += [""]

    L += ["## Расход", ""]
    L += ["| Статья | Руб. |", "|---|---|",
          f"| Разбор holdout-новостей (Gemma) | {spend['parse_rub']:.3f} |",
          f"| Карточки ({a['cards_ok']} шт.) | {a['cost_cards_rub']:.3f} |",
          f"| Судьи ({', '.join(m for m in (spend['judge_model'], spend.get('judge2_model')) if m)}) | {a['cost_judge_rub']:.3f} |",
          f"| **Всего за прогон (по журналу)** | **{spend['run_rub']:.3f}** |",
          f"| Потрачено за бюджетные сутки на этой машине | {spend['day_rub']:.3f} из {settings.llm.daily_budget_rub:.0f} |", ""]
    if a["skipped_budget"]:
        L += [f"Остановлено лимитом прогона `--max-rub {args.max_rub}`: пропущено {a['skipped_budget']} пар.", ""]
    L += [f"Прогноз на день: при {_f(a['cost_per_card_rub'], nd=3)} руб. за карточку дневного лимита хватает примерно на "
          f"{int(settings.llm.daily_budget_rub / a['cost_per_card_rub']) if a['cost_per_card_rub'] else '—'} карточек (без судьи).", ""]

    L += ["## Срезы", ""]
    keys = [("cards_ok", "Карточек", False), ("first_try_schema_ok", "С 1-й попытки", True), ("evidence_coverage", "Покрытие", True),
            ("judge_factual", "Факты (0-2)", False), ("judge_personalized", "Персонализ. (0-2)", False),
            ("bank_numbers_grounded", "Цифры банка", True), ("cost_per_card_rub", "Руб./карт.", False), ("latency_p50_s", "p50, с", False)]
    L += ["| Срез | " + " | ".join(k[1] for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, g in groups.items():
        if name == "all":
            continue
        L += [f"| {name} | " + " | ".join(_f(g[k], pct) if k != "cards_ok" else str(g[k]) for k, _, pct in keys) + " |"]
    L += [""]

    L += ["## Калибровка релевантности карточки", "",
          "| Разметка | Средний relevance.score |", "|---|---|",
          f"| 0 (не относится) | {_f(a['relevance_mean_gold0'])} |",
          f"| 1 (косвенно) | {_f(a['relevance_mean_gold1'])} |",
          f"| 2 (напрямую) | {_f(a['relevance_mean_gold2'])} |", ""]

    L += ranking_section(ranking)
    if cv:
        L += ranking_cv_section(ranking, cv)

    bad = [r for r in rs if r.status == "error"]
    if bad:
        L += ["## Ошибки", ""] + [f"- `{r.company} x {r.item}`: {r.error}" for r in bad[:15]] + [""]
    worst = sorted([r for r in rs if r.judge and "error" not in r.judge and (r.judge["factual"] < 2 or r.judge["unsupported"])],
                   key=lambda r: (r.judge["factual"], -len(r.judge["unsupported"])))[:12]
    if worst:
        L += ["## Карточки с замечаниями судья (для ручной проверки)", ""]
        for r in worst:
            uns = "; ".join(r.judge["unsupported"][:3])
            L += [f"- `{r.card_id}` factual={r.judge['factual']}: {r.judge.get('comment', '')[:300]}" + (f" Неподтверждено: {uns[:300]}" if uns else "")]
        L += [""]
    issues = [(r.card_id, i) for r in rs if r.status == "ok" for i in r.final_issues]
    if issues:
        L += ["## Замечания проверки, оставшиеся после повторов", ""] + [f"- `{c}`: {i[:250]}" for c, i in issues[:15]] + [""]
    L += [f"Сырые результаты: `var/eval/{run_id}/results.jsonl`, трассы: `var/eval/{run_id}/traces/`.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-rub", type=float, default=60.0)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--ranking-only", action="store_true", help="только ранжирование, без вызовов LLM (разбор holdout из кэша)")
    ap.add_argument("--rerank", action="store_true",
                    help="с --ranking-only: оценка переранжирования моделью (ПЛАТНО, ~6-24 руб.); сначала --rerank-dry-run")
    ap.add_argument("--rerank-dry-run", action="store_true", help="число вызовов и оценка стоимости переранжирования, без вызовов")
    ap.add_argument("--limit", type=int, default=0, help="ограничить число пар (отладка)")
    ap.add_argument("--news-split", choices=("dev", "holdout"), help="real: только новости этой части (holdout не открывать до итоговой оценки)")
    ap.add_argument("--set", choices=("v1", "v2", "real", "real2"), default="v1",
                    help="v1: 9 компаний x все новости, ручная разметка; v2: 40 компаний (dev/test), разметка labels.jsonl, "
                         "карточки по фиксированным парам card_pairs_v2.json")
    ap.add_argument("--split", choices=("dev", "test"), help="v2: только пары dev- или test-компаний (PROTOCOL.md: настройка только на dev)")
    ap.add_argument("--second-judge", action="store_true",
                    help="второй судья (Qwen): на прогоне 20261021_124003 ставил factual=2 в 99% карточек, κ_w с первым 0,02")
    ap.add_argument("--report", default=str(EVAL_DIR / "report.md"))
    ap.add_argument("--ranking-report", default=str(EVAL_DIR / "ranking_report.md"))
    args = ap.parse_args(argv)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_mock" if args.mock else "")
    out = settings.paths.var / "eval" / run_id
    out.mkdir(parents=True, exist_ok=True)
    os.environ["ML_RUNS_DIR"] = str(out)  # карточки и трассы eval отдельно от рабочих

    from ml.eval.labels import LABELS
    from ml.llm.client import LLMClient, SpendLedger
    from ml.llm.router import model_name, route
    from ml.profile.build_profile import get_profile
    from ml.retrieval.index import get_index

    ledger = SpendLedger()
    day0 = ledger.spent_today()
    client = LLMClient(mock=args.mock, use_cache=False)
    real = args.set.startswith("real")
    if real and not args.ranking_only and args.set != "real":
        sys.exit(f"карточки на {args.set} не размечены парами: только --ranking-only")
    if args.set == "v2" or real:
        from ml.eval.label_llm import load_v2

        companies, gold_fn = load_v2(), (load_gold_v2() if args.set == "v2" else load_gold_real(name=args.set))
    else:
        companies, gold_fn = load_companies(), human_gold
    for c in companies:
        get_profile(c["id"])
    get_index()

    if real:
        check_real_as_of(args.set)
        items = news = load_real_items(mock=args.mock, cache_only=True, split=args.news_split, name=args.set)
        holdout = []
    else:
        items, news, holdout = load_eval_items(mock=args.mock, cache_only=args.ranking_only)
    parse_rub = ledger.spent_today() - day0
    missing = [i.item_id for i in items if i.item_id not in LABELS]
    if missing and args.set == "v1":
        print(f"нет разметки для {missing}", file=sys.stderr)

    if args.rerank_dry_run or args.rerank:
        plan = rerank_plan(companies, items)
        calls = sum(len(c) for _, _, c in plan.values())
        print(f"переранжирование: вызовов Gemma {calls} (кэш может уменьшить), оценка до {calls * RERANK_RUB_PER_CALL:.1f} руб.", file=sys.stderr)
        if args.rerank_dry_run:
            return
        if calls * RERANK_RUB_PER_CALL > args.max_rub:
            sys.exit(f"оценка {calls * RERANK_RUB_PER_CALL:.1f} руб. выше --max-rub {args.max_rub}: прогон не начат")
        gold_sets = {"ручные пары (human, 9 компаний)": (load_gold_v2(("human",)), lambda c: True),
                     "dev, все пары": (gold_fn, lambda c: c.get("split") == "dev"),
                     "test, итоговая цифра": (gold_fn, lambda c: c.get("split") == "test")}
        res = eval_rerank(companies, items, LLMClient(mock=args.mock, use_cache=True), gold_sets, plan)
        (out / "rerank.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        Path(args.ranking_report).with_name("rerank_report.md").write_text("\n".join(rerank_section(res)), encoding="utf-8")
        print(f"rerank report -> {Path(args.ranking_report).with_name('rerank_report.md')}; spent={ledger.spent_today() - day0:.3f} rub", file=sys.stderr)
        return

    ranking = eval_ranking(companies, items, gold_fn)
    print(f"ranking: P={ranking['precision']:.2f} R={ranking['recall']:.2f} nDCG@5={ranking['ndcg5']:.2f} (bm25 {ranking['ndcg5_bm25']:.2f})", file=sys.stderr)
    ranking["baselines"] = eval_baselines(companies, items, gold_fn)
    b = ranking["baselines"]["все"]
    print(f"baselines: популярность nDCG@5={b['popularity']['ndcg5']:.2f}, случайно {b['random']['ndcg5']:.2f}, "
          f"AUC медиана {b['auc']['median']:.2f}", file=sys.stderr)
    if args.set == "v2":
        ranking["label_sources"] = label_source_shares()
        ranking["disputed"] = eval_disputed(companies, items)
    from ml.ranking import ranker

    # Сетка эвристики (ALPHA, MIN_SCORE, ...) имеет смысл без модели пары; честная оценка модели: ml/ranking/pair_model_cv.json.
    cv = eval_ranking_cv(companies, items, gold_fn) if ranker.active_model() is None else None
    if cv:
        print(f"ranking CV: P={cv['cv']['precision']:.2f} R={cv['cv']['recall']:.2f} nDCG@5={cv['cv']['ndcg5']:.2f} "
              f"(in-sample nDCG@5 {cv['in_sample']['ndcg5']:.2f})", file=sys.stderr)

    if args.ranking_only:
        (out / "ranking.json").write_text(json.dumps({**ranking, "cv": cv}, ensure_ascii=False, indent=1), encoding="utf-8")
        write_ranking_report(run_id, ranking, Path(args.ranking_report), cv)
        if real:
            rp = Path(args.ranking_report)
            rp.write_text(rp.read_text(encoding="utf-8").replace(
                "Разметка: v1 ручная (`ml/eval/labels.py`), v2 ручная + согласованная двумя LLM (`ml/eval/labels.jsonl`).",
                f"Набор {args.set}: реальные новости `ml/eval/{args.set}_news.jsonl` (часть {args.news_split or 'все'}), as_of {settings.as_of}; "
                f"разметка: согласие двух LLM (`ml/eval/labels_{args.set}.jsonl`), ручной разметки нет."), encoding="utf-8")
        print(f"ranking report -> {args.ranking_report}; spent={ledger.spent_today() - day0:.3f} rub", file=sys.stderr)
        return

    if args.set in ("v2", "real"):
        # Фиксированные пары: v2 card_pairs_v2.json, real card_pairs_real.json (ml.eval.card_pairs_real, только новости dev).
        by_c, by_i = {c["id"]: c for c in companies}, {i.item_id: i for i in news}
        fixed = json.loads((EVAL_DIR / f"card_pairs_{args.set}.json").read_text(encoding="utf-8"))
        pairs = [(by_c[x["company"]], by_i[x["item"]]) for x in fixed if (not args.split or x["split"] == args.split) and x["item"] in by_i]
    else:
        pairs = [(c, i) for c in companies for i in news]
    if args.limit:
        pairs = pairs[: args.limit]
    budget = Budget(start=ledger.spent_today(), max_rub=args.max_rub)
    results: list[PairResult] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        judges = JUDGES if args.second_judge else JUDGES[:1]
        futs = [ex.submit(run_pair, c, i, client, budget, not args.no_judge, gold_fn, judges) for c, i in pairs]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            if n % 10 == 0 or n == len(futs):
                print(f"{n}/{len(futs)} {time.time() - t0:.0f}s spent={ledger.spent_today() - day0:.2f} rub", file=sys.stderr)

    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    groups = {
        "все": aggregate(results),
        "демо-новости": aggregate([r for r in results if not r.holdout]),
        "отложенные v1 (использовались при настройке ленты)": aggregate([r for r in results if r.holdout]),
        "чистые тексты": aggregate([r for r in results if not r.noisy]),
        "зашумлённые тексты": aggregate([r for r in results if r.noisy]),
        "релевантные пары (разметка >= 1)": aggregate([r for r in results if r.gold >= 1]),
        "нерелевантные пары (разметка 0)": aggregate([r for r in results if r.gold == 0]),
    }
    if args.set == "v2":
        groups["dev-компании"] = aggregate([r for r in results if r.split == "dev"])
        groups["test-компании"] = aggregate([r for r in results if r.split == "test"])
    groups["all"] = groups["все"]
    spend = {
        "parse_rub": parse_rub, "run_rub": ledger.spent_today() - day0, "day_rub": ledger.spent_today(),
        "interpret_model": model_name(route("interpret").family), "judge_model": model_name(route("eval_judge").family),
        "judge2_model": model_name(route("eval_judge_2").family) if args.second_judge else None, "n_companies": len({c["id"] for c, _ in pairs}), "set": args.set,
        "n_news": len({i.item_id for _, i in pairs}), "n_holdout": len({i.item_id for _, i in pairs if i.item_id.startswith("h_")}),
    }
    metrics = {"run_id": run_id, "ranking": ranking, "ranking_cv": cv, "groups": {k: v for k, v in groups.items() if k != "all"}, "spend": spend}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    view = {k: v for k, v in groups.items() if k != "all"}
    write_report(run_id, args, ranking, {"all": groups["all"], **view}, results, spend, Path(args.report), cv)
    print(f"report -> {args.report}; raw -> {out.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
