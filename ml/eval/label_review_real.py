"""Ручная проверка машинной разметки реальных новостей.

    ML_AS_OF=2026-10-22 python -m ml.eval.label_review_real make     # -> ml/eval/review_real.md + review_real_key.json
    python -m ml.eval.label_review_real score                        # после заполнения review_real.md
    python -m ml.eval.label_review_real score --apply                # то же и дописать оценки в labels_real.jsonl (source="human")
    ML_AS_OF=2026-10-22 python -m ml.eval.label_review_real make --batch 2   # вторая партия: случайные пары -> review_real_2.md
    python -m ml.eval.label_review_real score --batch 2 [--apply]
    ML_AS_OF=<as_of real2> python -m ml.eval.label_review_real make --set real2 --batch 1   # итоговый набор: 150 случайных пар
    python -m ml.eval.label_review_real score --set real2 --batch 1 --apply                # -> labels_real2.jsonl (source="human")
    ML_AS_OF=2026-10-22 python -m ml.eval.label_review_real tiers   # уровни ленты против ручных оценок (внефолдовая модель)

Партия 2 и следующие: простая случайная выборка из всех пар real (dev и holdout, включая спорные, где модели разошлись),
кроме уже размеченных человеком; не больше BATCH_PER_ITEM пар на новость. Без слоёв: несмещённая оценка согласия и ленты.

Только новости dev. Выборка по слоям, где ошибка разметки сильнее всего сдвигает выводы о ленте:
- macro: материалы scope="economy" (разметка ставит их почти всем компаниям; правило ленты ranker.ECONOMY_MIN_IMPORTANCE опирается на это);
- kept_neg: лента оставила, разметка 0 (ложные срабатывания: возможно, ошибается разметка, а не лента);
- dropped_pos: лента отсеяла, разметка >= 1, без macro;
- random: случайные пары, для общей оценки согласия.
Оценки моделей и ленты в файл для человека не попадают (лежат в review_real_key.json), чтобы не влиять на оценку.
Спорные пары (модели разошлись) в выборку не входят: в метриках их нет.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys

from ml.eval.collect_real import labels_path
from ml.eval.label_llm import EVAL_DIR, agreement, load_v2

REVIEW = EVAL_DIR / "review_real.md"
KEY = EVAL_DIR / "review_real_key.json"
SEED = 20261022
QUOTAS = {"macro": 30, "kept_neg": 30, "dropped_pos": 15, "random": 25}
_PAIR = re.compile(r"^## (\d+)\. `([^`]+)` × `([^`]+)`", re.M)
_ANSWER = re.compile(r"^Оценка:[ \t]*([012?])?[ \t]*$", re.M)
BATCH_N, BATCH_PER_ITEM = 120, 4
REAL2_N = 150

CRITERIA = """## Как ставить оценку

В строке «Оценка:» каждой пары поставьте одну цифру: **0**, **1** или **2**. Не уверены — поставьте **?**: пара не войдёт
в подсчёт, это лучше догадки. Опирайтесь только на профиль компании ниже и текст новости, не на знания о реальной компании.

Идите по шагам сверху вниз: первый подошедший шаг даёт оценку.

1. **Не экономика** (политика, происшествия, спорт, светская хроника) или событие за рубежом без связи с Россией и с
   профилем → **0**.
2. **Событие прямо в отрасли компании**: совпадает её отрасль (строка «Отрасль»), основной рынок, ключевое сырьё или
   главная статья затрат → **2**. Примеры: цены на металлопрокат — металлотрейдеру; топливо — перевозчику; продажи
   новостроек — застройщику; продажи аптек — аптечной сети.
3. **Решение по цене продукта, который у компании есть:**
   - ключевая ставка, условия кредитования — при `loan` или `leasing` в «Продукты банка»: решение ЦБ или прямой прогноз
     ставки → **2**, обзор и комментарии → **1**;
   - курс рубля — при «Внешняя торговля: да»: фактическое движение или решение → **2**, комментарии → **1**.
4. **Общий макрофон через признак профиля** → **1**:
   - инфляция, ВВП, доходности ОФЗ, бюджет — при `loan` или `leasing`;
   - курс, мировые рынки, золото, санкции на торговлю — при «Внешняя торговля: да»;
   - зарплаты, МРОТ, дефицит кадров — при «Численность: Более 251»;
   - спрос населения, розничные продажи — если компания сама продаёт населению (розница, услуги населению).
5. **Тот же макрофон без подходящего признака** (нет продуктов банка, нет ВЭД, мало работников) → **0**. Широкая новость
   сама по себе пару релевантной не делает.
6. **Событие в соседней отрасли** → **1**, только если механизм виден без догадок: стройка покупает металл и бетон;
   розница покупает у оптовика тех же товаров; топливо — затраты перевозчика. Связь через одного контрагента или через
   отрасль покупателей, которой нет в профиле → **0**.
7. Всё остальное → **0**.

Уточнения:
- Слух, мнение, интервью без новых фактов — на ступень ниже, чем был бы факт (2 → 1, 1 → 0).
- Новость о другой компании: **1**, если это сигнал для всей отрасли компании (банкротство крупного игрока, новые правила
  для отрасли); **0**, если частное событие (смена директора, сделка).
- Налоги и регулирование: **2**, если правило прямо для отрасли компании; **1**, если для всех крупных компаний (налог на
  прибыль, отчётность); **0**, если для физлиц или для другой отрасли.
- «Продукты банка» — только факт наличия продукта; сумма, ставка и условия неизвестны.
- «Отрасли контрагентов» — платежи за неизвестный период по частичным данным; опирайтесь на них только в очевидных цепочках.
- Не знаете отрасль по названию ОКВЭД — смотрите на ключевые слова названия («торговля оптовая …», «производство …»,
  «аренда …»): этого достаточно для шагов 2 и 6.
"""


def _profile_lines(p) -> list[str]:
    bp = [k for k in ("loan", "leasing", "deposit", "guarantee", "factoring") if getattr(p.bank_products, k)]
    lines = [f"- Отрасль: {p.industry.okved_code} {p.industry.okved_name or ''}".rstrip(),
             f"- Продукты банка (кроме РКО и зарплатного проекта): {', '.join(bp) or 'нет'}",
             f"- Внешняя торговля: {'да, ' + ', '.join(p.geo.fts_countries[:5]) if p.geo.foreign_trade else 'нет'}",
             f"- Численность: {p.size.workers_band or 'нет данных'}"]
    if p.network.counterparty_sectors:
        lines.append(f"- Отрасли контрагентов: {', '.join(p.network.counterparty_sectors[:4])}")
    return lines


def pick(pairs: list[dict], seed: int = SEED, quotas: dict[str, int] = QUOTAS) -> list[dict]:
    """pairs: {company, item, layer-признаки...}. Слои не пересекаются: пара идёт в первый слой, куда подходит."""
    rnd = random.Random(seed)
    chosen: list[dict] = []
    used: set[tuple[str, str]] = set()
    for layer, n in quotas.items():
        pool = sorted((p for p in pairs if (p["company"], p["item"]) not in used and (layer == "random" or p["layer"] == layer)),
                      key=lambda p: (p["company"], p["item"]))
        rnd.shuffle(pool)
        # Не больше 3 пар на материал: иначе слой займут 2-3 материала.
        per_item: dict[str, int] = {}
        for p in pool:
            if len([c for c in chosen if c["sample"] == layer]) >= n:
                break
            if per_item.get(p["item"], 0) >= 3:
                continue
            per_item[p["item"]] = per_item.get(p["item"], 0) + 1
            chosen.append({**p, "sample": layer})
            used.add((p["company"], p["item"]))
    rnd.shuffle(chosen)
    return chosen


def make() -> None:
    from ml.eval.run_eval import check_real_as_of, load_real_items
    from ml.profile.build_profile import get_profile
    from ml.ranking import ranker

    check_real_as_of()
    # Слои kept_neg / dropped_pos по ленте до шага 2 (без модели и правил по масштабу), как при сборе выборки 2026-10-22.
    ranker.set_model(None)
    from ml import topics

    ranker.ECONOMY_MIN_IMPORTANCE, ranker.NARROW_NEEDS_SECTOR, ranker.IN_SECTOR_KEEPS, topics.TOPIC_OKVED = None, False, False, "all"
    items = {i.item_id: i for i in load_real_items(cache_only=True, split="dev")}
    rows = [json.loads(l) for l in labels_path().read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["source"] == "llm_agreed" and r["item"] in items]
    companies = {c["id"]: c for c in load_v2()}
    kept: dict[str, set[str]] = {}
    for cid in companies:
        scored, _ = ranker.score_all(get_profile(cid), list(items.values()))
        kept[cid] = {s.item.item_id for s in scored if s.personal >= ranker.MIN_SCORE}
    pairs = []
    for r in rows:
        it, k = items[r["item"]], r["item"] in kept[r["company"]]
        layer = ("macro" if it.scope == "economy" else "kept_neg" if k and r["grade"] == 0
                 else "dropped_pos" if not k and r["grade"] >= 1 else "other")
        pairs.append({"company": r["company"], "item": r["item"], "layer": layer, "llm": r["grade"], "kept": k,
                      "reason_a": r["reason_a"], "reason_b": r["reason_b"]})
    chosen = pick(pairs)
    L = ["# Ручная проверка разметки реальных новостей", "",
         "Для каждой пары поставьте оценку релевантности новости для руководителя компании в строке «Оценка:» (одна цифра 0/1/2).",
         "Шкала как у моделей (`ml/prompts/label_relevance.md`): 2 напрямую меняет условия бизнеса (отрасль, рынок, ключевое сырьё, "
         "цена продукта банка, который у компании есть); 1 косвенно, общий макрофон через признак профиля (кредит, ВЭД, трудоёмкость) "
         "или смежная отрасль с понятным механизмом; 0 не относится или связь натянута.",
         "Опирайтесь только на профиль ниже, не на знания о реальной компании. Оценки моделей скрыты.",
         f"Пар: {len(chosen)}. После заполнения: `python -m ml.eval.label_review_real score`.", ""]
    for n, p in enumerate(chosen, 1):
        it, prof = items[p["item"]], get_profile(p["company"])
        text = " ".join((it.summary or "").split())[:600]
        L += [f"## {n}. `{p['company']}` × `{p['item']}`", "", "**Компания**", *_profile_lines(prof), "",
              f"**Новость** ({it.source.publisher or ''}, {it.source.published_at}): {it.title}", "", f"> {text}", "", "Оценка: ", ""]
    REVIEW.write_text("\n".join(L), encoding="utf-8")
    KEY.write_text(json.dumps({"seed": SEED, "pairs": chosen}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{REVIEW.name}: {len(chosen)} пар; слои {{{', '.join(f'{k}: {sum(c['sample'] == k for c in chosen)}' for k in QUOTAS)}}}",
          file=sys.stderr)


def batch_paths(batch: int = 1, name: str = "real"):
    """real: партия 1 — review_real.md (слои), партии 2+ — review_real_<n>.md (случайные пары). Другой набор (real2):
    review_<name>_<n>.md, всегда случайные пары."""
    if name == "real" and batch == 1:
        return REVIEW, KEY
    return EVAL_DIR / f"review_{name}_{batch}.md", EVAL_DIR / f"review_{name}_{batch}_key.json"


def pick_random(pairs: list[dict], n: int, seed: int, per_item: int = BATCH_PER_ITEM) -> list[dict]:
    rnd = random.Random(seed)
    pool = sorted(pairs, key=lambda p: (p["company"], p["item"]))
    rnd.shuffle(pool)
    out, used = [], {}
    for p in pool:
        if len(out) >= n:
            break
        if used.get(p["item"], 0) >= per_item:
            continue
        used[p["item"]] = used.get(p["item"], 0) + 1
        out.append({**p, "sample": "random"})
    return out


def make_random(batch: int, n: int = BATCH_N, name: str = "real") -> None:
    from ml.eval.run_eval import check_real_as_of, load_real_items
    from ml.profile.build_profile import get_profile

    check_real_as_of(name)
    items = {i.item_id: i for i in load_real_items(cache_only=True, name=name)}
    rows = [json.loads(l) for l in labels_path(name).read_text(encoding="utf-8").splitlines() if l.strip()]
    human = {(r["company"], r["item"]) for r in rows if r["source"] == "human"}
    pairs, seen = [], set()
    for r in rows:
        k = (r["company"], r["item"])
        if r["source"] == "human" or k in human or k in seen or r["item"] not in items:
            continue
        seen.add(k)
        agreed = r["source"] == "llm_agreed"
        pairs.append({"company": r["company"], "item": r["item"], "llm": r["grade"] if agreed else None,
                      "grade_a": r["grade_a"], "grade_b": r["grade_b"], "disputed": not agreed, "kept": False,
                      "reason_a": r["reason_a"], "reason_b": r["reason_b"]})
    chosen = pick_random(pairs, n, SEED + batch)
    review, key = batch_paths(batch, name)
    L = [f"# Ручная разметка реальных новостей ({name}), партия {batch}", "",
         f"Пар: {len(chosen)}, выбраны случайно из всех пар компания × новость (кроме уже размеченных). Оценки моделей скрыты.",
         f"После заполнения: `python -m ml.eval.label_review_real score --set {name} --batch {batch}`.", "", CRITERIA]
    for num, p in enumerate(chosen, 1):
        it, prof = items[p["item"]], get_profile(p["company"])
        text = " ".join((it.summary or "").split())[:600]
        L += [f"## {num}. `{p['company']}` × `{p['item']}`", "", "**Компания**", *_profile_lines(prof), "",
              f"**Новость** ({it.source.publisher or ''}, {it.source.published_at}): {it.title}", "", f"> {text}", "", "Оценка: ", ""]
    review.write_text("\n".join(L), encoding="utf-8")
    key.write_text(json.dumps({"seed": SEED + batch, "pairs": chosen}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{review.name}: {len(chosen)} пар, новостей {len({c['item'] for c in chosen})}, компаний {len({c['company'] for c in chosen})}, "
          f"спорных у моделей {sum(c['disputed'] for c in chosen)}", file=sys.stderr)


def read_answers(text: str) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    heads = list(_PAIR.finditer(text))
    for k, h in enumerate(heads):
        block = text[h.end(): heads[k + 1].start() if k + 1 < len(heads) else len(text)]
        m = _ANSWER.search(block)
        if m and m.group(1) not in (None, "?"):
            out[(h.group(2), h.group(3))] = int(m.group(1))
    return out


def score(apply: bool = False, batch: int = 1, name: str = "real") -> dict:
    review, key_path = batch_paths(batch, name)
    key = json.loads(key_path.read_text(encoding="utf-8"))["pairs"]
    human = read_answers(review.read_text(encoding="utf-8"))
    done = [p for p in key if (p["company"], p["item"]) in human]
    if not done:
        sys.exit(f"в {review.name} нет заполненных оценок")
    res: dict = {"filled": len(done), "of": len(key), "by_layer": {}}
    disputed = [p for p in done if p.get("disputed")]
    if disputed:
        res["disputed_resolved"] = {"n": len(disputed),
                                    "human_sided_with_a": sum(human[(p["company"], p["item"])] == p["grade_a"] for p in disputed),
                                    "human_sided_with_b": sum(human[(p["company"], p["item"])] == p["grade_b"] for p in disputed)}
    for layer in [None, *(QUOTAS if name == "real" else ())]:
        sub = [p for p in done if (layer is None or p["sample"] == layer) and p["llm"] is not None]
        if not sub:
            continue
        h, m = [human[(p["company"], p["item"])] for p in sub], [p["llm"] for p in sub]
        binary = sum((x >= 1) == (y >= 1) for x, y in zip(h, m)) / len(sub)
        llm_pos = [x for x, y in zip(h, m) if y >= 1]
        stats = {**agreement(h, m), "binary": binary,
                 "llm_pos_confirmed": sum(x >= 1 for x in llm_pos) / len(llm_pos) if llm_pos else None,
                 "kept_by_feed_relevant": (sum(human[(p["company"], p["item"])] >= 1 for p in sub if p["kept"]) / max(1, sum(p["kept"] for p in sub)))}
        res["by_layer"][layer or "все"] = stats
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if apply:
        path = labels_path(name)
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        have = {(r["company"], r["item"]) for r in rows if r["source"] == "human"}
        by = {(r["company"], r["item"]): r for r in rows}
        add = [{**{k: v for k, v in by[(p["company"], p["item"])].items() if k in ("company", "key", "split", "item")},
                "source": "human", "grade": human[(p["company"], p["item"])], "grade_llm": p["llm"]}
               for p in done if (p["company"], p["item"]) not in have]
        # load_gold_real берёт последнюю строку пары: ручная оценка заменяет машинную.
        with path.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in add)
        print(f"дописано ручных оценок: {len(add)} -> {path.name}", file=sys.stderr)
    return res


def human_pairs(name: str = "real") -> dict[tuple[str, str], int]:
    """Пары с ручной оценкой (source="human") из labels_<name>.jsonl; при повторе берётся последняя."""
    path = labels_path(name)
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(r["company"], r["item"]): r["grade"] for r in rows if r["source"] == "human"}


def tiers_vs_human(sets: dict | None = None, main: float | None = None, maybe: float | None = None, macro_tier: str | None = None,
                   thresholds: tuple[float, ...] = ()) -> dict:
    """Уровни ленты против ручной разметки real: точность основной ленты, полнота обоих уровней,
    доля нужного в первой пятёрке. Модель: внефолдовые предсказания по дням (как в CV), настройки из pair_model_cv.json.
    thresholds: ещё те же метрики для других порогов основной ленты (этап 4)."""
    from ml.ranking import pair_model as pm
    from ml.ranking import ranker

    sets = sets or pm.load_sets()
    cv = json.loads(ranker.MODEL_PATH.with_name("pair_model_cv.json").read_text(encoding="utf-8"))["chosen"]
    train = tuple(cv["train"].split("+"))
    scored, _ = pm.out_of_fold_scores(sets["real"], list(sets["v2"]) if "v2" in train else [], cv["l2"], cv["beta"], cv["w_model"],
                                      cv["order_mix"])
    human = human_pairs()
    sub = [r for r in sets["real"] if (r.company, r.item) in human]
    kw = {"maybe": maybe if maybe is not None else ranker.MIN_SCORE, "macro_tier": macro_tier or ranker.MACRO_FLOOR_TIER}
    return {"pairs": len(sub), "relevant": sum(r.grade >= 1 for r in sub),
            "main": {str(t): pm.tier_metrics(sub, scored, t, **kw) for t in (main or ranker.MAIN_SCORE, *thresholds)}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("make", "score", "tiers"))
    ap.add_argument("--apply", action="store_true", help="score: дописать ручные оценки в labels_real.jsonl")
    ap.add_argument("--batch", type=int, default=1, help="real: 1 слои (review_real.md), 2+ случайная партия review_real_<n>.md")
    ap.add_argument("--set", dest="name", default="real", choices=("real", "real2"),
                    help="real2: итоговый набор, случайные пары review_real2_<n>.md; размечать до просмотра результатов ленты")
    ap.add_argument("--n", type=int, help=f"размер случайной партии (real {BATCH_N}, real2 {REAL2_N})")
    args = ap.parse_args(argv)
    if args.name != "real" and args.cmd == "tiers":
        ap.error("tiers считается на real (внефолдовая модель); для real2 см. ml.eval.final_eval")
    if args.cmd == "tiers":
        print(json.dumps(tiers_vs_human(), ensure_ascii=False, indent=1))
    elif args.cmd == "make":
        if args.name == "real" and args.batch == 1:
            make()
        else:
            make_random(args.batch, args.n or (REAL2_N if args.name == "real2" else BATCH_N), args.name)
    else:
        score(args.apply, args.batch, args.name)


if __name__ == "__main__":
    main()
