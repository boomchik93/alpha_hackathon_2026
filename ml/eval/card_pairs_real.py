"""Фиксированные пары «компания × реальная новость» для оценки карточек (STATUS_HANDOFF.md, шаг 4).

    ML_AS_OF=2026-10-22 python -m ml.eval.card_pairs_real      # -> ml/eval/card_pairs_real.json (перезапись только с --force)

Только новости dev и только пары, которые рабочая лента показывает компании (карточка в бою пишется по ленте):
N_REL пар с разметкой >= 1 и N_IRR ложных срабатываний ленты (разметка 0), чтобы видеть поведение карточки на
нерелевантной паре. Не больше MAX_PER_ITEM пар на материал и MAX_PER_COMPANY на компанию, seed фиксирован.
Пары выбираются до генерации карточек и после этого не меняются: прогоны сравниваются парно.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from ml.eval.label_llm import EVAL_DIR, load_v2

PATH = EVAL_DIR / "card_pairs_real.json"
SEED = 20261022
N_REL, N_IRR = 90, 30
MAX_PER_ITEM, MAX_PER_COMPANY = 8, 5


def pick(cands: list[dict], n: int, rnd: random.Random, used_item: dict, used_company: dict) -> list[dict]:
    out = []
    for c in cands:
        if len(out) >= n:
            break
        if used_item.get(c["item"], 0) >= MAX_PER_ITEM or used_company.get(c["company"], 0) >= MAX_PER_COMPANY:
            continue
        used_item[c["item"]] = used_item.get(c["item"], 0) + 1
        used_company[c["company"]] = used_company.get(c["company"], 0) + 1
        out.append(c)
    return out


def make() -> list[dict]:
    from ml.eval.run_eval import check_real_as_of, load_gold_real, load_real_items
    from ml.profile.build_profile import get_profile
    from ml.ranking import ranker

    check_real_as_of()
    items = load_real_items(cache_only=True, split="dev")
    gold = load_gold_real()
    cands = []
    for c in load_v2():
        kept, _ = ranker.rank(get_profile(c["id"]), items, limit=100)
        for s in kept:
            g = gold(c, s.item.item_id)
            if g is not None:
                cands.append({"company": c["id"], "key": c["key"], "split": "dev", "item": s.item.item_id, "gold": g})
    rnd = random.Random(SEED)
    cands.sort(key=lambda x: (x["company"], x["item"]))
    rnd.shuffle(cands)
    used_item: dict[str, int] = {}
    used_company: dict[str, int] = {}
    rel = pick([x for x in cands if x["gold"] >= 1], N_REL, rnd, used_item, used_company)
    irr = pick([x for x in cands if x["gold"] == 0], N_IRR, rnd, used_item, used_company)
    return sorted(rel + irr, key=lambda x: (x["company"], x["item"]))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if PATH.exists() and not args.force:
        sys.exit(f"{PATH.name} уже есть: пары фиксированы. Перевыбор только с --force (прежние прогоны станут несравнимы).")
    pairs = make()
    PATH.write_text(json.dumps(pairs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{PATH.name}: {len(pairs)} пар (релевантных {sum(p['gold'] >= 1 for p in pairs)}, "
          f"ложных срабатываний ленты {sum(p['gold'] == 0 for p in pairs)}), материалов {len({p['item'] for p in pairs})}, "
          f"компаний {len({p['company'] for p in pairs})}", file=sys.stderr)


if __name__ == "__main__":
    main()
