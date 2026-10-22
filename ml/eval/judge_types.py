"""Типы замечаний судьи и куда они попадают в карточке (грубая классификация правилами по тексту замечания).

    python -m ml.eval.judge_types var/eval/<run_a> [var/eval/<run_b>]   # на общих парах dev
"""
import json, re, sys, collections
from ml.retrieval.index import tokenize

RULES = [
    ("судья сам подтверждает", re.compile(r"(?<!не )подтвержда(ет|ется)|есть в профиле|что подтверждается", re.I)),
    ("приписано не тому источнику", re.compile(r"приписан|приписыва\w* банк|только в банковском|есть только в|подан\w* как (факт|прогноз|ожидани|рыночн)|как (позици|прогноз|ожидани)\w* банк|оценка аналитики банка|в событии (этого )?нет", re.I)),
    ("цифра", re.compile(r"\d")),
    ("додумано о компании", re.compile(r"в профиле|профил|интерпретац|роль|основн\w* партн|ассортимент|структур\w* (спроса|затрат)|крупный", re.I)),
    ("сильнее источника", re.compile(r"сильнее|увереннее|категорич|а не о том|не означает|преувелич", re.I)),
]

def kind(u):
    for name, rx in RULES:
        if rx.search(u):
            return name
    return "механизм без опоры"

def segs(card):
    out = [("inference", card["ai_inference"]["text"]), ("sourced", card["why_it_matters"]["text"]), ("sourced", card["what_happened"]["text"])]
    if card.get("bank_view"): out.append(("sourced", card["bank_view"]["text"]))
    for grp in (card["consequences"]["risks"], card["consequences"]["opportunities"], card["suggested_actions"]):
        for c in grp:
            out.append(("inference" if c.get("is_model_inference") or not c.get("evidence") else "sourced", c["text"]))
    return [(k, set(tokenize(t)), t) for k, t in out]

def where(u, ss):
    q = set(tokenize(u))
    best = max(ss, key=lambda s: len(q & s[1]) / (len(q | s[1]) or 1), default=None)
    if not best or not q: return "не найдено"
    j = len(q & best[1]) / len(q | best[1])
    return best[0] if j >= 0.15 else "не найдено"

def analyze(run, split="dev", pairs=None):
    rs = [json.loads(l) for l in open(f"{run}/results.jsonl")]
    rs = [r for r in rs if r.get("split") == split and r.get("status") == "ok" and r.get("judge") and "unsupported" in r["judge"]]
    if pairs is not None: rs = [r for r in rs if (r["company"], r["item"]) in pairs]
    kinds, places, both = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rs:
        cid = f"ins_{r['company']}_{r['item']}"
        try: card = json.load(open(f"{run}/cards/{cid}.json"))
        except FileNotFoundError: continue
        ss = segs(card)
        for u in r["judge"]["unsupported"]:
            k, w = kind(u), where(u, ss); kinds[k] += 1; places[w] += 1; both[(k, w)] += 1
    n = sum(kinds.values())
    return len(rs), n, kinds, places, both

if __name__ == "__main__":
    runs = sys.argv[1:]
    sets = [{(json.loads(l)["company"], json.loads(l)["item"]) for l in open(f"{r}/results.jsonl") if json.loads(l).get("split") == "dev" and json.loads(l).get("status") == "ok"} for r in runs]
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    for r in runs:
        cards, n, kinds, places, both = analyze(r, pairs=common)
        print(f"\n== {r}: карточек {cards}, замечаний {n} ({n/cards:.2f} на карточку)")
        for k, v in kinds.most_common(): print(f"   {k:30s} {v:4d}  {v/n:5.0%}")
        print("   где в карточке:", {k: f"{v/n:.0%}" for k, v in places.most_common()})
        inf = sum(v for (k, w), v in both.items() if w == "inference")
        print(f"   замечаний к утверждениям, уже помеченным как вывод модели: {inf} ({inf/n:.0%})")
