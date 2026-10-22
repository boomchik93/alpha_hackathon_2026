"""Ручная калибровка судей eval (IMPROVEMENTS.md, E4 п.3).

    python -m ml.eval.judge_review make var/eval/<run_id>    # -> ml/eval/review_judges.md + review_judges_key.json
    python -m ml.eval.judge_review score                     # после заполнения review_judges.md

Берутся только карточки dev-компаний (PROTOCOL.md: test не разбирается), поровну с
factual=2 и factual<2 у первого судьи, seed фиксирован. Оценки судей в файл для
человека не попадают, чтобы не влиять на его оценку; они лежат в review_judges_key.json.

Человек заполняет у каждой карточки строку «Оценка:» (factual 0/1/2, есть ли
неподтверждённое да/нет, выводы модели в bank_view да/нет). score считает согласие
каждого судьи с человеком: точность factual, κ_w, ложные срабатывания и пропуски.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
REVIEW = EVAL_DIR / "review_judges.md"
KEY = EVAL_DIR / "review_judges_key.json"
N_CARDS = 40
SEED = 20261021
_ANSWER = re.compile(r"^Оценка:\s*factual=(\S*)\s*;\s*неподтверждено=(\S*)\s*;\s*bank_view_выводы=(\S*)", re.M)
_CARD = re.compile(r"^## (\d+)\. `([^`]+)`", re.M)


def _pick(rows: list[dict]) -> list[dict]:
    rnd = random.Random(SEED)
    dev = [r for r in rows if r["status"] == "ok" and r.get("split") == "dev" and r["judge"] and "error" not in r["judge"]]
    good = sorted((r for r in dev if r["judge"]["factual"] == 2), key=lambda r: r["card_id"])
    bad = sorted((r for r in dev if r["judge"]["factual"] < 2), key=lambda r: r["card_id"])
    rnd.shuffle(good)
    rnd.shuffle(bad)
    picked = good[: N_CARDS // 2] + bad[: N_CARDS - N_CARDS // 2]
    rnd.shuffle(picked)
    return picked


def _card_block(n: int, r: dict, run: Path) -> list[str]:
    from ml.ingest.analytics import load_chunks
    from ml.ingest.store import get_item
    from ml.profile.build_profile import get_profile

    card = json.loads((run / "cards" / f"{r['card_id']}.json").read_text(encoding="utf-8"))
    prof = get_profile(r["company"])
    item = get_item(r["item"])
    chunks = {c.chunk_id: c for c in load_chunks()}
    used = {s["id"] for s in card["sources"]}
    bp = [k for k, v in prof.bank_products.model_dump().items() if v]
    L = [f"## {n}. `{r['card_id']}`", "",
         f"**Компания:** {prof.industry.okved_name}; продукты банка: {', '.join(bp) or 'нет'}; "
         f"ВЭД: {', '.join(prof.geo.fts_countries) or ('да' if prof.geo.foreign_trade else 'нет')}; "
         f"контрагенты: {', '.join(prof.network.counterparty_sectors[:3]) or 'нет данных'}.", "",
         f"**Событие:** {item.title}. " + " ".join(f"[{f.id}] {f.text}" for f in item.facts), ""]
    for cid in sorted(used & set(chunks)):
        c = chunks[cid]
        L += [f"**Фрагмент банка `{cid}`** ({c.title}, {c.published_at}): {c.text[:900]}", ""]
    L += ["**Карточка:**", "",
          f"- Релевантность {card['relevance']['score']}: {'; '.join(card['relevance']['reasons'])}",
          f"- Что произошло: {card['what_happened']['text']}",
          f"- Почему важно: {card['why_it_matters']['text']}"]
    L += [f"- Риск: {c['text']} (источники: {', '.join(c['evidence']) or 'вывод модели'})" for c in card["consequences"]["risks"]]
    L += [f"- Возможность: {c['text']} (источники: {', '.join(c['evidence']) or 'вывод модели'})" for c in card["consequences"]["opportunities"]]
    L += [f"- Позиция банка: {card['bank_view']['text'] if card['bank_view'] else 'нет'}",
          f"- Вывод модели: {card['ai_inference']['text']}"]
    L += [f"- Действие: {c['text']}" for c in card["suggested_actions"]]
    L += [f"- Ограничения: {'; '.join(card['limitations']) or 'нет'}", "",
          "Оценка: factual= ; неподтверждено= ; bank_view_выводы= ", "", "---", ""]
    return L


def make(run_dir: str) -> None:
    from ml.eval.run_eval import load_eval_items

    load_eval_items(cache_only=True)  # holdout-новости в память процесса, чтобы get_item их видел
    run = Path(run_dir)
    rows = [json.loads(l) for l in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = _pick(rows)
    L = ["# Ручная калибровка судей", "",
         f"Прогон `{run.name}`, {len(picked)} карточек dev-компаний. Оцените каждую карточку только по приведённым данным "
         "(профиль, событие, фрагменты банка) и заполните строку «Оценка:» после карточки:", "",
         "- `factual`: 2 = все конкретные утверждения и цифры подтверждаются данными или помечены как вывод модели и логично из них следуют; "
         "1 = мелкие неточности или натяжки; 0 = выдуманные факты, цифры или мнение банка;",
         "- `неподтверждено`: да/нет, есть ли утверждение, которого нет в данных (кроме явно помеченных выводов модели);",
         "- `bank_view_выводы`: да/нет, есть ли в «Позиции банка» то, чего нет во фрагментах банка.", "",
         "Пример: `Оценка: factual=1 ; неподтверждено=да ; bank_view_выводы=нет`. Можно заполнить не все карточки: "
         "считаются только заполненные.", "", "---", ""]
    key = []
    for n, r in enumerate(picked, 1):
        L += _card_block(n, r, run)
        key.append({"n": n, "card_id": r["card_id"], "judge": r["judge"], "judge2": r.get("judge2")})
    REVIEW.write_text("\n".join(L), encoding="utf-8")
    KEY.write_text(json.dumps({"run": run.name, "cards": key}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(picked)} карточек -> {REVIEW}; оценки судей -> {KEY}", file=sys.stderr)


def _yes(v: str) -> bool | None:
    v = v.strip().lower()
    return True if v in ("да", "yes", "1", "true") else False if v in ("нет", "no", "0", "false") else None


def score() -> None:
    from ml.eval.label_llm import cohen_kappa

    text = REVIEW.read_text(encoding="utf-8")
    key = {c["card_id"]: c for c in json.loads(KEY.read_text(encoding="utf-8"))["cards"]}
    heads = list(_CARD.finditer(text))
    human = {}
    for i, h in enumerate(heads):
        block = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        m = _ANSWER.search(block)
        if m and m.group(1).strip() in ("0", "1", "2"):
            human[h.group(2)] = {"factual": int(m.group(1)), "unsupported": _yes(m.group(2)), "bank_ai_mixed": _yes(m.group(3))}
    out = {"cards_scored": len(human)}
    for jn in ("judge", "judge2"):
        pairs = [(human[c], key[c][jn]) for c in human if key[c].get(jn) and "error" not in key[c][jn]]
        if not pairs:
            continue
        hf, jf = [h["factual"] for h, _ in pairs], [j["factual"] for _, j in pairs]
        hu = [(h["unsupported"], bool(j["unsupported"])) for h, j in pairs if h["unsupported"] is not None]
        out[jn] = {
            "n": len(pairs),
            "factual_exact": sum(a == b for a, b in zip(hf, jf)) / len(pairs),
            "factual_kappa_w": cohen_kappa(hf, jf),
            "human_factual_mean": sum(hf) / len(hf), "judge_factual_mean": sum(jf) / len(jf),
            # ложное срабатывание: судья нашёл неподтверждённое, человек нет; пропуск: наоборот.
            "unsupported_false_pos": sum(j and not h for h, j in hu) / max(1, sum(j for _, j in hu)),
            "unsupported_missed": sum(h and not j for h, j in hu) / max(1, sum(h for h, _ in hu)),
        }
    (EVAL_DIR / "review_judges_score.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "make":
        make(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "score":
        score()
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
