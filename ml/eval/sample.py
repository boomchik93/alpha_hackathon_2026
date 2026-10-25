"""Расширенный eval-набор компаний: стратифицированная выборка и dev/test по отраслям.

    python -m ml.eval.sample            # пишет ml/eval/companies_v2.json
    python -m ml.eval.sample --pairs    # после ml.eval.label_llm: ml/eval/card_pairs_v2.json

Страта: раздел ОКВЭД (2 цифры), для опта (46) группа из 3 цифр: опт слишком разный.
dev: страты исходных 9 компаний (на них подбирались параметры и промпты) плюс
случайная половина остальных страт; test: другая половина. Компании одной страты
всегда в одной части, поэтому test проверяет перенос на новые отрасли.
Внутри части компании и группы берутся по кругу по стратам, чтобы покрыть как
можно больше отраслей. Seed фиксирован.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from ml.eval.run_eval import load_companies
from ml.profile.build_profile import list_profiles
from ml.schemas import CompanyProfile

EVAL_DIR = Path(__file__).parent
OUT = EVAL_DIR / "companies_v2.json"
SEED = 20261021
PER_SPLIT = {"company": 16, "group": 4}
PAIRS_OUT = EVAL_DIR / "card_pairs_v2.json"
# Фиксированный набор пар для карточек: ошибки почти все в релевантных парах, нерелевантные нужны для калибровки relevance.
PAIRS_RELEVANT = 240
PAIRS_IRRELEVANT = 60


def stratum(p: CompanyProfile) -> str:
    code = (p.industry.okved_code or "").replace(".", "")
    if not code:
        return "NA"
    return code[:3] if code.startswith("46") else code[:2]


def _round_robin(pool: list[CompanyProfile], n: int, rnd: random.Random) -> list[CompanyProfile]:
    by: dict[str, list[CompanyProfile]] = {}
    for p in pool:
        by.setdefault(stratum(p), []).append(p)
    for v in by.values():
        rnd.shuffle(v)
    keys = sorted(by)
    rnd.shuffle(keys)
    out: list[CompanyProfile] = []
    while len(out) < n and any(by.values()):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out


def build() -> list[dict]:
    rnd = random.Random(SEED)
    profiles = list_profiles()
    original = load_companies()
    orig_ids = {c["id"] for c in original}
    by_id = {p.company_id: p for p in profiles}
    dev_strata = {stratum(by_id[c["id"]]) for c in original if c["id"] in by_id}
    rest = sorted({stratum(p) for p in profiles} - dev_strata)
    rnd.shuffle(rest)
    dev_strata |= set(rest[: len(rest) // 2])

    out = [{"id": c["id"], "key": c["key"], "split": "dev", "stratum": stratum(by_id[c["id"]]), "level": by_id[c["id"]].level,
            "okved": by_id[c["id"]].industry.okved_name, "labels": "human", "note": c["note"]} for c in original]
    for split in ("dev", "test"):
        for level, n in PER_SPLIT.items():
            have = sum(1 for x in out if x["split"] == split and x["level"] == level)
            pool = [p for p in profiles if p.level == level and p.company_id not in orig_ids
                    and (stratum(p) in dev_strata) == (split == "dev")]
            for p in _round_robin(pool, max(0, n - have), rnd):
                out.append({"id": p.company_id, "key": p.company_id, "split": split, "stratum": stratum(p), "level": level,
                            "okved": p.industry.okved_name, "labels": "llm", "note": ""})
    return out


def card_pairs() -> list[dict]:
    """Пары «компания x новость» с эталоном (human или llm_agreed), поровну dev и test, seed фиксирован."""
    rnd = random.Random(SEED)
    split = {c["id"]: c["split"] for c in json.loads(OUT.read_text(encoding="utf-8"))}
    labels = [json.loads(l) for l in (EVAL_DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    news = [r for r in labels if r["source"] in ("human", "llm_agreed") and not r["item"].startswith("bank_")]
    out = []
    for part in ("dev", "test"):
        for want, rel in ((PAIRS_RELEVANT // 2, True), (PAIRS_IRRELEVANT // 2, False)):
            pool = sorted((r for r in news if split[r["company"]] == part and (r["grade"] >= 1) == rel),
                          key=lambda r: (r["company"], r["item"]))
            rnd.shuffle(pool)
            out += [{"company": r["company"], "item": r["item"], "grade": r["grade"], "source": r["source"], "split": part}
                    for r in pool[:want]]
    return sorted(out, key=lambda r: (r["split"], r["company"], r["item"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="store_true")
    if ap.parse_args().pairs:
        pairs = card_pairs()
        PAIRS_OUT.write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
        from collections import Counter
        print(f"pairs={len(pairs)} {dict(Counter((p['split'], p['grade']) for p in pairs))} -> {PAIRS_OUT}", file=sys.stderr)
        return
    rows = build()
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for split in ("dev", "test"):
        part = [r for r in rows if r["split"] == split]
        print(f"{split}: {len(part)} ({sum(r['level'] == 'group' for r in part)} групп), страт {len({r['stratum'] for r in part})}",
              file=sys.stderr)
    print(f"-> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
