"""Смежность отраслей из платежей (IMPROVEMENTS.md, вариант B1).

Вместо ручной карты `ml.topics.ADJACENCY`: из `df_Uid_payment_hack.csv` берутся
платежи, у которых известны ОКВЭД плательщика и получателя, и агрегируются по
парам префиксов ОКВЭД (2, 3, 4 цифры). Событие в отрасли S касается компании
отрасли X, если в данных X заметно больше среднего платит в S (S поставщик X)
или получает от S (S клиент X).

Пороги не подбирались по метрикам eval: превышение средней доли (lift) не меньше
MIN_LIFT, доля не меньше MIN_SHARE, в самой паре не меньше MIN_PAIR_ROWS платежей,
у стороны компании не меньше MIN_ROWS. Главный критерий lift, а не доля: малые
отрасли (металлургия ~0,1% всех платежей) иначе не связываются ни с кем. Доля
сглаживается к средней по всем отраслям с весом PRIOR_ROWS платежей, чтобы малые
выборки не давали случайных связей.

`python -m ml.profile.flows` пересчитывает `var/flows.parquet`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from ml.config import settings

LEVELS = (2, 3, 4)
MIN_ROWS = 20
PRIOR_ROWS = 50
MIN_PAIR_ROWS = 10
MIN_SHARE = 0.01
MIN_LIFT = 3.0
# Платежи выше этого перцентиля считаются выбросами (максимум в данных 8e10 руб.).
OUTLIER_Q = 0.999


def _digits(code: str | None, n: int) -> str | None:
    if not code:
        return None
    d = code.replace(".", "")
    return d[:n] if len(d) >= n else None


@dataclass(frozen=True)
class Link:
    weight: float  # сглаженная доля, 0..1
    direction: str  # "supplier": отрасль события поставляет компании; "customer": покупает у неё
    share: float
    lift: float
    rows: int
    company_prefix: str
    event_prefix: str
    mechanism: str
    source: str = "payments"


def build_flows() -> pd.DataFrame:
    """Суммы платежей между префиксами ОКВЭД: одна строка на (уровень плательщика, уровень получателя, пара)."""
    from ml.profile.build_profile import company_index, load_payments

    pay = load_payments()
    pay = pay[pay["PAY_AMT"] <= pay["PAY_AMT"].quantile(OUTLIER_Q)]
    okved = company_index()["OKVED_NORM"].dropna()
    p = pay.assign(pk=pay["PAYER"].map(okved), bk=pay["BENEFICIARY"].map(okved)).dropna(subset=["pk", "bk"])
    frames = []
    for la in LEVELS:
        for lb in LEVELS:
            g = p.assign(a=p["pk"].map(lambda c: _digits(c, la)), b=p["bk"].map(lambda c: _digits(c, lb)))
            g = g.dropna(subset=["a", "b"]).groupby(["a", "b"]).agg(amt=("PAY_AMT", "sum"), n=("PAY_AMT", "size")).reset_index()
            frames.append(g.assign(la=la, lb=lb))
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=1)
def load_flows() -> pd.DataFrame:
    path = settings.paths.var / "flows.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = build_flows()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


@lru_cache(maxsize=1)
def _tables() -> dict:
    """Индексы для быстрых запросов: пары, итоги по плательщику/получателю, общий итог по уровням."""
    df = load_flows()
    pair = {(r.la, r.lb, r.a, r.b): (r.amt, r.n) for r in df.itertuples(index=False)}
    out_tot = df.groupby(["la", "lb", "a"]).agg(amt=("amt", "sum"), n=("n", "sum"))
    in_tot = df.groupby(["la", "lb", "b"]).agg(amt=("amt", "sum"), n=("n", "sum"))
    grand = df.groupby(["la", "lb"])["amt"].sum()
    return {"pair": pair, "out": out_tot.to_dict("index"), "in": in_tot.to_dict("index"), "grand": grand.to_dict()}


@lru_cache(maxsize=4096)
def okved_title(prefix: str) -> str:
    """Название отрасли по префиксу: самый частый MAINOKVED_NAME среди кодов, совпадающих с префиксом."""
    from ml.profile.build_profile import company_index

    idx = company_index()
    codes = idx["OKVED_NORM"].fillna("").str.replace(".", "", regex=False)
    names = idx.loc[codes == prefix, "MAINOKVED_NAME"].dropna()
    if names.empty:
        names = idx.loc[codes.str.startswith(prefix), "MAINOKVED_NAME"].dropna()
        if names.empty:
            return f"ОКВЭД {prefix}"
        return f"ОКВЭД {prefix} (например, {names.mode().iloc[0].lower()})"
    return names.mode().iloc[0]


def _one_way(company: str, event: str, lc: int, le: int, company_pays: bool) -> Link | None:
    t = _tables()
    if company_pays:  # компания платит отрасли события: та её поставщик
        key, tot, glob_key = (lc, le, company, event), t["out"].get((lc, le, company)), t["in"].get((lc, le, event))
    else:  # отрасль события платит компании: та её клиент
        key, tot, glob_key = (le, lc, event, company), t["in"].get((le, lc, company)), t["out"].get((le, lc, event))
    if not tot or tot["n"] < MIN_ROWS or not glob_key:
        return None
    grand = t["grand"][(lc, le) if company_pays else (le, lc)]
    amt, pair_rows = t["pair"].get(key, (0.0, 0))
    if pair_rows < MIN_PAIR_ROWS:
        return None
    base = glob_key["amt"] / grand
    raw = amt / tot["amt"] if tot["amt"] else 0.0
    share = (tot["n"] * raw + PRIOR_ROWS * base) / (tot["n"] + PRIOR_ROWS)
    lift = share / base if base else 0.0
    if share < MIN_SHARE or lift < MIN_LIFT:
        return None
    # Цифры доли в текст не идут: модель перенесла бы их в карточку, а проверить их по источникам нельзя.
    if company_pays:
        why = f"по платежам клиентов банка отрасль компании заметно чаще среднего платит отрасли «{okved_title(event)}» (поставщик)"
    else:
        why = f"по платежам клиентов банка отрасль компании заметно чаще среднего получает деньги от отрасли «{okved_title(event)}» (покупатель)"
    return Link(round(share, 4), "supplier" if company_pays else "customer", round(raw, 4), round(lift, 2),
                int(tot["n"]), company, event, why)


@lru_cache(maxsize=8192)
def adjacency(company_okved: str | None, event_okved: str) -> Link | None:
    """Самая сильная связь «отрасль компании <-> отрасль события» по платежам.

    Сторона компании от детального уровня к общему (46.72 -> 46.7), но не до раздела
    из двух цифр: раздел слишком неоднороден (46 это весь опт). Берётся первый
    уровень, где связь видна; на детальном уровне пары часто слишком редки.
    Сторона события на уровне, с которым пришла подсказка (не детальнее 4 цифр).
    """
    le = min(len(event_okved.replace(".", "")), max(LEVELS))
    if le < min(LEVELS):
        return None
    e = _digits(event_okved, le)
    for lc in sorted(LEVELS, reverse=True):
        c = _digits(company_okved, lc)
        if c is None or (lc == min(LEVELS) and _digits(company_okved, lc + 1)):
            continue
        found = [l for l in (_one_way(c, e, lc, le, True), _one_way(c, e, lc, le, False)) if l]
        if found:
            return max(found, key=lambda l: l.weight)
    return None


def covered(company_okved: str | None, event_okved: str) -> bool:
    """Хватает ли платежей, чтобы судить о связи: у отрасли компании и у отрасли события есть данные.

    Если хватает, а связи нет, ручное правило не применяется: данные говорят, что связи нет.
    """
    t = _tables()
    le = min(len(event_okved.replace(".", "")), max(LEVELS))
    e = _digits(event_okved, le)
    if e is None or le < min(LEVELS):
        return False
    event_rows = max((t["out"].get((le, lc, e)) or {}).get("n", 0) + (t["in"].get((lc, le, e)) or {}).get("n", 0) for lc in LEVELS)
    company_rows = max(
        (t["out"].get((lc, le, _digits(company_okved, lc))) or {}).get("n", 0)
        + (t["in"].get((le, lc, _digits(company_okved, lc))) or {}).get("n", 0)
        for lc in LEVELS if _digits(company_okved, lc)
    ) if company_okved else 0
    return event_rows >= MIN_ROWS and company_rows >= MIN_ROWS


def main() -> None:
    load_flows.cache_clear()
    path = settings.paths.var / "flows.parquet"
    path.unlink(missing_ok=True)
    df = load_flows()
    print(f"flows rows={len(df)} -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
