"""CSV -> CompanyProfile.

`python -m ml.profile.build_profile` строит профили целевой аудитории
(компании и группы с выручкой > 15 млрд руб.) в `var/profiles.jsonl`.
`build_profile(id)` строит профиль любой компании или группы на лету.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache

import pandas as pd

from ml.config import settings
from ml.errors import NotFound
from ml.schemas import (
    BankProducts, CompanyProfile, Counterparty, DataQuality, Geo, GroupInfo, Industry, MarketPosition, Network, Size,
)
from ml.topics import TOPICS, okved_matches

SEP = ";"
PRODUCTS = {
    "rko": "CL_PRODUCT_RKO_MARK",
    "guarantee": "CL_PRODUCT_GAR_MARK",
    "loan": "CL_PRODUCT_LOA_MARK",
    "leasing": "CL_PRODUCT_LEA_MARK",
    "deposit": "CL_PRODUCT_DEP_MARK",
    "payroll": "CL_PRODUCT_ZPP_MARK",
    "factoring": "FACTORING_MARK",
}
# Поля, пропуск которых важен для выводов. SPARK не берём: пусто в 99,8%.
TRACKED = ["MAINOKVED_CODE", "MAINOKVED_NAME", "ECO_MARK_LABEL", "AMOUNT_WORKERS_FNS_GRADATION", "CITY", "REGION"]
TOP_N_COUNTERPARTIES = 5


def normalize_okved(code: str | None) -> str | None:
    """`41.20` -> `41.2`, `01.70` -> `01.7`; `46.73.3` без изменений."""
    if not code or pd.isna(code):
        return None
    parts = str(code).strip().split(".")
    out = [parts[0].zfill(2)]
    for p in parts[1:]:
        out.append(p[:-1] if len(p) == 2 and p.endswith("0") else p)
    return ".".join(x for x in out if x)


def revenue_band(rub: float | None) -> str | None:
    if rub is None or pd.isna(rub):
        return None
    bn = rub / 1e9
    for hi, label in [(1, "< 1 млрд"), (5, "1-5 млрд"), (15, "5-15 млрд"), (30, "15-30 млрд"), (60, "30-60 млрд")]:
        if bn < hi:
            return label
    return "> 60 млрд"


def sector_tags(okved_code: str | None, okved_name: str | None) -> list[str]:
    tags = [t.title for t in TOPICS.values() if okved_matches(okved_code, t.okved_prefixes)]
    name = (okved_name or "").lower()
    for word, tag in [("оптов", "опт"), ("розничн", "розница"), ("строительств", "строительство"),
                      ("перевоз", "перевозки"), ("топлив", "топливо"), ("фармацев", "фармацевтика")]:
        if word in name:
            tags.append(tag)
    return sorted(set(tags))


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_companies() -> pd.DataFrame:
    df = pd.read_csv(settings.paths.structured / "df_Uid_data_hack.csv", sep=SEP, dtype=str)
    df["REVENUE_COMPANY"] = pd.to_numeric(df["REVENUE_COMPANY"], errors="coerce")
    df["OKVED_NORM"] = df["MAINOKVED_CODE"].map(normalize_okved)
    return df


@lru_cache(maxsize=1)
def duplicate_ids() -> frozenset[str]:
    c = load_companies()["SYNTHETIC_U"].value_counts()
    return frozenset(c[c > 1].index)


@lru_cache(maxsize=1)
def company_index() -> pd.DataFrame:
    """Одна строка на компанию: у дублей берём строку с наибольшей выручкой."""
    df = load_companies().sort_values("REVENUE_COMPANY", ascending=False)
    return df.drop_duplicates("SYNTHETIC_U").set_index("SYNTHETIC_U")


@lru_cache(maxsize=1)
def load_payments() -> pd.DataFrame:
    df = pd.read_csv(settings.paths.structured / "df_Uid_payment_hack.csv", sep=SEP, dtype={"BENEFICIARY": str, "PAYER": str})
    df["PAY_AMT"] = pd.to_numeric(df["PAY_AMT"], errors="coerce").fillna(0.0)
    df["PAY_CNT"] = pd.to_numeric(df["PAY_CNT"], errors="coerce").fillna(0).astype(int)
    return df[df["PAYER"] != df["BENEFICIARY"]]


@lru_cache(maxsize=1)
def group_stats() -> pd.DataFrame:
    idx = company_index()
    return idx.groupby("GROUP_ID").agg(
        group_size=("REVENUE_COMPANY", "size"),
        group_revenue=("REVENUE_COMPANY", "sum"),
        group_clients=("CL_TYPE", lambda s: int((s == "CLIENT").sum())),
    )


# ---------------------------------------------------------------------------
# Построение профиля
# ---------------------------------------------------------------------------


POSITION_DIGITS = (4, 3, 2)  # длина префикса ОКВЭД для сравнения, от узкой к широкой
POSITION_MIN_PEERS = 10
POSITION_NOTE = "по выручке среди компаний выборки банка; это не доля рынка РФ"
GROUP_POSITION_NOTE = ("выручка группы против компаний выборки банка с тем же префиксом ОКВЭД, что у крупнейшей компании группы; "
                       "члены группы в сравнении не учтены; это не доля рынка РФ")


@lru_cache(maxsize=1)
def _okved_groups() -> dict[str, list[tuple[str, float]]]:
    """Префикс ОКВЭД (цифры без точки, 2-4 знака) -> [(компания, выручка)] по одной строке на компанию. Считается один раз."""
    idx = company_index()
    out: dict[str, list[tuple[str, float]]] = {}
    for cid, code, rev in zip(idx.index, idx["OKVED_NORM"], idx["REVENUE_COMPANY"]):
        if pd.isna(code) or pd.isna(rev):
            continue
        digits = str(code).replace(".", "")
        for d in POSITION_DIGITS:
            if len(digits) >= d:
                out.setdefault(digits[:d], []).append((cid, float(rev)))
    return out


def market_position(okved_code: str | None, revenue: float | None, exclude: frozenset[str], note: str = POSITION_NOTE) -> MarketPosition | None:
    """Место по выручке среди компаний выборки банка с тем же префиксом ОКВЭД: 4 цифры, если сравнение не меньше 10 субъектов,
    иначе 3, иначе 2. `exclude`: сама компания или члены группы, их выручка в сравнение не входит."""
    if not okved_code or revenue is None or pd.isna(revenue):
        return None
    digits = okved_code.replace(".", "")
    chosen = None
    for d in POSITION_DIGITS:
        if len(digits) < d:
            continue
        others = [r for cid, r in _okved_groups().get(digits[:d], []) if cid not in exclude]
        chosen = (d, others)
        if len(others) + 1 >= POSITION_MIN_PEERS:
            break
    if chosen is None:
        return None
    d, others = chosen
    n = len(others) + 1
    total = revenue + sum(others)
    prefix = digits[:d]
    return MarketPosition(
        okved_prefix=prefix if d <= 2 else f"{prefix[:2]}.{prefix[2:]}", digits=d, peers=n,
        rank=1 + sum(r > revenue for r in others),
        percentile=sum(r < revenue for r in others) / (n - 1) if n > 1 else 1.0,
        revenue_share=revenue / total if total > 0 else None, note=note)


def _flag(v) -> bool | None:
    if v is None or pd.isna(v):
        return None
    return float(v) == 1.0


def _str(v) -> str | None:
    return None if v is None or pd.isna(v) else str(v)


def _countries(raw) -> list[str]:
    raw = _str(raw)
    return [c.strip().title() for c in raw.split(",") if c.strip()] if raw else []


def _network(ids: list[str], revenue: float | None) -> Network:
    pay = load_payments()
    idset = set(ids)
    out_ = pay[pay["PAYER"].isin(idset) & ~pay["BENEFICIARY"].isin(idset)]
    in_ = pay[pay["BENEFICIARY"].isin(idset) & ~pay["PAYER"].isin(idset)]
    if out_.empty and in_.empty:
        return Network(note="платежей в данных нет")

    idx = company_index()
    rows = []
    for frame, col, role in [(out_, "BENEFICIARY", "beneficiary"), (in_, "PAYER", "payer")]:
        agg = frame.groupby(col).agg(amt=("PAY_AMT", "sum"), cnt=("PAY_CNT", "sum")).reset_index()
        for r in agg.itertuples(index=False):
            cid = getattr(r, col)
            known = cid in idx.index
            rows.append(Counterparty(
                id=cid, role=role, amt_rub=float(r.amt), cnt=int(r.cnt), known=known,
                okved_name=_str(idx.at[cid, "MAINOKVED_NAME"]) if known else None,
            ))
    total = sum(c.amt_rub for c in rows) or 1.0
    known_amt = sum(c.amt_rub for c in rows if c.known)
    top = sorted(rows, key=lambda c: c.amt_rub, reverse=True)[:TOP_N_COUNTERPARTIES]

    sectors: dict[str, float] = {}
    for c in rows:
        if c.known and c.okved_name:
            sectors[c.okved_name] = sectors.get(c.okved_name, 0.0) + c.amt_rub
    notes = ["данные о платежах частичные, период не указан"]
    if revenue and max(c.amt_rub for c in rows) > 3 * revenue:
        notes.append("есть платежи больше трёх годовых выручек: вероятны выбросы")
    return Network(
        top_counterparties=top,
        counterparty_sectors=[s for s, _ in sorted(sectors.items(), key=lambda x: -x[1])[:5]],
        known_share=round(known_amt / total, 3),
        note="; ".join(notes),
    )


def _quality(row: pd.Series, cid: str) -> DataQuality:
    q = DataQuality()
    q.missing_fields = [f for f in TRACKED if pd.isna(row.get(f))]
    fts, countries = _flag(row.get("FTS_FLAG")), _countries(row.get("FTS_COUNTRY_NAME"))
    if fts and not countries:
        q.missing_fields.append("FTS_COUNTRY_NAME")
    if fts is False and countries:
        q.conflicts.append("FTS_FLAG=0, но указаны страны внешней торговли")
    if cid in duplicate_ids():
        q.conflicts.append("идентификатор встречается в нескольких строках с разными ОКВЭД; взята строка с наибольшей выручкой")
    raw, norm = _str(row.get("MAINOKVED_CODE")), _str(row.get("OKVED_NORM"))
    if raw and norm and raw != norm:
        q.notes.append(f"код ОКВЭД нормализован: {raw} -> {norm}")
    q.notes.append("NEGATIVEMARK не используется: смысл признака не уточнён")
    return q


def _company_profile(cid: str) -> CompanyProfile:
    idx = company_index()
    if cid not in idx.index:
        raise NotFound(f"компания {cid} не найдена")
    row = idx.loc[cid]
    gid = _str(row["GROUP_ID"])
    gs = group_stats().loc[gid] if gid in group_stats().index else None
    revenue = float(row["REVENUE_COMPANY"]) if not pd.isna(row["REVENUE_COMPANY"]) else None
    group_revenue = float(gs["group_revenue"]) if gs is not None else None
    fts = _flag(row["FTS_FLAG"])
    eligible = bool((revenue or 0) > settings.big_revenue_rub or (group_revenue or 0) > settings.big_revenue_rub)
    return CompanyProfile(
        company_id=cid,
        org_type=_str(row["ORG_TYPE"]),
        ownership=_str(row["OKFS_NAME"]),
        client_type=_str(row["CL_TYPE"]),
        stage=_str(row["ECO_MARK_LABEL"]),
        eligible_big=eligible,
        industry=Industry(
            okved_code=_str(row["OKVED_NORM"]), okved_raw=_str(row["MAINOKVED_CODE"]),
            okved_name=_str(row["MAINOKVED_NAME"]),
            okved_section=(_str(row["OKVED_NORM"]) or "")[:2] or None,
            sector_tags=sector_tags(_str(row["OKVED_NORM"]), _str(row["MAINOKVED_NAME"])),
        ),
        size=Size(revenue_rub=revenue, revenue_band=revenue_band(revenue),
                  workers_band=_str(row["AMOUNT_WORKERS_FNS_GRADATION"]),
                  age_band=_str(row["COMPANY_AGE_YEARS_GRADATION"])),
        geo=Geo(city=_str(row["CITY"]), region=_str(row["REGION"]), foreign_trade=fts,
                fts_countries=_countries(row["FTS_COUNTRY_NAME"]) if fts else []),
        bank_products=BankProducts(**{k: _flag(row[c]) for k, c in PRODUCTS.items()}),
        group=GroupInfo(group_id=gid,
                        group_size=int(gs["group_size"]) if gs is not None else None,
                        group_revenue_rub=group_revenue,
                        group_clients=int(gs["group_clients"]) if gs is not None else None),
        network=_network([cid], revenue),
        data_quality=_quality(row, cid),
        market_position=market_position(_str(row["OKVED_NORM"]), revenue, frozenset({cid})),
    )


def _group_profile(gid: str) -> CompanyProfile:
    idx = company_index()
    members = idx[idx["GROUP_ID"] == gid].sort_values("REVENUE_COMPANY", ascending=False)
    if members.empty:
        raise NotFound(f"группа {gid} не найдена")
    lead = _company_profile(members.index[0])
    total = float(members["REVENUE_COMPANY"].sum())
    countries = sorted({c for v in members["FTS_COUNTRY_NAME"] for c in _countries(v)})
    products = {k: bool((members[c].astype(float) == 1.0).any()) for k, c in PRODUCTS.items()}
    q = lead.data_quality.model_copy(deep=True)
    q.notes.insert(0, f"профиль группы: отрасль, регион и стадия взяты у крупнейшей компании {members.index[0]}")
    return lead.model_copy(update={
        "company_id": gid,
        "level": "group",
        "client_type": "CLIENT" if (members["CL_TYPE"] == "CLIENT").any() else lead.client_type,
        "eligible_big": total > settings.big_revenue_rub,
        "size": lead.size.model_copy(update={"revenue_rub": total, "revenue_band": revenue_band(total)}),
        "geo": lead.geo.model_copy(update={
            "foreign_trade": bool((members["FTS_FLAG"].astype(float) == 1.0).any()), "fts_countries": countries}),
        "bank_products": BankProducts(**products),
        "network": _network(list(members.index), total),
        "data_quality": q,
        "member_ids": list(members.index),
        "market_position": market_position(lead.industry.okved_code, total, frozenset(members.index), GROUP_POSITION_NOTE),
    })


def build_profile(company_id: str) -> CompanyProfile:
    if company_id.startswith("group_"):
        return _group_profile(company_id)
    return _company_profile(company_id)


def eligible_ids() -> tuple[list[str], list[str]]:
    """Компании с выручкой > порога и группы с суммарной выручкой > порога и хотя бы одним клиентом."""
    idx = company_index()
    companies = idx.index[idx["REVENUE_COMPANY"] > settings.big_revenue_rub].tolist()
    gs = group_stats()
    groups = gs.index[(gs["group_revenue"] > settings.big_revenue_rub) & (gs["group_clients"] > 0) & (gs["group_size"] > 1)].tolist()
    return companies, groups


def main() -> None:
    companies, groups = eligible_ids()
    out = settings.paths.profiles
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for cid in [*companies, *groups]:
            f.write(build_profile(cid).model_dump_json() + "\n")
            n += 1
    print(f"companies={len(companies)} groups={len(groups)} written={n} -> {out}", file=sys.stderr)


@lru_cache(maxsize=1)
def _stored() -> dict[str, dict]:
    p = settings.paths.profiles
    if not p.exists():
        return {}
    return {json.loads(l)["company_id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()}


def get_profile(company_id: str) -> CompanyProfile:
    """Профиль из предрасчёта, иначе строится на лету."""
    if (d := _stored().get(company_id)) is not None:
        return CompanyProfile.model_validate(d)
    return build_profile(company_id)


def list_profiles() -> list[CompanyProfile]:
    return [CompanyProfile.model_validate(d) for d in _stored().values()]


if __name__ == "__main__":
    main()
