"""Эталонная релевантность пар «компания x материал», размечено вручную.

Шкала: 2 = событие напрямую бьёт по бизнесу компании; 1 = косвенно (макрофон,
продукты банка, ВЭД); 0 = не относится. Отсутствующая пара = 0.

Ключи компаний см. `companies.json`:
R розница одеждой, T грузоперевозки, P фарма-опт, F топливо-опт, C стройка,
M металлы-опт, B стройматериалы-опт (трудный), W неспец. опт (трудный), G группа бетон.

Разметка опирается только на то, что есть в профиле (отрасль, продукты банка,
ВЭД), а не на знания о реальной компании. Устаревший материал (сталь 2021)
размечен 0 для всех: как актуальный его рекомендовать нельзя.
"""

ALL = "RTPFCMBWG"
CREDIT = "RPFG"     # есть кредит
LEASE_ONLY = "TC"   # только лизинг
FTS = "RTFMG"       # внешняя торговля


def _g(**grades: str) -> dict[str, int]:
    """_g(two="TF", one="RG") -> {"T": 2, "F": 2, "R": 1, "G": 1}."""
    out = {k: 1 for k in grades.get("one", "")}
    out.update({k: 2 for k in grades.get("two", "")})
    return out


LABELS: dict[str, dict[str, int]] = {
    # --- демо-новости ---
    "n_2026_09_11_cbr_rate": _g(two=CREDIT + LEASE_ONLY, one="MBW"),
    "n_2026_09_11_cbr_rate_dup": _g(two=CREDIT + LEASE_ONLY, one="MBW"),
    "n_2026_09_03_petrol": _g(two="FT", one="GC"),
    "n_2026_09_14_ofz": _g(one=CREDIT + LEASE_ONLY),
    "n_2026_09_08_retail": _g(two="R", one="PW"),
    "n_2026_09_10_trucking": _g(two="T", one="RPFMBWG"),
    "n_2026_09_09_wages": _g(two="TC", one="RPFMBWG"),
    "n_2026_09_12_gold": _g(one=FTS),
    "n_2026_09_13_rate_cut_rumor": _g(one=CREDIT + LEASE_ONLY),
    "n_2026_09_05_football": {},
    "n_2026_09_06_noisy": _g(two="CGB", one="MRFTW"),
    "n_2026_09_11_steel": _g(two="MC", one="GB"),
    # --- аналитика банка ---
    "bank_rate_2026_09_11": _g(two=CREDIT + LEASE_ONLY, one="MBW"),
    "bank_bond_brief_2026_09_14": _g(one=CREDIT + LEASE_ONLY),
    "bank_macro_7m26": _g(two="R", one="TPFCMBWG"),
    "bank_inflation_2026_09_02": _g(two="FT", one="RPWG"),
    "bank_gold_2026_08": _g(one=FTS),
    "bank_consumption_2026_08_05": _g(two="R", one="PW"),
    "bank_macro_1h26": _g(one="RTFCMBWG"),
    "bank_fi_strategy_3q26": _g(one=CREDIT + LEASE_ONLY),
    "bank_steel_1q21": {},
    # --- holdout: новости, которых нет в демо-корпусе ---
    "h_2026_09_09_housing": _g(two="CGB", one="M"),
    "h_2026_09_04_petrol_noisy": _g(two="FT", one="GC"),
    "h_2026_09_10_grain_duty": {},
    "h_2026_09_12_rub_fx": _g(two="R", one="TFMG"),
    "h_2026_09_08_pharma": _g(two="P"),
    "h_2026_09_11_metal_prices": _g(two="MC", one="GB"),
    "h_2026_09_11_rate_en": _g(two=CREDIT + LEASE_ONLY, one="MBW"),
}

# Пары-дубликаты: в ленте должен остаться один материал из пары.
DUPLICATES = [("n_2026_09_11_cbr_rate", "n_2026_09_11_cbr_rate_dup")]
# Зашумлённые тексты для замера устойчивости.
NOISY = {"n_2026_09_06_noisy", "h_2026_09_04_petrol_noisy", "h_2026_09_11_rate_en"}
# Тексты с попыткой prompt injection.
INJECTED = {"n_2026_09_06_noisy"}


def grade(company_key: str, item_id: str) -> int:
    return LABELS.get(item_id, {}).get(company_key, 0)
