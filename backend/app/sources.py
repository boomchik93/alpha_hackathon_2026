"""Первоисточники: PDF аналитики банка по `doc_id` из `SourceRef` карточки.

Белый список повторяет `ml/ingest/analytics.py, DOCS` (doc_id -> имя PDF без расширения); тест сверяет их.
Путь к файлу строится только из белого списка: из запроса берётся лишь ключ.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

BANK_DOCS: dict[str, str] = {
    "bank_rate_2026_09_11": "ALFA-Macro_Insights-Policy_Rate_11Sep",
    "bank_bond_brief_2026_09_14": "Бонд_Бриф_14092026",
    "bank_macro_7m26": "ALFA-7M26",
    "bank_inflation_2026_09_02": "Inflation_tracker_02",
    "bank_gold_2026_08": "Gold_digest_August26",
    "bank_consumption_2026_08_05": "ALFA-consumption",
    "bank_macro_1h26": "ALFA-macro_stat1H26",
    "bank_fi_strategy_3q26": "FI_Strategy_3Q26_22072026",
    "bank_steel_1q21": "ALFA_Russian_Steels_1К21PreviewUpdate_15Apr2021",
}


def pdf_path(analytics_dir: Path, doc_id: str) -> Path | None:
    stem = BANK_DOCS.get(doc_id)
    if stem is None or not analytics_dir.exists():
        return None
    # macOS хранит кириллицу в именах файлов в NFD: сравниваем в NFC, как ml/ingest/analytics.py.
    want = unicodedata.normalize("NFC", stem + ".pdf")
    return next((p for p in analytics_dir.iterdir() if unicodedata.normalize("NFC", p.name) == want), None)
