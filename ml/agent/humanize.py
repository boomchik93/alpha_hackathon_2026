"""Служебные обозначения в текстах для пользователя: пути полей профиля, id фактов и фрагментов, имена блоков промпта.

Модель видит данные с техническими ключами (`bank_products.loan`, `f5`, `ITEM`, `confidence: assumption`) и иногда
переносит их в текст карточки («есть кредитный продукт (bank_products.loan)»). Руководитель не должен их видеть.
Промпт это запрещает (interpret, правило 20), а код чистит тексты после проверки: и новые карточки, и сохранённые
(при загрузке), ответы на вопросы и сценарии.

Чистятся только тексты для чтения. Идентификаторы в служебных полях (evidence, source_ids, company_features_used,
sources, meta) и дословные цитаты банка (bank_view) не трогаются.
"""

from __future__ import annotations

import re
from typing import Any, TypeVar

from pydantic import BaseModel

# Подписи полей профиля для текста (frontend/src/dict/labels.ts, FEATURE_LABEL, в форме для середины предложения).
FEATURE_LABELS = {
    "bank_products.loan": "кредит в банке", "bank_products.leasing": "лизинг", "bank_products.deposit": "депозиты",
    "bank_products.rko": "расчётный счёт", "bank_products.guarantee": "банковские гарантии",
    "bank_products.payroll": "зарплатный проект", "bank_products.factoring": "факторинг", "bank_products": "продукты банка",
    "industry.okved_name": "отрасль", "industry.okved_code": "код ОКВЭД", "industry.sector_tags": "сектор",
    "industry": "отрасль", "size.revenue_rub": "выручка", "size.revenue_band": "размер компании",
    "size.workers_band": "число сотрудников", "geo.region": "регион", "geo.city": "город",
    "geo.foreign_trade": "внешняя торговля", "geo.fts_countries": "страны внешней торговли",
    "network.counterparty_sectors": "отрасли контрагентов", "network.top_counterparties": "ключевые контрагенты",
    "network.known_share": "доля известных контрагентов", "network.coverage": "данные о платежах",
    "network.note": "данные о платежах", "network": "контрагенты", "group.group_id": "группа компаний",
    "group.group_revenue_rub": "выручка группы", "group.group_size": "размер группы", "group": "группа компаний",
    "market_position": "положение в отрасли", "data_quality": "полнота данных", "eligible_big": "размер компании",
}
# Путь с точкой («geo.fts_countries») или ключ с подчёркиванием: обычные слова («X5 Group», «network») не трогаются.
_PATH = (r"(?:profile\.)?(?:(?:bank_products|industry|geo|network|size|group)(?:\.[a-z_]+)+"
         r"|bank_products|market_position|data_quality|eligible_big)\b")
_FACT = r"f\d+(?:\s*[–—,-]\s*f?\d+)*"

# Обозначения блоков промпта и полей карточки -> слова для читателя.
_WORDS = [
    (r"\bв ITEM\b", "в новости"), (r"\bиз ITEM\b", "из новости"), (r"\bITEM\b", "новость"),
    (r"\bRELATED_NEWS\b", "соседние новости"), (r"\bBANK_CHUNKS\b", "фрагменты аналитики банка"),
    (r"\bPROFILE\b", "профиль компании"), (r"\bPOSITION\b", "положение в отрасли"),
    (r"\bCONTRADICTIONS\b", "расхождение"), (r"\bbank_view\b", "позиция банка"), (r"\bai_inference\b", "вывод модели"),
    (r"\bwhy_it_matters\b", "«почему это важно»"), (r"\bwhat_happened\b", "«что произошло»"),
    (r"\bsuggested_actions\b", "действия"), (r"\bcompany_features_used\b", "признаки профиля"),
    (r"\bis_model_inference\b", "вывод модели"), (r"\brelevance\.score\b", "релевантность"),
    (r"\blimitations\b", "ограничения"), (r"\bevidence\b", "источники"), (r"\bstale\s*=\s*true\b", "устаревший"),
]
_CONFIDENCE = [
    (rf"\(\s*confidence\s*[:=]\s*[\"«]?assumption[\"»]?\s*\)", "(предположение)"),
    (rf"\(\s*confidence\s*[:=]\s*[\"«]?confirmed[\"»]?\s*\)", ""),
    (r"\bconfidence\s*[:=]\s*[\"«]?assumption[\"»]?", "статус предположения"),
    (r"\bconfidence\s*[:=]\s*[\"«]?confirmed[\"»]?", "статус подтверждённого факта"),
    (r"\bassumption\b", "предположение"), (r"\bconfirmed\b", "подтверждённый факт"), (r"\bconfidence\b", "статус"),
]


def _doc_titles() -> dict[str, str]:
    from ml.ingest.analytics import DOCS

    return {m.doc_id: m.title for m in DOCS.values()}


def _chunk_ref(m: re.Match) -> str:
    title = _doc_titles().get(m.group(1))
    return f"«{title}», стр. {m.group(2)}" if title else f"аналитика банка, стр. {m.group(2)}"


def clean(text: str) -> str:
    """Текст без служебных обозначений. Обычный текст проходит без изменений."""
    if not text or not re.search(r"[A-Za-z_]{2,}|f\d", text):
        return text
    s = text
    # id фрагмента аналитики банка -> документ и страница
    s = re.sub(r"\b(bank_[a-z0-9_]+):p(\d+):c\d+\b", _chunk_ref, s)
    s = re.sub(r"\bsrc_[a-z0-9_]+(?::f\d+)?\b", "новость", s)
    s = re.sub(r"\bn_(?:[0-9a-f]{8,}|20\d\d_\d\d_\d\d_[a-z_]+)\b", "новость", s)
    for pat, rep in _CONFIDENCE:
        s = re.sub(pat, rep, s, flags=re.I)
    # скобки только из служебных обозначений: «(bank_products.loan, geo.fts_countries)», «(f5)», «(f6–f10)»
    s = re.sub(rf"\s*\(\s*(?:(?:{_PATH}|{_FACT})\s*(?:[,;]|\bи\b)?\s*)+\)", "", s)
    s = re.sub(rf"\b(факт\w*)\s+{_FACT}\b", r"\1", s)
    s = re.sub(rf"(?<![\w.]){_FACT}\b", "", s)
    s = re.sub(rf"\b(по данным|согласно|в поле|поле)\s+{_PATH}", lambda m: "по данным профиля" if m.group(1) != "согласно" else "согласно профилю", s)
    s = re.sub(_PATH, lambda m: FEATURE_LABELS.get(m.group(0).removeprefix("profile."),
                                                   FEATURE_LABELS.get(m.group(0).removeprefix("profile.").split(".")[0], "профиль")),
               s)
    for pat, rep in _WORDS:
        s = re.sub(pat, rep, s)
    # пробелы и пунктуация после вырезания
    s = re.sub(r"\(\s*[,;]?\s*\)", "", s)
    s = re.sub(r"\s+([,.;:!?)])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r",\s*,", ",", s)
    return s.strip()


# Ключи, под которыми лежат тексты для чтения; остальное (id, пути, цитаты банка, источники, meta) не трогается.
_TEXT_KEYS = {"text", "reasons", "limitations", "answer", "horizon", "title", "assumption", "likelihood_basis", "impact",
              "signals", "rationale", "trigger", "summary", "key_uncertainty", "recommendation", "switch_signals"}
_SKIP_KEYS = {"bank_view", "sources", "meta", "channels", "quotes", "company_features_used", "evidence", "source_ids"}

M = TypeVar("M", bound=BaseModel)


def _walk(obj: Any, key: str | None = None) -> Any:
    if isinstance(obj, dict):
        return {k: v if k in _SKIP_KEYS else _walk(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(x, key) for x in obj]
    if isinstance(obj, str) and key in _TEXT_KEYS:
        return clean(obj)
    return obj


def humanize(model: M) -> M:
    """Копия карточки, ответа или набора сценариев с очищенными текстами."""
    data = model.model_dump(mode="json")
    cleaned = _walk(data)
    return model if cleaned == data else type(model).model_validate(cleaned)
