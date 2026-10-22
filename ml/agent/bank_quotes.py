"""bank_view из дословных цитат аналитики банка.

Модель выбирает цитаты, код проверяет, что каждая есть во фрагменте, и сам
собирает текст: цитата, документ, дата, страница. Пересказать банк своими
словами и приписать ему вывод модель так не может.
"""

from __future__ import annotations

import re
from datetime import date

from ml.schemas import BankQuote, BankView

MIN_QUOTE_CHARS = 20
MAX_QUOTE_CHARS = 400
MIN_PART_CHARS = 12
# Строка таблицы после OCR («Курс, конец года, 62 74 74 70 90 102»): цифр много, слов мало,
# к какому году и показателю относится число, из цитаты не понять.
MIN_QUOTE_WORDS = 6
MAX_DIGIT_SHARE = 0.25
_WORD = re.compile(r"[^\W\d_]{3,}")


def table_like(quote: str) -> bool:
    chars = [ch for ch in quote if not ch.isspace()]
    digits = sum(ch.isdigit() for ch in chars)
    return len(_WORD.findall(quote)) < MIN_QUOTE_WORDS or (bool(chars) and digits / len(chars) > MAX_DIGIT_SHARE)
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")


def _norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[«»\"“”„'‘’]", "", s)
    s = re.sub(r"[‐‑‒–—−]", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,;:")


def quote_found(quote: str, chunk_text: str) -> bool:
    """Цитата есть во фрагменте дословно (с точностью до пробелов, кавычек и регистра).

    Пропуск внутри цитаты можно обозначить многоточием: тогда каждая часть ищется отдельно.
    """
    hay = _norm(chunk_text)
    parts = [_norm(p) for p in _ELLIPSIS.split(quote)]
    parts = [p for p in parts if p]
    if not parts or not any(len(p) >= MIN_PART_CHARS for p in parts):
        return False
    return all(p in hay for p in parts)


def check_quotes(view: BankView | None, hits: list[dict]) -> tuple[list[BankQuote], list[str]]:
    """Проверенные цитаты и замечания по непроверенным."""
    if view is None:
        return [], []
    by_id = {h["id"]: h for h in hits}
    ok, issues = [], []
    if not view.quotes:
        issues.append("bank_view без quotes: приведи 1-3 дословные цитаты из BANK_CHUNKS в bank_view.quotes")
        return ok, issues
    for i, q in enumerate(view.quotes):
        h = by_id.get(q.source_id)
        text = q.text.strip()
        if h is None:
            issues.append(f"bank_view.quotes[{i}]: source_id {q.source_id} не из BANK_CHUNKS")
        elif len(text) < MIN_QUOTE_CHARS or len(text) > MAX_QUOTE_CHARS:
            issues.append(f"bank_view.quotes[{i}]: цитата должна быть от {MIN_QUOTE_CHARS} до {MAX_QUOTE_CHARS} символов")
        elif table_like(text):
            issues.append(f"bank_view.quotes[{i}]: это строка таблицы без подписей, а не позиция банка. Цитируй предложение с текстом")
        elif not quote_found(text, h["text"]):
            issues.append(f"bank_view.quotes[{i}]: цитаты нет дословно во фрагменте {q.source_id}. Копируй текст фрагмента без изменений")
        else:
            ok.append(BankQuote(source_id=q.source_id, text=text))
    return ok, issues


def _ru_date(iso: str) -> str:
    try:
        return date.fromisoformat(iso).strftime("%d.%m.%Y")
    except ValueError:
        return iso


def assemble(quotes: list[BankQuote], hits: list[dict]) -> BankView | None:
    """Текст bank_view из проверенных цитат, сгруппированных по документу."""
    if not quotes:
        return None
    by_id = {h["id"]: h for h in hits}
    groups: dict[str, list[BankQuote]] = {}
    for q in quotes:
        groups.setdefault(by_id[q.source_id]["doc_id"], []).append(q)
    parts = []
    for qs in groups.values():
        h = by_id[qs[0].source_id]
        pages = sorted({by_id[q.source_id]["page"] for q in qs})
        where = f"«{h['title']}», {_ru_date(h['published_at'])}, стр. {', '.join(map(str, pages))}"
        stale = ", материал устарел" if h.get("stale") else ""
        cited = " … ".join("«" + q.text.strip(' «»"') + "»" for q in qs)
        parts.append(f"Аналитика банка ({where}{stale}): {cited}")
    ids = list(dict.fromkeys(q.source_id for q in quotes))
    return BankView(text="\n".join(parts), source_ids=ids, quotes=quotes)
