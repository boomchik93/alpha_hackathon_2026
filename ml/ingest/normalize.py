"""Детерминированная чистка зашумлённых текстов новостей перед разбором.

Цифры не трогаем: меняются только пробелы, пунктуация, регистр и известные
сокращения. Флаг `noisy` уходит в разбор и в контекст интерпретации.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ABBREVIATIONS = {
    r"\bнед\.": "неделю",
    r"\bвнутр\.": "внутренний",
    r"\bрозничн\.": "розничные",
    r"\bмлрд\.": "млрд",
    r"\bмлн\.": "млн",
    r"\bдизтопливо\b": "дизельное топливо",
    r"\bг\.\s?г\.": "г/г",
}
# Короткие слова в верхнем регистре считаем аббревиатурами (ВВП, ЦБ, НПЗ), кроме служебных.
SHORT_COMMON = {"НА", "И", "В", "С", "ПО", "ДО", "ЗА", "ИЗ", "ОТ", "НЕ", "К", "О", "У", "А", "НО", "ИЛИ", "ДЛЯ", "ПРИ", "БЕЗ"}


def _lower_word(w: str) -> str:
    letters = [c for c in w if c.isalpha()]
    if not letters or not all(c.isupper() for c in letters):
        return w  # смешанный регистр («Аи-92») оставляем как есть
    core = re.sub(r"\W", "", w)
    if len(letters) <= 4 and core.upper() not in SHORT_COMMON:
        return w
    return w.lower()


@dataclass
class Normalized:
    text: str
    noisy: bool
    fixes: int


def _decaps(s: str) -> tuple[str, int]:
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 8 or sum(c.isupper() for c in letters) / len(letters) < 0.6:
        return s, 0
    words = [_lower_word(w) for w in s.split(" ")]
    out = " ".join(words)
    return out[:1].upper() + out[1:], 1


def normalize(text: str) -> Normalized:
    fixes = 0

    def sub(pattern: str, repl: str, s: str, flags: int = 0) -> str:
        nonlocal fixes
        new, n = re.subn(pattern, repl, s, flags=flags)
        fixes += n
        return new

    s = text
    for pat, repl in ABBREVIATIONS.items():
        s = sub(pat, repl, s, re.I)
    s = sub(r"\s*--\s*", " — ", s)
    s = sub(r"\s*;;+\s*", "; ", s)
    s = sub(r"([!?.])\1+", r"\1", s)
    s = sub(r"(\d)\s+%", r"\1%", s)
    s = sub(r"%(?=[а-яa-z])", "% ", s)
    s = sub(r"\s+([,.;:!?)])", r"\1", s)
    s = sub(r"\(\s+", "(", s)
    s = sub(r"[ \t]{2,}", " ", s)
    s, n = _decaps(s)
    fixes += n
    # Предложение с маленькой буквы после точки или в начале текста.
    s = sub(r"(^|[.!?]\s+)([а-яё])", lambda m: m.group(1) + m.group(2).upper(), s)  # type: ignore[arg-type]
    s = s.strip()
    return Normalized(s, noisy=fixes >= 3, fixes=fixes)
