"""Защита от prompt injection в недоверенных текстах (новости, OCR аналитики)."""

from __future__ import annotations

import re

INJECTION = re.compile(
    r"игнорируй|забудь (все|предыдущ)|предыдущие инструкции|ignore (all|previous)|system prompt|ты теперь|you are now",
    re.I,
)
_SENT = re.compile(r"(?<=[.!?])\s+")


def strip_injections(text: str) -> tuple[str, bool]:
    """Убирает предложения, похожие на инструкции модели. Возвращает текст и флаг, что что-то убрано."""
    kept, found = [], False
    for s in _SENT.split(text):
        if INJECTION.search(s):
            found = True
            continue
        kept.append(s)
    return " ".join(kept).strip(), found
