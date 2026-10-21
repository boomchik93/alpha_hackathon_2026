"""Общая важность материала для экономики: один вызов Gemma на материал, не на пару.

Не зависит от компании, поэтому масштабируется на любой поток и кэшируется клиентом. Используется лентой как приор порядка
(`ml.ranking.ranker.BETA`); при сбое вызова остаётся нейтральные 0,5, лента от этого не ломается.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ml.llm.client import LLMClient, LLMError, get_client
from ml.prompts import render
from ml.schemas import ParsedItem

NEUTRAL = 0.5
MAX_TEXT_CHARS = 2500


class ImportanceResult(BaseModel):
    importance: float = Field(ge=0, le=1)
    scope: Literal["economy", "multi_sector", "sector", "company", "none"] = "sector"
    reason: str = ""


def score_importance(item: ParsedItem, text: str, client: LLMClient | None = None, trace=None) -> float:
    """Важность материала 0..1. `text`: исходный текст новости или первые фрагменты документа банка."""
    client = client or get_client()
    p = render("importance", title=item.title, published_at=item.source.published_at.isoformat(), text=text[:MAX_TEXT_CHARS])
    mock_ctx = {"event_type": item.event_type, "topics": item.topics, "title": item.title}
    try:
        res, _ = client.complete_json("importance", p.system, p.user, ImportanceResult, mock_context=mock_ctx,
                                      prompt_version=p.version, trace=trace)
    except (ValueError, LLMError):
        return NEUTRAL
    return res.importance


def with_importance(item: ParsedItem, text: str, client: LLMClient | None = None) -> ParsedItem:
    return item.model_copy(update={"importance": score_importance(item, text, client)})
