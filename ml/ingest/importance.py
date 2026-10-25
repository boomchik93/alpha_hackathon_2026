"""Сигналы материала, не зависящие от компании:
один вызов Gemma на материал, не на пару.

- importance: общая важность для экономики, приор порядка ленты (`ml.ranking.ranker.BETA`); слух и мнение
  код ограничивает сверху (`SOFT_SOURCE_MAX`), потому что модель шкалу для них не держит;
- scope: масштаб события (вся экономика, несколько отраслей, отрасль, компания);
- channels: экономические каналы влияния (`ml.topics.IMPACT_CHANNELS`) вместо ручной карты тем `ml.topics.TOPICS`:
  по ним лента находит признаки профиля, через которые событие касается компании.

Масштабируется на любой поток и кэшируется клиентом. При сбое вызова остаются нейтральные значения: лента не ломается.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ml.llm.client import LLMClient, LLMError, get_client
from ml.prompts import render
from ml.schemas import ParsedItem
from ml.topics import IMPACT_CHANNELS

NEUTRAL = 0.5
MAX_TEXT_CHARS = 2500
# Прогнозы не ограничиваются: на реальных новостях прогнозы по ставке и инфляции релевантны почти всем компаниям.
SOFT_SOURCES = ("rumor", "opinion")
SOFT_SOURCE_MAX = 0.3


class ImportanceResult(BaseModel):
    importance: float = Field(ge=0, le=1)
    scope: Literal["economy", "multi_sector", "sector", "company", "none"] = "sector"
    source_type: Literal["official", "statistics", "market", "company", "forecast", "rumor", "opinion"] = "market"
    channels: list[str] = []
    reason: str = ""


def item_signals(item: ParsedItem, text: str, client: LLMClient | None = None, trace=None) -> ImportanceResult | None:
    """Сигналы материала; None при сбое вызова. `text`: исходный текст новости или первые фрагменты документа банка."""
    client = client or get_client()
    p = render("importance", title=item.title, published_at=item.source.published_at.isoformat(), text=text[:MAX_TEXT_CHARS],
               channels=[f"{c.key}: {c.meaning}" for c in IMPACT_CHANNELS.values()])
    mock_ctx = {"event_type": item.event_type, "topics": item.topics, "title": item.title}
    try:
        res, _ = client.complete_json("importance", p.system, p.user, ImportanceResult, mock_context=mock_ctx,
                                      prompt_version=p.version, trace=trace)
    except (ValueError, LLMError):
        return None
    channels = [c for c in dict.fromkeys(res.channels) if c in IMPACT_CHANNELS][:3]
    imp = min(res.importance, SOFT_SOURCE_MAX) if res.source_type in SOFT_SOURCES else res.importance
    return res.model_copy(update={"importance": imp, "channels": channels})


def score_importance(item: ParsedItem, text: str, client: LLMClient | None = None, trace=None) -> float:
    """Важность материала 0..1 (нейтральные 0,5 при сбое)."""
    res = item_signals(item, text, client, trace)
    return res.importance if res else NEUTRAL


def with_importance(item: ParsedItem, text: str, client: LLMClient | None = None) -> ParsedItem:
    res = item_signals(item, text, client)
    if res is None:
        return item.model_copy(update={"importance": NEUTRAL})
    return item.model_copy(update={"importance": res.importance, "scope": res.scope, "channels": res.channels})
