"""Новость -> ParsedItem (Gemma, дёшево; результат кэшируется клиентом)."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from ml.ingest.analytics import freshness
from ml.ingest.normalize import normalize
from ml.ingest.news_fetch import RawItem
from ml.llm.client import LLMClient, get_client
from ml.prompts import render
from ml.safety import strip_injections
from ml.schemas import Fact, ItemSource, ParsedItem, SourceKind
from ml.profile.build_profile import normalize_okved
from ml.topics import EVENT_TYPES, TOPICS


class ParseResult(BaseModel):
    title: str
    summary: str
    event_type: str
    topics: list[str] = []
    sectors: list[str] = []
    okved_hints: list[str] = []
    regions: list[str] = []
    facts: list[Fact] = []


def parse_item(raw: RawItem, client: LLMClient | None = None, trace=None) -> ParsedItem:
    client = client or get_client()
    # Сначала нормализация шума (пробелы, регистр, сокращения; цифры не меняются), затем вырезание инструкций:
    # инструкции из источника не должны доходить ни до промпта, ни до карточки.
    title, body = normalize(raw.title), normalize(raw.text)
    noisy = title.noisy or body.noisy
    text, injected = strip_injections(body.text)
    if injected and trace is not None:
        trace.log("injection_stripped", item_id=raw.id)
    raw = RawItem(**{**raw.__dict__, "title": title.text, "text": text})
    p = render("parse_item", title=raw.title, published_at=raw.published_at, text=raw.text,
               event_types=EVENT_TYPES, topics=list(TOPICS),
               noise_note="Текст был зашумлён и автоматически почищен: формулировки фактов приводи к нормальному русскому языку, цифры переноси точно." if noisy else "")
    res, _ = client.complete_json("parse_item", p.system, p.user, ParseResult,
                                  mock_context={"raw": raw}, prompt_version=p.version, trace=trace)
    published = date.fromisoformat(raw.published_at)
    return ParsedItem(
        item_id=raw.id,
        source=ItemSource(kind=SourceKind(raw.kind), title=raw.title, publisher=raw.publisher, url=raw.url, published_at=published),
        title=raw.title,
        summary=res.summary,
        event_type=res.event_type if res.event_type in EVENT_TYPES else "other",
        topics=[t for t in res.topics if t in TOPICS],
        sectors=res.sectors,
        okved_hints=[h for h in (normalize_okved(x) for x in res.okved_hints) if h],
        regions=res.regions,
        facts=res.facts,
        freshness=freshness(published),
        noisy=noisy,
    )
