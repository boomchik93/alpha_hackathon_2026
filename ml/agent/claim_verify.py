"""Проверка утверждений карточки по одному.

Один вызов дешёвой модели на карточку: каждое утверждение сверяется с теми
источниками, на которые оно ссылается. Выводы модели (ai_inference,
why_it_matters, утверждения без evidence) проверяются на натяжку.

Политика: неподтверждённые риски, возможности и действия код снимает сам, без
повтора генерации; натяжка в why_it_matters или ai_inference идёт в обратную
связь для повтора. Включается ML_CLAIM_VERIFY=1 (по умолчанию выключено: на eval
не снизило долю карточек с замечаниями судьи, а цену и время карточки подняло).

Модель проверки Qwen (ml.llm.router): на выборке карточек Gemma признавала
подтверждёнными почти все утверждения, в том числе те, где судья eval нашёл
выдуманные прогнозы банка. Qwen дороже, поэтому проверяются только карточки
с relevance.score >= MIN_RELEVANCE: короткие карточки нерелевантных пар нет.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel

from ml.llm.client import LLMClient
from ml.prompts import render
from ml.schemas import CompanyProfile, InsightCard, ParsedItem

BAD = {"unsupported", "stretch"}
MIN_RELEVANCE = 0.3
REMOVED_NOTE = "часть утверждений снята автоматической проверкой: источники их не подтверждают"


class Verdict(BaseModel):
    id: str
    verdict: Literal["supported", "partial", "unsupported", "stretch"]
    reason: str = ""


class Verdicts(BaseModel):
    verdicts: list[Verdict] = []


def enabled(card: InsightCard | None = None) -> bool:
    if os.getenv("ML_CLAIM_VERIFY", "0") != "1":
        return False
    return card is None or card.relevance.score >= MIN_RELEVANCE


def claims_of(card: InsightCard) -> list[dict]:
    out = [
        {"id": "why", "where": "why_it_matters", "kind": "inference", "text": card.why_it_matters.text, "evidence": ["profile"]},
        {"id": "ai", "where": "ai_inference", "kind": "inference", "text": card.ai_inference.text, "evidence": card.ai_inference.source_ids},
    ]
    groups = [("risks", card.consequences.risks), ("opportunities", card.consequences.opportunities),
              ("suggested_actions", card.suggested_actions)]
    for name, claims in groups:
        for i, c in enumerate(claims):
            kind = "inference" if c.is_model_inference or not c.evidence else "sourced"
            out.append({"id": f"{name}[{i}]", "where": name, "kind": kind, "text": c.text, "evidence": c.evidence})
    return out


def verify_claims(client: LLMClient, card: InsightCard, profile: CompanyProfile, item: ParsedItem,
                  bank_hits: list[dict], item_source_id: str, trace=None) -> dict[str, Verdict]:
    claims = claims_of(card)
    used = card.referenced_source_ids()
    p = render("claim_verify", profile=profile.prompt_view(), item_source_id=item_source_id,
               item=item.model_dump(mode="json", include={"title", "summary", "facts"}),
               bank_chunks=[{"id": h["id"], "date": h["published_at"], "text": h["text"]} for h in bank_hits if h["id"] in used] or "нет",
               claims=[{k: c[k] for k in ("id", "kind", "text", "evidence")} for c in claims])
    res, _ = client.complete_json("claim_verify", p.system, p.user, Verdicts,
                                  mock_context={"claims": claims}, prompt_version=p.version, trace=trace)
    known = {c["id"] for c in claims}
    return {v.id: v for v in res.verdicts if v.id in known}


def feedback_issues(verdicts: dict[str, Verdict]) -> list[str]:
    """Замечания, ради которых стоит переписать текст: натяжки в why_it_matters и ai_inference."""
    out = []
    for cid, where in (("why", "why_it_matters"), ("ai", "ai_inference")):
        v = verdicts.get(cid)
        if v and v.verdict in BAD:
            out.append(f"{where}: проверка утверждений считает это натяжкой ({v.reason}). "
                       "Опирайся только на событие, фрагменты банка и записанные признаки профиля; механизм без источника подай как гипотезу")
    return out


def prune(card: InsightCard, verdicts: dict[str, Verdict]) -> tuple[InsightCard, list[str]]:
    """Снимает неподтверждённые риски, возможности и действия. Возвращает карточку и список снятого."""
    removed: list[str] = []

    def keep(name: str, claims: list) -> list:
        kept = []
        for i, c in enumerate(claims):
            v = verdicts.get(f"{name}[{i}]")
            if v and v.verdict in BAD:
                removed.append(f"{name}[{i}]: {v.reason}")
            else:
                kept.append(c)
        return kept

    risks = keep("risks", card.consequences.risks)
    opps = keep("opportunities", card.consequences.opportunities)
    actions = keep("suggested_actions", card.suggested_actions)
    if not actions:  # хотя бы одно действие обязательно: оставляем первое как вопрос для анализа
        first = card.suggested_actions[0]
        actions = [first.model_copy(update={"type": "question_for_analysis", "is_model_inference": True, "evidence": []})]
        removed = [r for r in removed if not r.startswith("suggested_actions[0]")]
    if not removed:
        return card, []
    limitations = card.limitations if REMOVED_NOTE in card.limitations else [*card.limitations, REMOVED_NOTE]
    card = card.model_copy(update={
        "consequences": card.consequences.model_copy(update={"risks": risks, "opportunities": opps}),
        "suggested_actions": actions, "limitations": limitations,
    })
    return card, removed
