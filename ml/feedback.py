"""Обратная связь пользователей по карточкам (REVIEW_PLAN.md, T16): запись в `var/feedback.jsonl` и сводка по версиям промпта.

Оценка привязана к версии промпта и модели карточки, поэтому по сводке видно, стал ли новый промпт хуже старого.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from ml.agent.orchestrator import load_card
from ml.config import settings

MAX_COMMENT_CHARS = 1000
_LOCK = threading.Lock()

Reason = Literal["wrong_fact", "not_relevant", "obvious", "other"]


class FeedbackIn(BaseModel):
    rating: Literal["up", "down"]
    reason: Reason | None = None
    comment: str | None = Field(None, max_length=MAX_COMMENT_CHARS)
    role: str | None = Field(None, max_length=64, description="роль пользователя: owner, cfo, treasurer и т. п.")


class FeedbackSaved(BaseModel):
    ok: bool = True
    card_id: str
    trace_id: str


def record(card_id: str, fb: FeedbackIn) -> FeedbackSaved:
    """Дописывает оценку; NotFound, если карточки нет."""
    card = load_card(card_id)
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "card_id": card.card_id, "company_id": card.company_id,
           "item_ids": card.item_ids, "trace_id": card.trace_id, "prompt_version": card.meta.prompt_version, "model": card.meta.model,
           **fb.model_dump()}
    path = settings.paths.feedback
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return FeedbackSaved(card_id=card.card_id, trace_id=card.trace_id)


def load_rows() -> list[dict]:
    path = settings.paths.feedback
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(rows: list[dict]) -> dict:
    """Доля «down» и причины по версии промпта; ключ «все» — по всем версиям."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["prompt_version"], []).append(r)
    out = {}
    for name, rs in {**groups, "все": rows}.items():
        down = [r for r in rs if r["rating"] == "down"]
        out[name] = {"total": len(rs), "down": len(down), "down_share": len(down) / len(rs) if rs else None,
                     "reasons": dict(Counter(r.get("reason") or "не указана" for r in down))}
    return out


def summary() -> dict:
    return summarize(load_rows())
