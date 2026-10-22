"""Сверка противоречий: факты материала против фрагментов аналитики банка.

Отдельный дешёвый вызов до интерпретации. Если найдено расхождение, оно передаётся в промпт интерпретации блоком
CONTRADICTIONS, а `rule_check` требует назвать его в `limitations`. Включается ML_CONSISTENCY=1; по умолчанию выключено
до замера (`python -m ml.eval.eval_consistency`, затем парное сравнение карточек на dev).
"""

from __future__ import annotations

import os
import re
from typing import Literal

from pydantic import BaseModel

from ml.llm.client import LLMClient, LLMError
from ml.prompts import render
from ml.schemas import InsightCard, ParsedItem

MAX_CHUNKS = 6
CONTRADICTION_WORD = re.compile(r"расхожд|противореч", re.I)


class Pair(BaseModel):
    fact_id: str
    chunk_id: str
    note: str = ""


class ConsistencyResult(BaseModel):
    status: Literal["consistent", "contradicts", "unrelated"] = "unrelated"
    pairs: list[Pair] = []


def enabled() -> bool:
    return os.getenv("ML_CONSISTENCY", "0") == "1"


def clean(res: ConsistencyResult, fact_ids: set[str], chunk_ids: set[str]) -> ConsistencyResult:
    """Пары с неизвестными id отбрасываются; `contradicts` без валидных пар считается `unrelated`."""
    pairs = [p for p in res.pairs if p.fact_id in fact_ids and p.chunk_id in chunk_ids]
    if res.status == "contradicts" and not pairs:
        return ConsistencyResult(status="unrelated")
    return ConsistencyResult(status=res.status, pairs=pairs if res.status == "contradicts" else [])


def check(client: LLMClient, title: str, facts: list[dict], chunks: list[dict], trace=None) -> ConsistencyResult:
    """facts: [{id, text, confidence}], chunks: [{id, text}] (trusted_text). Без фактов или фрагментов вызова нет."""
    chunks = chunks[:MAX_CHUNKS]
    if not facts or not chunks:
        return ConsistencyResult()
    p = render("consistency", title=title, facts=facts, chunks=[{"id": c["id"], "text": c["text"]} for c in chunks])
    try:
        res, _ = client.complete_json("consistency", p.system, p.user, ConsistencyResult,
                                      mock_context={"facts": facts, "chunks": chunks}, prompt_version=p.version, trace=trace)
    except (ValueError, LLMError):
        return ConsistencyResult()
    return clean(res, {f["id"] for f in facts}, {c["id"] for c in chunks})


def check_item(client: LLMClient, item: ParsedItem, bank_hits: list[dict], trace=None) -> ConsistencyResult:
    facts = [{"id": f.id, "text": f.text, "confidence": f.confidence} for f in item.facts]
    res = check(client, item.title, facts, [h for h in bank_hits if not h["stale"]], trace)
    if trace is not None:
        trace.log("consistency", status=res.status, pairs=[p.model_dump() for p in res.pairs])
    return res


def prompt_block(res: ConsistencyResult) -> str:
    if res.status != "contradicts":
        return ""
    lines = [f"- факт {p.fact_id} против фрагмента {p.chunk_id}: {p.note}" for p in res.pairs]
    return ("CONTRADICTIONS (расхождение событие/банк установлено кодом; назови его в ai_inference и в limitations, "
            "не выбирай сторону новости):\n" + "\n".join(lines))


def limitations_issue(card: InsightCard) -> list[str]:
    if any(CONTRADICTION_WORD.search(x) for x in card.limitations):
        return []
    return ["материал расходится с позицией банка (CONTRADICTIONS), но в limitations нет строки о расхождении: "
            "добавь её со словом «расхождение» и назови обе позиции"]
