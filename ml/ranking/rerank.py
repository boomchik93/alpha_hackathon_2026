"""Переранжирование топ-N модели. Эксперимент: включается ML_RERANK=1, по умолчанию выключено.

Релевантность, которую Gemma ставит по `prompts/relevance.md`, отделяет нерелевантное намного лучше ленты
(P/R 95/97% против 85/81% при пороге 0,3), но вызов на пару стоит денег. Поэтому модель оценивает только топ-N ленты и
материалы у порога (личный скор в [MIN_SCORE - RERANK_BAND; MIN_SCORE)); итог = смесь личного скора и оценки модели.
Включать по умолчанию только по результатам `python -m ml.eval.run_eval --ranking-only --set v2 --rerank` на ручных парах.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from pydantic import BaseModel, Field

from ml.llm.client import LLMClient, LLMError, get_client
from ml.prompts import render
from ml.schemas import CompanyProfile

RERANK_N = 10
RERANK_BAND = 0.1
RERANK_WEIGHT = 0.5  # вес оценки модели в смеси; подбирается на dev (run_eval --rerank)
WORKERS = 6


class ModelRelevance(BaseModel):
    score: float = Field(ge=0, le=1)
    reasons: list[str] = []
    features: list[str] = []


def enabled() -> bool:
    return os.getenv("ML_RERANK", "0") == "1"


def candidates(scored: list, min_score: float, n: int = RERANK_N) -> list:
    """Топ-N по порядку ленты и материалы у порога отсева."""
    return scored[:n] + [s for s in scored[n:] if min_score - RERANK_BAND <= s.personal < min_score]


def model_score(profile: CompanyProfile, s, client: LLMClient) -> float | None:
    """Оценка модели 0..1 для пары; None при сбое (материал остаётся со своим скором)."""
    item = s.item.model_dump(mode="json", include={"title", "summary", "facts", "event_type", "topics", "okved_hints"})
    p = render("relevance", profile=profile.prompt_view(), item=item)
    try:
        res, _ = client.complete_json("relevance", p.system, p.user, ModelRelevance, mock_context={"reasons": s.reasons},
                                      prompt_version=p.version)
    except (ValueError, LLMError):
        return None
    return res.score


def model_scores(profile: CompanyProfile, cands: list, client: LLMClient, workers: int = WORKERS) -> dict[str, float]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(lambda s: model_score(profile, s, client), cands))
    return {s.item.item_id: v for s, v in zip(cands, out) if v is not None}


def apply(scored: list, model: dict[str, float], weight: float, beta: float) -> list:
    """Личный скор оценённых материалов становится смесью с оценкой модели; порядок пересчитывается."""
    out = []
    for s in scored:
        if s.item.item_id in model:
            personal = round((1 - weight) * s.personal + weight * model[s.item.item_id], 3)
            s = replace(s, personal=personal, score=round(beta * s.item.importance + (1 - beta) * personal, 3))
        out.append(s)
    return sorted(out, key=lambda s: -s.score)


def rerank(profile: CompanyProfile, scored: list, min_score: float, beta: float, n: int = RERANK_N,
           weight: float = RERANK_WEIGHT, client: LLMClient | None = None) -> list:
    client = client or get_client()
    return apply(scored, model_scores(profile, candidates(scored, min_score, n), client), weight, beta)
