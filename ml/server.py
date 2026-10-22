"""HTTP-обёртка над `ml.api` для бэкенда и фронта.

`uvicorn ml.server:app --port 8001`
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml import api, feedback
from ml.agent.orchestrator import InsightError
from ml.agent.trace import load_trace
from ml.errors import NotFound
from ml.roles import Role
from ml.llm.client import BudgetExceeded, LLMError, SpendLedger
from ml.schemas import CompanyProfile, Feed, FollowupAnswer, InsightCard

app = FastAPI(title="alpha-hack ML", version="0.1")


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotFound as e:
        raise HTTPException(404, str(e)) from e
    except BudgetExceeded as e:
        raise HTTPException(429, str(e)) from e
    except (InsightError, LLMError) as e:
        raise HTTPException(502, f"модель не вернула годный ответ: {str(e)[:300]}") from e


class CompanyBrief(BaseModel):
    company_id: str
    level: str
    okved_name: str | None
    revenue_rub: float | None
    region: str | None
    client_type: str | None


@app.get("/companies", response_model=list[CompanyBrief])
def companies():
    return [CompanyBrief(company_id=p.company_id, level=p.level, okved_name=p.industry.okved_name,
                         revenue_rub=p.size.revenue_rub, region=p.geo.region, client_type=p.client_type)
            for p in api.list_companies()]


@app.get("/companies/{company_id}", response_model=CompanyProfile)
def profile(company_id: str):
    return _guard(api.build_profile, company_id)


@app.get("/companies/{company_id}/feed", response_model=Feed)
def ranked_feed(company_id: str, limit: int = 20, tier: Literal["main", "all"] = "all", maybe_limit: int = 10):
    return _guard(api.get_ranked_feed, company_id, limit, tier, maybe_limit)


@app.get("/companies/{company_id}/insights/{item_id}", response_model=InsightCard)
def insight(company_id: str, item_id: str, refresh: bool = False, role: Role | None = None):
    return _guard(api.get_insight, company_id, item_id, refresh, role)


class Question(BaseModel):
    question: str


@app.post("/cards/{card_id}/followup", response_model=FollowupAnswer)
def followup(card_id: str, q: Question, role: Role | None = None):
    return _guard(api.answer_followup, card_id, q.question, role)


@app.post("/cards/{card_id}/feedback", response_model=feedback.FeedbackSaved)
def card_feedback(card_id: str, fb: feedback.FeedbackIn):
    return _guard(feedback.record, card_id, fb)


@app.get("/feedback/summary")
def feedback_summary():
    return feedback.summary()


@app.get("/traces/{trace_id}")
def trace(trace_id: str):
    t = load_trace(trace_id)
    if t is None:
        raise HTTPException(404, "trace not found")
    return t


@app.get("/health")
def health():
    from ml.config import settings
    return {"ok": True, "llm_mock": settings.llm.mock, "spent_today_rub": round(SpendLedger().spent_today(), 2),
            "budget_rub": settings.llm.daily_budget_rub}
