"""Ручки API для фронта под префиксом `/api`. Ответы ML отдаются как есть (модели в `ml/schemas.py`)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.app.config import Settings
from backend.app.errors import ERROR_RESPONSES, ApiError, ErrorBody, not_found
from backend.app.jobs import Job, JobManager
from backend.app.ml_client import MLClient

router = APIRouter(prefix="/api", responses=ERROR_RESPONSES)

# Идентификаторы подставляются в путь запроса к ML: только безопасные символы, без «/» и «..».
Id = Annotated[str, Path(pattern=r"^[A-Za-z0-9_\-:.]{1,128}$")]
# Роли ml/roles.py. Пока нет авторизации, роль приходит параметром; потом бэкенд будет брать её из сессии.
Role = Literal["owner", "ceo", "cfo", "treasurer", "chief_accountant", "other"]
MAX_QUESTION_CHARS = 500   # как ml/api.py, MAX_QUESTION_CHARS
MAX_COMMENT_CHARS = 1000   # как ml/feedback.py, MAX_COMMENT_CHARS


def _ml(request: Request) -> MLClient:
    return request.app.state.ml


def _jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def _settings(request: Request) -> Settings:
    return request.app.state.settings


# --- служебное -----------------------------------------------------------------

@router.get("/health")
async def health(request: Request) -> dict:
    """Бэкенд жив всегда, когда отвечает; состояние ML отдельным полем, чтобы фронт показал понятный баннер."""
    try:
        ml = await _ml(request).get("/health", timeout_s=3)
    except ApiError as exc:
        return {"ok": True, "ml_ok": False, "ml_error": exc.code}
    return {"ok": True, "ml_ok": True, "llm_mock": ml.get("llm_mock"),
            "spent_today_rub": ml.get("spent_today_rub"), "budget_rub": ml.get("budget_rub")}


# --- компании и лента ----------------------------------------------------------------

@router.get("/companies")
async def companies(request: Request) -> list[dict[str, Any]]:
    return await _ml(request).get("/companies")


@router.get("/companies/{company_id}")
async def company_profile(request: Request, company_id: Id) -> dict[str, Any]:
    return await _ml(request).get(f"/companies/{company_id}")


@router.get("/companies/{company_id}/feed")
async def company_feed(
    request: Request,
    company_id: Id,
    tier: Literal["main", "all"] = "all",
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    maybe_limit: Annotated[int, Query(ge=0, le=50)] = 10,
) -> dict[str, Any]:
    """Основная лента и блок «может быть интересно» (`entries[].tier == "maybe"`), см. ml/schemas.py, Feed."""
    return await _ml(request).get(f"/companies/{company_id}/feed",
                                  params={"tier": tier, "limit": limit, "maybe_limit": maybe_limit})


# --- карточки -----------------------------------------------------------------------

class JobView(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    poll_url: str
    created_at: datetime
    updated_at: datetime
    card: dict[str, Any] | None = Field(None, description="InsightCard, когда status=done")
    error: ErrorBody | None = Field(None, description="ошибка в общем формате, когда status=failed")


def _view(job: Job) -> JobView:
    ts = lambda t: datetime.fromtimestamp(t, timezone.utc)  # noqa: E731
    return JobView(job_id=job.id, status=job.status, poll_url=f"/api/jobs/{job.id}", created_at=ts(job.created_at),
                   updated_at=ts(job.updated_at), card=job.result if job.status == "done" else None,
                   error=job.error.body() if job.status == "failed" and job.error else None)


@router.get(
    "/companies/{company_id}/insights/{item_id}",
    responses={200: {"description": "InsightCard: карточка готова (из кэша ML или успела сгенерироваться)"},
               202: {"model": JobView, "description": "карточка генерируется: опрашивайте poll_url раз в 1-2 с"}},
)
async def insight(request: Request, company_id: Id, item_id: Id, role: Role | None = None):
    """Карточка влияния события. Готовая отдаётся сразу (200), иначе 202 с задачей.

    Ошибка генерации, случившаяся за время ожидания, отдаётся сразу в общем формате (404, 429, 502, 504).
    """
    ml, settings = _ml(request), _settings(request)

    async def work() -> dict:
        return await ml.get(f"/companies/{company_id}/insights/{item_id}", params={"role": role},
                            timeout_s=settings.ml_insight_timeout_s)

    jobs = _jobs(request)
    job = await jobs.wait(jobs.submit(("insight", company_id, item_id, role), work), settings.insight_sync_wait_s)
    if job.status == "done":
        return job.result
    if job.status == "failed" and job.error is not None:
        raise job.error
    return JSONResponse(_view(job).model_dump(mode="json"), status_code=202, headers={"Retry-After": "1"})



@router.get("/jobs/{job_id}", response_model=JobView)
async def job_status(request: Request, job_id: Id) -> JobView:
    """Состояние генерации. Ошибка самой генерации приходит в поле `error` со статусом 200: запрос опроса удался."""
    job = _jobs(request).get(job_id)
    if job is None:
        raise not_found(f"задача {job_id} не найдена или устарела")
    return _view(job)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


@router.post("/cards/{card_id}/followup")
async def followup(request: Request, card_id: Id, q: Question, role: Role | None = None) -> dict[str, Any]:
    """Ответ на уточняющий вопрос по карточке (FollowupAnswer). Цифры ответа проверяет ML."""
    if not q.question.strip():
        raise ApiError(422, "validation_error", "вопрос пустой")
    return await _ml(request).post(f"/cards/{card_id}/followup", params={"role": role}, json={"question": q.question},
                                   timeout_s=_settings(request).ml_followup_timeout_s)


class FeedbackIn(BaseModel):
    rating: Literal["up", "down"]
    reason: Literal["wrong_fact", "not_relevant", "obvious", "other"] | None = None
    comment: str | None = Field(None, max_length=MAX_COMMENT_CHARS)
    role: Role | None = None


@router.post("/cards/{card_id}/feedback")
async def feedback(request: Request, card_id: Id, fb: FeedbackIn) -> dict[str, Any]:
    """Оценка карточки «полезно / не полезно». Пишется в ML (`var/feedback.jsonl`), сводка по версиям промпта там же."""
    return await _ml(request).post(f"/cards/{card_id}/feedback", json=fb.model_dump())
