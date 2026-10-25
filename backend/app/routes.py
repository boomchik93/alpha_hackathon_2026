"""Ручки API для фронта под префиксом `/api`. Ответы ML отдаются как есть (модели в `ml/schemas.py`),
лента дополняется признаком «новое» и фильтрами.

Все ручки, кроме `/api/health` и `/api/auth/login`, требуют `Authorization: Bearer <token>`.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path as FsPath
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Path, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.app.auth import CurrentUser, PrivilegedUser, User, effective_role, ensure_company, can_access, forbidden, \
    bearer_token, unauthorized
from backend.app.config import Settings
from backend.app.db import DB, iso
from backend.app.errors import ERROR_RESPONSES, ApiError, ErrorBody, not_found
from backend.app.jobs import Job, JobManager
from backend.app.ml_client import MLClient
from backend.app.ratelimit import RateLimiter
from backend.app.sources import pdf_path

router = APIRouter(prefix="/api", responses={401: {"model": ErrorBody}, 403: {"model": ErrorBody}, **ERROR_RESPONSES})

# Идентификаторы подставляются в путь запроса к ML: только безопасные символы, без «/» и без точки в начале («..»).
Id = Annotated[str, Path(pattern=r"^[A-Za-z0-9_\-:][A-Za-z0-9_\-:.]{0,127}$")]
Role = Literal["owner", "ceo", "cfo", "treasurer", "chief_accountant", "other"]  # ml/roles.py
MAX_QUESTION_CHARS = 500   # как ml/api.py, MAX_QUESTION_CHARS
MAX_COMMENT_CHARS = 1000   # как ml/feedback.py, MAX_COMMENT_CHARS


def _ml(request: Request) -> MLClient:
    return request.app.state.ml


def _jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def _db(request: Request) -> DB:
    return request.app.state.db


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _limiter(request: Request) -> RateLimiter:
    return request.app.state.limiter


LIMIT_WINDOW_S = 600   # окно лимитов на карточки и вопросы
LOGIN_WINDOW_S = 300   # окно лимита неудачных входов
USAGE_DAYS = 7         # окно вовлечённости на экране мониторинга


# --- служебное и вход ------------------------------------------------------------

@router.get("/health")
async def health(request: Request) -> dict:
    """Бэкенд жив всегда, когда отвечает; состояние ML отдельным полем, чтобы фронт показал понятный баннер."""
    try:
        ml = await _ml(request).get("/health", timeout_s=3)
    except ApiError as exc:
        return {"ok": True, "ml_ok": False, "ml_error": exc.code}
    return {"ok": True, "ml_ok": True, "llm_mock": ml.get("llm_mock"),
            "spent_today_rub": ml.get("spent_today_rub"), "budget_rub": ml.get("budget_rub")}


class Me(BaseModel):
    login: str
    display_name: str
    role: Role
    access: Literal["user", "expert", "admin"]
    companies: list[str] = Field(description="привязанные компании и группы; эксперт и админ видят любые")


def _me(user: User) -> Me:
    return Me(login=user.login, display_name=user.display_name, role=user.role, access=user.access, companies=list(user.companies))


class LoginIn(BaseModel):
    login: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class LoginOut(BaseModel):
    token: str
    expires_at: datetime
    user: Me


@router.post("/auth/login", response_model=LoginOut)
async def login(request: Request, body: LoginIn) -> LoginOut:
    db, settings, limiter = _db(request), _settings(request), _limiter(request)
    ip = request.client.host if request.client else "unknown"
    by_login, by_ip = ("login", body.login.lower()), ("login_ip", ip)
    message = "слишком много неудачных попыток входа, подождите"
    limiter.check(by_login, settings.limit_login_failures_per_login, LOGIN_WINDOW_S, message)
    limiter.check(by_ip, settings.limit_login_failures_per_ip, LOGIN_WINDOW_S, message)
    row = db.authenticate(body.login, body.password)
    if row is None:
        limiter.record(by_login, LOGIN_WINDOW_S)
        limiter.record(by_ip, LOGIN_WINDOW_S)
        db.audit("login_failed", detail=f"{body.login} {ip}")
        raise unauthorized("неверный логин или пароль")
    limiter.reset(by_login)
    db.prune_sessions()
    token, expires = db.create_session(row["id"], request.app.state.session_ttl)
    db.audit("login", row["id"])
    user = User(row["id"], row["login"], row["display_name"], row["role"], row["access"], tuple(db.user_companies(row["id"])))
    return LoginOut(token=token, expires_at=expires, user=_me(user))


@router.post("/auth/logout")
async def logout(request: Request, user: CurrentUser) -> dict:
    _db(request).delete_session(bearer_token(request))
    return {"ok": True}


@router.get("/me", response_model=Me)
async def me(user: CurrentUser) -> Me:
    return _me(user)


# --- компании и лента ----------------------------------------------------------------

@router.get("/companies")
async def companies(request: Request, user: CurrentUser) -> list[dict[str, Any]]:
    """Компании и группы, доступные пользователю (эксперту и админу все)."""
    return [c for c in await _ml(request).get("/companies") if await can_access(request, user, c["company_id"])]


@router.get("/companies/{company_id}")
async def company_profile(request: Request, user: CurrentUser, company_id: Id) -> dict[str, Any]:
    await ensure_company(request, user, company_id)
    return await _ml(request).get(f"/companies/{company_id}")


async def _feed(request: Request, company_id: str, tier: str, limit: int, maybe_limit: int) -> dict[str, Any]:
    return await _ml(request).get(f"/companies/{company_id}/feed",
                                  params={"tier": "main" if tier == "main" else "all", "limit": limit, "maybe_limit": maybe_limit})


@router.get("/companies/{company_id}/feed")
async def company_feed(
    request: Request,
    user: CurrentUser,
    company_id: Id,
    tier: Literal["main", "maybe", "all"] = "all",
    source_kind: Literal["news", "bank_analytics"] | None = None,
    topic: Annotated[str | None, Query(max_length=64, description="ключ темы из entries[].topics")] = None,
    date_from: date | None = None,
    date_to: date | None = None,
    hide_stale: bool = False,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    maybe_limit: Annotated[int, Query(ge=0, le=50)] = 10,
) -> dict[str, Any]:
    """Лента `Feed` из ML с фильтрами и признаком «новое» (`entries[].is_new`: пользователь ещё не открывал карточку).

    `limit` и `maybe_limit` ограничивают выдачу ML до фильтров, поэтому после фильтров записей может быть меньше.
    `new_count`: новые записи основной ленты без учёта фильтров.
    """
    await ensure_company(request, user, company_id)
    feed = await _feed(request, company_id, tier, limit, maybe_limit)
    viewed = _db(request).viewed_items(user.id, company_id)
    entries = [{**e, "is_new": e["item_id"] not in viewed} for e in feed["entries"]]
    new_count = sum(1 for e in entries if e["is_new"] and e.get("tier", "main") == "main")

    def keep(e: dict) -> bool:
        published = date.fromisoformat(e["published_at"])
        return ((tier == "all" or e.get("tier", "main") == tier)
                and (source_kind is None or e["source_kind"] == source_kind)
                and (topic is None or topic in e.get("topics", []))
                and (date_from is None or published >= date_from)
                and (date_to is None or published <= date_to)
                and not (hide_stale and e.get("stale")))

    entries = [e for e in entries if keep(e)]
    return {**feed, "entries": entries, "maybe_count": sum(1 for e in entries if e.get("tier") == "maybe"),
            "new_count": new_count, "fetched_at": iso(datetime.now(timezone.utc))}


@router.get("/me/summary")
async def me_summary(request: Request, user: CurrentUser,
                     company_id: Annotated[str | None, Query(pattern=r"^[A-Za-z0-9_\-:][A-Za-z0-9_\-:.]{0,127}$")] = None) -> dict:
    """Виджет главного экрана (К09): сколько новых материалов в основной ленте и главный из них. Карточки не генерируются."""
    company_id = company_id or (user.companies[0] if user.companies else None)
    if company_id is None:
        raise ApiError(422, "validation_error", "укажите company_id: к пользователю не привязана компания")
    await ensure_company(request, user, company_id)
    feed = await _feed(request, company_id, "main", 20, 0)
    viewed = _db(request).viewed_items(user.id, company_id)
    new = [e for e in feed["entries"] if e["item_id"] not in viewed]
    top = (new or feed["entries"] or [None])[0]
    return {
        "company_id": company_id,
        "new_count": len(new),
        "top": None if top is None else {k: top[k] for k in ("item_id", "title", "score", "published_at")} | {
            "is_new": top["item_id"] not in viewed},
        "as_of": feed["as_of"],
        "updated_at": iso(datetime.now(timezone.utc)),
    }


# --- карточки -----------------------------------------------------------------------

class JobView(BaseModel):
    job_id: str
    kind: Literal["insight", "scenarios"] = Field("insight", description="что генерируется: карточка или набор сценариев")
    status: Literal["queued", "running", "done", "failed"]
    poll_url: str
    created_at: datetime
    updated_at: datetime
    card: dict[str, Any] | None = Field(None, description="InsightCard, когда kind=insight и status=done")
    scenarios: dict[str, Any] | None = Field(None, description="ScenarioSet, когда kind=scenarios и status=done")
    error: ErrorBody | None = Field(None, description="ошибка в общем формате, когда status=failed")


def _view(job: Job) -> JobView:
    ts = lambda t: datetime.fromtimestamp(t, timezone.utc)  # noqa: E731
    kind = job.key[0]  # ключ задачи начинается с вида: ("insight", ...) или ("scenarios", ...)
    done = job.status == "done"
    return JobView(job_id=job.id, kind=kind, status=job.status, poll_url=f"/api/jobs/{job.id}", created_at=ts(job.created_at),
                   updated_at=ts(job.updated_at), card=job.result if done and kind == "insight" else None,
                   scenarios=job.result if done and kind == "scenarios" else None,
                   error=job.error.body() if job.status == "failed" and job.error else None)


@router.get(
    "/companies/{company_id}/insights/{item_id}",
    responses={200: {"description": "InsightCard: карточка готова (из кэша ML или успела сгенерироваться)"},
               202: {"model": JobView, "description": "карточка генерируется: опрашивайте poll_url раз в 1-2 с"}},
)
async def insight(request: Request, user: CurrentUser, company_id: Id, item_id: Id,
                  role: Annotated[Role | None, Query(description="только эксперт и админ: карточка для другой роли")] = None):
    """Карточка влияния события для роли пользователя. Готовая отдаётся сразу (200), иначе 202 с задачей.

    Ошибка генерации, случившаяся за время ожидания, отдаётся сразу в общем формате (404, 429, 502, 504).
    """
    await ensure_company(request, user, company_id)
    ml, settings, db = _ml(request), _settings(request), _db(request)
    card_role = effective_role(user, role)

    async def work() -> dict:
        card = await ml.get(f"/companies/{company_id}/insights/{item_id}", params={"role": card_role},
                            timeout_s=settings.ml_insight_timeout_s)
        db.save_card(card, company_id, item_id)  # по card_id потом проверяется доступ к вопросу, оценке и трассе
        return card

    db.audit("insight", user.id, company_id=company_id, item_id=item_id, detail=card_role)
    jobs = _jobs(request)
    key = ("insight", company_id, item_id, card_role)
    # Лимит только на новую генерацию: присоединение к идущей и повторное открытие готовой карточки бесплатны.
    if jobs.active(key) is None and not db.has_card(company_id, item_id, card_role):
        _limiter(request).hit(("insight", user.id), settings.limit_insights_per_10m, LIMIT_WINDOW_S,
                              "слишком много новых карточек за 10 минут, подождите")
    job = jobs.submit(key, work)
    job.viewers.add(user.id)
    await jobs.wait(job, settings.insight_sync_wait_s)
    if job.status == "failed" and job.error is not None:
        raise job.error
    # Материал просмотрен, когда карточка открыта или генерируется: несуществующий (404) в историю не попадает.
    db.mark_viewed(user.id, company_id, item_id)
    if job.status == "done":
        return job.result
    return JSONResponse(_view(job).model_dump(mode="json"), status_code=202, headers={"Retry-After": "1"})


@router.get(
    "/companies/{company_id}/insights/{item_id}/scenarios",
    responses={200: {"description": "ScenarioSet: три ветки (good, base, bad) и разбор решений; готов из кэша ML"},
               202: {"model": JobView, "description": "сценарии строятся: опрашивайте poll_url раз в 2-3 с, результат в поле scenarios"},
               422: {"model": ErrorBody, "description": "code=scenarios_not_applicable: связь карточки с компанией слабая, кнопку скрыть"}},
)
async def scenarios(request: Request, user: CurrentUser, company_id: Id, item_id: Id,
                    role: Annotated[Role | None, Query(description="только эксперт и админ: сценарии для другой роли")] = None):
    """Сценарии для карточки: развилка, три ветки (good, base, bad) и разбор решений по ним. Готовые отдаются сразу (200),
    иначе 202 с задачей, как у карточки.

    Сценарии строятся поверх карточки той же роли: если карточки ещё нет, ML сначала делает её, бэкенд её сохраняет, и вопрос
    по ней потом принимается. Для карточек со слабой связью (relevance ниже 0,3) ответ 422 `scenarios_not_applicable`.
    """
    await ensure_company(request, user, company_id)
    ml, settings, db = _ml(request), _settings(request), _db(request)
    card_role = effective_role(user, role)

    async def work() -> dict:
        card = await ml.get(f"/companies/{company_id}/insights/{item_id}", params={"role": card_role},
                            timeout_s=settings.ml_insight_timeout_s)
        db.save_card(card, company_id, item_id)
        result = await ml.get(f"/companies/{company_id}/insights/{item_id}/scenarios", params={"role": card_role},
                              timeout_s=settings.ml_scenarios_timeout_s)
        db.save_scenarios(result, company_id, item_id)  # по set_id потом проверяется доступ к трассе
        return result

    db.audit("scenarios", user.id, company_id=company_id, item_id=item_id, detail=card_role)
    jobs = _jobs(request)
    key = ("scenarios", company_id, item_id, card_role)
    if jobs.active(key) is None and not db.has_scenarios(company_id, item_id, card_role):  # как у карточки
        _limiter(request).hit(("scenarios", user.id), settings.limit_scenarios_per_10m, LIMIT_WINDOW_S,
                              "слишком много новых сценариев за 10 минут, подождите")
    job = jobs.submit(key, work)
    job.viewers.add(user.id)
    await jobs.wait(job, settings.insight_sync_wait_s)
    if job.status == "failed" and job.error is not None:
        raise job.error
    if job.status == "done":
        return job.result
    return JSONResponse(_view(job).model_dump(mode="json"), status_code=202, headers={"Retry-After": "2"})


@router.get("/jobs/{job_id}", response_model=JobView)
async def job_status(request: Request, user: CurrentUser, job_id: Id) -> JobView:
    """Состояние генерации. Ошибка самой генерации приходит в поле `error` со статусом 200: запрос опроса удался."""
    job = _jobs(request).get(job_id)
    if job is None or (user.id not in job.viewers and not user.privileged):
        raise not_found(f"задача {job_id} не найдена или устарела")
    return _view(job)


async def _card_for(request: Request, user: User, card_id: str):
    """Карточка, которую бэкенд уже отдавал, и проверка доступа к её компании."""
    card = _db(request).card(card_id)
    if card is None:
        raise not_found(f"карточка {card_id} не найдена: сначала откройте её через ленту")
    await ensure_company(request, user, card["company_id"])
    return card


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


@router.post("/cards/{card_id}/followup")
async def followup(request: Request, user: CurrentUser, card_id: Id, q: Question,
                   role: Annotated[Role | None, Query(description="только эксперт и админ")] = None) -> dict[str, Any]:
    """Ответ на уточняющий вопрос по карточке (FollowupAnswer). Цифры ответа проверяет ML."""
    if not q.question.strip():
        raise ApiError(422, "validation_error", "вопрос пустой")
    card = await _card_for(request, user, card_id)
    card_role = effective_role(user, role)
    _limiter(request).hit(("followup", user.id), _settings(request).limit_followups_per_10m, LIMIT_WINDOW_S,
                          "слишком много вопросов за 10 минут, подождите")
    answer = await _ml(request).post(f"/cards/{card_id}/followup", params={"role": card_role},
                                     json={"question": q.question}, timeout_s=_settings(request).ml_followup_timeout_s)
    _db(request).audit("followup", user.id, company_id=card["company_id"], card_id=card_id,
                       trace_id=answer.get("trace_id"), detail=q.question)
    return answer


class FeedbackIn(BaseModel):
    rating: Literal["up", "down"]
    reason: Literal["wrong_fact", "not_relevant", "obvious", "other"] | None = None
    comment: str | None = Field(None, max_length=MAX_COMMENT_CHARS)


@router.post("/cards/{card_id}/feedback")
async def feedback(request: Request, user: CurrentUser, card_id: Id, fb: FeedbackIn) -> dict[str, Any]:
    """Оценка «полезно / не полезно». Пишется в базу бэкенда (с ролью и уровнем доступа: оценки экспертов отдельно)
    и дублируется в ML для сводки по версии промпта. Если ML недоступен, оценка всё равно сохранена: `synced_to_ml=false`."""
    card = await _card_for(request, user, card_id)
    db = _db(request)
    fid = db.add_feedback(user.id, user.access, user.role, card, fb.rating, fb.reason, fb.comment)
    db.audit("feedback", user.id, company_id=card["company_id"], card_id=card_id, trace_id=card["trace_id"], detail=fb.rating)
    try:
        await _ml(request).post(f"/cards/{card_id}/feedback", json={**fb.model_dump(), "role": user.role})
    except ApiError:
        return {"ok": True, "card_id": card_id, "synced_to_ml": False}
    db.mark_feedback_synced(fid)
    return {"ok": True, "card_id": card_id, "synced_to_ml": True}


@router.get("/feedback/summary")
async def feedback_summary(request: Request, user: PrivilegedUser) -> dict[str, Any]:
    """Мониторинг (К08): доля «не полезно» и причины отдельно для пользователей и экспертов, с разбивкой по ролям.
    `ml`: сводка ML по версиям промпта (null, если ML недоступен)."""
    groups: dict[str, list[dict]] = {}
    for r in _db(request).feedback_rows():
        groups.setdefault("expert" if r["access"] in ("expert", "admin") else "user", []).append(r)

    def stats(rows: list[dict]) -> dict:
        down = [r for r in rows if r["rating"] == "down"]
        return {"total": len(rows), "down": len(down), "down_share": round(len(down) / len(rows), 3) if rows else None,
                "reasons": dict(Counter(r["reason"] or "не указана" for r in down))}

    out: dict[str, Any] = {name: {**stats(rows), "by_role": {role: stats([r for r in rows if r["role"] == role])
                                                             for role in sorted({r["role"] for r in rows})}}
                           for name, rows in groups.items()}
    try:
        out["ml"] = await _ml(request).get("/feedback/summary")
    except ApiError:
        out["ml"] = None
    # Вовлечённость: что пользователи делают после открытия карточки (вопрос, первоисточник, трасса, оценка).
    actions = _db(request).audit_counts(USAGE_DAYS)
    opened = actions.get("insight", {}).get("count", 0)
    after = sum(actions.get(a, {}).get("count", 0) for a in ("followup", "source", "trace", "feedback"))
    out["usage"] = {"days": USAGE_DAYS, "actions": actions,
                    "actions_per_card": round(after / opened, 2) if opened else None}
    # Поступление материалов по дням: остановка сборщика новостей видна в тот же день.
    try:
        out["items_by_day"] = await _ml(request).get("/items/daily")
    except ApiError:
        out["items_by_day"] = None
    return out


def _user_scenarios_trace(trace: dict, scenarios: dict) -> dict:
    """Упрощённый вид трассы сценариев: сколько попыток и замечаний на каждом этапе, источники, модель. Без промптов и ответов модели."""
    events = trace.get("events", [])

    def stage(name: str, **match) -> list[dict]:
        return [{"attempt": e.get("attempt"), "issues": len(e.get("issues") or [])} for e in events
                if e.get("event") == name and all(e.get(k) == v for k, v in match.items())]

    return {
        "trace_id": trace.get("trace_id"),
        "kind": trace.get("kind"),
        "duration_s": trace.get("duration_s"),
        "retries": trace.get("retries", 0),
        "fork": stage("scenario_fork"),
        "branches": {kind: stage("scenario_branch", kind=kind) for kind in ("good", "base", "bad")},
        "analysis": stage("scenario_analysis"),
        "sources": [{k: s.get(k) for k in ("id", "kind", "title", "published_at", "doc_id", "page", "url")}
                    for s in scenarios.get("sources", [])],
        "model": scenarios.get("meta", {}).get("model"),
        "prompt_version": scenarios.get("meta", {}).get("prompt_version"),
    }


@router.get("/scenarios/{set_id}/trace")
async def scenarios_trace(request: Request, user: CurrentUser, set_id: Id) -> dict[str, Any]:
    """Трасса набора сценариев. Пользователь видит упрощённый вид; эксперт и админ ещё и полную трассу с расходом (`raw`)."""
    row = _db(request).scenarios(set_id)
    if row is None:
        raise not_found(f"сценарии {set_id} не найдены: сначала откройте их через карточку")
    await ensure_company(request, user, row["company_id"])
    trace = await _ml(request).get(f"/traces/{row['trace_id']}")
    out = _user_scenarios_trace(trace, json.loads(row["set_json"]))
    if user.privileged:
        out["cost_rub"] = trace.get("cost_rub")
        out["raw"] = trace
    _db(request).audit("trace", user.id, company_id=row["company_id"], item_id=row["item_id"], card_id=row["card_id"],
                       trace_id=row["trace_id"], detail=set_id)
    return out


# --- трасса и первоисточник ---------------------------------------------------------------

def _user_trace(trace: dict, card: dict) -> dict:
    """Упрощённый вид для пользователя (К12): план, инструменты, повторы, источники и признаки профиля.
    Только перечисленные поля: аргументы инструментов, промпты и ответы модели сюда не попадают."""
    events = trace.get("events", [])
    plan = next((e.get("steps", []) for e in events if e.get("event") == "plan"), [])
    return {
        "trace_id": trace.get("trace_id"),
        "kind": trace.get("kind"),
        "duration_s": trace.get("duration_s"),
        "retries": trace.get("retries", 0),
        "plan": [{"tool": s.get("tool"), "why": s.get("why")} for s in plan],
        "tool_calls": [{"tool": e.get("tool"), "ms": e.get("ms")} for e in events if e.get("event") == "tool_call"],
        "replans": [{"trigger": e.get("trigger"), "why": e.get("why")} for e in events if e.get("event") == "replan"],
        "checks": [{"attempt": e.get("attempt"), "issues": len(e.get("issues") or [])} for e in events if e.get("event") == "verify"],
        "sources": [{k: s.get(k) for k in ("id", "kind", "title", "published_at", "doc_id", "page", "url")}
                    for s in card.get("sources", [])],
        "company_features_used": card.get("why_it_matters", {}).get("company_features_used", []),
        "model": card.get("meta", {}).get("model"),
        "prompt_version": card.get("meta", {}).get("prompt_version"),
    }


@router.get("/cards/{card_id}/trace")
async def card_trace(request: Request, user: CurrentUser, card_id: Id) -> dict[str, Any]:
    """Трасса карточки. Пользователь видит упрощённый вид; эксперт и админ ещё и полную трассу с расходом (`raw`)."""
    card_row = await _card_for(request, user, card_id)
    trace = await _ml(request).get(f"/traces/{card_row['trace_id']}")
    out = _user_trace(trace, json.loads(card_row["card_json"]))
    if user.privileged:
        out["cost_rub"] = trace.get("cost_rub")
        out["raw"] = trace
    _db(request).audit("trace", user.id, company_id=card_row["company_id"], card_id=card_id, trace_id=card_row["trace_id"])
    return out


@router.get("/sources/{doc_id}/pdf", response_class=FileResponse,
            responses={200: {"content": {"application/pdf": {}}, "description": "PDF аналитики банка"}})
async def source_pdf(request: Request, user: CurrentUser, doc_id: Id):
    """PDF аналитики банка по `doc_id` из `SourceRef`; страница из `SourceRef.page` открывается фрагментом `#page=N`.
    Ссылка требует токен: фронт скачивает файл с заголовком и открывает blob-ссылку с `#page=N`."""
    path = pdf_path(FsPath(_settings(request).analytics_dir), doc_id)
    if path is None:
        raise not_found(f"документ {doc_id} не найден")
    _db(request).audit("source", user.id, detail=doc_id)
    return FileResponse(path, media_type="application/pdf", filename=path.name, content_disposition_type="inline")


@router.get("/admin/audit")
async def audit_log(request: Request, user: PrivilegedUser, limit: Annotated[int, Query(ge=1, le=1000)] = 200) -> list[dict]:
    """Журнал обращений: кто, какую компанию и карточку открыл, какой вопрос задал, какой trace_id. Только админ."""
    if user.access != "admin":
        raise forbidden("журнал доступен только администратору")
    return _db(request).audit_rows(limit)
