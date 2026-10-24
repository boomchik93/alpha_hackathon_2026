"""Единый формат ошибок для фронта: `{"code", "message", "retry_after"?}` с HTTP-статусом.

Коды (поле `code`), на которые фронт может опираться:
    unauthorized       401  нет токена, токен истёк, неверный логин или пароль
    forbidden          403  нет доступа к компании или ручке
    not_found          404  компания, материал, карточка или задача не найдены
    validation_error   422  неверные параметры запроса
    rate_limited       429  превышен лимит частоты (карточки, вопросы, неудачные входы)
    queue_full         429  слишком много карточек генерируется, повторите позже
    budget_exceeded    429  дневной лимит расходов на модели исчерпан
    model_failed       502  модель не дала годный ответ, можно повторить
    ml_error           502  ML-сервис ответил неожиданно
    ml_unavailable     503  ML-сервис недоступен
    ml_timeout         504  ML-сервис не ответил вовремя
    internal_error     500  ошибка самого бэкенда
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("backend.errors")


class ErrorBody(BaseModel):
    code: str
    message: str
    retry_after: int | None = None


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after

    def body(self) -> dict:
        return ErrorBody(code=self.code, message=self.message, retry_after=self.retry_after).model_dump(exclude_none=True)

    def response(self) -> JSONResponse:
        headers = {"Retry-After": str(self.retry_after)} if self.retry_after is not None else None
        return JSONResponse(self.body(), status_code=self.status, headers=headers)


def not_found(message: str) -> ApiError:
    return ApiError(404, "not_found", message)


# Для описания ответов в OpenAPI: фронт видит схему ошибки у каждой ручки.
ERROR_RESPONSES = {s: {"model": ErrorBody} for s in (404, 422, 429, 502, 503, 504)}


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return ApiError(422, "validation_error", problems[:500]).response()

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code = "not_found" if exc.status_code == 404 else "http_error"
        return ApiError(exc.status_code, code, str(exc.detail)).response()

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.exception("необработанная ошибка %s %s", request.method, request.url.path)
        return ApiError(500, "internal_error", "внутренняя ошибка сервера").response()
