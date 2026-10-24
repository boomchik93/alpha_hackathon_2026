"""HTTP-клиент ML-сервиса (`ml/server.py`). Ответы ML переводятся в единый формат ошибок бэкенда.

Ответы не перепроверяются по `ml.schemas`: ML-сервис уже отдаёт их через response_model,
а бэкенд не импортирует пакет `ml`, чтобы жить в отдельном лёгком контейнере.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from backend.app.errors import ApiError

log = logging.getLogger("backend.ml")

MSK = timezone(timedelta(hours=3))


def seconds_until_budget_reset(now: datetime | None = None) -> int:
    """Бюджет у организаторов сбрасывается в 3:00 МСК (ml/llm/client.py, budget_day)."""
    now = (now or datetime.now(timezone.utc)).astimezone(MSK)
    reset = now.replace(hour=3, minute=0, second=0, microsecond=0)
    if reset <= now:
        reset += timedelta(days=1)
    return int((reset - now).total_seconds())


def _detail(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail if detail is not None else resp.text)[:300]


def _to_api_error(resp: httpx.Response) -> ApiError:
    detail = _detail(resp)
    if resp.status_code == 404:
        return ApiError(404, "not_found", detail or "не найдено")
    if resp.status_code == 429:
        return ApiError(429, "budget_exceeded", "дневной лимит расходов на модели исчерпан, доступны сохранённые карточки",
                        retry_after=seconds_until_budget_reset())
    if resp.status_code == 422:
        return ApiError(422, "validation_error", detail)
    log.warning("ML %s %s -> %s: %s", resp.request.method, resp.request.url.path, resp.status_code, detail)
    if resp.status_code == 502:
        # Текст ошибки модели остаётся в логах: пользователю он ничего не скажет.
        return ApiError(502, "model_failed", "модель не дала годный ответ, повторите запрос")
    return ApiError(502, "ml_error", f"ML-сервис ответил с ошибкой {resp.status_code}")


class MLClient:
    def __init__(self, base_url: str, timeout_s: float, transport: httpx.AsyncBaseTransport | None = None):
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def request(self, method: str, path: str, *, params: dict | None = None, json: Any = None,
                      timeout_s: float | None = None) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        kwargs: dict[str, Any] = {"params": params, "json": json}
        if timeout_s is not None:
            kwargs["timeout"] = timeout_s
        try:
            resp = await self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            log.warning("ML %s %s: таймаут", method, path)
            raise ApiError(504, "ml_timeout", "ML-сервис не ответил вовремя, повторите запрос") from exc
        except httpx.TransportError as exc:
            log.warning("ML %s %s: %s", method, path, exc)
            raise ApiError(503, "ml_unavailable", "ML-сервис недоступен") from exc
        if resp.is_error:
            raise _to_api_error(resp)
        try:
            return resp.json()
        except ValueError as exc:
            raise ApiError(502, "ml_error", "ML-сервис вернул не JSON") from exc

    async def get(self, path: str, **kwargs) -> Any:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> Any:
        return await self.request("POST", path, **kwargs)
