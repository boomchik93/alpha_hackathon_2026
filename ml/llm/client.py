"""Тонкий клиент LLM: OpenAI-совместимый chat/completions.

Адрес, ключ и имена моделей берутся из `.env`. Клиент:
- пускает только модели из списка кейса;
- кэширует ответы на диск;
- считает токены и рубли, пишет журнал расходов и останавливает платные
  вызовы при приближении к дневному лимиту;
- возвращает ответ, провалидированный pydantic-схемой, с одной попыткой
  исправления невалидного JSON.

Без `LLM_BASE_URL` работает мок (`ml.llm.mock`), остальной код от этого не меняется.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ml.config import settings
from ml.llm import router
from ml.llm.cache import DiskCache, make_key

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Руб. за 1M токенов (вход, выход). Заглушка до получения тарифов от организаторов:
# завышено, чтобы счётчик ошибался в безопасную сторону.
DEFAULT_PRICE_RUB_PER_M = (60.0, 240.0)


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass
class CallInfo:
    task: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_rub: float = 0.0
    cached: bool = False
    mock: bool = False
    latency_s: float = 0.0
    repaired: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _prices() -> dict[str, tuple[float, float]]:
    raw = os.getenv("LLM_PRICES_RUB_PER_M", "").strip()
    if not raw:
        return {}
    return {k: (float(v[0]), float(v[1])) for k, v in json.loads(raw).items()}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = _prices().get(model, DEFAULT_PRICE_RUB_PER_M)
    return (prompt_tokens * pin + completion_tokens * pout) / 1e6


def extract_json(text: str) -> dict:
    """Достаёт JSON-объект из ответа модели (с ```json ... ``` или без)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("в ответе нет JSON-объекта")
        text = text[start : end + 1]
    return json.loads(text)


MSK = timezone(timedelta(hours=3))


def budget_day() -> str:
    """Бюджет у организаторов сбрасывается в 3:00 МСК: сутки считаем от этого момента."""
    return (datetime.now(MSK) - timedelta(hours=3)).date().isoformat()


class SpendLedger:
    """Журнал расходов за день (JSONL), общий для всех процессов команды на этой машине.

    Ключ общий на команду: расходы коллег здесь не видны, поэтому лимит в
    клиенте страхует только от собственных прогонов.
    """

    def __init__(self) -> None:
        self.path = settings.paths.spend_log

    def spent_today(self) -> float:
        if not self.path.exists():
            return 0.0
        today = budget_day()
        total = 0.0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("day") == today:
                total += rec.get("cost_rub", 0.0)
        return total

    _lock = threading.Lock()

    def record(self, info: CallInfo) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"day": budget_day(), "ts": time.time(), **info.as_dict()}
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def check(self) -> None:
        limit = settings.llm.daily_budget_rub * settings.llm.budget_stop_ratio
        spent = self.spent_today()
        if spent >= limit:
            raise BudgetExceeded(f"дневной расход {spent:.2f} руб. достиг порога {limit:.2f} руб.")


MockResponder = Callable[[dict[str, Any]], dict]


class LLMClient:
    def __init__(self, mock: bool | None = None, use_cache: bool = True, cache_only: bool = False):
        self.mock = settings.llm.mock if mock is None else mock
        # cache_only: только ответы из кэша, без платных вызовов (прогоны без LLM).
        self.cache_only = cache_only
        self.cache = DiskCache(settings.paths.llm_cache) if use_cache else None
        self.ledger = SpendLedger()
        self._http: httpx.Client | None = None

    # -- низкий уровень --------------------------------------------------

    def _client(self) -> httpx.Client:
        if self._http is None:
            if not settings.llm.base_url:
                raise LLMError("LLM_BASE_URL не задан")
            self._http = httpx.Client(
                base_url=settings.llm.base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {settings.llm.api_key}"},
                timeout=settings.llm.timeout_s,
            )
        return self._http

    def _chat(self, model: str, messages: list[dict], temperature: float, max_tokens: int,
              reasoning: str | None = None, json_mode: bool = True) -> tuple[str, dict]:
        if model not in router.allowed_model_names():
            raise LLMError(f"модель {model!r} не входит в список разрешённых кейсом")
        self.ledger.check()
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning is not None:
            body["reasoning_effort"] = reasoning
        if json_mode and os.getenv("LLM_JSON_MODE", "1") == "1":
            body["response_format"] = {"type": "json_object"}
        delay = 2.0
        for attempt in range(4):
            try:
                r = self._client().post("/chat/completions", json=body)
            except httpx.TransportError as e:
                if attempt == 3:
                    raise LLMError(f"сеть: {e}") from e
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 400 and "response_format" in body:
                # Провайдер может не поддерживать json-режим: повторяем без него.
                body.pop("response_format")
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code >= 400:
                if "budget has been exceeded" in r.text.lower():
                    raise BudgetExceeded("бюджет ключа исчерпан, сброс в 3:00 МСК")
                raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            if not content.strip() and choice.get("finish_reason") == "length":
                raise LLMError(f"пустой ответ: max_tokens={max_tokens} ушли на рассуждения или обрезаны")
            return content, data.get("usage", {})
        raise LLMError("исчерпаны повторы")

    # -- основной вход ---------------------------------------------------

    def complete_json(
        self,
        task: str,
        system: str,
        user: str,
        schema: type[T],
        *,
        mock_context: dict[str, Any] | None = None,
        prompt_version: str = "v0",
        trace: Any | None = None,
    ) -> tuple[T, CallInfo]:
        rt = router.route(task)
        model = router.model_name(rt.family)
        info = CallInfo(task=task, model=model, mock=self.mock)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        key = make_key(task, model, prompt_version, rt.temperature, messages, schema.__name__, self.mock)

        if self.cache and (hit := self.cache.get(key)) is not None:
            info.cached = True
            result = schema.model_validate(hit)
            self._log(trace, info)
            return result, info

        if self.cache_only:
            raise LLMError(f"cache_only: нет ответа в кэше для задачи {task}")
        t0 = time.time()
        if self.mock:
            from ml.llm.mock import respond

            payload = respond(task, mock_context or {})
            result = schema.model_validate(payload)
        else:
            text, usage = self._chat(model, messages, rt.temperature, rt.max_tokens, rt.reasoning)
            self._account(info, usage)
            try:
                result = schema.model_validate(extract_json(text))
            except (ValueError, ValidationError) as err:
                # Одна попытка исправления: возвращаем модели её ответ и ошибку.
                info.repaired = True
                fix = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"Ответ не прошёл проверку схемы:\n{err}\nВерни исправленный JSON целиком, без пояснений."},
                ]
                text, usage = self._chat(model, fix, rt.temperature, rt.max_tokens, rt.reasoning)
                self._account(info, usage)
                result = schema.model_validate(extract_json(text))
        info.latency_s = round(time.time() - t0, 3)

        if self.cache:
            self.cache.set(key, result.model_dump(mode="json"))
        if not self.mock:
            self.ledger.record(info)
        self._log(trace, info)
        return result, info

    # -- картинки --------------------------------------------------------

    def _chat_stream(self, model: str, messages: list[dict], temperature: float, max_tokens: int,
                     stop_check: Callable[[str], str | None] | None) -> tuple[str, dict, str | None]:
        """Ответ потоком: stop_check вызывается на растущем тексте и может оборвать генерацию.

        Нужен для VL-модели: на плотных графиках она зацикливается и без обрыва съедает весь max_tokens.
        Возвращает (текст, usage, причина обрыва). Причина «max_tokens», если ответ обрезан лимитом.
        """
        if model not in router.allowed_model_names():
            raise LLMError(f"модель {model!r} не входит в список разрешённых кейсом")
        self.ledger.check()
        body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
        delay = 2.0
        for attempt in range(4):
            try:
                with self._client().stream("POST", "/chat/completions", json=body) as r:
                    if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                        r.read()
                        time.sleep(delay)
                        delay *= 2
                        continue
                    if r.status_code >= 400:
                        r.read()
                        if "budget has been exceeded" in r.text.lower():
                            raise BudgetExceeded("бюджет ключа исчерпан, сброс в 3:00 МСК")
                        raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
                    parts: list[str] = []
                    usage: dict = {}
                    reason: str | None = None
                    deltas = last_check = 0
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        d = json.loads(payload)
                        usage = d.get("usage") or usage
                        for ch in d.get("choices") or []:
                            if piece := (ch.get("delta") or {}).get("content"):
                                parts.append(piece)
                                deltas += 1
                            if ch.get("finish_reason") == "length":
                                reason = "max_tokens"
                        if stop_check and deltas - last_check >= 25:
                            last_check = deltas
                            if (why := stop_check("".join(parts))) is not None:
                                reason = why
                                break
                    if not usage:
                        # Обрыв: провайдер не прислал расход. Оцениваем: картинка страницы около 3000 токенов.
                        usage = {"prompt_tokens": 3000, "completion_tokens": deltas}
                    return "".join(parts), usage, reason
            except httpx.TransportError as e:
                if attempt == 3:
                    raise LLMError(f"сеть: {e}") from e
                time.sleep(delay)
                delay *= 2
        raise LLMError("исчерпаны повторы")

    def complete_vision(
        self,
        task: str,
        prompt: str,
        image: bytes,
        *,
        mime: str = "image/png",
        stop_check: Callable[[str], str | None] | None = None,
        prompt_version: str = "v0",
        mock_text: str = "",
        trace: Any | None = None,
    ) -> tuple[str, str | None, CallInfo]:
        """Текст по картинке (не JSON). Возвращает (текст, причина обрыва или None, информация о вызове)."""
        rt = router.route(task)
        model = router.model_name(rt.family)
        info = CallInfo(task=task, model=model, mock=self.mock)
        key = make_key(task, model, prompt_version, rt.temperature, rt.max_tokens, hashlib.sha256(image).hexdigest(),
                       hashlib.sha256(prompt.encode()).hexdigest(), "vision", self.mock)
        if self.cache and (hit := self.cache.get(key)) is not None:
            info.cached = True
            self._log(trace, info)
            return hit["text"], hit.get("aborted"), info
        if self.cache_only:
            raise LLMError(f"cache_only: нет ответа в кэше для задачи {task}")
        t0 = time.time()
        aborted: str | None = None
        if self.mock:
            text = mock_text
        else:
            url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                     {"type": "image_url", "image_url": {"url": url}}]}]
            text, usage, aborted = self._chat_stream(model, messages, rt.temperature, rt.max_tokens, stop_check)
            self._account(info, usage)
        info.latency_s = round(time.time() - t0, 3)
        if self.cache:
            self.cache.set(key, {"text": text, "aborted": aborted})
        if not self.mock:
            self.ledger.record(info)
        self._log(trace, info)
        return text, aborted, info

    def _account(self, info: CallInfo, usage: dict) -> None:
        pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        info.prompt_tokens += pt
        info.completion_tokens += ct
        # Прокси (LiteLLM) отдаёт точную стоимость в рублях в usage.cost; оценка только запасной путь.
        cost = usage.get("cost")
        info.cost_rub += float(cost) if cost is not None else estimate_cost(info.model, pt, ct)

    @staticmethod
    def _log(trace: Any | None, info: CallInfo) -> None:
        if trace is not None:
            trace.log("llm_call", **info.as_dict())
            trace.add_cost(info.cost_rub)


_default: LLMClient | None = None


def get_client() -> LLMClient:
    global _default
    if _default is None:
        _default = LLMClient()
    return _default
