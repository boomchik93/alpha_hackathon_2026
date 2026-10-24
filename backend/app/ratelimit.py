"""Лимиты частоты в памяти процесса: скользящее окно на ключ.

Зачем: дневной бюджет моделей общий на команду (500 руб.), а генерация карточки и ответ на вопрос платные.
Без лимита один пользователь перебором материалов съедает бюджет всей демо. Вход ограничен по неудачным
попыткам, чтобы пароли нельзя было подбирать.

Бэкенд запускается одним процессом, поэтому счётчики в памяти достаточно; после перезапуска они обнуляются.
"""
from __future__ import annotations

import math
import time
from collections import deque

from backend.app.errors import ApiError


class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._hits: dict[tuple, deque[float]] = {}

    def _window(self, key: tuple, window_s: float) -> deque[float]:
        q = self._hits.setdefault(key, deque())
        cutoff = self._clock() - window_s
        while q and q[0] <= cutoff:
            q.popleft()
        return q

    def check(self, key: tuple, limit: int, window_s: float, message: str) -> None:
        """429 rate_limited, если за окно уже было `limit` событий; само событие не записывает."""
        q = self._window(key, window_s)
        if len(q) >= limit:
            retry = max(1, math.ceil(q[0] + window_s - self._clock()))
            raise ApiError(429, "rate_limited", message, retry_after=retry)

    def record(self, key: tuple, window_s: float) -> None:
        self._window(key, window_s).append(self._clock())

    def hit(self, key: tuple, limit: int, window_s: float, message: str) -> None:
        self.check(key, limit, window_s, message)
        self.record(key, window_s)

    def reset(self, key: tuple) -> None:
        self._hits.pop(key, None)
