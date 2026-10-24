"""Фоновая генерация карточек: задача в памяти процесса, статусы queued/running/done/failed.

Почему так: первая генерация карточки на настоящей модели идёт 15-30 с, а бесплатные хостинги и прокси
рвут запрос на 30-60 с. Фронт получает `job_id` и опрашивает `GET /api/jobs/{id}`.

- Одинаковые запросы (компания, материал, роль) склеиваются в одну задачу: повторный клик не тратит бюджет.
- Одновременно идёт не больше `max_parallel` генераций, остальные ждут в очереди; очередь ограничена.
- Завершённые задачи живут `ttl_s` секунд. Перезапуск процесса их теряет: карточка при этом
  остаётся в кэше ML, и новый запрос вернёт её сразу.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from backend.app.errors import ApiError

log = logging.getLogger("backend.jobs")

Status = Literal["queued", "running", "done", "failed"]
ACTIVE: tuple[Status, ...] = ("queued", "running")


@dataclass
class Job:
    key: tuple
    id: str = field(default_factory=lambda: "job_" + uuid.uuid4().hex[:12])
    status: Status = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Any = None
    error: ApiError | None = None
    task: asyncio.Task | None = field(default=None, repr=False)

    def set(self, status: Status) -> None:
        self.status, self.updated_at = status, time.time()


class JobManager:
    def __init__(self, max_parallel: int, max_queued: int, ttl_s: float):
        self._sem = asyncio.Semaphore(max_parallel)
        self._max_queued = max_queued
        self._ttl_s = ttl_s
        self._jobs: dict[str, Job] = {}
        self._active_by_key: dict[tuple, Job] = {}

    def submit(self, key: tuple, work: Callable[[], Awaitable[Any]]) -> Job:
        self._prune()
        if (job := self._active_by_key.get(key)) is not None:
            return job
        if len(self._active_by_key) >= self._max_queued:
            raise ApiError(429, "queue_full", "сейчас генерируется слишком много карточек, повторите позже", retry_after=10)
        job = Job(key=key)
        self._jobs[job.id] = job
        self._active_by_key[key] = job
        job.task = asyncio.create_task(self._run(job, work), name=job.id)
        return job

    async def _run(self, job: Job, work: Callable[[], Awaitable[Any]]) -> None:
        try:
            async with self._sem:
                job.set("running")
                job.result = await work()
            job.set("done")
        except ApiError as exc:
            job.error = exc
            job.set("failed")
        except Exception:  # noqa: BLE001 - задача не должна падать молча
            log.exception("задача %s упала", job.id)
            job.error = ApiError(500, "internal_error", "внутренняя ошибка сервера")
            job.set("failed")
        finally:
            if self._active_by_key.get(job.key) is job:
                del self._active_by_key[job.key]

    async def wait(self, job: Job, timeout_s: float) -> Job:
        """Ждёт задачу не дольше timeout_s; задача при этом не отменяется."""
        if job.task is not None and job.status in ACTIVE and timeout_s > 0:
            await asyncio.wait({job.task}, timeout=timeout_s)
        return job

    def get(self, job_id: str) -> Job | None:
        self._prune()
        return self._jobs.get(job_id)

    def _prune(self) -> None:
        cutoff = time.time() - self._ttl_s
        for jid in [j.id for j in self._jobs.values() if j.status not in ACTIVE and j.updated_at < cutoff]:
            del self._jobs[jid]

    async def shutdown(self) -> None:
        tasks = [j.task for j in self._jobs.values() if j.task is not None and not j.task.done()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
