"""Трасса запуска: план, вызовы инструментов и LLM, проверки, повторы, стоимость."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ml.config import settings


@dataclass
class Trace:
    kind: str
    trace_id: str = field(default_factory=lambda: "tr_" + uuid.uuid4().hex[:10])
    started_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    cost_rub: float = 0.0
    retries: int = 0

    def log(self, event: str, **data: Any) -> None:
        self.events.append({"t": round(time.time() - self.started_at, 3), "event": event, **data})

    def add_cost(self, rub: float) -> None:
        self.cost_rub += rub

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "kind": self.kind,
            "started_at": self.started_at,
            "duration_s": round(time.time() - self.started_at, 3),
            "cost_rub": round(self.cost_rub, 4),
            "retries": self.retries,
            "events": self.events,
        }

    def save(self) -> None:
        d = settings.paths.traces
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{self.trace_id}.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )


def load_trace(trace_id: str) -> dict | None:
    p = settings.paths.traces / f"{trace_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
