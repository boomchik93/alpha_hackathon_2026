"""Тесты бэкенда против настоящего `ml.server` в том же процессе, без сети и без вызовов модели.

Функции `ml.api` подменяются: в тестах нет `data/` и `var/`, а генерация карточки должна управляться
(задержка, ошибка бюджета, сбой модели). HTTP-обёртка ML, коды ошибок и схемы ответов остаются настоящими.
"""
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# До импорта ml: карточки и трассы не должны попасть в рабочий var/.
os.environ.setdefault("ML_RUNS_DIR", tempfile.mkdtemp(prefix="backend_ml_runs_"))

import httpx  # noqa: E402
import pytest  # noqa: E402

from ml import api as ml_api, feedback as ml_feedback, server as ml_server  # noqa: E402
from ml.schemas import Feed, FeedEntry, FollowupAnswer  # noqa: E402

from .fakes import COMPANY, ITEM, FakeML, NotFound, make_client, profile  # noqa: E402


@pytest.fixture()
def fake_ml(monkeypatch) -> FakeML:
    fake = FakeML()
    monkeypatch.setattr(ml_api, "list_companies", lambda: [profile()])
    monkeypatch.setattr(ml_api, "build_profile", profile)
    monkeypatch.setattr(ml_api, "get_ranked_feed", lambda cid, limit, tier, maybe_limit: Feed(
        company_id=profile(cid).company_id, as_of=date(2026, 10, 24), maybe_count=0,
        entries=[FeedEntry(item_id=ITEM, title="ЦБ сохранил ставку", published_at=date(2026, 10, 23),
                           source_kind="news", score=0.8, reasons=[f"limit={limit} tier={tier}"])]))
    monkeypatch.setattr(ml_api, "get_insight", fake.get_insight)

    def followup(card_id, question, role=None):
        if not card_id.startswith("ins_"):
            raise NotFound(f"карточка {card_id} не найдена")
        return FollowupAnswer(card_id=card_id, question=question, answer="Ответ по карточке.", trace_id="tr_q")

    monkeypatch.setattr(ml_api, "answer_followup", followup)

    def record(card_id, fb):
        if not card_id.startswith("ins_"):
            raise NotFound(f"карточка {card_id} не найдена")
        return ml_feedback.FeedbackSaved(card_id=card_id, trace_id="tr_test")

    monkeypatch.setattr(ml_feedback, "record", record)
    return fake


@pytest.fixture()
def client(fake_ml):
    with make_client(httpx.ASGITransport(app=ml_server.app)) as c:
        yield c

