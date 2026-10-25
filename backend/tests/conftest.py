"""Тесты бэкенда против настоящего `ml.server` в том же процессе, без сети и без вызовов модели.

Функции `ml.api` подменяются: в тестах нет `data/` и `var/`, а генерация карточки должна управляться
(задержка, ошибка бюджета, сбой модели). HTTP-обёртка ML, коды ошибок и схемы ответов остаются настоящими.
"""
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# До импорта ml: карточки и трассы не должны попасть в рабочий var/.
os.environ.setdefault("ML_RUNS_DIR", tempfile.mkdtemp(prefix="backend_ml_runs_"))

import pytest  # noqa: E402

from ml import api as ml_api, feedback as ml_feedback, server as ml_server  # noqa: E402
from ml.schemas import FollowupAnswer  # noqa: E402

from .fakes import SCENARIOS_TRACE, TRACE, FakeML, NotFound, feed, login, make_client, ml_transport, profile  # noqa: E402


@pytest.fixture()
def fake_ml(monkeypatch) -> FakeML:
    fake = FakeML()
    monkeypatch.setattr(ml_api, "list_companies", lambda: [profile(c) for c in ("U_1", "U_2", "U_3", "group_9")])
    monkeypatch.setattr(ml_api, "build_profile", profile)
    monkeypatch.setattr(ml_api, "get_ranked_feed", feed)
    monkeypatch.setattr(ml_api, "get_insight", fake.get_insight)
    monkeypatch.setattr(ml_api, "get_scenarios", fake.get_scenarios)
    traces = {TRACE["trace_id"]: TRACE, SCENARIOS_TRACE["trace_id"]: SCENARIOS_TRACE}
    monkeypatch.setattr(ml_server, "load_trace", traces.get)

    def followup(card_id, question, role=None):
        if not card_id.startswith("ins_"):
            raise NotFound(f"карточка {card_id} не найдена")
        return FollowupAnswer(card_id=card_id, question=question, answer="Ответ по карточке.", trace_id="tr_q")

    monkeypatch.setattr(ml_api, "answer_followup", followup)
    fake.feedback_calls = []

    def record(card_id, fb):
        fake.feedback_calls.append((card_id, fb))
        return ml_feedback.FeedbackSaved(card_id=card_id, trace_id="tr_test")

    monkeypatch.setattr(ml_feedback, "record", record)
    return fake


@pytest.fixture()
def client(fake_ml, tmp_path):
    with make_client(ml_transport(), tmp_path) as c:
        yield c


@pytest.fixture()
def alice(client) -> dict:
    return login(client, "alice")
