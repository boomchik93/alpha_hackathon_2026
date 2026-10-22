"""Обратная связь по карточкам."""

import json

import pytest
from fastapi.testclient import TestClient

from ml import server
from ml.config import settings
from ml.eval.feedback_export import export
from ml.feedback import summarize
from ml.tests.conftest import needs_data


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_RUNS_DIR", str(tmp_path))  # карточки и обратная связь теста отдельно
    return TestClient(server.app)


@needs_data
def test_feedback_recorded_with_card_meta(client):
    card = client.get("/companies/U_1595/insights/n_2026_09_11_cbr_rate").json()
    r = client.post(f"/cards/{card['card_id']}/feedback", json={"rating": "down", "reason": "wrong_fact", "comment": "цифра не из источника", "role": "cfo"})
    assert r.status_code == 200 and r.json()["trace_id"] == card["trace_id"]
    row = json.loads(settings.paths.feedback.read_text(encoding="utf-8").splitlines()[-1])
    assert row["prompt_version"] == card["meta"]["prompt_version"] and row["model"] == card["meta"]["model"]
    assert row["rating"] == "down" and row["reason"] == "wrong_fact" and row["role"] == "cfo" and row["ts"]


@needs_data
def test_feedback_validation(client):
    card = client.get("/companies/U_1595/insights/n_2026_09_11_cbr_rate").json()
    url = f"/cards/{card['card_id']}/feedback"
    assert client.post(url, json={"rating": "up", "comment": "x" * 1000}).status_code == 200
    assert client.post(url, json={"rating": "up", "comment": "x" * 1001}).status_code == 422
    assert client.post(url, json={"rating": "meh"}).status_code == 422
    assert client.post(url, json={"rating": "down", "reason": "бред"}).status_code == 422
    assert client.post("/cards/ins_nope/feedback", json={"rating": "up"}).status_code == 404


@needs_data
def test_feedback_summary_by_prompt_version(client):
    card = client.get("/companies/U_1595/insights/n_2026_09_11_cbr_rate").json()
    url = f"/cards/{card['card_id']}/feedback"
    for body in ({"rating": "up"}, {"rating": "down", "reason": "obvious"}, {"rating": "down", "reason": "obvious"}, {"rating": "down"}):
        client.post(url, json=body)
    s = client.get("/feedback/summary").json()
    g = s[card["meta"]["prompt_version"]]
    assert g["total"] == 4 and g["down"] == 3 and g["down_share"] == 0.75 and g["reasons"] == {"obvious": 2, "не указана": 1}
    assert s["все"]["total"] == 4


def test_summary_groups_and_empty():
    rows = [{"prompt_version": "interpret@v8", "rating": "down", "reason": "wrong_fact"},
            {"prompt_version": "interpret@v9", "rating": "up"}]
    s = summarize(rows)
    assert s["interpret@v8"]["down_share"] == 1.0 and s["interpret@v9"]["down_share"] == 0.0 and s["все"]["total"] == 2
    assert summarize([])["все"]["down_share"] is None


def test_export_splits_by_reason():
    rows = [
        {"rating": "down", "reason": "not_relevant", "company_id": "U_1", "item_ids": ["n_a", "n_b"], "card_id": "c1", "ts": "t"},
        {"rating": "down", "reason": "wrong_fact", "card_id": "c2", "trace_id": "tr", "prompt_version": "v", "model": "m", "ts": "t"},
        {"rating": "up", "reason": "not_relevant", "company_id": "U_1", "item_ids": ["n_a"], "card_id": "c3", "ts": "t"},
        {"rating": "down", "reason": "obvious", "card_id": "c4", "ts": "t"},
    ]
    cand, queue = export(rows)
    assert [c["item_id"] for c in cand] == ["n_a"] and [q["card_id"] for q in queue] == ["c2"]
