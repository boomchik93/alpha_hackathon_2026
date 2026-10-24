"""API бэкенда: прокси к ML, единый формат ошибок, фоновая генерация карточек."""
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from backend.app.ml_client import seconds_until_budget_reset
from ml.agent.orchestrator import InsightError
from ml.llm.client import BudgetExceeded
from ml import server as ml_server

from .fakes import COMPANY, ITEM, make_client

INSIGHT = f"/api/companies/{COMPANY}/insights/{ITEM}"


def poll(client, job: dict, timeout_s: float = 5) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = client.get(job["poll_url"]).json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"задача не завершилась: {job}")


def assert_error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["code"] == code and body["message"], body
    return body


# --- прокси -----------------------------------------------------------------

def test_companies_profile_and_feed(client):
    assert client.get("/api/companies").json()[0]["company_id"] == COMPANY
    assert client.get(f"/api/companies/{COMPANY}").json()["bank_products"]["loan"] is True
    feed = client.get(f"/api/companies/{COMPANY}/feed", params={"tier": "main", "limit": 5}).json()
    assert feed["entries"][0]["item_id"] == ITEM
    assert feed["entries"][0]["reasons"] == ["limit=5 tier=main"]  # параметры дошли до ML


def test_unknown_company_is_404_in_common_format(client):
    body = assert_error(client.get("/api/companies/U_404"), 404, "not_found")
    assert "U_404" in body["message"]
    assert_error(client.get("/api/companies/U_404/feed"), 404, "not_found")


def test_validation_errors_in_common_format(client):
    assert_error(client.get(f"/api/companies/{COMPANY}/feed", params={"limit": 0}), 422, "validation_error")
    assert_error(client.get(f"/api/companies/{COMPANY}/feed", params={"tier": "x"}), 422, "validation_error")
    assert_error(client.get(INSIGHT, params={"role": "king"}), 422, "validation_error")
    assert_error(client.get("/api/companies/U$1"), 422, "validation_error")  # в путь к ML только безопасные символы
    assert_error(client.get("/api/nope"), 404, "not_found")


# --- карточки ---------------------------------------------------------------

def test_ready_card_returns_200_with_role(client, fake_ml):
    resp = client.get(INSIGHT, params={"role": "cfo"})
    assert resp.status_code == 200
    assert resp.json()["card_id"] == f"ins_{COMPANY}_{ITEM}_cfo" and resp.json()["role"] == "cfo"


def test_slow_card_goes_to_background_and_is_deduplicated(client, fake_ml):
    fake_ml.delay_s = 2.0  # больше двух ожиданий insight_sync_wait_s по 0,5 с
    first = client.get(INSIGHT)
    assert first.status_code == 202 and first.headers["Retry-After"] == "1"
    job = first.json()
    assert job["status"] in ("queued", "running") and job["card"] is None
    again = client.get(INSIGHT)  # повторный клик, пока карточка генерируется
    assert again.status_code == 202 and again.json()["job_id"] == job["job_id"]

    done = poll(client, job)
    assert done["status"] == "done" and done["card"]["card_id"] == f"ins_{COMPANY}_{ITEM}"
    assert fake_ml.insight_calls == 1


def test_different_roles_are_different_jobs(client, fake_ml):
    fake_ml.delay_s = 0.8
    a, b = client.get(INSIGHT, params={"role": "cfo"}).json(), client.get(INSIGHT, params={"role": "ceo"}).json()
    assert a["job_id"] != b["job_id"]
    assert poll(client, a)["card"]["role"] == "cfo" and poll(client, b)["card"]["role"] == "ceo"


def test_budget_exceeded_is_429_with_retry_after(client, fake_ml):
    fake_ml.error = BudgetExceeded("дневной бюджет 500 руб. исчерпан")
    resp = client.get(INSIGHT)
    body = assert_error(resp, 429, "budget_exceeded")
    assert 0 < body["retry_after"] <= 24 * 3600 and resp.headers["Retry-After"] == str(body["retry_after"])


def test_model_failure_is_502_without_internal_details(client, fake_ml):
    fake_ml.error = InsightError("JSON не прошёл схему: внутренняя подробность")
    body = assert_error(client.get(INSIGHT), 502, "model_failed")
    assert "подробность" not in body["message"]


def test_unknown_item_is_404(client):
    assert_error(client.get(f"/api/companies/{COMPANY}/insights/x_missing"), 404, "not_found")


def test_background_failure_is_reported_in_job(client, fake_ml):
    fake_ml.delay_s, fake_ml.error = 0.8, InsightError("сбой")
    job = client.get(INSIGHT).json()
    done = poll(client, job)
    assert done["status"] == "failed" and done["card"] is None and done["error"]["code"] == "model_failed"
    # Неудачная задача не мешает новой попытке.
    fake_ml.error, fake_ml.delay_s = None, 0
    assert client.get(INSIGHT).status_code == 200


def test_queue_limit(fake_ml):
    fake_ml.delay_s = 1.0
    with make_client(httpx.ASGITransport(app=ml_server.app), max_parallel_generations=1, max_queued_jobs=1,
                     insight_sync_wait_s=0) as client:
        assert client.get(INSIGHT).status_code == 202
        assert_error(client.get(f"/api/companies/{COMPANY}/insights/n_other"), 429, "queue_full")


def test_unknown_job_is_404(client):
    assert_error(client.get("/api/jobs/job_missing"), 404, "not_found")


# --- вопрос и обратная связь -------------------------------------------------

def test_followup(client):
    card_id = f"ins_{COMPANY}_{ITEM}"
    resp = client.post(f"/api/cards/{card_id}/followup", json={"question": "Что с кредитами?"}, params={"role": "cfo"})
    assert resp.status_code == 200 and resp.json()["answer"]
    assert_error(client.post(f"/api/cards/{card_id}/followup", json={"question": "x" * 501}), 422, "validation_error")
    assert_error(client.post(f"/api/cards/{card_id}/followup", json={"question": "   "}), 422, "validation_error")
    assert_error(client.post("/api/cards/bad/followup", json={"question": "?"}), 404, "not_found")


def test_feedback(client):
    card_id = f"ins_{COMPANY}_{ITEM}"
    resp = client.post(f"/api/cards/{card_id}/feedback", json={"rating": "down", "reason": "obvious", "role": "cfo"})
    assert resp.status_code == 200 and resp.json()["card_id"] == card_id
    assert_error(client.post(f"/api/cards/{card_id}/feedback", json={"rating": "meh"}), 422, "validation_error")


# --- сбои ML ------------------------------------------------------------------

def _failing(exc: Exception) -> httpx.MockTransport:
    def handler(request):
        raise exc
    return httpx.MockTransport(handler)


def test_ml_down_is_503_and_health_says_so():
    with make_client(_failing(httpx.ConnectError("connection refused"))) as client:
        assert_error(client.get("/api/companies"), 503, "ml_unavailable")
        health = client.get("/api/health").json()
        assert health["ok"] is True and health["ml_ok"] is False and health["ml_error"] == "ml_unavailable"


def test_ml_timeout_is_504():
    with make_client(_failing(httpx.ReadTimeout("timed out"))) as client:
        assert_error(client.get("/api/companies"), 504, "ml_timeout")
        assert_error(client.get(INSIGHT), 504, "ml_timeout")


def test_ml_unexpected_error_is_502():
    with make_client(httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))) as client:
        assert_error(client.get("/api/companies"), 502, "ml_error")


def test_health_with_ml(client):
    assert client.get("/api/health").json()["ml_ok"] is True


def test_cors_preflight(client):
    resp = client.options("/api/companies", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


@pytest.mark.parametrize("now_msk, expected_h", [("2026-10-24T02:00", 1), ("2026-10-24T03:00", 24), ("2026-10-24T15:30", 11.5)])
def test_budget_reset_at_3_msk(now_msk, expected_h):
    now = datetime.fromisoformat(now_msk).replace(tzinfo=timezone(timedelta(hours=3)))
    assert seconds_until_budget_reset(now) == int(expected_h * 3600)
