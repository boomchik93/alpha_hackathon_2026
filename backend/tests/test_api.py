"""API бэкенда: прокси к ML, единый формат ошибок, фоновая генерация карточек."""
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from backend.app.ml_client import seconds_until_budget_reset
from ml.agent.orchestrator import InsightError
from ml.llm.client import BudgetExceeded

from .fakes import COMPANY, ITEM, login, make_client, ml_transport

INSIGHT = f"/api/companies/{COMPANY}/insights/{ITEM}"
CARD_ID = f"ins_{COMPANY}_{ITEM}_cfo"   # alice: финансовый директор


def poll(client, job: dict, headers: dict, timeout_s: float = 5) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = client.get(job["poll_url"], headers=headers).json()
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

def test_profile_and_feed(client, alice):
    assert client.get(f"/api/companies/{COMPANY}", headers=alice).json()["bank_products"]["loan"] is True
    feed = client.get(f"/api/companies/{COMPANY}/feed", params={"tier": "main", "limit": 5}, headers=alice).json()
    assert feed["entries"][0]["item_id"] == ITEM
    assert feed["entries"][0]["reasons"] == ["limit=5 tier=main"]  # параметры дошли до ML


def test_validation_errors_in_common_format(client, alice):
    assert_error(client.get(f"/api/companies/{COMPANY}/feed", params={"limit": 0}, headers=alice), 422, "validation_error")
    assert_error(client.get(f"/api/companies/{COMPANY}/feed", params={"tier": "x"}, headers=alice), 422, "validation_error")
    assert_error(client.get("/api/companies/U$1", headers=alice), 422, "validation_error")  # в путь к ML только безопасные символы
    assert_error(client.get("/api/companies/..", headers=alice), 404, "not_found")  # «..» не доходит до ML
    assert_error(client.get("/api/companies/.x", headers=alice), 422, "validation_error")
    assert_error(client.get("/api/nope"), 404, "not_found")


# --- карточки ---------------------------------------------------------------

def test_ready_card_returns_200_with_role_from_profile(client, alice):
    resp = client.get(INSIGHT, headers=alice)
    assert resp.status_code == 200
    assert resp.json()["card_id"] == CARD_ID and resp.json()["role"] == "cfo"


def test_slow_card_goes_to_background_and_is_deduplicated(client, fake_ml, alice):
    fake_ml.delay_s = 2.0  # больше двух ожиданий insight_sync_wait_s по 0,5 с
    first = client.get(INSIGHT, headers=alice)
    assert first.status_code == 202 and first.headers["Retry-After"] == "1"
    job = first.json()
    assert job["status"] in ("queued", "running") and job["card"] is None
    again = client.get(INSIGHT, headers=alice)  # повторный клик, пока карточка генерируется
    assert again.status_code == 202 and again.json()["job_id"] == job["job_id"]

    done = poll(client, job, alice)
    assert done["status"] == "done" and done["card"]["card_id"] == CARD_ID
    assert fake_ml.insight_calls == 1


def test_different_roles_are_different_jobs(client, fake_ml, alice):
    fake_ml.delay_s = 0.8
    bob = login(client, "bob")
    a, b = client.get(INSIGHT, headers=alice).json(), client.get(INSIGHT, headers=bob).json()
    assert a["job_id"] != b["job_id"]
    assert poll(client, a, alice)["card"]["role"] == "cfo" and poll(client, b, bob)["card"]["role"] == "owner"


def test_budget_exceeded_is_429_with_retry_after(client, fake_ml, alice):
    fake_ml.error = BudgetExceeded("дневной бюджет 500 руб. исчерпан")
    resp = client.get(INSIGHT, headers=alice)
    body = assert_error(resp, 429, "budget_exceeded")
    assert 0 < body["retry_after"] <= 24 * 3600 and resp.headers["Retry-After"] == str(body["retry_after"])


def test_model_failure_is_502_without_internal_details(client, fake_ml, alice):
    fake_ml.error = InsightError("JSON не прошёл схему: внутренняя подробность")
    body = assert_error(client.get(INSIGHT, headers=alice), 502, "model_failed")
    assert "подробность" not in body["message"]


def test_unknown_item_is_404(client, alice):
    assert_error(client.get(f"/api/companies/{COMPANY}/insights/x_missing", headers=alice), 404, "not_found")


def test_background_failure_is_reported_in_job(client, fake_ml, alice):
    fake_ml.delay_s, fake_ml.error = 0.8, InsightError("сбой")
    job = client.get(INSIGHT, headers=alice).json()
    done = poll(client, job, alice)
    assert done["status"] == "failed" and done["card"] is None and done["error"]["code"] == "model_failed"
    # Неудачная задача не мешает новой попытке.
    fake_ml.error, fake_ml.delay_s = None, 0
    assert client.get(INSIGHT, headers=alice).status_code == 200


def test_queue_limit(fake_ml, tmp_path):
    fake_ml.delay_s = 1.0
    with make_client(ml_transport(), tmp_path, max_parallel_generations=1, max_queued_jobs=1,
                     insight_sync_wait_s=0) as client:
        alice = login(client, "alice")
        assert client.get(INSIGHT, headers=alice).status_code == 202
        assert_error(client.get(f"/api/companies/{COMPANY}/insights/n_other", headers=alice), 429, "queue_full")


def test_unknown_job_is_404(client, alice):
    assert_error(client.get("/api/jobs/job_missing", headers=alice), 404, "not_found")


# --- вопрос и обратная связь -------------------------------------------------

def test_followup(client, alice):
    assert client.get(INSIGHT, headers=alice).status_code == 200
    resp = client.post(f"/api/cards/{CARD_ID}/followup", json={"question": "Что с кредитами?"}, headers=alice)
    assert resp.status_code == 200 and resp.json()["answer"]
    assert_error(client.post(f"/api/cards/{CARD_ID}/followup", json={"question": "x" * 501}, headers=alice), 422, "validation_error")
    assert_error(client.post(f"/api/cards/{CARD_ID}/followup", json={"question": "   "}, headers=alice), 422, "validation_error")
    # Карточка, которую бэкенд не отдавал, не найдена.
    assert_error(client.post("/api/cards/ins_bad/followup", json={"question": "?"}, headers=alice), 404, "not_found")


# --- сбои ML ------------------------------------------------------------------

def _failing(exc: Exception) -> httpx.MockTransport:
    def handler(request):
        raise exc
    return httpx.MockTransport(handler)


def test_ml_down_is_503_and_health_says_so(tmp_path):
    with make_client(_failing(httpx.ConnectError("connection refused")), tmp_path) as client:
        alice = login(client, "alice")
        assert_error(client.get(f"/api/companies/{COMPANY}/feed", headers=alice), 503, "ml_unavailable")
        health = client.get("/api/health").json()
        assert health["ok"] is True and health["ml_ok"] is False and health["ml_error"] == "ml_unavailable"


def test_ml_timeout_is_504(tmp_path):
    with make_client(_failing(httpx.ReadTimeout("timed out")), tmp_path) as client:
        alice = login(client, "alice")
        assert_error(client.get(f"/api/companies/{COMPANY}/feed", headers=alice), 504, "ml_timeout")
        assert_error(client.get(INSIGHT, headers=alice), 504, "ml_timeout")


def test_ml_unexpected_error_is_502(tmp_path):
    with make_client(httpx.MockTransport(lambda r: httpx.Response(500, text="boom")), tmp_path) as client:
        assert_error(client.get(f"/api/companies/{COMPANY}", headers=login(client, "alice")), 502, "ml_error")


def test_health_with_ml(client):
    assert client.get("/api/health").json()["ml_ok"] is True


def test_cors_preflight(client):
    resp = client.options("/api/companies", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


@pytest.mark.parametrize("now_msk, expected_h", [("2026-10-24T02:00", 1), ("2026-10-24T03:00", 24), ("2026-10-24T15:30", 11.5)])
def test_budget_reset_at_3_msk(now_msk, expected_h):
    now = datetime.fromisoformat(now_msk).replace(tzinfo=timezone(timedelta(hours=3)))
    assert seconds_until_budget_reset(now) == int(expected_h * 3600)
