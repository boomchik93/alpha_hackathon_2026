"""Сценарии: прокси к ML, фоновая задача, ошибки, лимит, доступ, трасса, журнал."""
from ml.agent.orchestrator import InsightError
from ml.llm.client import BudgetExceeded

from .fakes import COMPANY, ITEM, OTHER, WEAK_ITEM, login, make_client, ml_transport
from .test_api import CARD_ID, INSIGHT, assert_error, poll

SCENARIOS = f"{INSIGHT}/scenarios"
SET_ID = f"scn_{COMPANY}_{ITEM}_cfo"   # alice: финансовый директор


def test_ready_set_returns_200_with_three_branches(client, alice):
    resp = client.get(SCENARIOS, headers=alice)
    assert resp.status_code == 200
    body = resp.json()
    assert body["set_id"] == SET_ID and body["role"] == "cfo"
    assert [b["kind"] for b in body["branches"]] == ["good", "base", "bad"]
    assert body["analysis"]["decisions"][0]["type"] == "robust"


def test_card_is_saved_so_followup_works_without_opening_it_first(client, alice):
    assert client.post(f"/api/cards/{CARD_ID}/followup", json={"question": "?"}, headers=alice).status_code == 404
    assert client.get(SCENARIOS, headers=alice).status_code == 200
    assert client.post(f"/api/cards/{CARD_ID}/followup", json={"question": "Что с кредитами?"}, headers=alice).status_code == 200


def test_slow_set_goes_to_background_and_is_deduplicated(client, fake_ml, alice):
    fake_ml.scenario_delay_s = 2.0
    first = client.get(SCENARIOS, headers=alice)
    assert first.status_code == 202 and first.headers["Retry-After"] == "2"
    job = first.json()
    assert job["kind"] == "scenarios" and job["status"] in ("queued", "running") and job["scenarios"] is None and job["card"] is None
    again = client.get(SCENARIOS, headers=alice)
    assert again.status_code == 202 and again.json()["job_id"] == job["job_id"]

    done = poll(client, job, alice)
    assert done["status"] == "done" and done["scenarios"]["set_id"] == SET_ID and done["card"] is None
    assert fake_ml.scenario_calls == 1


def test_scenarios_and_card_are_different_jobs(client, fake_ml, alice):
    fake_ml.delay_s = fake_ml.scenario_delay_s = 0.8
    card_job = client.get(INSIGHT, headers=alice).json()
    scn_job = client.get(SCENARIOS, headers=alice).json()
    assert card_job["job_id"] != scn_job["job_id"] and card_job["kind"] == "insight" and scn_job["kind"] == "scenarios"
    assert poll(client, card_job, alice)["card"]["card_id"] == CARD_ID
    assert poll(client, scn_job, alice)["scenarios"]["set_id"] == SET_ID


def test_weak_card_is_422_with_own_code(client, alice):
    body = assert_error(client.get(f"/api/companies/{COMPANY}/insights/{WEAK_ITEM}/scenarios", headers=alice), 422,
                        "scenarios_not_applicable")
    assert "слабая" in body["message"]


def test_budget_and_model_failure(client, fake_ml, alice):
    fake_ml.scenario_error = BudgetExceeded("дневной бюджет 500 руб. исчерпан")
    body = assert_error(client.get(SCENARIOS, headers=alice), 429, "budget_exceeded")
    assert body["retry_after"] > 0
    fake_ml.scenario_error = InsightError("JSON не прошёл схему: внутренняя подробность")
    body = assert_error(client.get(SCENARIOS, headers=alice), 502, "model_failed")
    assert "подробность" not in body["message"]


def test_background_failure_is_reported_in_job(client, fake_ml, alice):
    fake_ml.scenario_delay_s, fake_ml.scenario_error = 0.8, InsightError("сбой")
    done = poll(client, client.get(SCENARIOS, headers=alice).json(), alice)
    assert done["status"] == "failed" and done["scenarios"] is None and done["error"]["code"] == "model_failed"


def test_unknown_item_is_404(client, alice):
    assert_error(client.get(f"/api/companies/{COMPANY}/insights/x_missing/scenarios", headers=alice), 404, "not_found")


def test_other_company_is_403_and_role_is_fixed_by_profile(client, alice):
    assert_error(client.get(f"/api/companies/{OTHER}/insights/{ITEM}/scenarios", headers=alice), 403, "forbidden")
    assert_error(client.get(SCENARIOS, params={"role": "owner"}, headers=alice), 403, "forbidden")
    assert client.get(SCENARIOS, params={"role": "owner"}, headers=login(client, "exp")).json()["role"] == "owner"


def test_job_of_scenarios_is_visible_only_to_requester(client, fake_ml, alice):
    fake_ml.scenario_delay_s = 0.8
    job = client.get(SCENARIOS, headers=alice).json()
    assert_error(client.get(job["poll_url"], headers=login(client, "grp")), 404, "not_found")
    poll(client, job, alice)


def test_scenarios_limit_counts_only_new_generations(fake_ml, tmp_path):
    with make_client(ml_transport(), tmp_path, limit_scenarios_per_10m=1, insight_sync_wait_s=0) as client:
        alice = login(client, "alice")
        fake_ml.scenario_delay_s = 0.5
        first = client.get(SCENARIOS, headers=alice)
        assert first.status_code == 202
        assert client.get(SCENARIOS, headers=alice).status_code == 202   # присоединение к идущей задаче лимит не тратит
        body = assert_error(client.get(f"{INSIGHT.replace(ITEM, 'n_fx')}/scenarios", headers=alice), 429, "rate_limited")
        assert body["retry_after"] > 0
        poll(client, first.json(), alice)


def test_trace_user_view_hides_internals(client, alice):
    assert_error(client.get(f"/api/scenarios/{SET_ID}/trace", headers=alice), 404, "not_found")   # набор ещё не отдавали
    client.get(SCENARIOS, headers=alice)
    trace = client.get(f"/api/scenarios/{SET_ID}/trace", headers=alice).json()
    assert trace["fork"] == [{"attempt": 0, "issues": 0}]
    assert trace["branches"]["base"] == [{"attempt": 0, "issues": 1}, {"attempt": 1, "issues": 0}]
    assert trace["branches"]["good"] == [{"attempt": 0, "issues": 0}] and trace["analysis"] == [{"attempt": 0, "issues": 0}]
    assert trace["sources"][0]["id"] == "src_n_rate" and trace["retries"] == 1
    assert "СЕКРЕТНЫЙ" not in str(trace) and "raw" not in trace and "cost_rub" not in trace
    full = client.get(f"/api/scenarios/{SET_ID}/trace", headers=login(client, "exp")).json()
    assert full["cost_rub"] == 0.99 and full["raw"]["events"]
    assert_error(client.get(f"/api/scenarios/{SET_ID}/trace", headers=login(client, "grp")), 403, "forbidden")


def test_audit_records_scenarios(client, alice):
    client.get(SCENARIOS, headers=alice)
    rows = client.get("/api/admin/audit", headers=login(client, "adm")).json()
    assert any(r["action"] == "scenarios" and r["company_id"] == COMPANY and r["item_id"] == ITEM and r["detail"] == "cfo" for r in rows)
