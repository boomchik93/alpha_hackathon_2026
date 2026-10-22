"""Роли пользователей (REVIEW_PLAN.md, T17)."""

import pytest
from fastapi.testclient import TestClient

from ml import api, server
from ml.agent.orchestrator import run_insight
from ml.llm.client import LLMClient
from ml.roles import ROLE_FOCUS, Role, card_suffix, role_block
from ml.tests.conftest import needs_data


class Spy(LLMClient):
    def __init__(self):
        super().__init__(mock=True, use_cache=False)
        self.prompts = {}

    def complete_json(self, task, system, user, *a, **kw):
        self.prompts.setdefault(task, []).append(user)
        return super().complete_json(task, system, user, *a, **kw)


def test_role_blocks():
    assert role_block(None) == "" and role_block(Role.other) == ""
    assert "ликвидность" in role_block(Role.treasurer) and "налоги" in role_block(Role.chief_accountant)
    assert "стоимость долга" in ROLE_FOCUS[Role.cfo] and "стоимость долга" in ROLE_FOCUS[Role.owner]
    assert card_suffix(None) == "" and card_suffix(Role.cfo) == "_cfo"


@needs_data
def test_role_goes_to_prompt_and_card_id_differs():
    plain, spy = Spy(), Spy()
    a = run_insight("U_1595", "n_2026_09_11_cbr_rate", plain)
    b = run_insight("U_1595", "n_2026_09_11_cbr_rate", spy, role=Role.treasurer)
    assert a.card_id == "ins_U_1595_n_2026_09_11_cbr_rate" and a.role is None
    assert b.card_id == "ins_U_1595_n_2026_09_11_cbr_rate_treasurer" and b.role == "treasurer"
    assert "ROLE:" not in plain.prompts["interpret"][0] and "ликвидность" in spy.prompts["interpret"][0]
    # факты и источники от роли не зависят
    assert a.what_happened.text == b.what_happened.text and [s.id for s in a.sources] == [s.id for s in b.sources]


@needs_data
def test_cached_cards_are_per_role_and_followup_inherits_role(monkeypatch):
    c1 = api.get_insight("U_1595", "n_2026_09_11_cbr_rate", refresh=True, role=Role.cfo)
    assert api.get_insight("U_1595", "n_2026_09_11_cbr_rate", role=Role.cfo).trace_id == c1.trace_id
    c2 = api.get_insight("U_1595", "n_2026_09_11_cbr_rate", role=Role.owner)
    assert c2.card_id != c1.card_id
    spy = Spy()
    monkeypatch.setattr(api, "get_client", lambda: spy)
    api.answer_followup(c1.card_id, "Что делать с долгом?")
    assert "стоимость долга" in spy.prompts["followup_qa"][0]
    api.answer_followup(c1.card_id, "Что делать с долгом?", role=Role.treasurer)
    assert "ликвидность" in spy.prompts["followup_qa"][1]


@needs_data
def test_http_role_query():
    c = TestClient(server.app)
    r = c.get("/companies/U_1595/insights/n_2026_09_11_cbr_rate", params={"role": "cfo"})
    assert r.status_code == 200 and r.json()["role"] == "cfo" and r.json()["card_id"].endswith("_cfo")
    assert c.get("/companies/U_1595/insights/n_2026_09_11_cbr_rate", params={"role": "президент"}).status_code == 422
    assert c.get("/companies/U_1595/insights/n_2026_09_11_cbr_rate").json()["role"] is None
