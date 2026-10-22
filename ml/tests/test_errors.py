"""Ошибки LLM и коды ответа API (REVIEW_PLAN.md, T5)."""

import pytest
from fastapi.testclient import TestClient

from ml import server
from ml.agent.orchestrator import InsightError, run_insight
from ml.errors import NotFound
from ml.llm.client import BudgetExceeded, LLMClient, LLMError
from ml.tests.conftest import needs_data


class FlakyClient(LLMClient):
    """Первые `fails` вызовов интерпретатора падают с заданной ошибкой."""

    def __init__(self, exc: Exception, fails: int):
        super().__init__(mock=True, use_cache=False)
        self.exc, self.fails = exc, fails

    def complete_json(self, task, *args, **kwargs):
        if task == "interpret" and self.fails > 0:
            self.fails -= 1
            raise self.exc
        return super().complete_json(task, *args, **kwargs)


@needs_data
@pytest.mark.parametrize("exc", [ValueError("не JSON"), LLMError("пустой ответ")])
def test_bad_model_answer_is_retried_not_500(exc):
    card = run_insight("U_1595", "n_2026_09_11_cbr_rate", FlakyClient(exc, fails=1))
    assert card.meta.retries == 1


@needs_data
def test_always_failing_model_gives_insight_error():
    with pytest.raises(InsightError, match="не разобран"):
        run_insight("U_1595", "n_2026_09_11_cbr_rate", FlakyClient(LLMError("сбой"), fails=99))


@needs_data
def test_budget_exceeded_is_not_swallowed():
    with pytest.raises(BudgetExceeded):
        run_insight("U_1595", "n_2026_09_11_cbr_rate", FlakyClient(BudgetExceeded("лимит"), fails=1))


def test_http_codes(monkeypatch):
    def boom(exc):
        def f(*a, **kw):
            raise exc
        return f

    c = TestClient(server.app, raise_server_exceptions=False)
    for exc, code in [(LLMError("сбой"), 502), (InsightError("нет карточки"), 502), (BudgetExceeded("лимит"), 429),
                      (NotFound("компания X не найдена"), 404), (KeyError("нет маршрута"), 500)]:
        monkeypatch.setattr(server.api, "get_insight", boom(exc))
        assert c.get("/companies/U_1/insights/n_x").status_code == code, exc
    monkeypatch.setattr(server.api, "get_insight", boom(NotFound("компания X не найдена")))
    assert c.get("/companies/U_1/insights/n_x").json()["detail"] == "компания X не найдена"


@needs_data
def test_unknown_company_and_item_are_404():
    c = TestClient(server.app)
    assert c.get("/companies/U_NOPE").status_code == 404
    assert c.get("/companies/U_1595/insights/n_nope").status_code == 404
    assert c.post("/cards/ins_nope/followup", json={"question": "почему?"}).status_code == 404


@needs_data
def test_http_502_when_model_always_fails(monkeypatch):
    from ml.llm import client as client_mod

    monkeypatch.setattr(client_mod, "_default", FlakyClient(LLMError("сбой"), fails=99))
    r = TestClient(server.app).get("/companies/U_1595/insights/n_2026_09_11_cbr_rate", params={"refresh": True})
    assert r.status_code == 502 and "модель не вернула годный ответ" in r.json()["detail"]


@needs_data
def test_stale_card_is_regenerated(monkeypatch):
    from ml import api
    from ml.agent.orchestrator import load_card, save_card
    from ml.prompts import prompt_version

    assert prompt_version("interpret").startswith("interpret@v")
    first = api.get_insight("U_1595", "n_2026_09_11_cbr_rate", refresh=True)
    calls = []
    real = api.run_insight
    monkeypatch.setattr(api, "run_insight", lambda *a, **kw: calls.append(a) or real(*a, **kw))
    assert api.get_insight("U_1595", "n_2026_09_11_cbr_rate").trace_id == first.trace_id and not calls  # текущая версия: из файла
    for field, value in (("prompt_version", "interpret@v0"), ("model", "другая/модель")):
        save_card(first.model_copy(update={"meta": first.meta.model_copy(update={field: value})}))
        calls.clear()
        again = api.get_insight("U_1595", "n_2026_09_11_cbr_rate")
        assert calls and again.trace_id != first.trace_id, field
        assert load_card(first.card_id).meta.prompt_version == prompt_version("interpret")
        first = again


class _QAClient:
    """Клиент-заглушка: ответы followup_qa по очереди, промпты сохраняются."""

    mock = True

    def __init__(self, answers):
        self.answers, self.users = list(answers), []

    def complete_json(self, task, system, user, schema, **kw):
        self.users.append(user)
        return schema(answer=self.answers.pop(0), source_ids=[], is_model_inference=False), None


@needs_data
def test_followup_invented_number_retried_then_flagged(monkeypatch):
    from ml import api

    card = api.get_insight("U_1595", "n_2026_09_11_cbr_rate")
    fake = _QAClient(["Выручка вырастет на 77,7% уже в следующем квартале.", "Эффект составит 88,8 млрд руб."])
    monkeypatch.setattr(api, "get_client", lambda: fake)
    ans = api.answer_followup(card.card_id, "Что будет с выручкой?")
    assert len(fake.users) == 2 and "ИСПРАВЬ ПРЕДЫДУЩИЙ ОТВЕТ" in fake.users[1] and "77.7" in fake.users[1]
    assert ans.verifier_issues and "88.8" in ans.verifier_issues[0] and ans.is_model_inference


@needs_data
def test_followup_fixed_on_retry_and_clean_answer_untouched(monkeypatch):
    from ml import api

    card = api.get_insight("U_1595", "n_2026_09_11_cbr_rate")
    fake = _QAClient(["Выручка вырастет на 77,7%.", "Точных цифр в источниках нет."])
    monkeypatch.setattr(api, "get_client", lambda: fake)
    ans = api.answer_followup(card.card_id, "Что будет с выручкой?")
    assert len(fake.users) == 2 and ans.verifier_issues == []
    fake = _QAClient(["Ключевая ставка 14,0%, как в материале."])
    monkeypatch.setattr(api, "get_client", lambda: fake)
    ans = api.answer_followup(card.card_id, "Какая ставка?")
    assert len(fake.users) == 1 and ans.verifier_issues == []
