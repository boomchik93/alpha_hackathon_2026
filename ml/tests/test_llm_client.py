import pytest

from ml.llm import client as client_mod
from ml.llm.client import BudgetExceeded, CallInfo, LLMClient, LLMError, SpendLedger, estimate_cost, extract_json


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Вот ответ: {"a": {"b": 2}} спасибо') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("нет json")


def test_disallowed_model_rejected():
    c = LLMClient(mock=False, use_cache=False)
    with pytest.raises(LLMError, match="не входит в список"):
        c._chat("gpt-4o", [], 0.0, 10)


def test_budget_stop(tmp_path, monkeypatch):
    ledger = SpendLedger()
    monkeypatch.setattr(ledger, "path", tmp_path / "spend.jsonl")
    ledger.record(CallInfo(task="t", model="m", cost_rub=1000.0))
    with pytest.raises(BudgetExceeded):
        ledger.check()


def test_cost_estimate_positive():
    assert estimate_cost("glm-5.3-flash", 1000, 1000) > 0


def test_mock_returns_validated_schema():
    from ml.ingest.news_fetch import RawItem
    from ml.ingest.parse_item import parse_item

    raw = RawItem(id="n_x", title="ЦБ сохранил ключевую ставку 14%", text="Совет директоров Банка России сохранил ключевую ставку на уровне 14%. Аналитики ожидают паузы.",
                  published_at="2026-09-11", kind="synthetic_test")
    item = parse_item(raw, client=client_mod.get_client())
    assert "key_rate" in item.topics
    assert any(f.confidence == "assumption" for f in item.facts)
