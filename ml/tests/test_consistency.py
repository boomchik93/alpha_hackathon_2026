"""Сверка противоречий событие/банк (REVIEW_PLAN.md, T15)."""

import pytest

from ml.agent import consistency
from ml.agent.consistency import ConsistencyResult, Pair
from ml.agent.orchestrator import run_insight
from ml.agent.trace import load_trace
from ml.eval.eval_consistency import load, score
from ml.llm.client import LLMClient
from ml.tests.conftest import needs_data


def test_unknown_ids_dropped_and_empty_contradiction_is_unrelated():
    res = ConsistencyResult(status="contradicts", pairs=[Pair(fact_id="f1", chunk_id="k1", note="x"), Pair(fact_id="f9", chunk_id="k1")])
    got = consistency.clean(res, {"f1"}, {"k1"})
    assert got.status == "contradicts" and [(p.fact_id, p.chunk_id) for p in got.pairs] == [("f1", "k1")]
    bad = consistency.clean(ConsistencyResult(status="contradicts", pairs=[Pair(fact_id="f9", chunk_id="k9")]), {"f1"}, {"k1"})
    assert bad.status == "unrelated" and bad.pairs == []
    assert consistency.clean(ConsistencyResult(status="consistent", pairs=[Pair(fact_id="f1", chunk_id="k1")]), {"f1"}, {"k1"}).pairs == []


def test_no_facts_or_chunks_means_no_call():
    class Boom(LLMClient):
        def complete_json(self, *a, **kw):
            raise AssertionError("вызова быть не должно")

    c = Boom(mock=True)
    assert consistency.check(c, "t", [], [{"id": "k", "text": "x"}]).status == "unrelated"
    assert consistency.check(c, "t", [{"id": "f1", "text": "x", "confidence": "confirmed"}], []).status == "unrelated"


def test_mock_contradiction_and_prompt_block():
    rumor = [{"id": "f1", "text": "Источники допускают снижение ключевой ставки", "confidence": "assumption"}]
    bank = [{"id": "k1", "text": "Банк ожидает паузу: ставка останется прежней."}]
    res = consistency.check(LLMClient(mock=True, use_cache=False), "слух", rumor, bank)
    assert res.status == "contradicts" and res.pairs[0].fact_id == "f1"
    block = consistency.prompt_block(res)
    assert "CONTRADICTIONS" in block and "k1" in block
    assert consistency.prompt_block(ConsistencyResult()) == ""


def test_target_set_shape_and_scoring():
    rows = load()
    assert 8 <= len(rows) <= 10
    assert sum(r["expected"] == "contradicts" for r in rows) == len(rows) // 2
    perfect = score([r["expected"] for r in rows], [r["expected"] for r in rows])
    assert perfect["precision"] == perfect["recall"] == perfect["accuracy"] == 1.0
    s = score(["contradicts", "contradicts", "consistent"], ["contradicts", "consistent", "contradicts"])
    assert s["precision"] == 0.5 and s["recall"] == 0.5


class SpyClient(LLMClient):
    def __init__(self):
        super().__init__(mock=True, use_cache=False)
        self.interpret_prompts = []

    def complete_json(self, task, system, user, *a, **kw):
        if task == "interpret":
            self.interpret_prompts.append(user)
        return super().complete_json(task, system, user, *a, **kw)


@needs_data
def test_contradiction_goes_to_prompt_and_limitations_rule(monkeypatch):
    monkeypatch.setenv("ML_CONSISTENCY", "1")
    client = SpyClient()
    card = run_insight("U_1595", "n_2026_09_13_rate_cut_rumor", client)
    tr = load_trace(card.trace_id)
    ev = [e for e in tr["events"] if e["event"] == "consistency"]
    assert ev and ev[0]["status"] == "contradicts"
    assert "CONTRADICTIONS" in client.interpret_prompts[0]
    first_verify = next(e for e in tr["events"] if e["event"] == "verify")
    assert any("расхожд" in i for i in first_verify["issues"])


@needs_data
def test_flag_off_by_default():
    client = SpyClient()
    card = run_insight("U_1595", "n_2026_09_13_rate_cut_rumor", client)
    assert not [e for e in load_trace(card.trace_id)["events"] if e["event"] == "consistency"]
    assert "CONTRADICTIONS" not in client.interpret_prompts[0]
