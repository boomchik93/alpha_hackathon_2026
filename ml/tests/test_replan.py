"""Перепланирование по данным и инструмент search_news."""

from datetime import date

import pytest

from ml.agent import tools as tools_mod
from ml.agent.orchestrator import run_insight
from ml.agent.tools import RunContext, ToolDenied, ToolRegistry
from ml.agent.trace import Trace, load_trace
from ml.ingest.store import get_item, load_items
from ml.schemas import Fact, ItemSource, ParsedItem
from ml.tests.conftest import needs_data


def _thin_item(item_id: str, facts: int) -> ParsedItem:
    return ParsedItem(
        item_id=item_id, source=ItemSource(kind="news", title="Банк России ключевая ставка решение", published_at=date(2026, 9, 11)),
        title="Банк России принял решение по ключевой ставке", summary="Совет директоров Банка России принял решение по ключевой ставке.",
        event_type="monetary_policy", topics=["key_rate"],
        facts=[Fact(id=f"f{i}", text="Совет директоров Банка России принял решение по ключевой ставке", confidence="confirmed") for i in range(facts)])


@pytest.fixture
def thin(request):
    """Материал в памяти процесса; после теста убирается."""
    item = _thin_item("n_test_thin", request.param)
    load_items()[item.item_id] = item
    yield item
    load_items().pop(item.item_id, None)


def _events(card, name):
    return [e for e in load_trace(card.trace_id)["events"] if e["event"] == name]


@needs_data
@pytest.mark.parametrize("thin", [1], indirect=True)
def test_few_facts_triggers_search_news(thin):
    card = run_insight("U_1595", thin.item_id)
    replans = _events(card, "replan")
    assert [e["trigger"] for e in replans if e["trigger"] == "few_facts"] and all(e["why"] for e in replans)
    calls = [e for e in _events(card, "tool_call") if e["tool"] == "search_news"]
    assert len(calls) == 1 and calls[0]["result"] not in ("list[0]",)
    # в item_ids только материалы, на которые карточка сослалась
    used = card.referenced_source_ids()
    assert card.item_ids[0] == thin.item_id and all(f"src_{i}" in used for i in card.item_ids[1:])


@needs_data
def test_bank_document_does_not_search_news():
    card = run_insight("U_1595", "bank_rate_2026_09_11")
    assert not [e for e in _events(card, "replan") if e["trigger"] == "few_facts"]
    assert not [e for e in _events(card, "tool_call") if e["tool"] == "search_news"]
    assert card.item_ids == ["bank_rate_2026_09_11"]


@needs_data
@pytest.mark.parametrize("thin", [3], indirect=True)
def test_enough_facts_no_news_search(thin):
    card = run_insight("U_1595", thin.item_id)
    assert not [e for e in _events(card, "tool_call") if e["tool"] == "search_news"]
    assert not [e for e in _events(card, "replan") if e["trigger"] == "few_facts"]


@needs_data
@pytest.mark.parametrize("thin", [3], indirect=True)
def test_empty_bank_search_triggers_second_search(thin, monkeypatch):
    monkeypatch.setattr(ToolRegistry, "_search_bank_analytics_multi", lambda self, **kw: [])
    card = run_insight("U_1595", thin.item_id)
    replans = [e for e in _events(card, "replan") if e["trigger"] == "bank_miss"]
    assert len(replans) == 1 and replans[0]["query"]
    second = [e for e in _events(card, "tool_call") if e["tool"] == "search_bank_analytics"]
    assert second and second[0]["args"]["k"] == 6 and second[0]["args"]["include_stale"] is False


@needs_data
def test_search_news_excludes_self_cluster_and_window():
    ctx = RunContext("U_1595", "n_2026_09_11_cbr_rate")
    reg = ToolRegistry(ctx, Trace("t"))
    it = reg.call("get_item", item_id="n_2026_09_11_cbr_rate")
    found = reg.call("search_news", item_id=it.item_id, days=3, limit=2)
    assert len(found) <= 2
    for f in found:
        x = get_item(f["item_id"])
        assert x.item_id != it.item_id and x.dedup_cluster != it.dedup_cluster
        assert abs((x.source.published_at - it.source.published_at).days) <= 3 and f["id"] == f"src_{x.item_id}"
    assert reg.call("search_news", item_id=it.item_id, days=0, limit=2) == [] or all(
        get_item(f["item_id"]).source.published_at == it.source.published_at for f in reg.call("search_news", item_id=it.item_id, days=0))


@needs_data
def test_search_news_is_whitelisted_and_company_scoped():
    reg = ToolRegistry(RunContext("U_1595"), Trace("t"))
    assert "search_news" in reg.names
    with pytest.raises(ToolDenied):
        reg.call("search_news", item_id="n_2026_09_11_cbr_rate", company_id="U_OTHER")


@needs_data
def test_core_topic_doc_is_added_when_noisy_topics_crowd_it_out():
    """Новость о курсе рубля с темой key_rate: фрагменты обзора золота (размечены key_rate) вытесняли обзор банка
    по ставке; с core_topics поиск добирает фрагмент профильного документа сверх k (ml/eval/analysis.md, 25.10)."""
    from ml.agent.orchestrator import _bank_queries
    from ml.ingest.analytics import core_docs
    from ml.profile.build_profile import get_profile

    title = "ЦБ установил курс доллара ниже 81 руб. Впервые с августа"
    item = ParsedItem(
        item_id="n_test_fx", source=ItemSource(kind="news", title=title, published_at=date(2026, 9, 14)), title=title,
        summary="Банк России установил официальный курс доллара на уровне 80,53 руб./$.", event_type="fx_commodities",
        topics=["fx_gold", "key_rate"],
        facts=[Fact(id="f1", text="Банк России установил официальный курс доллара на уровне 80,53 руб./$, евро — 90,24 руб./€",
                    confidence="confirmed"),
               Fact(id="f2", text="Банк России второй раз подряд сохранил ключевую ставку на уровне 14%", confidence="confirmed")])
    queries = _bank_queries(get_profile("U_1595"), item)

    def docs(core_topics):
        hits = ToolRegistry(RunContext("U_1595"), Trace("t")).call(
            "search_bank_analytics_multi", queries=queries, k=1, per_doc=1, core_topics=core_topics)
        return {h["doc_id"] for h in hits}, len(hits)

    # k=1: единственное место занимает обзор золота, профильный документ по ставке вытеснен.
    rate = core_docs("key_rate")
    without, n = docs(None)
    assert n == 1 and not without & rate
    with_core, n = docs(item.topics)
    assert with_core & rate and with_core >= without and n == 2
