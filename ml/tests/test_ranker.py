"""Приор важности материала в ленте: отсев по личному скору, порядок по смеси."""

from datetime import date

import pytest

from ml.ingest.importance import NEUTRAL, score_importance, with_importance
from ml.llm.client import LLMClient, LLMError
from ml.ranking import ranker
from ml.schemas import (BankProducts, CompanyProfile, DataQuality, Geo, GroupInfo, Industry, Network, ParsedItem, Size)


def _profile() -> CompanyProfile:
    return CompanyProfile(company_id="U_1", eligible_big=True, industry=Industry(okved_code="47.71"), size=Size(), geo=Geo(),
                          bank_products=BankProducts(loan=True), group=GroupInfo(), network=Network(), data_quality=DataQuality())


def _item(item_id: str, topics: list[str], importance: float, event_type: str = "other") -> ParsedItem:
    return ParsedItem.model_validate({
        "item_id": item_id, "source": {"kind": "news", "title": item_id, "published_at": date.today().isoformat()},
        "title": f"Материал {item_id} о рынке и банках", "event_type": event_type, "topics": topics, "importance": importance, "freshness": 1.0})


@pytest.fixture
def beta(monkeypatch):
    def set_beta(v):
        monkeypatch.setattr(ranker, "BETA", v)
    return set_beta


def test_zero_beta_keeps_personal_order(beta):
    beta(0.0)
    items = [_item("a", ["key_rate"], 0.1), _item("b", ["fx_gold"], 0.9)]
    scored, _ = ranker.score_all(_profile(), items)
    assert all(s.score == s.personal for s in scored)


def test_blend_reorders_but_filter_uses_personal_score(beta):
    beta(0.5)
    items = [_item("rate", ["key_rate"], 0.2), _item("rate_big", ["key_rate"], 0.95), _item("weak", [], 1.0)]
    scored, _ = ranker.score_all(_profile(), items)
    by_id = {s.item.item_id: s for s in scored}
    assert by_id["rate"].personal == by_id["rate_big"].personal > 0  # одинаковая связь с компанией
    assert by_id["rate_big"].score > by_id["rate"].score
    assert [s.item.item_id for s in scored][0] == "rate_big"
    # материал без связи с профилем не проходит отсев, как бы он ни был важен
    kept, _ = ranker.rank(_profile(), items)
    assert "weak" not in [s.item.item_id for s in kept] and by_id["weak"].personal < ranker.MIN_SCORE
    assert by_id["weak"].score == pytest.approx(0.5 * 1.0 + 0.5 * by_id["weak"].personal, abs=1e-3)


def test_important_reason_in_feed(beta):
    beta(0.2)
    f = ranker.feed(_profile(), [_item("big", ["key_rate"], 0.9), _item("small", ["key_rate"], 0.3)])
    reasons = {e.item_id: e.reasons for e in f.entries}
    assert ranker.IMPORTANT_REASON in reasons["big"] and ranker.IMPORTANT_REASON not in reasons["small"]
    assert all(0 <= e.score <= 1 for e in f.entries)


def test_importance_mock_is_deterministic_and_scaled():
    client = LLMClient(mock=True, use_cache=False)
    rate = _item("r", ["key_rate"], 0.5, "monetary_policy")
    corp = _item("c", [], 0.5, "company_news")
    assert score_importance(rate, "текст", client) == score_importance(rate, "текст", client) >= 0.9
    assert score_importance(corp, "текст", client) < 0.3
    assert with_importance(corp, "текст", client).importance == score_importance(corp, "текст", client)


def test_importance_failure_is_neutral():
    class Broken(LLMClient):
        def complete_json(self, *a, **kw):
            raise LLMError("cache_only: нет ответа")

    assert score_importance(_item("x", [], 0.9), "текст", Broken(mock=True)) == NEUTRAL


def _frozen():
    from ml.ranking.bm25 import FrozenBM25
    from ml.retrieval.index import tokenize

    docs = [tokenize(t) for t in ("ключевая ставка банк кредит лизинг", "инфляция цены рост потребление розница",
                                  "курс валюта юань доллар экспорт", "сталь металлургия прокат цена", "ставка кредит ипотека")]
    return FrozenBM25.fit(docs, [tokenize("кредит ставка лизинг розница")])


def test_frozen_bm25_score_does_not_depend_on_pool(monkeypatch):
    frozen = _frozen()
    monkeypatch.setattr(ranker, "load_background", lambda: frozen)
    base = [_item("rate", ["key_rate"], 0.5), _item("fx", ["fx_gold"], 0.5)]
    noise = [ParsedItem.model_validate({
        "item_id": f"noise{i}", "source": {"kind": "news", "title": "x", "published_at": date.today().isoformat()},
        "title": f"Посторонний материал номер {i} про кредит ставка лизинг и разное {'слово' * (i % 5)}", "event_type": "other",
        "freshness": 1.0}) for i in range(50)]
    before, _ = ranker.score_all(_profile(), base)
    after, _ = ranker.score_all(_profile(), base + noise)
    a = {s.item.item_id: (s.score, s.bm25, s.personal) for s in before}
    b = {s.item.item_id: (s.score, s.bm25, s.personal) for s in after}
    for iid, vals in a.items():
        assert all(abs(x - y) <= 1e-9 for x, y in zip(vals, b[iid])), iid


def test_frozen_bm25_saturation_and_unseen_terms():
    from ml.retrieval.index import tokenize

    f = _frozen()
    q = tokenize("кредит ставка лизинг")
    close, far = f.score(q, tokenize("ставка кредит лизинг банк")), f.score(q, tokenize("сталь прокат"))
    assert 0 < close < 1 and far == 0.0 and f.k > 0
    assert f.raw(tokenize("неизвестноеслово"), tokenize("неизвестноеслово")) > 0  # неизвестный термин получает максимальный idf
    assert f.score(q, tokenize("ставка")) < close
    again = type(f).from_json(f.to_json())
    assert again.score(q, tokenize("ставка кредит")) == f.score(q, tokenize("ставка кредит"))
