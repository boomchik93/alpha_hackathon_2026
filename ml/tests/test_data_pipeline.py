"""Сквозные проверки на реальных данных (нужны data/ и собранный var/)."""

from ml import api
from ml.agent.trace import load_trace
from ml.ingest.analytics import clean_text, freshness, is_stale
from ml.ingest.store import load_items
from ml.profile.build_profile import build_profile, normalize_okved
from ml.schemas import PROFILE_SOURCE_ID
from ml.tests.conftest import needs_csv, needs_data
from datetime import date


def test_normalize_okved():
    assert normalize_okved("41.20") == "41.2"
    assert normalize_okved("41.2") == "41.2"
    assert normalize_okved("46.73.3") == "46.73.3"
    assert normalize_okved("1.70") == "01.7"


def test_ocr_cleanup_and_freshness():
    assert "г/г" in clean_text("рост 5% r/r в 1126")
    assert "1П26" in clean_text("рост 5% r/r в 1126")
    assert "по 25 б.п. до конца года" in clean_text("мы ждем еще два снижения по 25 6. п. до конца года")
    assert "до 25 б.п., ставка" in clean_text("ЦБ уменьшил шаг до 25 6.п., ставка была снижена")
    assert is_stale(date(2021, 4, 15))
    assert freshness(date(2021, 4, 15)) < 0.01


@needs_csv
def test_duplicate_company_flagged():
    p = build_profile("U_428697")
    assert any("нескольких строках" in c for c in p.data_quality.conflicts)


@needs_csv
def test_big_client_profile():
    p = build_profile("U_1595")
    assert p.eligible_big and p.bank_products.loan and p.geo.foreign_trade
    assert p.geo.fts_countries


@needs_data
def test_feed_filters_irrelevant_and_dedups():
    f = api.get_ranked_feed("U_1595", limit=50)
    ids = [e.item_id for e in f.entries]
    assert "n_2026_09_05_football" not in ids
    assert not ({"n_2026_09_11_cbr_rate", "n_2026_09_11_cbr_rate_dup"} <= set(ids))
    assert f.filtered_out >= 1


@needs_data
def test_stale_steel_marked():
    f = api.get_ranked_feed("U_1595", limit=50)
    steel = [e for e in f.entries if e.item_id == "bank_steel_1q21"]
    assert all(e.stale for e in steel)


@needs_data
def test_insight_end_to_end_with_retry():
    card = api.get_insight("U_1595", "n_2026_09_11_cbr_rate", refresh=True)
    known = {s.id for s in card.sources} | {PROFILE_SOURCE_ID}
    for claim in [*card.consequences.risks, *card.consequences.opportunities, *card.suggested_actions]:
        assert claim.evidence or claim.is_model_inference
        assert set(claim.evidence) <= known
    assert card.bank_view is not None
    trace = load_trace(card.trace_id)
    events = [e["event"] for e in trace["events"]]
    assert "plan" in events and "tool_call" in events and "retry" in events


@needs_data
def test_prompt_injection_not_in_card():
    card = api.get_insight("U_1595", "n_2026_09_06_noisy", refresh=True)
    assert "ИГНОРИРУЙ" not in card.model_dump_json()


@needs_data
def test_followup_answer_has_sources():
    card = api.get_insight("U_1595", "n_2026_09_11_cbr_rate")
    ans = api.answer_followup(card.card_id, "Что банк думает о ставке?")
    assert ans.answer and ans.trace_id


@needs_data
def test_all_items_parsed():
    items = load_items()
    assert any(i.source.kind.value == "bank_analytics" for i in items.values())
    assert any(i.source.kind.value == "synthetic_test" for i in items.values())


def test_normalize_noisy_keeps_numbers():
    from ml.ingest.normalize import normalize

    n = normalize("БЕНЗИН  Аи-92  ПОДОРОЖАЛ НА 0,4%н/н ,  дизель--+0,3 %")
    assert n.text == "Бензин Аи-92 подорожал на 0,4% н/н, дизель — +0,3%"
    assert n.noisy
    assert not normalize("Банк России сохранил ключевую ставку на уровне 14%.").noisy


@needs_data
def test_stale_bank_analytics_not_in_feed():
    f = api.get_ranked_feed("U_200890", limit=50)
    assert "bank_steel_1q21" not in [e.item_id for e in f.entries]


def test_okved_hints_normalized_in_ranking():
    from ml.ranking.ranker import heuristic_reasons
    from ml.schemas import BankProducts, CompanyProfile, DataQuality, Geo, GroupInfo, Industry, Network, ParsedItem, Size

    prof = CompanyProfile(company_id="U_1", eligible_big=True, industry=Industry(okved_code="41.2.9"), size=Size(), geo=Geo(),
                          bank_products=BankProducts(), group=GroupInfo(), network=Network(), data_quality=DataQuality())
    item = ParsedItem.model_validate({"item_id": "n", "source": {"kind": "news", "title": "t", "published_at": "2026-09-09"},
                                      "title": "t", "event_type": "other", "okved_hints": ["41.20"]})
    assert any(path == "industry.okved_code" for _, _, path in heuristic_reasons(prof, item))


def test_adjacency_map():
    from ml.topics import adjacency_reason

    assert adjacency_reason("46.73", ("41.2",)) is not None       # новостройки -> опт стройматериалами
    assert adjacency_reason("23.64", ("41.2",)) is not None       # новостройки -> бетон
    assert adjacency_reason("46.46", ("47.73",)) is not None      # аптеки -> фарма-опт
    assert adjacency_reason("46.46", ("47.31",)) is None          # топливная розница не связана с фарма-оптом
    assert adjacency_reason("41.2.9", ("41.2",)) is None          # своя отрасль, не смежная
    assert adjacency_reason("46.71", ("01.11",)) is None          # зерно не связано с топливом


@needs_csv
def test_adjacency_from_payments():
    from ml.profile.flows import adjacency
    from ml.topics import adjacency_link

    link = adjacency("23.64", "41")                               # стройка -> бетон: стройка покупает бетон
    assert link is not None and link.direction == "customer"
    assert adjacency("46.71", "01.11") is None                    # зерно не связано с топливом
    assert adjacency_link("41.2.9", ("41.2",)) is None            # своя отрасль, не смежная
    assert adjacency_link("23.64", ("41",))[1] == "payments"      # связь из платежей, не из ручной карты


@needs_data
def test_insight_with_channels(monkeypatch):
    monkeypatch.setenv("ML_CHANNELS", "1")
    card = api.get_insight("U_1595", "n_2026_09_11_cbr_rate", refresh=True)
    assert [c.channel for c in card.channels][:1] == ["cost_of_money"]   # мок: ставка при кредите
    assert all(c.evidence and c.profile_features for c in card.channels)
    assert "fx" not in {c.channel for c in card.channels}                # канал без источника отброшен
    trace = load_trace(card.trace_id)
    assert any(e["event"] == "channels" for e in trace["events"])


@needs_csv
def test_market_position_company_and_group():
    from ml.profile.build_profile import build_profile, company_index
    from ml.agent.verifier import profile_numbers, ungrounded_numbers

    p = build_profile("U_1595")
    m = p.market_position
    assert m is not None and m.peers >= 10 and 1 <= m.rank <= m.peers and 0 < m.revenue_share <= 1
    assert m.digits in (4, 3, 2) and "не доля рынка" in m.note
    share = f"{m.revenue_share * 100:.1f}".replace(".", ",")
    assert ungrounded_numbers(f"доля выручки среди компаний выборки {share}%", [], profile_numbers(p)) == []

    idx = company_index()
    gid = next(g for g in idx["GROUP_ID"].dropna().unique() if (idx["GROUP_ID"] == g).sum() > 1 and str(g).startswith("group_"))
    g = build_profile(gid)
    gm = g.market_position
    assert gm is not None and "члены группы" in gm.note and 1 <= gm.rank <= gm.peers
    # члены группы не считаются пирами: перебор без них даёт то же число субъектов
    from ml.profile.build_profile import _okved_groups
    digits = g.industry.okved_code.replace(".", "")[:gm.digits]
    outside = [c for c, _ in _okved_groups()[digits] if c not in set(g.member_ids)]
    assert gm.peers == len(outside) + 1
