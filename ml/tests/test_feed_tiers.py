"""Два уровня ленты: основная и «может быть интересно» (FEED_TIERS_PLAN.md, этап 1)."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from ml import api, server
from ml.ranking import ranker
from ml.schemas import BankProducts, CompanyProfile, DataQuality, FeedEntry, Geo, GroupInfo, Industry, Network, ParsedItem, Size


def _profile(okved: str = "47.71", **bp) -> CompanyProfile:
    return CompanyProfile(company_id="U_1", eligible_big=True, industry=Industry(okved_code=okved), size=Size(), geo=Geo(),
                          bank_products=BankProducts(**bp), group=GroupInfo(), network=Network(), data_quality=DataQuality())


def _item(item_id: str, scope: str = "sector", importance: float = 0.5, hints=()) -> ParsedItem:
    return ParsedItem.model_validate({
        "item_id": item_id, "source": {"kind": "news", "title": item_id, "published_at": date.today().isoformat()},
        "title": f"Материал {item_id}", "event_type": "other", "topics": [], "importance": importance, "freshness": 1.0,
        "scope": scope, "okved_hints": list(hints)})


def _scored(item: ParsedItem, personal: float, score: float | None = None) -> ranker.Scored:
    return ranker.Scored(item, personal if score is None else score, [], personal=personal)


@pytest.mark.parametrize("personal, tier", [(0.45, "main"), (0.40, "main"), (0.35, "maybe"), (0.30, "maybe"), (0.25, None)])
def test_tier_by_personal_score(personal, tier):
    assert ranker.tier_of(_profile(), _scored(_item("a"), personal)) == tier


def test_event_in_company_sector_is_always_main():
    p = _profile("47.71")
    own = _item("own", hints=("47",))
    assert ranker.tier_of(p, _scored(own, ranker.MIN_SCORE)) == "main"
    assert ranker.tier_of(p, _scored(_item("other", hints=("01.1",)), ranker.MIN_SCORE)) == "maybe"
    assert ranker.tier_of(p, _scored(own, 0.1)) is None  # ниже отсева даже своя отрасль не возвращается: пол ставит скор сам


def test_macro_without_hook_is_maybe_by_default_and_main_on_request():
    p = _profile()  # нет продуктов банка и ВЭД
    s = ranker.score_item(p, _item("e", "economy", importance=0.7))
    assert s.personal == pytest.approx(ranker.MIN_SCORE) and ranker.economy_floor(p, s.item)
    assert ranker.tier_of(p, s) == "maybe"
    assert ranker.tier_of(p, s, macro_tier="main") == "main"
    assert ranker.tier_of(p, _scored(s.item, 0.2), macro_tier="main") is None  # ниже отсева макро не поднимается


def test_main_score_override():
    assert ranker.tier_of(_profile(), _scored(_item("a"), 0.35), main_score=0.35) == "main"
    assert ranker.tier_of(_profile(), _scored(_item("a"), 0.45), main_score=0.5) == "maybe"


def _patch_scores(monkeypatch, personals: dict[str, float]):
    items = [_item(k) for k in personals]
    # порядок score_all: по убыванию score; у maybe score выше, чем у части main, чтобы проверить порядок уровней
    scored = sorted((_scored(i, personals[i.item_id], score=1 - n * 0.01) for n, i in enumerate(items)), key=lambda s: -s.score)
    monkeypatch.setattr(ranker, "score_all", lambda p, its: (scored, 2))
    return items


def test_rank_main_first_then_maybe_with_limits(monkeypatch):
    items = _patch_scores(monkeypatch, {"m1": 0.35, "a1": 0.8, "a2": 0.6, "m2": 0.31, "a3": 0.5, "x": 0.1, "m3": 0.33})
    kept, dropped = ranker.rank(_profile(), items)
    assert [s.item.item_id for s in kept] == ["a1", "a2", "a3", "m1", "m2", "m3"]
    assert [s.tier for s in kept] == ["main"] * 3 + ["maybe"] * 3 and dropped == 3
    kept, _ = ranker.rank(_profile(), items, limit=2, maybe_limit=1)
    assert [s.item.item_id for s in kept] == ["a1", "a2", "m1"]
    assert [s.item.item_id for s in ranker.rank(_profile(), items, maybe_limit=0)[0]] == ["a1", "a2", "a3"]


def test_feed_entries_carry_tier_and_weak_reason(monkeypatch):
    items = _patch_scores(monkeypatch, {"m1": 0.35, "a1": 0.8})
    f = ranker.feed(_profile(), items)
    assert [(e.item_id, e.tier) for e in f.entries] == [("a1", "main"), ("m1", "maybe")] and f.maybe_count == 1
    assert f.entries[1].reasons[0] == ranker.WEAK_REASON and ranker.WEAK_REASON not in f.entries[0].reasons
    only_main = ranker.feed(_profile(), items, tier="main")
    assert [e.item_id for e in only_main.entries] == ["a1"] and only_main.maybe_count == 0


def test_schema_default_tier_is_main():
    e = FeedEntry.model_validate({"item_id": "a", "title": "t", "published_at": "2026-10-23", "source_kind": "news", "score": 0.5, "reasons": []})
    assert e.tier == "main"
    assert api.Feed.model_validate({"company_id": "U_1", "as_of": "2026-10-23", "entries": []}).maybe_count == 0


def test_api_tier_parameter(monkeypatch):
    seen = {}

    def fake_feed(p, items, limit, maybe_limit, tier):
        seen.update(limit=limit, maybe_limit=maybe_limit, tier=tier)
        return api.Feed(company_id="U_1", as_of=date.today(), entries=[])

    monkeypatch.setattr(api, "feed", fake_feed)
    monkeypatch.setattr(api, "get_profile", lambda cid: _profile())
    monkeypatch.setattr(api, "load_items", lambda: {})
    api.get_ranked_feed("U_1")
    assert seen == {"limit": 20, "maybe_limit": 10, "tier": "all"}
    c = TestClient(server.app)
    assert c.get("/companies/U_1/feed", params={"tier": "main", "limit": 5}).status_code == 200
    assert seen == {"limit": 5, "maybe_limit": 10, "tier": "main"}
    assert c.get("/companies/U_1/feed", params={"tier": "bogus"}).status_code == 422


def _row(company: str, item: str, grade: int, keep: bool = False, floor: bool = False):
    from ml.ranking import pair_model as pm

    return pm.Row(company, item, "d", "real", {}, grade, 0.5, False, keep=keep, floor=floor)


def test_tier_metrics():
    from ml.ranking import pair_model as pm

    rows = [_row("A", "1", 2), _row("A", "2", 0), _row("A", "3", 1), _row("A", "4", 0), _row("A", "5", 1), _row("A", "6", 0),
            _row("A", "7", 1, keep=True)]
    #  скор:  0.9 main+, 0.5 main-, 0.35 maybe+, 0.32 maybe-, 0.2 отсев нужный, 0.1 отсев, 0.3 отрасль -> main
    scored = {("A", i): (s, s) for i, s in zip("1234567", (0.9, 0.5, 0.35, 0.32, 0.2, 0.1, 0.3))}
    t = pm.tier_metrics(rows, scored, main=0.4, maybe=0.3)
    assert (t["n_main"], t["n_maybe"], t["n_rel"]) == (3, 2, 4)
    assert t["main"]["precision"] == pytest.approx(2 / 3) and t["main"]["recall"] == pytest.approx(2 / 4)
    assert t["main"]["f05"] == pytest.approx(1.25 * (2 / 3) * 0.5 / (0.25 * (2 / 3) + 0.5))
    assert t["both"]["recall"] == pytest.approx(3 / 4) and t["both"]["precision"] == pytest.approx(3 / 5)
    assert t["maybe_share_of_rel"] == pytest.approx(1 / 4) and t["maybe_precision"] == pytest.approx(1 / 2)
    assert t["main"]["precision5"] == pytest.approx(2 / 3)
    # макро по полу: в main по просьбе, но только если прошло отсев
    macro = [_row("A", "m", 0, floor=True), _row("A", "n", 1, floor=True)]
    sc = {("A", "m"): (0.3, 0.3), ("A", "n"): (0.2, 0.2)}
    assert pm.tier_metrics(macro, sc, 0.4, 0.3, "maybe")["n_main"] == 0
    assert pm.tier_metrics(macro, sc, 0.4, 0.3, "main")["n_main"] == 1


def test_tier_of_row_matches_ranker():
    from ml.ranking import pair_model as pm

    for personal in (0.1, 0.3, 0.35, 0.4, 0.8):
        for keep in (False, True):
            r = _row("A", "x", 1, keep=keep)
            assert pm.tier_of_row(r, personal, 0.4, 0.3, "maybe") == ranker.tier_of(
                _profile("47.71"), _scored(_item("x", hints=("47",) if keep else ()), personal), 0.4, "maybe"), (personal, keep)


def test_platt_calibration_and_thresholds():
    import math

    from ml.ranking import pair_model as pm

    scores = [0.1, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 0.6, 0.7, 0.8] * 3
    rel = [s > 0.4 or (s == 0.35 and n % 2 == 0) for n, s in enumerate(scores)]
    a, b = pm.fit_platt(scores, rel)
    assert a > 0
    t_main, t_maybe = pm.prob_thresholds(a, b)
    assert t_maybe < t_main and 1 / (1 + math.exp(-(a * t_main + b))) == pytest.approx(0.5)
    assert 1 / (1 + math.exp(-(a * t_maybe + b))) == pytest.approx(0.25)


def test_tier_metrics_accepts_per_group_thresholds():
    from ml.ranking import pair_model as pm

    r1, r2 = _row("A", "1", 1), _row("A", "2", 1)
    r1.group, r2.group = "d1", "d2"
    sc = {("A", "1"): (0.45, 0.5), ("A", "2"): (0.45, 0.5)}
    t = pm.tier_metrics([r1, r2], sc, {"d1": 0.4, "d2": 0.5}, {"d1": 0.3, "d2": 0.3})
    assert (t["n_main"], t["n_maybe"]) == (1, 1)


def test_review_paths_per_set():
    from ml.eval import label_review_real as lr

    assert lr.batch_paths(1)[0].name == "review_real.md" and lr.batch_paths(2)[0].name == "review_real_2.md"
    assert lr.batch_paths(1, "real2")[0].name == "review_real2_1.md"
    assert lr.batch_paths(1, "real2")[1].name == "review_real2_1_key.json"


def test_final_eval_refuses_real2_without_human_labels(monkeypatch):
    import pytest

    from ml.eval import final_eval, label_review_real

    monkeypatch.setattr(label_review_real, "human_pairs", lambda name="real": {})
    with pytest.raises(SystemExit, match="нет ручной разметки"):
        final_eval.main(["--name", "real2"])
