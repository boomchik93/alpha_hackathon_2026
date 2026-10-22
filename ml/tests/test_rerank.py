"""Переранжирование моделью (REVIEW_PLAN.md, T13)."""

from dataclasses import dataclass

import pytest

from ml.eval.run_eval import _paired_ci, eval_rerank
from ml.ranking import rerank
from ml.ranking.ranker import MIN_SCORE, Scored
from ml.schemas import ParsedItem


def _scored(i: int, personal: float) -> Scored:
    item = ParsedItem.model_validate({"item_id": f"i{i}", "source": {"kind": "news", "title": "t", "published_at": "2026-09-10"},
                                      "title": f"t{i}", "event_type": "other", "importance": 0.5})
    return Scored(item, personal, [], personal=personal)


def test_candidates_are_top_n_plus_band_below_threshold():
    scored = [_scored(i, 0.9 - 0.04 * i) for i in range(20)]  # от 0,9 до 0,14
    cands = [s.item.item_id for s in rerank.candidates(scored, MIN_SCORE, n=3)]
    assert cands[:3] == ["i0", "i1", "i2"]
    assert all(MIN_SCORE - rerank.RERANK_BAND <= s.personal < MIN_SCORE for s in scored if s.item.item_id in cands[3:])
    assert set(cands[3:]) == {s.item.item_id for s in scored[3:] if MIN_SCORE - 0.1 <= s.personal < MIN_SCORE}


def test_apply_mixes_scores_and_reorders_only_scored_items():
    scored = [_scored(0, 0.8), _scored(1, 0.6), _scored(2, 0.3)]
    out = rerank.apply(scored, {"i0": 0.1, "i1": 0.9}, weight=0.5, beta=0.0)
    by = {s.item.item_id: s for s in out}
    assert by["i0"].personal == pytest.approx(0.45) and by["i1"].personal == pytest.approx(0.75)
    assert by["i2"].personal == 0.3  # без оценки модели остаётся как был
    assert [s.item.item_id for s in out] == ["i1", "i0", "i2"]
    assert [s.score for s in rerank.apply(scored, {}, 0.5, 0.0)] == [0.8, 0.6, 0.3]


def test_weight_zero_equals_current_feed():
    scored = [_scored(0, 0.8), _scored(1, 0.6)]
    same = rerank.apply(scored, {"i0": 0.0, "i1": 1.0}, weight=0.0, beta=0.0)
    assert [(s.item.item_id, s.personal) for s in same] == [("i0", 0.8), ("i1", 0.6)]


def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("ML_RERANK", raising=False)
    assert not rerank.enabled()
    monkeypatch.setenv("ML_RERANK", "1")
    assert rerank.enabled()


def test_paired_ci_covers_zero_for_identical_and_not_for_shift():
    a = [0.5, 0.6, 0.7, 0.8, 0.9]
    d, lo, hi = _paired_ci(a, a, n=200)
    assert d == 0 and lo == 0 == hi
    d, lo, hi = _paired_ci([x + 0.1 for x in a], a, n=200)
    assert d == pytest.approx(0.1) and lo > 0.09


@pytest.mark.parametrize("flag", [True])
def test_rerank_in_feed_with_mock(monkeypatch, flag):
    from ml.ranking import ranker
    from ml.tests.test_ranker import _item, _profile

    monkeypatch.setenv("ML_RERANK", "1")
    kept, _ = ranker.rank(_profile(), [_item("rate", ["key_rate"], 0.5), _item("fx", ["fx_gold"], 0.5)])
    assert kept  # мок-клиент по умолчанию (conftest), платных вызовов нет
