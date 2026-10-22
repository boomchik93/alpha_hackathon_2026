"""Предрасчёт карточек (REVIEW_PLAN.md, T18)."""

import pytest

from ml import api, precompute
from ml.tests.conftest import needs_data


@pytest.fixture(autouse=True)
def own_runs_dir(tmp_path, monkeypatch):
    """Свой каталог карточек: карточки других тестов не должны считаться готовыми."""
    monkeypatch.setenv("ML_RUNS_DIR", str(tmp_path))


@needs_data
def test_dry_run_reports_estimate_and_makes_no_cards(capsys):
    precompute.main(["--companies", "U_1595,group_134463_sb", "--top", "3", "--dry-run"])
    err = capsys.readouterr().err
    assert "компаний 2" in err and "оценка" in err and "руб." in err
    todo, cached = precompute.plan(["U_1595"], 3, None)
    assert len(todo) + cached == 3 and cached == 0


@needs_data
def test_precompute_two_companies_on_mock_and_skips_cached(capsys):
    ids = ["U_1595", "group_134463_sb"]
    todo, _ = precompute.plan(ids, 2, None)
    assert todo
    res = precompute.run(todo, max_rub=50, workers=2, role=None)
    assert res["made"] == len(todo) and not res["failed"] and res["skipped_budget"] == 0
    for cid, item in todo:
        assert api.cached_insight(cid, item) is not None
    todo2, cached2 = precompute.plan(ids, 2, None)
    assert todo2 == [] and cached2 == len(todo)  # готовые пропускаются


@needs_data
def test_budget_stops_launching_cards(monkeypatch):
    class Ledger:
        n = 0

        def spent_today(self):
            Ledger.n += 1
            return 0.0 if Ledger.n == 1 else 99.0  # после старта расход сразу выше лимита

    monkeypatch.setattr(precompute, "SpendLedger", Ledger)
    res = precompute.run([("U_1595", "n_2026_09_11_cbr_rate")] * 3, max_rub=1.0, workers=1, role=None)
    assert res["made"] == 0 and res["skipped_budget"] == 3


def test_demo_is_30_largest():
    ids = precompute.resolve_companies("demo")
    revs = {p.company_id: p.size.revenue_rub or 0 for p in api.list_companies()}
    assert len(ids) == min(30, len(revs)) and all(revs[a] >= revs[b] for a, b in zip(ids, ids[1:]))
    assert precompute.resolve_companies("U_1,U_2") == ["U_1", "U_2"]
