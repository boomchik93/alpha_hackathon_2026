"""Правила по масштабу события и модель пары в ленте (STATUS_HANDOFF.md, шаг 2)."""

from datetime import date

import numpy as np
import pytest

from ml.ranking import pair_model as pm
from ml.ranking import ranker
from ml.schemas import BankProducts, CompanyProfile, DataQuality, Geo, GroupInfo, Industry, Network, ParsedItem, Size


def _profile(okved: str = "47.71", **bp) -> CompanyProfile:
    return CompanyProfile(company_id="U_1", eligible_big=True, industry=Industry(okved_code=okved), size=Size(), geo=Geo(),
                          bank_products=BankProducts(**bp), group=GroupInfo(), network=Network(), data_quality=DataQuality())


def _item(item_id: str, topics: list[str], scope: str, importance: float = 0.5, hints=(), channels=()) -> ParsedItem:
    return ParsedItem.model_validate({
        "item_id": item_id, "source": {"kind": "news", "title": item_id, "published_at": date.today().isoformat()},
        "title": f"Материал {item_id}", "event_type": "other", "topics": topics, "importance": importance, "freshness": 1.0,
        "scope": scope, "okved_hints": list(hints), "channels": list(channels)})


def test_narrow_event_without_sector_link_gets_no_topic_reasons():
    p = _profile(loan=True)
    other_sector = ranker.score_item(p, _item("s", ["gdp_industry", "key_rate"], "sector", hints=("01.1",)))
    assert other_sector.personal < ranker.MIN_SCORE and not [r for r in other_sector.reasons if r[2] != "industry.okved_name"]
    own_sector = ranker.score_item(p, _item("o", ["consumption"], "sector", hints=("47",)))
    assert own_sector.personal >= ranker.MIN_SCORE  # своя отрасль: темы считаются


def test_economy_event_passes_filter_for_company_without_hooks(monkeypatch):
    p = _profile()  # нет продуктов банка и ВЭД
    big = ranker.score_item(p, _item("e", ["labor"], "economy", importance=0.7))
    small = ranker.score_item(p, _item("e2", ["labor"], "economy", importance=0.3))
    assert big.personal >= ranker.MIN_SCORE > small.personal
    assert any(r[1] == ranker.ECONOMY_REASON for r in big.reasons)
    monkeypatch.setattr(ranker, "ECONOMY_MIN_IMPORTANCE", None)
    assert ranker.score_item(p, _item("e", ["labor"], "economy", importance=0.7)).personal < ranker.MIN_SCORE


def test_fit_logistic_separates_and_threshold_maximizes_f1():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] + 0.1 * rng.normal(size=400) > 0).astype(float)
    w = pm.fit_logistic(X, y, l2=1.0)
    assert w[1] > 3 * abs(w[2])
    assert pm.best_threshold([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) <= 0.8


def _rows(n_days: int = 3) -> list[pm.Row]:
    """Синтетические пары: релевантно, если событие всей экономики или своя отрасль."""
    rows, p = [], _profile(loan=True)
    items = [_item(f"i{d}_{k}", ["inflation"] if k % 2 else ["gdp_industry"], "economy" if k % 2 else "sector",
                   importance=0.7 if k % 2 else 0.3, hints=() if k % 2 else ("01.1",)) for d in range(n_days) for k in range(6)]
    for c in ("A", "B"):
        for it in items:
            f = ranker.pair_features(p, it, 0.1, 0.2)
            rows.append(pm.Row(c, it.item_id, it.item_id.split("_")[0], "real", f, 1 if it.scope == "economy" else 0,
                               it.importance, False, "dev", 0.4 if it.scope == "economy" else 0.1))
    return rows


def test_model_blend_in_score_item_and_cross_val():
    rows = _rows()
    model = pm.fit(rows, l2=1.0, beta=0.3, w_model=0.5, order_mix=0.5)
    assert set(model["w_rel"]) == {"bias", *model["features"]} and 0 < model["threshold"] < 1
    cv = pm.cross_val(rows, [], 1.0, 0.3, 0.5, 0.5)
    assert cv["f1"] == pytest.approx(1.0) and cv["companies"] == 2
    ranker.set_model(model)
    p = _profile(loan=True)
    eco = ranker.score_item(p, _item("e", ["inflation"], "economy", importance=0.7))
    sec = ranker.score_item(p, _item("s", ["gdp_industry"], "sector", importance=0.3, hints=("01.1",)))
    none = ranker.score_item(p, _item("n", ["inflation"], "none", importance=0.7))
    assert eco.personal >= ranker.min_score() > sec.personal and none.personal == 0.0
    assert eco.score > sec.score


def test_paired_delta_sign():
    a = {"per_company": {"x": [1, 1, 1, 0.5], "y": [1, 1, 1, 0.5]}}
    b = {"per_company": {"x": [2, 0, 0, 0.9], "y": [2, 0, 0, 0.7]}}
    d = pm.paired_delta(a, b, n=200)
    assert d["f1"]["delta"] > 0 and d["ndcg5"]["delta"] == pytest.approx(0.3) and d["ndcg5"]["ci95"][0] > 0


def test_superlative_check_and_position_phrase():
    from ml.agent.verifier import SUPERLATIVE
    from ml.schemas import MarketPosition

    assert SUPERLATIVE.search("Компания — крупнейший по выручке игрок выборки банка")
    assert SUPERLATIVE.search("компания является одним из лидеров рынка")
    assert not SUPERLATIVE.search("платёжеспособность крупнейших плательщиков по данным сети")
    p = _profile()
    assert "положение компании не упоминай" in p.position_phrase()
    p = p.model_copy(update={"market_position": MarketPosition(okved_prefix="52.29", digits=4, peers=1544, rank=1, percentile=1.0, note="n")})
    assert p.position_phrase() == "по выручке 1-я из 1544 компаний выборки банка с кодом ОКВЭД 52.29"
    assert not SUPERLATIVE.search(p.position_phrase())


def test_caption_chunks_rank_below_text():
    from ml.retrieval.index import caption_factor, tokenize

    caption = "Рис. 1: Динамика инфляции, % н/н Источники: Росстат, Альфа-Банк Рис. 2: Инфляционные ожидания"
    text = "Ставка без изменения на 14,0%: ЦБ РФ принял решение взять паузу и оставить ставку неизменной, риторика стала жестче"
    assert caption_factor(caption) < 0.5 < caption_factor(text) == 1.0
    assert tokenize("ставка 14,0%") == ["ставка", "14"]  # цифры остаются токенами (однозначные отбрасываются)


def test_bank_attribution_numbers_must_come_from_bank_chunks():
    from ml.agent.verifier import BANK_ATTRIBUTION, bank_attribution_issues

    assert BANK_ATTRIBUTION.search("По прогнозу банка, инфляция 5,5%") and not BANK_ATTRIBUTION.search("Банк России сохранил ставку 14%")
    assert not BANK_ATTRIBUTION.search("Банки ужесточат кредитование")
    from types import SimpleNamespace as NS

    card = NS(ai_inference=NS(text="Банк ожидает ставку 12,0% на конец года. Банк России сохранил 14,0%."),
              why_it_matters=NS(text="есть кредит"), consequences=NS(risks=[], opportunities=[]), suggested_actions=[])
    item = _item("n_x", ["key_rate"], "economy")
    hits = [{"title": "Ставка", "text": "Мы ожидаем снижения ставки до 13,0% к концу года."}]
    issues = bank_attribution_issues(card, _profile(loan=True), item, hits)
    assert len(issues) == 1 and "12" in issues[0] and "14" not in issues[0]
    hits.append({"title": "Ставка", "text": "базовый сценарий: 12,0% на конец года"})
    assert bank_attribution_issues(card, _profile(loan=True), item, hits) == []


def test_fx_gold_facets_separate_ruble_rate_from_gold():
    from ml.topics import facet_titles, facets_overlap, topic_facets

    news = topic_facets(["fx_gold"], "ЦБ понизил официальный курс доллара: 81,5 рубля")
    gold = topic_facets(["fx_gold"], "Рынок золота: доллар США укрепился до 99 пунктов, золотое обеспечение юаня, покупки золота ЦБ")
    assert news == {"fx_gold:fx"} and gold == {"fx_gold:gold"} and not facets_overlap(news, gold)
    whole = topic_facets(["fx_gold"], "Рынок акций открылся без изменений")
    assert whole == {"fx_gold"} and facets_overlap(whole, gold) and facets_overlap(news, whole)
    assert facets_overlap({"key_rate", "fx_gold:gold"}, {"key_rate"})
    assert facet_titles(["fx_gold", "key_rate"], "курс рубля") == ["Валютный курс рубля", "Ключевая ставка и стоимость денег"]


def test_empty_bank_view_allowed_when_card_says_no_bank_position():
    from ml.agent.verifier import NO_BANK_POSITION

    for t in ("Позиции банка по курсу рубля во фрагментах нет.", "Фрагменты аналитики банка о другом: рынок золота.",
              "Во фрагментах отсутствует позиция банка по теме события", "позиция банка по теме не найдена"):
        assert NO_BANK_POSITION.search(t), t
    assert not NO_BANK_POSITION.search("Банк ожидает сохранения ставки на уровне 14%")


def test_no_position_excuse_not_accepted_with_contradiction():
    import inspect

    from ml.agent import verifier

    src = inspect.getsource(verifier.rule_check)
    assert "no_position = not contradicts and" in src


def test_spelled_units_and_ranges_confirm_points():
    from ml.agent.verifier import ungrounded_numbers

    src = ["экономист предполагает снижение ставки на 25 базисных пунктов до 13,75%",
           "повышение траектории ставки в пределах 0,5-1 процентного пункта"]
    assert ungrounded_numbers("допускает снижение на 25 б.п. до 13,75%", src) == []
    assert ungrounded_numbers("пересмотр на 0,5-1 п.п.", src) == []
    assert ungrounded_numbers("снижение на 0,25 п.п.", src) == []
    assert ungrounded_numbers("снижение на 75 б.п.", src) == ["75"]  # 50 б.п. = 0,5 п.п. есть в источнике
    assert ungrounded_numbers("ставка 14%", ["стр. 14"]) == ["14"]  # голое целое процент по-прежнему не подтверждает


def test_absent_bank_product_is_flagged():
    from types import SimpleNamespace as NS

    from ml.agent.verifier import product_issues

    def card(t):
        return NS(why_it_matters=NS(text=t), ai_inference=NS(text=""), consequences=NS(risks=[], opportunities=[]), suggested_actions=[])

    p = _profile(leasing=True)  # кредита нет
    assert product_issues(card("У компании есть кредитный продукт в банке"), p)
    assert not product_issues(card("У компании есть лизинговый продукт в банке"), p)
    assert not product_issues(card("Стоит проверить, есть ли кредит у дочерних компаний; предложить кредитную линию"), p)
    assert not product_issues(card("Отсутствие действующих кредитных продуктов в банке; у компании нет кредитных продуктов"), p)
    assert not product_issues(card("компания не использует кредитные продукты"), p)


def test_review_batch_answers_and_random_pick(tmp_path, monkeypatch):
    import json

    from ml.eval import label_review_real as lr

    text = "## 1. `U_1` × `r_a`\n\nОценка: 1\n\n## 2. `U_2` × `r_a`\n\nОценка: ?\n\n## 3. `U_3` × `r_b`\n\nОценка: \n"
    assert lr.read_answers(text) == {("U_1", "r_a"): 1}  # «?» и пустая строка не считаются
    pairs = [{"company": f"U_{i}", "item": f"r_{i % 3}", "llm": 0} for i in range(30)]
    got = lr.pick_random(pairs, 10, seed=1, per_item=2)
    assert len(got) == 6 and max(sum(g["item"] == it for g in got) for it in ("r_0", "r_1", "r_2")) == 2
    review, key = tmp_path / "r.md", tmp_path / "k.json"
    key.write_text(json.dumps({"pairs": [{"company": "U_1", "item": "r_a", "llm": 0, "sample": "random", "kept": False},
                                         {"company": "U_2", "item": "r_a", "llm": None, "grade_a": 0, "grade_b": 1,
                                          "disputed": True, "sample": "random", "kept": False}]}))
    review.write_text("## 1. `U_1` × `r_a`\n\nОценка: 1\n\n## 2. `U_2` × `r_a`\n\nОценка: 1\n")
    monkeypatch.setattr(lr, "batch_paths", lambda b, name="real": (review, key))
    res = lr.score(batch=2)
    assert res["filled"] == 2 and res["disputed_resolved"]["human_sided_with_b"] == 1
    assert res["by_layer"]["все"]["n"] == 1 and res["by_layer"]["все"]["binary"] == 0.0
