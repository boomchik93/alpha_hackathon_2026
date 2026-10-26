"""Выдача ленты: одно событие одной записью, устаревшая аналитика банка не выше новости, которая её сменила."""

from datetime import date, timedelta

from ml.ingest.dedup import event_groups
from ml.ranking import ranker
from ml.schemas import BankProducts, CompanyProfile, DataQuality, Geo, GroupInfo, Industry, Network, ParsedItem, Size

DAY = date(2026, 10, 23)


def _item(item_id: str, title: str, topics=("key_rate",), event_type: str = "monetary_policy", kind: str = "news",
          day: date = DAY, summary: str = "") -> ParsedItem:
    return ParsedItem.model_validate({
        "item_id": item_id, "source": {"kind": kind, "title": title, "published_at": day.isoformat(),
                                       **({"doc_id": item_id} if kind == "bank_analytics" else {})},
        "title": title, "summary": summary, "event_type": event_type, "topics": list(topics), "scope": "economy"})


DECISION = _item("n_dec", "Банк России принял решение сохранить ключевую ставку на уровне 14,00% годовых")
DELIBERATED = _item("n_only", "ЦБ предметно рассматривал на заседании только сохранение уровня ключевой ставки")
MARKET = _item("n_mkt", "Реакцию рынка на решение Банка России по ставке объяснили")
WAGES = _item("n_wages", "Рост зарплат в России замедлился", topics=("labor",), event_type="labor_market")


def test_one_decision_is_one_event():
    g = event_groups([DECISION, DELIBERATED, MARKET, WAGES])
    assert g["n_only"] == g["n_mkt"] == g["n_dec"] == "n_dec"
    assert g["n_wages"] == "n_wages"


def test_shared_words_alone_do_not_make_an_event():
    # Одна тема и тип события, но разные события: общих слов заголовка меньше двух.
    fuel = dict(topics=("fuel",), event_type="fuel_prices")
    a = _item("a", "Подсчитаны масштабы глобального дефицита дизеля", **fuel)
    b = _item("b", "В России рассказали об отсутствии документов на поставки дизтоплива в США", **fuel)
    assert len(set(event_groups([a, b]).values())) == 2
    # Тот же заголовок через неделю — уже другое событие.
    later = _item("n_later", DECISION.title, day=DAY + timedelta(days=7))
    assert len(set(event_groups([DECISION, later]).values())) == 2


def test_bank_analytics_is_not_grouped():
    doc = _item("bank_rate", "Ключевая ставка ЦБ РФ: ставка без изменения на 14,0%", kind="bank_analytics")
    assert event_groups([DECISION, doc])["n_dec"] == "n_dec" and "bank_rate" not in event_groups([DECISION, doc])


def _profile() -> CompanyProfile:
    return CompanyProfile(company_id="U_1", eligible_big=True, industry=Industry(okved_code="47.71"), size=Size(), geo=Geo(),
                          bank_products=BankProducts(loan=True), group=GroupInfo(), network=Network(), data_quality=DataQuality())


def _rank(monkeypatch, items_scores: list[tuple[ParsedItem, float]], flag: bool = True):
    scored = [ranker.Scored(i, s, [], personal=s) for i, s in items_scores]
    monkeypatch.setattr(ranker, "score_all", lambda p, its: (scored, 0))
    monkeypatch.setattr(ranker, "FEED_EVENTS", flag)
    return ranker.rank(_profile(), [i for i, _ in items_scores])


def test_feed_collapses_event_and_keeps_best_placed(monkeypatch):
    kept, dropped = _rank(monkeypatch, [(DELIBERATED, 0.9), (WAGES, 0.8), (DECISION, 0.7), (MARKET, 0.35)])
    assert [s.item.item_id for s in kept] == ["n_only", "n_wages"]
    assert kept[0].related == ["n_dec", "n_mkt"]  # из блока «может быть» тоже сворачивается в запись основной ленты
    assert dropped == 0  # свёрнутые не считаются отсеянными
    kept, _ = _rank(monkeypatch, [(DELIBERATED, 0.9), (WAGES, 0.8), (DECISION, 0.7)], flag=False)
    assert [s.item.item_id for s in kept] == ["n_only", "n_wages", "n_dec"]


def test_old_bank_analytics_goes_below_news_that_superseded_it(monkeypatch):
    old = _item("bank_rate", "Ключевая ставка ЦБ РФ: ставка без изменения на 14,0%", kind="bank_analytics", event_type="other",
                day=DAY - timedelta(days=42))
    fresh_doc = _item("bank_fresh", "Обзор решения по ставке", kind="bank_analytics", event_type="other", day=DAY - timedelta(days=5))
    other_topic = _item("n_fx", "Курс доллара ниже 81 руб.", topics=("fx_gold", "key_rate"), event_type="fx_commodities")
    kept, _ = _rank(monkeypatch, [(old, 0.95), (fresh_doc, 0.9), (other_topic, 0.85), (WAGES, 0.8), (DECISION, 0.7)])
    # Старый обзор — сразу после решения 23.10; обзор недельной давности и новость с key_rate не главной темой его не двигают.
    assert [s.item.item_id for s in kept] == ["bank_fresh", "n_fx", "n_wages", "n_dec", "bank_rate"]
