"""Служебные обозначения не попадают в тексты для пользователя (ml/agent/humanize.py)."""

from ml.agent.humanize import clean, humanize
from ml.schemas import FollowupAnswer


def test_profile_paths_in_parentheses_are_removed():
    t = "приходятся на Турцию и Нидерланды (geo.fts_countries). Есть кредитный и лизинговый продукт (bank_products.loan, bank_products.leasing), а"
    assert clean(t) == "приходятся на Турцию и Нидерланды. Есть кредитный и лизинговый продукт, а"


def test_bare_paths_become_words():
    assert clean("по данным network.top_counterparties компания платит") == "по данным профиля компания платит"
    assert clean("учитывая market_position компании") == "учитывая положение в отрасли компании"


def test_fact_ids_confidence_and_prompt_blocks():
    assert clean("Прогнозы курса в новости (f6–f10) помечены как assumption") == "Прогнозы курса в новости помечены как предположение"
    assert clean("Прогноз дефицита в ITEM имеет confidence=assumption") == "Прогноз дефицита в новости имеет статус предположения"
    assert clean("ожидание экспертов (confidence: assumption), не факт") == "ожидание экспертов (предположение), не факт"
    assert clean("поэтому bank_view не приводится") == "поэтому позиция банка не приводится"


def test_chunk_id_becomes_document_and_page():
    assert clean("Фрагмент bank_rate_2026_09_11:p3:c7 без подписей").startswith("Фрагмент «Ключевая ставка ЦБ РФ")
    assert "стр. 3" in clean("Фрагмент bank_rate_2026_09_11:p3:c7 без подписей")


def test_plain_text_is_untouched():
    for t in ["Ставка 14% сохранена; кредит дороже на 2,5 п.п.", "Выручка X5 Group выросла", "IT-сектор и M&A в fashion-ритейле"]:
        assert clean(t) == t


def test_humanize_keeps_ids_and_cleans_text():
    a = FollowupAnswer(card_id="c1", question="q", answer="Есть кредит (bank_products.loan).", source_ids=["bank_rate_2026_09_11:p1:c2"],
                       trace_id="t1")
    h = humanize(a)
    assert h.answer == "Есть кредит." and h.source_ids == a.source_ids and h.question == "q"
