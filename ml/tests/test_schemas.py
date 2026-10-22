"""Эталонный пример карточки: схема валидна, у каждого утверждения есть источник."""

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from ml.schemas import PROFILE_SOURCE_ID, Claim, InsightCard

REFERENCE = {
    "relevance": {"score": 0.9, "reasons": ["есть кредитные продукты", "инвестиции частично за счёт заёмных средств"]},
    "what_happened": {"text": "Банк России сохранил ключевую ставку на уровне 14% после серии снижений. Решение принято 11 сентября 2026.", "facts": ["f1"]},
    "why_it_matters": {"text": "Кредитная нагрузка компании остаётся высокой: удешевление заёмных денег откладывается.", "company_features_used": ["bank_products.loan"]},
    "consequences": {
        "risks": [{"text": "Стоимость обслуживания долга не снизится в ближайшей перспективе.", "horizon": "6-12 мес", "evidence": ["src_bank_rate"]}],
        "opportunities": [{"text": "Свободную ликвидность выгоднее размещать на депозитах.", "horizon": "3-6 мес", "evidence": ["profile"]}],
    },
    "bank_view": {"text": "По оценке аналитиков банка, риторика регулятора стала жестче.", "source_ids": ["src_bank_rate"]},
    "ai_inference": {"text": "Разумно пересмотреть график заимствований.", "source_ids": ["profile", "src_bank_rate"]},
    "suggested_actions": [
        {"type": "action", "text": "Проверить условия рефинансирования действующих кредитов.", "evidence": ["profile"]},
        {"type": "question_for_analysis", "text": "Какая доля инвестпрограммы зависит от заёмного финансирования?", "is_model_inference": True},
    ],
    "limitations": ["Нет данных о валютной структуре долга"],
    "card_id": "ins_U_1_n_rate",
    "company_id": "U_1",
    "item_ids": ["n_rate"],
    "sources": [
        {"id": "src_bank_rate", "kind": "bank_analytics", "title": "Ключевая ставка ЦБ РФ", "published_at": date(2026, 9, 11), "page": 1},
        {"id": "src_n_rate", "kind": "news", "title": "ЦБ сохранил ставку"},
    ],
    "trace_id": "tr_test",
    "meta": {"model": "glm-5.3-flash", "prompt_version": "interpret@v1", "generated_at": datetime.now(timezone.utc)},
}


def test_reference_card_valid_and_sourced():
    card = InsightCard.model_validate(REFERENCE)
    known = {s.id for s in card.sources} | {PROFILE_SOURCE_ID}
    for claim in [*card.consequences.risks, *card.consequences.opportunities, *card.suggested_actions]:
        assert claim.evidence or claim.is_model_inference
        assert set(claim.evidence) <= known
    assert card.bank_view and set(card.bank_view.source_ids) <= known


def test_claim_without_evidence_must_be_flagged():
    with pytest.raises(ValidationError):
        Claim(text="голое утверждение")
    assert Claim(text="вывод модели", is_model_inference=True)


def test_unknown_source_rejected():
    bad = {**REFERENCE, "ai_inference": {"text": "x", "source_ids": ["src_missing"]}}
    with pytest.raises(ValidationError, match="неизвестные источники"):
        InsightCard.model_validate(bad)


def test_bank_view_must_cite_bank_analytics():
    bad = {**REFERENCE, "bank_view": {"text": "x", "source_ids": ["src_n_rate"]}}
    with pytest.raises(ValidationError, match="не на аналитику банка"):
        InsightCard.model_validate(bad)


def test_bank_view_may_be_null():
    InsightCard.model_validate({**REFERENCE, "bank_view": None})


def test_normalize_draft_fixes_refs():
    from ml.agent.orchestrator import item_source_id, normalize_draft
    from ml.schemas import InsightDraft, ParsedItem

    item = ParsedItem.model_validate({"item_id": "n_x", "source": {"kind": "news", "title": "t", "published_at": "2026-09-11"},
                                      "title": "t", "event_type": "other"})
    from ml.schemas import BankProducts, CompanyProfile, DataQuality, Geo, GroupInfo, Industry, Network, Size

    prof = CompanyProfile(company_id="U_1", eligible_big=True, industry=Industry(okved_code="47.71", sector_tags=["розница"]),
                          size=Size(), geo=Geo(fts_countries=[]), bank_products=BankProducts(loan=True),
                          group=GroupInfo(), network=Network(), data_quality=DataQuality())
    raw = {k: v for k, v in REFERENCE.items() if k in InsightDraft.model_fields}
    raw = {**raw, "why_it_matters": {"text": "x", "company_features_used": ["sector_tags", "geo.fts_countries"]},
           "consequences": {"risks": [{"text": "r", "evidence": ["f3", "n_x:f1", "src_bank_rate"]}], "opportunities": []}}
    d = normalize_draft(InsightDraft.model_validate(raw), prof, item, {"src_bank_rate"})
    assert d.consequences.risks[0].evidence == [item_source_id(item), "src_bank_rate"]
    assert d.why_it_matters.company_features_used == ["industry.sector_tags"]


def test_ungrounded_numbers():
    from ml.agent.verifier import ungrounded_numbers

    src = ["ставка 14,0% до конца года, прогноз на 2027 г. 12,0%"]
    assert ungrounded_numbers("ставка 14% и 12% в 2027", src) == []
    assert ungrounded_numbers("ранее ожидалось 13,5%", src) == ["13.5"]


def test_ungrounded_numbers_rounding_and_profile():
    from ml.agent.verifier import ungrounded_numbers

    assert ungrounded_numbers("доходности до 16,3%", ["выросли до 16,26%"]) == []
    assert ungrounded_numbers("ставка 14%", ["ставка 13,5%"]) == ["14"]
    assert ungrounded_numbers("выручка 16,5 млрд", [], {16461799989 / 1e9}) == []


def test_dates_are_not_numbers():
    from ml.agent.verifier import ungrounded_numbers

    assert ungrounded_numbers("решение от 11.09.2026", []) == []


def test_normalize_draft_profile_path_as_evidence():
    from ml.agent.orchestrator import normalize_draft
    from ml.schemas import (BankProducts, CompanyProfile, DataQuality, Geo, GroupInfo, Industry, InsightDraft, Network,
                            ParsedItem, Size)

    item = ParsedItem.model_validate({"item_id": "n_x", "source": {"kind": "news", "title": "t", "published_at": "2026-09-11"},
                                      "title": "t", "event_type": "other"})
    prof = CompanyProfile(company_id="U_1", eligible_big=True, industry=Industry(sector_tags=["розница"]), size=Size(), geo=Geo(),
                          bank_products=BankProducts(loan=True), group=GroupInfo(), network=Network(), data_quality=DataQuality())
    raw = {k: v for k, v in REFERENCE.items() if k in InsightDraft.model_fields}
    raw = {**raw, "ai_inference": {"text": "x", "source_ids": ["industry.sector_tags", "bank_products.loan"]}}
    d = normalize_draft(InsightDraft.model_validate(raw), prof, item, {"src_bank_rate"})
    assert d.ai_inference.source_ids == ["profile"]


def test_okved_code_counts_as_profile_number():
    from ml.agent.verifier import profile_numbers, ungrounded_numbers
    from ml.schemas import BankProducts, CompanyProfile, DataQuality, Geo, GroupInfo, Industry, Network, Size

    prof = CompanyProfile(company_id="U_1", eligible_big=True, industry=Industry(okved_code="49.41"), size=Size(), geo=Geo(),
                          bank_products=BankProducts(), group=GroupInfo(), network=Network(), data_quality=DataQuality())
    assert ungrounded_numbers("ОКВЭД 49.41", [], profile_numbers(prof)) == []


def test_bank_quotes_verbatim_only():
    from ml.agent.bank_quotes import assemble, check_quotes
    from ml.schemas import BankView

    hits = [{"id": "c1", "doc_id": "d", "title": "Ставка", "published_at": "2026-09-11", "page": 2, "stale": False,
             "text": "Теперь мы ожидаем   сохранения ставки на уровне 14,0% до конца года. Прогноз на 2027 г. 12,0%."}]
    view = BankView(source_ids=["c1"], quotes=[
        {"source_id": "c1", "text": "мы ожидаем сохранения ставки на уровне 14,0% до конца года"},
        {"source_id": "c1", "text": "банк ожидает снижения ставки до 12% в этом году"},
    ])
    ok, issues = check_quotes(view, hits)
    assert len(ok) == 1 and len(issues) == 1
    bv = assemble(ok, hits)
    assert "11.09.2026" in bv.text and "стр. 2" in bv.text and bv.source_ids == ["c1"]


def test_claim_prune_keeps_one_action():
    from ml.agent.claim_verify import REMOVED_NOTE, Verdict, prune

    card = InsightCard.model_validate(REFERENCE)
    verdicts = {"risks[0]": Verdict(id="risks[0]", verdict="unsupported"),
                "suggested_actions[0]": Verdict(id="suggested_actions[0]", verdict="stretch"),
                "suggested_actions[1]": Verdict(id="suggested_actions[1]", verdict="stretch")}
    pruned, removed = prune(card, verdicts)
    assert not pruned.consequences.risks and len(pruned.suggested_actions) == 1
    assert pruned.suggested_actions[0].type == "question_for_analysis" and REMOVED_NOTE in pruned.limitations
    assert len(removed) == 2


def test_table_row_is_not_a_bank_quote():
    from ml.agent.bank_quotes import table_like

    assert table_like("Курс, конец года, 62 74 74 70 90 102 78,2 87,2 95,0")
    assert not table_like("После заседания на дальнем участке кривой ОФЗ доходность выросла до 16,3%.")


def _profile_with_loan():
    from ml.schemas import BankProducts, CompanyProfile, DataQuality, Geo, GroupInfo, Industry, Network, Size

    return CompanyProfile(company_id="U_1", eligible_big=True, industry=Industry(okved_code="49.41", okved_name="Перевозки"),
                          size=Size(), geo=Geo(foreign_trade=False), bank_products=BankProducts(loan=True), group=GroupInfo(),
                          network=Network(), data_quality=DataQuality())


def test_channels_validated_by_code():
    from ml.agent.channels import validate
    from ml.schemas import ImpactChannel

    raw = [
        ImpactChannel(channel="cost_of_money", strength="strong", profile_features=["loan"], evidence=["src_n", "profile"]),
        ImpactChannel(channel="fx", strength="strong", profile_features=["geo.fts_countries"], evidence=["src_n"]),  # нет ВЭД
        ImpactChannel(channel="demand", strength="weak", profile_features=["industry.okved_name"], evidence=[]),     # нет источника
        ImpactChannel(channel="weather", strength="strong", profile_features=["industry.okved_name"], evidence=["src_n"]),
        ImpactChannel(channel="labor_costs", strength="none"),
    ]
    kept, dropped = validate(raw, _profile_with_loan(), {"src_n", "bank:c1"})
    assert [c.channel for c in kept] == ["cost_of_money"]
    assert kept[0].profile_features == ["bank_products.loan"] and kept[0].evidence == ["src_n"]
    assert len(dropped) == 4


def test_channels_cap_relevance():
    from ml.agent.channels import NO_CHANNEL_MAX, WEAK_ONLY_MAX, enforce
    from ml.schemas import ImpactChannel

    card = InsightCard.model_validate(REFERENCE)
    assert enforce(card, [], ["Спрос: нет источника"]).relevance.score == NO_CHANNEL_MAX
    weak = [ImpactChannel(channel="demand", strength="weak", profile_features=["industry.okved_name"], evidence=["src_n_rate"])]
    capped = enforce(card, weak, [])
    assert capped.relevance.score == WEAK_ONLY_MAX and capped.channels == weak
    assert any("Спрос" in l for l in enforce(card, [], ["Спрос: нет источника"]).limitations)


def test_integer_amounts_with_units_are_checked():
    from ml.agent.verifier import ungrounded_numbers

    assert ungrounded_numbers("импорт вырос до $340 млрд", ["прочее"]) == ["340"]
    assert ungrounded_numbers("импорт вырос до $340 млрд", ["таблица: Импорт 340"]) == []
    assert ungrounded_numbers("вырос до 340 млрд долл.", ["импорт 340 млрд долл."]) == []
    assert ungrounded_numbers("в 2 раза", ["рост 3,5%"]) == ["2"]
    assert ungrounded_numbers("в 2 раза", ["выручка выросла в 2 раза"]) == []
    assert ungrounded_numbers("маржа выросла на 3 п.п.", ["маржа 21 п.п."]) == ["3"]


def test_integer_amounts_rounding_scale_only():
    from ml.agent.verifier import profile_numbers, ungrounded_numbers

    assert ungrounded_numbers("выручка 16 млрд", [], {16461799989 / 1e9}) == []
    assert ungrounded_numbers("выручка 16 млрд", [], {19.2}) == ["16"]
    assert ungrounded_numbers("курс 95 руб. за доллар", ["курс 95,0"]) == []
    assert ungrounded_numbers("14 п.п.", ["13,5 п.п."]) == ["14"]
    # проценты как раньше: целое подтверждается только точным совпадением
    assert ungrounded_numbers("ставка 14%", ["ставка 13,5%"]) == ["14"]


def test_percent_not_confirmed_by_bare_integer():
    from ml.agent.verifier import ungrounded_numbers

    # голое целое источника (дата, номер страницы) процент не подтверждает
    assert ungrounded_numbers("ставка 14%", ["заседание 14 декабря"]) == ["14"]
    assert ungrounded_numbers("ставка 14%", ["см. стр. 14, ставка 13,5%"]) == ["14"]
    assert ungrounded_numbers("рост на 3 п.п.", ["таблица 3"]) == ["3"]
    assert ungrounded_numbers("в 2 раза", ["пункт 2"]) == ["2"]
    # процент, дробь или та же единица подтверждают
    assert ungrounded_numbers("ставка 14%", ["ставка 14,0%"]) == []
    assert ungrounded_numbers("ставка 14%", ["ставка 14 %"]) == []
    assert ungrounded_numbers("рост на 3 п.п.", ["выросла на 3 п.п."]) == []
    # суммы по-прежнему подтверждаются и целым из таблицы
    assert ungrounded_numbers("импорт $340 млрд", ["таблица 340"]) == []


def test_years_dates_and_codes_are_not_claims():
    from ml.agent.verifier import claimed_numbers, ungrounded_numbers

    for text in ("в 2026 году", "по итогам 1П26", "прогноз на 7М26", "с 2019 по 2024 год", "ОКВЭД 46 и 49", "решение от 11.09.2026"):
        assert ungrounded_numbers(text, []) == [], text
    assert claimed_numbers("в 2026 году") == []


def test_thousands_separator_amounts():
    from ml.agent.verifier import ungrounded_numbers

    assert ungrounded_numbers("долг 12 000 руб.", ["долг 12 000"]) == []
    assert ungrounded_numbers("долг 12 000 руб.", ["долг 4 500"]) == ["12000"]


def test_topics_match_word_start_only():
    from ml.topics import detect_topics

    assert "key_rate" not in detect_topics("Сроки поставки оборудования, доставка задерживается, выставка ставок", 1)
    for text in ("ключевой ставки", "ставку", "ставка"):
        assert "key_rate" in detect_topics(text, 1), text
    assert "key_rate" in detect_topics("Совет директоров снизил ключевую ставку")
    assert detect_topics("Ключевая Ставка") == ["key_rate"]
