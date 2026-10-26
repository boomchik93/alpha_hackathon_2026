"""Самопроверка карточки: правила без LLM, опционально LLM-проверка (ML_LLM_VERIFY=1)."""

from __future__ import annotations

import os
import re

from pydantic import BaseModel

from ml.llm.client import LLMClient
from ml.prompts import render
from ml.safety import INJECTION
from ml.schemas import CompanyProfile, InsightCard, ParsedItem, SourceKind
from ml.topics import facets_overlap, topic_facets

BANK_HIT_MIN_SCORE = 0.35
LOW_RELEVANCE = 0.3


class VerifyResult(BaseModel):
    ok: bool
    issues: list[str] = []


def _path_value(profile: CompanyProfile, path: str):
    obj = profile.model_dump()
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


_NUM = re.compile(r"\d+(?:[.,]\d+)?(?=\s*%)|\d+[.,]\d+")
_DATE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")
# Компания названа крупнейшей или лидером (interpret, правило 12): в профиле только место по выручке среди компаний выборки банка.
# «Крупнейшие плательщики/контрагенты» о сети платежей не ловятся: нужен субъект-компания или роль на рынке.
SUPERLATIVE = re.compile(
    r"компани[яи]\s*(?:[—–-]|является|выступает)?\s*(?:одн\w+ из\s+)?(?:крупнейш\w*|лидер\w*|ведущ\w*)"
    r"|(?:является|выступает)\s+(?:одн\w+ из\s+)?(?:крупнейш\w*|лидер\w*|ведущ\w*)"
    r"|(?:крупнейш\w*|лидер\w*|ведущ\w*)\s+(?:по выручке\s+)?(?:игрок\w*|участник\w*|производител\w*|поставщик\w*|оптов\w*|"
    r"перевозчик\w*|трейдер\w*|дистрибьютор\w*|ритейлер\w*|фармпосредник\w*|сет\w+)", re.I)
# Утверждение приписано банку (не Банку России и не банкам вообще): «банк ожидает», «по прогнозу банка», «аналитики Альфа-Банка».
BANK_ATTRIBUTION = re.compile(
    r"Альфа-Банк\w*|(?<![А-Яа-яЁё])(?:банк|банка|банку|банком)(?![А-Яа-яЁё])(?!\s+России)", re.I)
# «Позиции банка по теме во фрагментах нет», «фрагменты банка не относятся к событию» (interpret, правило 25).
NO_BANK_POSITION = re.compile(
    r"позици\w*\s+(?:Альфа-)?банка[^.;]{0,80}?(?:нет|отсутству\w*|не\s+(?:найден\w*|приведен\w*|содерж\w*|относ\w*|касает\w*))"
    r"|(?:нет|отсутству\w*)[^.;]{0,40}?позици\w*\s+(?:Альфа-)?банка"
    r"|фрагмент\w*\s+(?:аналитики\s+)?банка[^.;]{0,80}?(?:не\s+(?:относ\w*|касает\w*|по\s+теме|о\s+теме)|о\s+другом)", re.I)
# Продукт банка, которого у компании нет по профилю (bank_products.X = false), но карточка пишет «у компании есть кредит».
# «Есть ли кредит» (вопрос) не считается.
PRODUCT_WORDS = {"loan": r"кредит\w*", "leasing": r"лизинг\w*", "deposit": r"депозит\w*", "guarantee": r"гаранти\w*",
                 "factoring": r"факторинг\w*"}
_HAS = r"(?:есть(?!\s+ли\b)|имеет\w*|использует)\s+(?:[а-яё]+\s+){0,2}"
_NEG_BEFORE = re.compile(r"(?:\bнет|\bне|\bбез|отсутств\w*)\s*(?:[а-яё]+\s+){0,2}$", re.I)
_SENT = re.compile(r"[^.!?;]+(?:[.!?;]+|$)")
ADVICE = re.compile(r"\b(стоит|следует|целесообразно|рекомендуем\w*|рекомендуется|необходимо|для компании)\b", re.I)

# Целое число с единицей или знаком валюты: «340 млрд», «95 руб.», «$340», «в 2 раза», «3 п.п.». Левая граница: не буква, не цифра,
# не точка и не запятая («1П26», «7М26», коды ОКВЭД); справа не десятичная часть (дроби ловит _NUM).
_INT = r"\d{1,3}(?:[ \u00a0]\d{3})+|\d+"
_SCALE_UNIT = r"(?:млрд|млн|трлн|тыс)\w*"
_MONEY_UNIT = r"(?:руб(?:л[а-яё]*|\.|(?![а-яё]))|долл\w*|юан\w*|евро)"
_EXACT_UNIT = r"(?:п\.\s?п\.?|б\.\s?п\.?|раз(?:а|ов)?(?![а-яё]))"
_INT_UNIT = re.compile(
    rf"(?<![\w.,])(?P<cur>[$€₽])?\s?(?P<n>{_INT})(?![.,]?\d)"
    rf"(?:\s*(?P<unit>{_SCALE_UNIT}|{_MONEY_UNIT}|{_EXACT_UNIT}|[$€₽]))?", re.I)
# Единицы процентных и базисных пунктов в источнике бывают словами («25 базисных пунктов», «0,5-1 процентного пункта»),
# а число бывает концом диапазона: без этого «25 б.п.» и «1 п.п.» в карточке считались выдуманными (real, what_happened).
_SPELLED_UNIT = r"(?:процентн\w*\s+пункт\w*|базисн\w*\s+пункт\w*)"
_STRICT_SRC = re.compile(
    rf"(?<![\w.,])(?P<a>\d+(?:[.,]\d+)?)(?:\s*(?:[-–—]|до)\s*(?P<b>\d+(?:[.,]\d+)?))?\s*(?P<unit>{_EXACT_UNIT}|{_SPELLED_UNIT}|%)", re.I)
_ANY_NUM = re.compile(rf"{_INT}(?:[.,]\d+)?|\d+(?:[.,]\d+)?")


def _clean_float(n: str) -> tuple[str, float, int]:
    """Запись числа в нормальном виде: (запись, значение, знаков после запятой)."""
    n = n.replace("\u00a0", "").replace(" ", "").replace(",", ".")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n, float(n), len(n.split(".")[1]) if "." in n else 0


def claimed_numbers(text: str) -> list[tuple[str, float, int, bool, bool]]:
    """Что проверять в тексте карточки: (запись, значение, знаков после запятой, допустимо ли округление источника,
    строгая сверка).

    Дроби и проценты как раньше, плюс целые числа с единицей (млрд, млн, тыс, трлн, руб., долл., $, п.п., б.п., раз)
    или знаком валюты. Годы без денежной единицы и даты не проверяются. Округление допускается для сумм
    (и масштаба млрд, млн, тыс), но не для процентов, п.п., б.п. и «раз». Проценты, п.п., б.п. и «раз» сверяются
    строго (`strict_source_numbers`): «14%» не подтверждается номером страницы «14» или датой «14 декабря»."""
    out = []
    text = _DATE.sub(" ", text)
    for m in _NUM.finditer(text):
        n, v, dec = _clean_float(m.group())
        out.append((n, v, dec, False, text[m.end():].lstrip().startswith("%")))
    for m in _INT_UNIT.finditer(text):
        unit = (m.group("unit") or "").lower()
        if not (m.group("cur") or unit):
            continue
        n, v, dec = _clean_float(m.group("n"))
        exact = bool(re.fullmatch(_EXACT_UNIT, unit, re.I))
        if 1900 <= v <= 2099 and not (m.group("cur") or (unit and not exact)):
            continue
        out.append((n, v, dec, not exact, exact))
    return out


def strict_source_numbers(text: str) -> list[float]:
    """Подтверждение для процентов, п.п., б.п. и «раз»: как до проверки целых чисел (проценты и дроби источника)
    плюс целые числа источника с той же единицей («на 3 п.п.», «в 2 раза»). Голые целые сюда не входят."""
    text = _DATE.sub(" ", text)
    out = [_clean_float(m)[1] for m in _NUM.findall(text)]
    for m in _INT_UNIT.finditer(text):
        if re.fullmatch(_EXACT_UNIT, (m.group("unit") or "").lower(), re.I):
            out.append(_clean_float(m.group("n"))[1])
    for m in _STRICT_SRC.finditer(text):
        unit = m.group("unit").lower()
        for x in (m.group("a"), m.group("b")):
            if x is None:
                continue
            v = _clean_float(x)[1]
            out.append(v)
            if unit.startswith(("б", "базисн")):
                out.append(v / 100)  # «25 б.п.» подтверждает «0,25 п.п.»
            elif unit.startswith(("п", "процентн")):
                out.append(v * 100)  # «0,25 п.п.» подтверждает «25 б.п.»
    return out


def source_numbers(text: str) -> list[tuple[str, float]]:
    """Что считается подтверждением в источниках: все числа, включая целые без единиц (в таблице «Импорт 340» подтверждает
    «$340 млрд»), кроме дат. Число с пробелом-разделителем разрядов даётся и целиком, и частями."""
    out = []
    for m in _ANY_NUM.findall(_DATE.sub(" ", text)):
        n, v, _ = _clean_float(m)
        out.append((n, v))
        if " " in m.strip() or "\u00a0" in m:
            out += [(p, float(p)) for p in re.findall(r"\d+", m)]
    return out


def _nums(text: str) -> set[str]:
    """Записи чисел источника: «14,0%» и «14%» считаются одним числом."""
    return {n for n, _ in source_numbers(text)}


def profile_numbers(profile: CompanyProfile) -> set[float]:
    """Числа профиля в тех масштабах, в которых их пишет модель: руб., тыс., млн, млрд, доли в процентах."""
    out: set[float] = set()

    def walk(obj) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            out.update({float(obj), obj / 1e3, obj / 1e6, obj / 1e9, obj * 100})
        elif isinstance(obj, str):
            out.update(v for _, v in source_numbers(obj))  # коды ОКВЭД вроде «49.41»

    walk(profile.prompt_view())
    return out


def ungrounded_numbers(text: str, sources: list[str], extra: set[float] | frozenset[float] = frozenset()) -> list[str]:
    """Числа текста, которых нет в источниках. Округление допускается, но только до знаков, записанных в тексте:
    «16,3» подтверждается «16,26», а «14%» не подтверждается «13,5%». Целая сумма с масштабом или валютой («16 млрд»)
    подтверждается и округлённым нецелым значением источника (16,46). Проценты, п.п., б.п. и «раз» подтверждаются
    только числами источника с процентом, дробью или той же единицей; числа профиля подходят для всех."""
    loose = {v for s in sources for _, v in source_numbers(s)} | set(extra)
    strict = {v for s in sources for v in strict_source_numbers(s)} | set(extra)
    loose |= strict  # пересчёт б.п. ↔ п.п. («25 базисных пунктов» подтверждает «0,25 п.п.»)
    bad = set()
    for n, v, dec, rounds, is_strict in claimed_numbers(text):
        allowed = strict if is_strict else loose
        if dec:
            ok = any(abs(round(a, dec) - v) < 1e-9 for a in allowed)
        else:
            ok = any(abs(a - v) < 1e-9 or (rounds and a != int(a) and abs(a - v) <= 0.5) for a in allowed)
        if not ok:
            bad.add(n)
    return sorted(bad)


def card_texts(card: InsightCard) -> list[str]:
    """Все тексты карточки: из них ответ на уточняющий вопрос может брать цифры."""
    claims = [*card.consequences.risks, *card.consequences.opportunities, *card.suggested_actions]
    return [card.what_happened.text, card.why_it_matters.text, card.ai_inference.text, *(c.text for c in claims),
            *([card.bank_view.text, *(q.text for q in card.bank_view.quotes)] if card.bank_view else []), *card.limitations]


def item_texts(item: ParsedItem) -> list[str]:
    return [item.title, item.summary or "", *(f"{f.text} {f.span or ''}" for f in item.facts)]


def own_doc_texts(item: ParsedItem, bank_hits: list[dict]) -> list[str]:
    """Материал — сам документ аналитики банка: в ITEM только его сводка и факты, а событие целиком лежит во фрагментах
    того же документа, которые модель видела в BANK_CHUNKS. Для новости пусто."""
    doc = item.source.doc_id if item.source.kind == SourceKind.bank_analytics else None
    return [h["text"] for h in bank_hits if doc and h.get("doc_id") == doc]


def item_topic_text(item: ParsedItem) -> str:
    return " ".join(item_texts(item))


def good_bank_hits(bank_hits: list[dict], item: ParsedItem) -> list[dict]:
    """Релевантные фрагменты банка. Скор BM25 нормирован на лучший результат, поэтому сам по себе не доказывает релевантность:
    требуем свежесть, порог и общую тему фрагмента с темой материала (с подтемами: обзор золота не тема новости о курсе)."""
    mine = topic_facets(item.topics, item_topic_text(item))
    return [h for h in bank_hits if not h["stale"] and h["score"] >= BANK_HIT_MIN_SCORE
            and facets_overlap(topic_facets(h.get("topics", []), f"{h.get('title', '')} {h['text']}"), mine)]


def number_issues(card: InsightCard, profile: CompanyProfile, item: ParsedItem, bank_hits: list[dict],
                  extra_items: list[ParsedItem] | tuple = ()) -> list[str]:
    """Каждая цифра карточки должна найтись там, откуда её можно было взять. Дополнительные материалы (search_news)
    входят в пул цифр, кроме what_happened: оно пишется только по ITEM (у документа банка — и по его фрагментам)."""
    issues = []
    ev = item_texts(item)
    wh = ungrounded_numbers(card.what_happened.text, ev + own_doc_texts(item, bank_hits))
    if wh:
        issues.append(f"what_happened: цифр {wh} нет в событии (ITEM). Пиши what_happened только по событию, цифры банка не переноси")
    pool = ev + [t for x in extra_items for t in item_texts(x)] + [h["text"] for h in bank_hits]
    pnums = profile_numbers(profile)
    rest = {
        "why_it_matters": card.why_it_matters.text,
        "ai_inference": card.ai_inference.text,
        **{f"risks[{i}]": c.text for i, c in enumerate(card.consequences.risks)},
        **{f"opportunities[{i}]": c.text for i, c in enumerate(card.consequences.opportunities)},
        **{f"suggested_actions[{i}]": c.text for i, c in enumerate(card.suggested_actions)},
    }
    for where, text in rest.items():
        bad = ungrounded_numbers(text, pool, pnums)
        if bad:
            issues.append(f"{where}: цифр {bad} нет ни в событии, ни во фрагментах банка, ни в профиле. Убери их или возьми цифры из источников")
    return issues


def bank_attribution_issues(card: InsightCard, profile: CompanyProfile, item: ParsedItem, bank_hits: list[dict]) -> list[str]:
    """Цифры, которые карточка приписывает банку вне bank_view (ai_inference, почему важно, последствия, действия), должны
    быть во фрагментах банка (или в самом событии, если предложение их смешивает). Цифра из соседней новости или из головы
    модели, поданная как прогноз банка, это смешение позиции банка и выводов модели (судья: bank_ai_mixed)."""
    pool = [f"{h.get('title', '')} {h['text']}" for h in bank_hits] + item_texts(item)
    fields = {"ai_inference": card.ai_inference.text, "why_it_matters": card.why_it_matters.text,
              **{f"risks[{i}]": c.text for i, c in enumerate(card.consequences.risks)},
              **{f"opportunities[{i}]": c.text for i, c in enumerate(card.consequences.opportunities)},
              **{f"suggested_actions[{i}]": c.text for i, c in enumerate(card.suggested_actions)}}
    pnums = profile_numbers(profile)  # «кредитный продукт банка … выручка 33 млрд»: цифра профиля, не банка
    issues = []
    for where, text in fields.items():
        bad = sorted({n for m in _SENT.finditer(text or "") if BANK_ATTRIBUTION.search(m.group(0))
                      for n in ungrounded_numbers(m.group(0), pool, pnums)})
        if bad:
            issues.append(f"{where}: банку приписаны цифры {bad}, которых нет во фрагментах банка (BANK_CHUNKS). "
                          "Цифры банка бери только из BANK_CHUNKS; цифру из другого источника не называй позицией банка")
    return issues


def product_issues(card: InsightCard, profile: CompanyProfile) -> list[str]:
    text = " ".join([card.why_it_matters.text, card.ai_inference.text,
                     *(c.text for c in card.consequences.risks + card.consequences.opportunities + card.suggested_actions)])
    absent = [k for k, rx in PRODUCT_WORDS.items() if not getattr(profile.bank_products, k)
              and any(not _NEG_BEFORE.search(text[max(0, m.start() - 40):m.start()]) for m in re.finditer(_HAS + rx, text, re.I))]
    if not absent:
        return []
    return [f"компании приписаны продукты банка {absent}, которых у неё нет (в профиле bank_products.{absent[0]} = false). "
            "Не пиши, что у компании есть этот продукт; можно предложить его как действие"]


def _all_text(card: InsightCard) -> str:
    return card.model_dump_json(include={"what_happened", "why_it_matters", "consequences", "bank_view", "ai_inference", "suggested_actions"})


def rule_check(card: InsightCard, profile: CompanyProfile, item: ParsedItem, bank_hits: list[dict],
               extra_items: list[ParsedItem] | tuple = (), contradicts: bool = False) -> list[str]:
    issues: list[str] = []
    if contradicts:
        from ml.agent.consistency import limitations_issue

        issues += limitations_issue(card)
    # Пустой bank_view при релевантных фрагментах — замечание, если и сама карточка не низкой релевантности.
    good_hits = good_bank_hits(bank_hits, item)
    # Модель может честно сказать, что фрагменты не о теме события (interpret, правило 25): тогда пустой bank_view не ошибка.
    # Кроме установленного расхождения с позицией банка (CONTRADICTIONS, правило 21): там фрагмент по теме по построению.
    no_position = not contradicts and any(NO_BANK_POSITION.search(t) for t in [card.ai_inference.text, *card.limitations])
    if card.bank_view is None and good_hits and card.relevance.score >= LOW_RELEVANCE and not no_position:
        ids = ", ".join(h["id"] for h in good_hits[:3])
        issues.append(f"bank_view пустой, хотя есть релевантные фрагменты аналитики банка: {ids}")
    if card.bank_view:
        stale = {h["id"] for h in bank_hits if h["stale"]}
        used = stale & set(card.bank_view.source_ids)
        if used and not re.search(r"20[0-2]\d", card.bank_view.text):
            issues.append(f"bank_view опирается на устаревший материал {sorted(used)} без указания его даты")
    if card.bank_view:
        # Название документа тоже текст банка: код ставит его в bank_view рядом с цитатой.
        by_id = {h["id"]: f"{h.get('title', '')} {h['text']}" for h in bank_hits}
        cited = [by_id[i] for i in card.bank_view.source_ids if i in by_id]
        missing = ungrounded_numbers(card.bank_view.text, cited)
        if missing:
            others = [h["id"] for h in bank_hits if h["id"] not in card.bank_view.source_ids and set(missing) & _nums(h["text"])]
            hint = f"; они есть во фрагментах {others}" if others else "; в фрагментах банка их нет, убери их"
            issues.append(f"в bank_view цифры {missing} не найдены в указанных source_ids{hint}")
    if card.bank_view:
        cited_text = " ".join(by_id.get(i, "") for i in card.bank_view.source_ids).lower()
        # Слово допустимо, если банк сам так пишет во фрагменте («рекомендуем покупать ОФЗ»).
        advice = sorted({m.group(0).lower() for m in ADVICE.finditer(card.bank_view.text)} - set(cited_text.split()))
        if advice:
            issues.append(f"bank_view содержит рекомендации или выводы для компании ({advice}), которых нет во фрагментах банка. "
                          "В bank_view только позиция банка из фрагментов; выводы для компании перенеси в ai_inference")
    issues += number_issues(card, profile, item, bank_hits, extra_items)
    issues += bank_attribution_issues(card, profile, item, bank_hits)
    issues += product_issues(card, profile)
    bad = [p for p in card.why_it_matters.company_features_used if p != "eligible_big" and _path_value(profile, p) in (None, [], "")]
    if bad:
        issues.append(f"company_features_used ссылается на пустые или несуществующие поля профиля: {bad}")
    fact_ids = {f.id for f in item.facts}
    unknown = [f for f in card.what_happened.facts if f not in fact_ids]
    if unknown:
        issues.append(f"what_happened.facts содержит неизвестные id фактов: {unknown}")
    if profile.data_quality.missing_fields and not card.limitations:
        issues.append(f"в профиле есть пропуски {profile.data_quality.missing_fields}, но limitations пустой")
    sup = SUPERLATIVE.search(" ".join([card.what_happened.text, card.why_it_matters.text, card.ai_inference.text,
                                       *(c.text for c in card.consequences.risks + card.consequences.opportunities)]))
    if sup:
        issues.append(f"компания названа «{sup.group(0)}»: в профиле есть только место по выручке среди компаний выборки банка. "
                      f"Используй дословно: «{profile.position_phrase()}» или не упоминай положение")
    if INJECTION.search(_all_text(card)):
        issues.append("в карточку попал текст, похожий на инструкцию из источника (prompt injection)")
    return issues


def llm_check(client: LLMClient, card: InsightCard, profile: CompanyProfile, bank_hits: list[dict], trace=None) -> list[str]:
    if os.getenv("ML_LLM_VERIFY", "0") != "1":
        return []
    p = render("verify", card=card.model_dump(mode="json", exclude={"meta"}),
               bank_chunks=[{k: h[k] for k in ("id", "text")} for h in bank_hits], profile=profile.prompt_view())
    res, _ = client.complete_json("verify", p.system, p.user, VerifyResult, prompt_version=p.version, trace=trace)
    return res.issues
