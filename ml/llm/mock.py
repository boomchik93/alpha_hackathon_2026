"""Мок LLM: детерминированные ответы по контексту вызова.

Нужен, пока нет доступа к API, и для тестов. Ответы правдоподобны по форме,
но не по качеству: качество проверять только на реальной модели.
"""

from __future__ import annotations

import re
from typing import Any

from ml.topics import TOPICS, detect_topics

_SENT = re.compile(r"(?<=[.!?])\s+")
_ASSUMPTION = ("источник", "ожида", "может", "предполож", "прогноз", "вероятно")
_EVENT_BY_TOPIC = {
    "key_rate": "monetary_policy", "bonds": "debt_market", "inflation": "inflation", "fuel": "fuel_prices",
    "consumption": "consumption", "labor": "labor_market", "fx_gold": "fx_commodities",
    "gdp_industry": "macro_growth", "steel": "industry",
}


def _parse_item(ctx: dict[str, Any]) -> dict:
    raw = ctx["raw"]
    text = f"{raw.title}. {raw.text}"
    topics = detect_topics(text, min_hits=1)[:3]
    facts = []
    for i, s in enumerate(_SENT.split(raw.text.strip())[:4], start=1):
        s = s.strip()
        if len(s) < 20 or "ИНСТРУКЦИ" in s.upper():
            continue
        conf = "assumption" if any(w in s.lower() for w in _ASSUMPTION) else "confirmed"
        facts.append({"id": f"f{i}", "text": s, "span": s[:120], "confidence": conf})
    hints = sorted({p for t in topics for p in TOPICS[t].okved_prefixes})
    return {
        "title": raw.title,
        "summary": raw.text[:300],
        "event_type": _EVENT_BY_TOPIC.get(topics[0], "other") if topics else "other",
        "topics": topics,
        "sectors": [TOPICS[t].title for t in topics],
        "okved_hints": hints,
        "regions": ["РФ"],
        "facts": facts,
    }


def _interpret(ctx: dict[str, Any]) -> dict:
    profile, item = ctx["profile"], ctx["item"]
    chunks = ctx.get("bank_chunks", [])
    reasons = ctx.get("reasons", [])
    attempt = ctx.get("attempt", 0)
    src = ctx["item_source_id"]
    features = [r[2] for r in reasons if r[2] != "eligible_big"] or ["industry.okved_name"]
    fresh = [c for c in chunks if not c["stale"]]
    score = min(1.0, sum(r[0] for r in reasons))
    name = profile.industry.okved_name or "компания"
    facts = [f.id for f in item.facts if f.confidence == "confirmed"][:2]
    bank_ids = [c["id"] for c in fresh[:2]]
    # Первая попытка мока намеренно без bank_view: так в трассе виден реальный цикл verify -> retry.
    bank_view = None
    if bank_ids and attempt > 0:
        bank_view = {"source_ids": bank_ids[:1], "quotes": [{"source_id": bank_ids[0], "text": fresh[0]["text"][:120]}]}
    limitations = [f"Нет данных: {f}" for f in profile.data_quality.missing_fields] or []
    if profile.network.note:
        limitations.append(profile.network.note)
    return {
        "relevance": {"score": round(score, 2), "reasons": [r[1] for r in reasons] or ["связь с профилем слабая"]},
        "what_happened": {"text": item.summary or item.title, "facts": facts},
        "why_it_matters": {
            "text": f"[MOCK] Для компании ({name}) событие важно: " + "; ".join(r[1] for r in reasons[:2]),
            "company_features_used": features,
        },
        "consequences": {
            "risks": [{"text": "[MOCK] Риск роста издержек по направлению, связанному с событием.", "horizon": "3-6 мес",
                       "evidence": bank_ids[:1] or [src], "is_model_inference": False}],
            "opportunities": [{"text": "[MOCK] Возможность пересмотреть условия по банковским продуктам.", "horizon": "3-6 мес",
                               "evidence": ["profile"], "is_model_inference": False}],
        },
        "bank_view": bank_view,
        "ai_inference": {"text": "[MOCK] Вывод модели: стоит оценить чувствительность бизнеса к событию.", "source_ids": [src, "profile"]},
        "suggested_actions": [
            {"text": "[MOCK] Проверить условия действующих продуктов банка.", "type": "action", "evidence": ["profile"], "is_model_inference": False},
            {"text": "[MOCK] Какая доля затрат зависит от этого фактора?", "type": "question_for_analysis", "evidence": [], "is_model_inference": True},
        ],
        "limitations": limitations,
    }


def _channels(ctx: dict[str, Any]) -> dict:
    profile, src, bank = ctx["profile"], ctx["item_source_id"], ctx.get("bank_ids", [])
    out = [{"channel": "demand", "strength": "weak", "direction": "unclear", "mechanism": "[MOCK] спрос в отрасли компании",
            "profile_features": ["industry.okved_name"], "evidence": [src]},
           # Канал без источника: код должен его отбросить.
           {"channel": "fx", "strength": "strong", "direction": "negative", "mechanism": "[MOCK] курс",
            "profile_features": ["geo.fts_countries"], "evidence": []}]
    if profile.bank_products.loan or profile.bank_products.leasing:
        out.insert(0, {"channel": "cost_of_money", "strength": "strong", "direction": "negative", "mechanism": "[MOCK] ставка",
                       "profile_features": ["loan" if profile.bank_products.loan else "leasing"], "evidence": [src, *bank[:1]]})
    return {"channels": out}


def _followup(ctx: dict[str, Any]) -> dict:
    sources = ctx.get("sources", [])
    if not sources:
        return {"answer": "[MOCK] В источниках нет ответа на этот вопрос.", "source_ids": [], "is_model_inference": True}
    top = sources[0]
    return {"answer": f"[MOCK] По материалу «{top['title']}»: {top['text'][:300]}…", "source_ids": [top["id"]], "is_model_inference": False}


_IMPORTANCE_BY_EVENT = {
    "monetary_policy": 0.95, "inflation": 0.75, "macro_growth": 0.65, "debt_market": 0.6, "fx_commodities": 0.6,
    "fuel_prices": 0.5, "consumption": 0.5, "labor_market": 0.5, "regulation": 0.5, "industry": 0.4, "company_news": 0.2, "other": 0.1,
}


def _importance(ctx: dict[str, Any]) -> dict:
    """Детерминированно по event_type; несколько тем добавляют вес."""
    base = _IMPORTANCE_BY_EVENT.get(ctx.get("event_type", "other"), 0.3) + 0.05 * max(0, len(ctx.get("topics", [])) - 1)
    imp = round(min(base, 1.0), 2)
    scope = "economy" if imp >= 0.9 else "multi_sector" if imp >= 0.6 else "sector" if imp >= 0.3 else "company" if imp > 0 else "none"
    return {"importance": imp, "scope": scope, "reason": "[MOCK] по типу события"}


_CUT = re.compile(r"снизи|сниж|смягч|уменьш", re.I)
_HOLD = re.compile(r"пауз|без изменени|неизменн|сохран|жестк|ужесточ", re.I)


def _consistency(ctx: dict[str, Any]) -> dict:
    """Детерминированно: слух о снижении против паузы или жёсткой позиции банка = расхождение."""
    facts, chunks = ctx.get("facts", []), ctx.get("chunks", [])
    f = next((x for x in facts if _CUT.search(x["text"])), None)
    c = next((x for x in chunks if _HOLD.search(x["text"])), None)
    if f and c:
        return {"status": "contradicts", "pairs": [{"fact_id": f["id"], "chunk_id": c["id"], "note": "[MOCK] снижение против паузы"}]}
    return {"status": "consistent" if facts and chunks else "unrelated", "pairs": []}


def _relevance(ctx: dict[str, Any]) -> dict:
    reasons = ctx.get("reasons", [])
    return {"score": min(1.0, sum(r[0] for r in reasons)), "reasons": [r[1] for r in reasons], "features": [r[2] for r in reasons]}


RESPONDERS = {
    "parse_item": _parse_item,
    "interpret": _interpret,
    "verify": lambda ctx: {"ok": True, "issues": []},
    "claim_verify": lambda ctx: {"verdicts": [{"id": c["id"], "verdict": "supported", "reason": "[MOCK]"} for c in ctx.get("claims", [])]},
    "followup_qa": _followup,
    "channels": _channels,
    "label_a": lambda ctx: {"labels": [{"item_id": m["item_id"], "grade": 0, "reason": "[MOCK]"} for m in ctx.get("materials", [])]},
    "label_b": lambda ctx: {"labels": [{"item_id": m["item_id"], "grade": 0, "reason": "[MOCK]"} for m in ctx.get("materials", [])]},
    "relevance": _relevance,
    "importance": _importance,
    "consistency": _consistency,
    "eval_judge_2": lambda ctx: {"factual": 2, "unsupported": [], "personalized": 1, "retells_news": False,
                                 "bank_ai_mixed": False, "actionable": 1, "relevance_ok": True, "comment": "[MOCK]"},
    "eval_judge": lambda ctx: {"factual": 2, "unsupported": [], "personalized": 1, "retells_news": False,
                               "bank_ai_mixed": False, "actionable": 1, "relevance_ok": True, "comment": "[MOCK]"},
}


def respond(task: str, ctx: dict[str, Any]) -> dict:
    if task not in RESPONDERS:
        raise KeyError(f"нет мок-ответа для задачи {task!r}")
    return RESPONDERS[task](ctx)
