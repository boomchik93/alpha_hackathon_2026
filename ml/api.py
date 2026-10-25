"""Входы для бэкенда (ML_SPEC.md, раздел 11).

    build_profile(company_id) -> CompanyProfile
    list_companies()          -> list[CompanyProfile]
    get_ranked_feed(company_id, limit) -> Feed
    get_insight(company_id, item_id, refresh=False) -> InsightCard
    get_scenarios(company_id, item_id, refresh=False) -> ScenarioSet
    answer_followup(card_id, question) -> FollowupAnswer

Все ответы pydantic-модели из `ml.schemas`; `model_dump(mode="json")` даёт JSON для фронта.
"""

from __future__ import annotations

from pydantic import BaseModel

from ml.agent.orchestrator import load_card, run_insight
from ml.agent.scenarios import cached_scenarios, run_scenarios
from ml.agent.trace import Trace
from ml.agent.verifier import card_texts, item_texts, profile_numbers, ungrounded_numbers
from ml.errors import NotFound
from ml.ingest.analytics import chunk_by_id
from ml.ingest.store import get_item, load_items
from ml.llm.client import get_client
from ml.llm.router import model_name, route
from ml.profile.build_profile import get_profile, list_profiles
from ml.prompts import prompt_version, render
from ml.roles import Role, card_suffix, role_block
from ml.ranking.ranker import feed
from ml.retrieval.index import get_index
from ml.schemas import CompanyProfile, Feed, FollowupAnswer, InsightCard, ScenarioSet, SourceKind, SourceRef

MAX_QUESTION_CHARS = 500
MAX_FOLLOWUP_ATTEMPTS = 2  # ответ и один повтор после замечания по цифрам


def build_profile(company_id: str) -> CompanyProfile:
    return get_profile(company_id)


def list_companies() -> list[CompanyProfile]:
    return list_profiles()


def get_ranked_feed(company_id: str, limit: int = 20, tier: str = "all", maybe_limit: int = 10) -> Feed:
    """tier="main": только основная лента; "all" (по умолчанию): основная и блок «может быть интересно» (entries[].tier)."""
    return feed(get_profile(company_id), list(load_items().values()), limit, maybe_limit, tier)


def cached_insight(company_id: str, item_id: str, role: Role | None = None) -> InsightCard | None:
    """Сохранённая карточка, если она сделана текущими промптом и моделью; иначе None (устарела или её нет)."""
    try:
        card = load_card(f"ins_{company_id}_{item_id}{card_suffix(role)}")
    except NotFound:
        return None
    # Карточка, сделанная другим промптом или другой моделью, считается устаревшей.
    if card.meta.prompt_version == prompt_version("interpret") and card.meta.model == model_name(route("interpret").family):
        return card
    return None


def get_insight(company_id: str, item_id: str, refresh: bool = False, role: Role | None = None) -> InsightCard:
    if not refresh and (card := cached_insight(company_id, item_id, role)) is not None:
        return card
    return run_insight(company_id, item_id, role=role)


def get_scenarios(company_id: str, item_id: str, refresh: bool = False, role: Role | None = None) -> ScenarioSet:
    """Три ветки (good, base, bad) и разбор решений по карточке пары. Карточка берётся из `get_insight` (её кэш и устаревание
    те же); набор сценариев пересобирается, если карточка новая, сменились промпты или модель, либо `refresh`.
    Для карточки со слабой связью (relevance < 0,3) бросает `NotApplicable`."""
    card = get_insight(company_id, item_id, role=role)
    if not refresh and (cached := cached_scenarios(card)) is not None:
        return cached
    return run_scenarios(card, role=role)


class _QA(BaseModel):
    answer: str
    source_ids: list[str] = []
    is_model_inference: bool = True


def answer_followup(card_id: str, question: str, role: Role | None = None) -> FollowupAnswer:
    question = question.strip()[:MAX_QUESTION_CHARS]
    card = load_card(card_id)
    role = role or (Role(card.role) if card.role else None)  # без роли в запросе берём роль карточки
    trace = Trace("followup")
    trace.log("start", card_id=card_id, question=question)

    pool: dict[str, dict] = {}
    for s in card.sources:
        # В карточке хранится короткая цитата; для ответа берём фрагмент аналитики целиком.
        full = chunk_by_id(s.id)
        pool[s.id] = {"id": s.id, "title": s.title, "date": str(s.published_at), "page": s.page,
                      "text": full.text if full else (s.quote or "")}
    # Сначала соседние фрагменты тех же документов банка, что в карточке, затем общий поиск.
    card_docs = {s.doc_id for s in card.sources if s.kind == SourceKind.bank_analytics and s.doc_id}
    hits = (get_index().search(question, k=3, doc_ids=card_docs) if card_docs else []) + get_index().search(question, k=3)
    for h in hits:
        c = h.chunk
        pool.setdefault(c.chunk_id, {"id": c.chunk_id, "title": c.title, "date": c.published_at, "page": c.page, "text": c.text})
    sources = list(pool.values())
    trace.log("tool_call", tool="search_bank_analytics", args={"query": question}, result=f"list[{len(sources)}]")

    # Цифры ответа берутся только из карточки, источников, материала и профиля.
    allowed = [s["text"] for s in sources] + card_texts(card)
    for item_id in card.item_ids:
        try:
            allowed += item_texts(get_item(item_id))
        except NotFound:
            pass
    extra = profile_numbers(get_profile(card.company_id))
    feedback, issues = "", []
    for attempt in range(MAX_FOLLOWUP_ATTEMPTS):
        p = render("followup_qa", card=card.model_dump(mode="json", exclude={"meta", "sources"}), sources=sources,
                   question=question, feedback=feedback, role_block=role_block(role))
        res, _ = get_client().complete_json("followup_qa", p.system, p.user, _QA,
                                            mock_context={"sources": sources, "question": question}, prompt_version=p.version, trace=trace)
        bad = ungrounded_numbers(res.answer, allowed, extra)
        issues = [f"в ответе цифр {bad} нет в карточке, источниках и профиле"] if bad else []
        trace.log("verify", attempt=attempt, issues=issues)
        if not issues:
            break
        feedback = (f"ИСПРАВЬ ПРЕДЫДУЩИЙ ОТВЕТ: цифр {bad} нет в карточке, источниках и профиле. "
                    "Убери их или замени словами, не придумывай цифры.")
        if attempt + 1 < MAX_FOLLOWUP_ATTEMPTS:
            trace.log("retry", attempt=attempt + 1, reason=issues)
    ids = [i for i in res.source_ids if i in pool]
    refs = []
    for i in ids:
        existing = next((s for s in card.sources if s.id == i), None)
        if existing:
            refs.append(existing)
        elif (c := chunk_by_id(i)) is not None:
            refs.append(SourceRef(id=c.chunk_id, kind=SourceKind.bank_analytics, title=c.title, published_at=c.published_at,
                                  doc_id=c.doc_id, page=c.page, quote=c.text[:400], stale=c.stale))
    trace.save()
    return FollowupAnswer(card_id=card_id, question=question, answer=res.answer, source_ids=ids, sources=refs,
                          is_model_inference=res.is_model_inference or not ids or bool(issues),
                          verifier_issues=issues, trace_id=trace.trace_id)
