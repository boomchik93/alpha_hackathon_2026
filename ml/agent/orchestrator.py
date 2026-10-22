"""Агентный цикл: plan -> tools -> context -> interpret -> verify -> retry.

Один оркестратор, роли разделены по шагам: планирование и инструменты (код),
интерпретация (сложная модель), проверка (правила + опционально LLM).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from pydantic import ValidationError

from ml.agent import channels as impact_channels
from ml.agent import claim_verify, consistency
from ml.agent.bank_quotes import assemble as assemble_bank_view, check_quotes
from ml.agent.tools import RunContext, ToolRegistry
from ml.agent.trace import Trace
from ml.agent.verifier import good_bank_hits, item_topic_text, llm_check, rule_check
from ml.config import settings
from ml.errors import NotFound
from ml.llm.client import LLMClient, LLMError, get_client
from ml.llm.router import model_name, route
from ml.prompts import render
from ml.roles import Role, card_suffix, role_block
from ml.ranking.ranker import topic_titles
from ml.topics import TOPICS, facet_titles
from ml.schemas import (
    PROFILE_SOURCE_ID, CardMeta, CompanyProfile, InsightCard, InsightDraft, ParsedItem, SourceKind, SourceRef,
)

MAX_RETRIES = 2
# Фрагментов банка в контексте: меньше соседних кусков, меньше «подтягивания» тезисов (IMPROVEMENTS.md, A3 п.4, F2).
BANK_K = 3
BANK_PER_DOC = 2
# Перепланирование по данным (REVIEW_PLAN.md, T14): окно соседних материалов и сколько их брать.
NEWS_WINDOW_DAYS = 3
NEWS_LIMIT = 2
MIN_CONFIRMED_FACTS = 2


class InsightError(RuntimeError):
    pass


def item_source_id(item: ParsedItem) -> str:
    return f"src_{item.item_id}"


def make_plan(profile: CompanyProfile, item: ParsedItem) -> list[dict]:
    """Какие данные нужны для этой пары. Решение записывается в трассу."""
    steps = [
        {"tool": "compute_exposure", "why": "найти признаки профиля, через которые событие влияет на компанию"},
        {"tool": "search_bank_analytics", "why": "найти позицию банка по темам: " + (", ".join(_titles(item)) or "тема события")},
    ]
    if item.source.kind == SourceKind.bank_analytics:
        steps.insert(1, {"tool": "search_bank_analytics", "why": "материал сам из аналитики банка: взять его фрагменты", "doc": item.item_id})
    if profile.network.top_counterparties:
        steps.append({"tool": "get_counterparty_network", "why": "событие может затронуть контрагентов компании"})
    else:
        steps.append({"tool": None, "why": "связей по платежам нет: вывод только по профилю, указать в limitations"})
    steps.append({"tool": None, "why": "интерпретация, затем самопроверка; при провале до 2 повторов с правкой"})
    return steps


def _titles(item: ParsedItem) -> list[str]:
    """Названия тем материала для запросов к аналитике банка; у широкой темы — подтема (ml.topics.FACETS)."""
    return facet_titles(item.topics, item_topic_text(item))


def _bank_query(profile: CompanyProfile, item: ParsedItem) -> str:
    facts = " ".join(f.text for f in item.facts[:3])
    return f"{item.title} {facts} {' '.join(_titles(item))} {profile.industry.okved_name or ''}"


def _bank_queries(profile: CompanyProfile, item: ParsedItem) -> list[str]:
    """Общий запрос плюс по запросу на тему события: разные темы тянут разные документы банка."""
    qs = [_bank_query(profile, item)]
    for title in _titles(item)[:3]:
        qs.append(f"{title} {item.title}")
    return qs


def _sources(item: ParsedItem, hits: list[dict], draft: InsightDraft, extra_items: list[ParsedItem] = ()) -> list[SourceRef]:
    used = draft.referenced_source_ids()
    out = [SourceRef(
        id=item_source_id(item), kind=item.source.kind, title=item.title, url=item.source.url,
        published_at=item.source.published_at, doc_id=item.source.doc_id, page=item.source.page,
        quote=(item.facts[0].span if item.facts else item.summary),
    )]
    for x in extra_items:
        if item_source_id(x) in used:
            out.append(SourceRef(
                id=item_source_id(x), kind=x.source.kind, title=x.title, url=x.source.url, published_at=x.source.published_at,
                quote=(x.facts[0].span if x.facts else x.summary),
            ))
    for h in hits:
        if h["id"] in used:
            out.append(SourceRef(
                id=h["id"], kind=SourceKind.bank_analytics, title=h["title"], published_at=h["published_at"],
                doc_id=h["doc_id"], page=h["page"], quote=h["text"][:400], stale=h["stale"],
            ))
    return out


def _assemble(ctx: RunContext, draft: InsightDraft, trace: Trace, prompt_version: str, issues: list[str],
              role: Role | None = None) -> InsightCard:
    item, profile = ctx.item, ctx.profile
    return InsightCard(
        **draft.model_dump(),
        card_id=f"ins_{profile.company_id}_{item.item_id}{card_suffix(role)}",
        role=role.value if role else None,
        company_id=profile.company_id,
        # Соседние материалы входят в item_ids, только если карточка на них сослалась: остальные были лишь контекстом поиска.
        item_ids=[item.item_id, *(x.item_id for x in ctx.extra_items if item_source_id(x) in draft.referenced_source_ids())],
        sources=_sources(item, ctx.bank_hits, draft, ctx.extra_items),
        trace_id=trace.trace_id,
        meta=CardMeta(
            model=model_name(route("interpret").family), prompt_version=prompt_version,
            generated_at=datetime.now(timezone.utc), retries=trace.retries, cost_rub=round(trace.cost_rub, 4),
            verifier_issues=issues,
        ),
    )


_FACT_REF = re.compile(r"(^|:)f\d+$")


def _profile_paths(profile: CompanyProfile) -> set[str]:
    out: set[str] = set()

    def walk(obj, prefix: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        out.add(prefix)

    walk(profile.prompt_view(), "")
    out.discard("")
    return out


def normalize_draft(draft: InsightDraft, profile: CompanyProfile, item: ParsedItem, hit_ids: set[str]) -> InsightDraft:
    """Чинит типовые огрехи модели в ссылках, не трогая смысл.

    Модель часто ставит в evidence id факта (`f3`, `src_x:f1`) вместо id источника
    и пишет признак без префикса (`sector_tags`). Это не ошибка по существу, повтор
    на такое тратить дорого.
    """
    src = item_source_id(item)
    known = hit_ids | {src, PROFILE_SOURCE_ID}
    paths = _profile_paths(profile)

    def fix_ref(r: str) -> str:
        if r in known:
            return r
        if r in paths or r.startswith("profile."):
            return PROFILE_SOURCE_ID  # путь поля профиля вместо id источника
        if r in (item.item_id, src) or _FACT_REF.search(r) or r.startswith((item.item_id, src)):
            return src
        return r

    def fix_list(refs: list[str]) -> list[str]:
        return list(dict.fromkeys(fix_ref(r) for r in refs))

    def fix_path(path: str) -> str:
        if path in paths:
            return path
        cands = [p for p in paths if p.endswith("." + path)]
        return cands[0] if len(cands) == 1 else path

    d = draft.model_dump()
    for group in (d["consequences"]["risks"], d["consequences"]["opportunities"], d["suggested_actions"]):
        for claim in group:
            claim["evidence"] = fix_list(claim["evidence"])
    if d["bank_view"]:
        d["bank_view"]["source_ids"] = fix_list(d["bank_view"]["source_ids"])
    d["ai_inference"]["source_ids"] = fix_list(d["ai_inference"]["source_ids"])
    feats = list(dict.fromkeys(fix_path(x) for x in d["why_it_matters"]["company_features_used"]))
    # Пустой список в профиле (например, fts_countries без внешней торговли) признаком не считаем.
    view = profile.prompt_view()

    def empty(path: str) -> bool:
        obj = view
        for part in path.split("."):
            obj = obj.get(part) if isinstance(obj, dict) else None
        return obj in ([], "")

    d["why_it_matters"]["company_features_used"] = [f for f in feats if not empty(f)] or feats
    return InsightDraft.model_validate(d)


def _bank_block(hits: list[dict]) -> list[dict]:
    return [{"id": h["id"], "title": h["title"], "date": h["published_at"], "page": h["page"], "stale": h["stale"], "text": h["text"]} for h in hits]


def _related_block(extra_items: list[ParsedItem]) -> list[dict] | str:
    return [{"id": item_source_id(x), "date": str(x.source.published_at), "title": x.title, "summary": x.summary,
             "facts": [{"id": f.id, "text": f.text, "confidence": f.confidence} for f in x.facts]} for x in extra_items] or "нет"


def replan(profile: CompanyProfile, item: ParsedItem, ctx: RunContext, tools: ToolRegistry, trace: Trace) -> None:
    """Шаги по результатам предыдущих шагов, каждый с причиной в трассе (событие `replan`)."""
    if item.topics and not good_bank_hits(ctx.bank_hits, item):
        q = " ".join([*_titles(item), item.event_type, profile.industry.okved_name or ""]).strip()
        trace.log("replan", trigger="bank_miss", why="нет фрагмента банка с достаточным скором и общей темой: переформулировать запрос", query=q)
        tools.call("search_bank_analytics", query=q, k=6, include_stale=False)
    # У документа банка фактов нет по построению (ml.ingest.analytics.doc_items): его содержание приходит фрагментами
    # самого документа, соседние новости карточке о нём не нужны.
    if item.source.kind == SourceKind.bank_analytics:
        return
    confirmed = sum(f.confidence == "confirmed" for f in item.facts)
    if confirmed < MIN_CONFIRMED_FACTS or item.noisy:
        why = "текст был зашумлён" if item.noisy and confirmed >= MIN_CONFIRMED_FACTS else f"подтверждённых фактов {confirmed} < {MIN_CONFIRMED_FACTS}"
        trace.log("replan", trigger="few_facts", why=f"{why}: поискать соседние по времени материалы")
        tools.call("search_news", item_id=item.item_id, days=NEWS_WINDOW_DAYS, limit=NEWS_LIMIT)


def run_insight(company_id: str, item_id: str, client: LLMClient | None = None, role: Role | None = None) -> InsightCard:
    client = client or get_client()
    trace = Trace("insight")
    ctx = RunContext(company_id, item_id)
    tools = ToolRegistry(ctx, trace)
    trace.log("start", company_id=company_id, item_id=item_id, mock=client.mock)

    profile = tools.call("get_company_profile", company_id=company_id)
    item = tools.call("get_item", item_id=item_id)
    plan = make_plan(profile, item)
    trace.log("plan", steps=plan)

    for step in plan:
        if step["tool"] == "compute_exposure":
            tools.call("compute_exposure", company_id=company_id, item_id=item_id)
        elif step["tool"] == "search_bank_analytics" and "doc" in step:
            tools.call("search_bank_analytics", query=_bank_query(profile, item), k=3, doc_ids=[step["doc"]])
        elif step["tool"] == "search_bank_analytics":
            tools.call("search_bank_analytics_multi", queries=_bank_queries(profile, item), k=BANK_K, per_doc=BANK_PER_DOC)
        elif step["tool"] == "get_counterparty_network":
            tools.call("get_counterparty_network", company_id=company_id)
        elif step["tool"] is None and "связей" in step["why"]:
            ctx.notes.append("нет данных о платежах компании")

    replan(profile, item, ctx, tools, trace)
    contradiction = consistency.check_item(client, item, ctx.bank_hits, trace) if consistency.enabled() else None
    contradicts = bool(contradiction and contradiction.status == "contradicts")

    # Стадия 1 (IMPROVEMENTS.md, A3): каналы влияния до текста. None: стадия выключена.
    kept_channels, dropped_channels = None, []
    if impact_channels.enabled():
        kept_channels, dropped_channels = impact_channels.extract(
            client, profile, item, item_source_id(item), _bank_block(ctx.bank_hits), [r[1] for r in ctx.exposure], trace)

    feedback = ""
    card: InsightCard | None = None
    issues: list[str] = []
    verdicts: dict = {}
    for attempt in range(MAX_RETRIES + 1):
        p = render(
            "interpret",
            profile=profile.prompt_view(), position=profile.position_phrase(),
            reasons=[r[1] for r in ctx.exposure] or ["явных связей с профилем не найдено"],
            item=item.model_dump(mode="json", exclude={"dedup_cluster"}), item_source_id=item_source_id(item),
            bank_chunks=_bank_block(ctx.bank_hits) or "нет подходящих фрагментов",
            run_context=ctx.summary(), feedback=feedback, role_block=role_block(role), related_news=_related_block(ctx.extra_items),
            contradictions=consistency.prompt_block(contradiction) if contradiction else "",
            allowed_source_ids=[item_source_id(item), PROFILE_SOURCE_ID, *(item_source_id(x) for x in ctx.extra_items),
                                *(h["id"] for h in ctx.bank_hits)],
            **impact_channels.prompt_vars(kept_channels, dropped_channels),
        )
        mock_ctx = {"profile": profile, "item": item, "bank_chunks": _bank_block(ctx.bank_hits), "reasons": ctx.exposure,
                    "attempt": attempt, "item_source_id": item_source_id(item)}
        try:
            draft, _ = client.complete_json("interpret", p.system, p.user, InsightDraft,
                                            mock_context=mock_ctx, prompt_version=p.version, trace=trace)
            draft = normalize_draft(draft, profile, item, {h["id"] for h in ctx.bank_hits} | {item_source_id(x) for x in ctx.extra_items})
            # bank_view собирает код из дословных цитат; непроверенные цитаты отбрасываются.
            quotes, quote_issues = check_quotes(draft.bank_view, ctx.bank_hits)
            if draft.bank_view is not None:
                trace.log("bank_quotes", attempt=attempt, given=len(draft.bank_view.quotes), verified=len(quotes))
            draft = draft.model_copy(update={"bank_view": assemble_bank_view(quotes, ctx.bank_hits)})
            card = _assemble(ctx, draft, trace, p.version, [], role)
            verdicts = {}
            # Повтор из-за цитат только если не осталось ни одной проверенной.
            issues = (quote_issues if not quotes else []) + rule_check(card, profile, item, ctx.bank_hits, ctx.extra_items, contradicts) \
                + llm_check(client, card, profile, ctx.bank_hits, trace)
            if claim_verify.enabled(card):
                try:
                    verdicts = claim_verify.verify_claims(client, card, profile, item, ctx.bank_hits, item_source_id(item), trace)
                except Exception as e:  # noqa: BLE001  проверка не должна ронять карточку
                    trace.log("claim_verify_error", error=f"{type(e).__name__}: {e}"[:300])
                trace.log("claim_verify", attempt=attempt, verdicts={k: v.verdict for k, v in verdicts.items()})
                if attempt == 0:  # натяжки в тексте переписываем не больше одного раза
                    issues += claim_verify.feedback_issues(verdicts)
        except ValidationError as e:
            # В повтор отдаём только суть ошибок: полный дамп pydantic с input раздувает промпт в разы.
            issues = [f"ответ не прошёл схему: {'.'.join(map(str, err['loc']))} {err['msg']}".strip() for err in e.errors()[:5]]
        except (ValueError, LLMError) as e:
            # Невалидный JSON после попытки починки клиентом, пустой ответ, сбой вызова: обычный повтор, а не 500.
            issues = [f"ответ модели не разобран: {type(e).__name__}: {str(e)[:200]}"]
        trace.log("verify", attempt=attempt, issues=issues)
        if not issues:
            break
        if attempt == MAX_RETRIES:
            break
        trace.retries += 1
        # Адаптация: если не хватает позиции банка, расширяем поиск, а не только переписываем текст.
        if any(i.startswith("bank_view пустой") for i in issues):
            tools.call("search_bank_analytics", query=" ".join(_titles(item)) + " " + item.title, k=6, include_stale=False)
        feedback = "ИСПРАВЬ ПРОБЛЕМЫ ПРЕДЫДУЩЕГО ОТВЕТА:\n- " + "\n- ".join(issues)
        trace.log("retry", attempt=attempt + 1, reason=issues)

    if card is None:
        trace.save()
        raise InsightError(f"не удалось получить валидную карточку: {issues}")
    if kept_channels is not None:
        card = impact_channels.enforce(card, kept_channels, dropped_channels)
    if verdicts:
        card, removed = claim_verify.prune(card, verdicts)
        if removed:
            trace.log("claims_removed", removed=removed)
    card = card.model_copy(update={"meta": card.meta.model_copy(update={
        "retries": trace.retries, "cost_rub": round(trace.cost_rub, 4), "verifier_issues": issues})})
    trace.log("done", card_id=card.card_id, issues=issues)
    trace.save()
    save_card(card)
    return card


def save_card(card: InsightCard) -> None:
    settings.paths.cards.mkdir(parents=True, exist_ok=True)
    (settings.paths.cards / f"{card.card_id}.json").write_text(card.model_dump_json(indent=1), encoding="utf-8")


def load_card(card_id: str) -> InsightCard:
    p = settings.paths.cards / f"{card_id}.json"
    if not p.exists():
        raise NotFound(f"карточка {card_id} не найдена")
    return InsightCard.model_validate(json.loads(p.read_text(encoding="utf-8")))


__all__ = ["run_insight", "load_card", "InsightError", "PROFILE_SOURCE_ID"]
