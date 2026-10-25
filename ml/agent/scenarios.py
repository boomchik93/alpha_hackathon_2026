"""Ветвление сценариев: развилка, три ветки (good, base, bad), разбор решений по веткам.

Строится поверх готовой карточки (`InsightCard`): её факты, источники и каналы
уже проверены, сценарии их уточняют, а не пересчитывают. Этапы:

1. Развилка (`scenario_fork`): одна внешняя неопределённость и три исхода. Общая ось
   для всех веток, чтобы они не расходились по разным переменным.
2. Ветки (`scenario`): по вызову на ветку, параллельно. Каждая ветка проходит
   проверки кода (цифры только из источников, ссылки, продукты банка, «лидер»,
   инъекции) и при замечаниях переписывается один раз.
3. Разбор (`scenario_analysis`): решения из всех веток сведены в одну таблицу
   «решение x ветка» (помогает, нейтрально, вредит). Код следит за согласованностью:
   «устойчивое» решение не может вредить ни в одной ветке, у условного и страхового
   решения должен быть сигнал запуска.

Вероятности числом нигде нет: только low, medium, high с пояснением. Условие
ветки это гипотеза, и по цифрам она проверяется так же строго, как остальной текст.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import ValidationError

from ml.agent.orchestrator import InsightError, _FACT_REF, _profile_paths, _related_block, item_source_id
from ml.agent.trace import Trace
from ml.agent.verifier import (
    BANK_ATTRIBUTION, BANK_HIT_MIN_SCORE, PRODUCT_WORDS, SUPERLATIVE, _HAS, _NEG_BEFORE, _SENT, _path_value, card_texts,
    item_texts, profile_numbers, ungrounded_numbers,
)
from ml.config import settings
from ml.errors import NotApplicable, NotFound
from ml.ingest.analytics import chunk_by_id
from ml.ingest.store import get_item
from ml.llm.client import LLMClient, LLMError, get_client
from ml.llm.router import model_name, route
from ml.profile.build_profile import get_profile
from ml.prompts import prompt_version, render
from ml.retrieval.index import get_index
from ml.roles import ROLE_FOCUS, ROLE_TITLES, Role
from ml.safety import INJECTION
from ml.schemas import (
    PROFILE_SOURCE_ID, SCENARIO_KINDS, BranchDraft, CompanyProfile, InsightCard, ParsedItem, ScenarioAnalysis, ScenarioBranch,
    ScenarioFork, ScenarioMeta, ScenarioSet, SourceKind, SourceRef,
)

MIN_RELEVANCE = 0.3           # ниже: связь карточки с компанией слабая, веткам не на чем стоять
STAGE_RETRIES = 1             # повтор каждого этапа после замечаний (ветки дорогие, поэтому один)
EXTRA_BANK_PER_BRANCH = 2     # фрагментов банка сверх карточки на ветку: поиск по условию ветки
MAX_DECISIONS = 6
DECISION_ORDER = {"robust": 0, "hedge": 1, "conditional": 2}   # сначала то, что делать сейчас

# То, что SUPERLATIVE пропускает: «статус крупнейшей компании в выборке банка», «лидер среди компаний» (interpret, правило 12).
LEADER_WORDS = re.compile(r"статус\w*\s+(?:крупнейш|лидер|ведущ)\w*|крупнейш\w*\s+(?:[а-яё]+\s+){0,3}компани\w*", re.I)
# Идентификатор источника в тексте для читателя: src_n_2026_..., bank_rate_2026_09_11:p1:c4.
SOURCE_ID_IN_TEXT = re.compile(r"\bsrc_\w+|\b\w+:p\d+:c\d+", re.I)

KIND_TITLES = {"good": "Благоприятный", "base": "Базовый", "bad": "Неблагоприятный"}
KIND_HINTS = {
    "good": "исход, выгодный именно этой компании",
    "base": "наиболее вероятный исход (по прогнозу банка или ожиданию из материала)",
    "bad": "реалистичный исход, неблагоприятный именно для этой компании",
}
PROMPTS = ("scenario_fork", "scenario", "scenario_analysis")


class ScenarioError(InsightError):
    pass


@dataclass
class _Ctx:
    """Всё, из чего пишутся сценарии одной карточки."""

    card: InsightCard
    profile: CompanyProfile
    item: ParsedItem
    related: list[ParsedItem]
    bank: dict[str, dict]
    role: Role
    extra_bank: dict[str, dict] = field(default_factory=dict)   # фрагменты, найденные поиском под ветки

    @property
    def src(self) -> str:
        return item_source_id(self.item)


def prompt_signature() -> str:
    return " ".join(prompt_version(n) for n in PROMPTS)


def set_id_for(card: InsightCard) -> str:
    return "scn_" + card.card_id.removeprefix("ins_")


def effective_role(card: InsightCard, role: Role | None) -> Role:
    """Сценарии пишутся для главы компании: без указанной роли берём роль карточки, а без неё генерального директора."""
    return role or (Role(card.role) if card.role else Role.ceo)


def _role_block(role: Role) -> str:
    if not ROLE_FOCUS[role]:
        return ""
    return (f"ROLE: ветки читает {ROLE_TITLES[role]}. Акцент: {ROLE_FOCUS[role]}. "
            "Это меняет акцент impact и порядок actions. Факты, цифры и источники от роли не зависят.")


# ---------------------------------------------------------------------------
# Контекст
# ---------------------------------------------------------------------------


def _hit(chunk, doc_id: str | None = None) -> dict:
    return {"id": chunk.chunk_id, "title": chunk.title, "date": str(chunk.published_at), "page": chunk.page,
            "stale": chunk.stale, "text": chunk.trusted_text(), "doc_id": chunk.doc_id}


def build_context(card: InsightCard, role: Role | None) -> _Ctx:
    profile = get_profile(card.company_id)
    items = [get_item(i) for i in card.item_ids]
    bank: dict[str, dict] = {}
    for s in card.sources:
        if s.kind != SourceKind.bank_analytics:
            continue
        chunk = chunk_by_id(s.id)
        bank[s.id] = _hit(chunk) if chunk else {"id": s.id, "title": s.title, "date": str(s.published_at), "page": s.page,
                                                "stale": s.stale, "text": s.quote or "", "doc_id": s.doc_id}
    return _Ctx(card, profile, items[0], items[1:], bank, effective_role(card, role))


def _prompt_bank(hits: list[dict]) -> list[dict] | str:
    return [{k: h[k] for k in ("id", "title", "date", "page", "stale", "text")} for h in hits] or "нет подходящих фрагментов"


def _base_card_block(card: InsightCard) -> dict:
    return {
        "relevance": card.relevance.score,
        "what_happened": card.what_happened.text,
        "why_it_matters": card.why_it_matters.text,
        "risks": [c.text for c in card.consequences.risks],
        "opportunities": [c.text for c in card.consequences.opportunities],
        "bank_view": card.bank_view.text if card.bank_view else None,
        "ai_inference": card.ai_inference.text,
        "channels": [{"channel": c.channel, "strength": c.strength, "direction": c.direction, "mechanism": c.mechanism}
                     for c in card.channels],
        "limitations": card.limitations,
    }


def _pool_texts(ctx: _Ctx, bank: list[dict], extra: list[str] = ()) -> list[str]:
    """Откуда можно брать цифры: материалы, фрагменты банка, уже проверенная карточка."""
    return [*item_texts(ctx.item), *(t for x in ctx.related for t in item_texts(x)), *(h["text"] for h in bank),
            *card_texts(ctx.card), *extra]


# ---------------------------------------------------------------------------
# Правила проверки (без LLM)
# ---------------------------------------------------------------------------


def _product_issues(text: str, profile: CompanyProfile) -> list[str]:
    absent = [k for k, rx in PRODUCT_WORDS.items() if not getattr(profile.bank_products, k)
              and any(not _NEG_BEFORE.search(text[max(0, m.start() - 40):m.start()]) for m in re.finditer(_HAS + rx, text, re.I))]
    if not absent:
        return []
    return [f"компании приписаны продукты банка {absent}, которых у неё нет (в профиле bank_products.{absent[0]} = false). "
            "Не пиши, что у компании есть этот продукт; можно предложить его как действие"]


def text_issues(fields: dict[str, str], ctx: _Ctx, pool: list[str], bank: list[dict]) -> list[str]:
    """Общие правила для любого текста сценариев: цифры, приписывание банку, продукты, «лидер», инъекции."""
    pnums = profile_numbers(ctx.profile)
    issues: list[str] = []
    for where, text in fields.items():
        if bad := ungrounded_numbers(text, pool, pnums):
            issues.append(f"{where}: цифр {bad} нет ни в событии, ни во фрагментах банка, ни в профиле, ни в развилке. "
                          "Убери их или опиши словами: процентов вероятности и своих значений не придумывай")
    bank_pool = [f"{h['title']} {h['text']}" for h in bank] + item_texts(ctx.item)
    for where, text in fields.items():
        bad = sorted({n for m in _SENT.finditer(text) if BANK_ATTRIBUTION.search(m.group(0))
                      for n in ungrounded_numbers(m.group(0), bank_pool, pnums)})
        if bad:
            issues.append(f"{where}: банку приписаны цифры {bad}, которых нет во фрагментах банка. "
                          "Цифру из другого источника не называй позицией банка")
    joined = " ".join(fields.values())
    issues += _product_issues(joined, ctx.profile)
    if sup := SUPERLATIVE.search(joined):
        issues.append(f"компания названа «{sup.group(0)}»: в профиле только место по выручке среди компаний выборки банка. "
                      f"Используй дословно: «{ctx.profile.position_phrase()}» или не упоминай положение")
    if (lead := LEADER_WORDS.search(joined)) and not SUPERLATIVE.search(joined):
        issues.append(f"компания названа «{lead.group(0)}»: в профиле только место по выручке среди компаний выборки банка. "
                      f"Используй дословно: «{ctx.profile.position_phrase()}» или не упоминай положение")
    if (sid := SOURCE_ID_IN_TEXT.search(joined)):
        issues.append(f"в тексте идентификатор источника «{sid.group(0)}»: id пишутся только в evidence, в тексте назови источник словами")
    if INJECTION.search(joined):
        issues.append("в текст попало похожее на инструкцию из источника (prompt injection)")
    return issues


def branch_fields(b: BranchDraft) -> dict[str, str]:
    out = {"assumption": b.assumption, "likelihood_basis": b.likelihood_basis, "impact": b.impact}
    for name, items in (("risks", b.risks), ("opportunities", b.opportunities), ("actions", b.actions)):
        out.update({f"{name}[{i}]": c.text for i, c in enumerate(items)})
    out.update({f"signals[{i}]": s for i, s in enumerate(b.signals)})
    return out


def branch_issues(b: BranchDraft, kind: str, ctx: _Ctx, bank: list[dict], fork: ScenarioFork) -> list[str]:
    issues: list[str] = []
    if not (b.title.strip() and b.assumption.strip() and b.impact.strip()):
        issues.append("title, assumption и impact не должны быть пустыми")
    if kind == "bad" and not b.risks:
        issues.append("в неблагоприятной ветке нет ни одного риска")
    if kind == "good" and not b.opportunities:
        issues.append("в благоприятной ветке нет ни одной возможности")
    bad_paths = [p for p in b.company_features_used if p != "eligible_big" and _path_value(ctx.profile, p) in (None, [], "")]
    if bad_paths:
        issues.append(f"company_features_used ссылается на пустые или несуществующие поля профиля: {bad_paths}")
    pool = _pool_texts(ctx, bank, [fork.variable, *fork.outcomes.values()])
    return issues + text_issues(branch_fields(b), ctx, pool, bank)


def _validate_refs(draft: BranchDraft, allowed: set[str]) -> list[str]:
    unknown = sorted(draft.referenced_source_ids() - allowed)
    return [f"evidence ссылается на неизвестные источники: {unknown}. Допустимы только {sorted(allowed)}"] if unknown else []


# ---------------------------------------------------------------------------
# Нормализация ответов модели
# ---------------------------------------------------------------------------


def _fix_ref_factory(ctx: _Ctx, allowed: set[str]):
    paths = _profile_paths(ctx.profile)
    src = ctx.src

    def fix(r: str) -> str:
        if r in allowed:
            return r
        if r in paths or r.startswith("profile."):
            return PROFILE_SOURCE_ID
        if r in (ctx.item.item_id, src) or _FACT_REF.search(r) or r.startswith((ctx.item.item_id, src)):
            return src
        return r

    return fix, paths


def normalize_branch(draft: BranchDraft, ctx: _Ctx, allowed: set[str]) -> BranchDraft:
    """Чинит типовые огрехи ссылок (id факта вместо источника, признак без префикса), не трогая смысл."""
    fix, paths = _fix_ref_factory(ctx, allowed)

    def fix_path(p: str) -> str:
        if p in paths:
            return p
        cands = [x for x in paths if x.endswith("." + p)]
        return cands[0] if len(cands) == 1 else p

    d = draft.model_dump()
    for group in (d["risks"], d["opportunities"], d["actions"]):
        for c in group:
            c["evidence"] = list(dict.fromkeys(fix(r) for r in c["evidence"]))
    d["company_features_used"] = list(dict.fromkeys(fix_path(p) for p in d["company_features_used"]))
    return BranchDraft.model_validate(d)


def normalize_fork(fork: ScenarioFork, ctx: _Ctx, allowed: set[str], trace: Trace) -> ScenarioFork:
    """Оставляет только допустимые источники и приводит base_basis в соответствие с ними."""
    fix, _ = _fix_ref_factory(ctx, allowed)
    evidence = [e for e in dict.fromkeys(fix(r) for r in fork.evidence) if e in allowed and e != PROFILE_SOURCE_ID]
    basis = fork.base_basis
    if basis == "bank_forecast" and not any(e in ctx.bank for e in evidence):
        basis = "item_expectation" if ctx.src in evidence else "assumption"
    if basis == "item_expectation" and ctx.src not in evidence:
        basis = "assumption"
    if basis != fork.base_basis:
        trace.log("scenario_fork_basis_fixed", was=fork.base_basis, now=basis)
    return fork.model_copy(update={"evidence": evidence, "base_basis": basis})


def normalize_analysis(a: ScenarioAnalysis, branches: list[ScenarioBranch]) -> ScenarioAnalysis:
    """Согласует тип решения с его эффектами и заполняет сигнал запуска из сигналов веток.

    «Устойчивое» решение не может вредить ни в одной ветке: такое понижается до условного. У условного и страхового
    решения должен быть trigger: берём первый сигнал ветки, в которой решение помогает (для страховки сначала bad).
    """
    sig = {b.kind: b.signals[0] for b in branches}
    out, seen = [], set()
    for d in a.decisions:
        key = re.sub(r"\W+", " ", d.text.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        kind, trigger = d.type, d.trigger
        if kind == "robust" and "hurts" in d.effects.values():
            kind = "conditional"
        if kind in ("conditional", "hedge") and not (trigger or "").strip():
            order = ("bad", "base", "good") if kind == "hedge" else SCENARIO_KINDS
            helps = [k for k in order if d.effects[k] == "helps"]
            trigger = sig[helps[0]] if helps else None
        out.append(d.model_copy(update={"type": kind, "trigger": trigger,
                                        "from_branches": list(dict.fromkeys(d.from_branches))}))
    out.sort(key=lambda d: DECISION_ORDER[d.type])
    return a.model_copy(update={"decisions": out[:MAX_DECISIONS]})


def analysis_issues(a: ScenarioAnalysis, ctx: _Ctx, branches: list[ScenarioBranch], fork: ScenarioFork, bank: list[dict]) -> list[str]:
    issues: list[str] = []
    for i, d in enumerate(a.decisions):
        helps = "helps" in d.effects.values()
        if d.type == "robust" and not helps:
            issues.append(f"decisions[{i}]: устойчивое решение не помогает ни в одной ветке. Уточни effects или убери решение")
        if d.type in ("conditional", "hedge") and not (d.trigger or "").strip():
            issues.append(f"decisions[{i}]: для типа {d.type} нужен trigger (сигнал из signals веток)")
    fields = {"summary": a.summary, "key_uncertainty": a.key_uncertainty, "recommendation": a.recommendation,
              **{f"decisions[{i}].text": d.text for i, d in enumerate(a.decisions)},
              **{f"decisions[{i}].rationale": d.rationale for i, d in enumerate(a.decisions)},
              **{f"decisions[{i}].trigger": d.trigger for i, d in enumerate(a.decisions) if d.trigger},
              **{f"switch_signals[{i}]": s for i, s in enumerate(a.switch_signals)}}
    branch_texts = [t for b in branches for t in branch_fields(b).values()]
    pool = _pool_texts(ctx, bank, [fork.variable, *fork.outcomes.values(), *branch_texts])
    return issues + text_issues(fields, ctx, pool, bank)


# ---------------------------------------------------------------------------
# Этапы
# ---------------------------------------------------------------------------


def _feedback(issues: list[str]) -> str:
    return "ИСПРАВЬ ПРОБЛЕМЫ ПРЕДЫДУЩЕГО ОТВЕТА:\n- " + "\n- ".join(issues) if issues else ""


def _parse_issues(e: Exception) -> list[str]:
    if isinstance(e, ValidationError):
        # Только суть: полный дамп pydantic с input раздувает промпт в разы.
        return [f"ответ не прошёл схему: {'.'.join(map(str, err['loc']))} {err['msg']}".strip() for err in e.errors()[:5]]
    return [f"ответ модели не разобран: {type(e).__name__}: {str(e)[:200]}"]


def make_fork(client: LLMClient, ctx: _Ctx, trace: Trace) -> tuple[ScenarioFork, list[str], int]:
    bank = list(ctx.bank.values())
    allowed = {ctx.src, PROFILE_SOURCE_ID, *(item_source_id(x) for x in ctx.related), *ctx.bank}
    feedback, fork, fork_issues, issues, retries = "", None, [], [], 0
    for attempt in range(STAGE_RETRIES + 1):
        p = render("scenario_fork", profile=ctx.profile.prompt_view(), item=ctx.item.model_dump(mode="json", exclude={"dedup_cluster"}),
                   item_source_id=ctx.src, allowed_source_ids=sorted(allowed), related_news=_related_block(ctx.related),
                   bank_chunks=_prompt_bank(bank), base_card=_base_card_block(ctx.card), feedback=feedback)
        try:
            res, _ = client.complete_json("scenario_fork", p.system, p.user, ScenarioFork, prompt_version=p.version, trace=trace,
                                          mock_context={"item_source_id": ctx.src, "bank_ids": list(ctx.bank)})
            fork = normalize_fork(res, ctx, allowed, trace)
            fork_issues = text_issues({"variable": fork.variable, **{f"outcomes.{k}": v for k, v in fork.outcomes.items()}},
                                      ctx, _pool_texts(ctx, bank), bank)
            issues = fork_issues
        except (ValidationError, ValueError, LLMError) as e:
            issues = _parse_issues(e)
        trace.log("scenario_fork", attempt=attempt, issues=issues, variable=fork.variable if fork else None)
        if not issues or attempt == STAGE_RETRIES:
            break
        retries += 1
        feedback = _feedback(issues)
    if fork is None:
        raise ScenarioError(f"не удалось построить развилку: {issues}")
    return fork, fork_issues, retries


def _branch_bank(ctx: _Ctx, fork: ScenarioFork, kind: str) -> list[dict]:
    """Фрагменты карточки плюс найденные по условию ветки: у плохого и хорошего исхода своя аналитика."""
    hits = dict(ctx.bank)
    query = f"{fork.variable} {fork.outcomes[kind]} {ctx.item.title}"
    for h in get_index().search(query, k=EXTRA_BANK_PER_BRANCH + len(ctx.bank), include_stale=False):
        if h.score >= BANK_HIT_MIN_SCORE and h.chunk.chunk_id not in hits and len(hits) < len(ctx.bank) + EXTRA_BANK_PER_BRANCH:
            hits[h.chunk.chunk_id] = _hit(h.chunk)
    return list(hits.values())


def write_branch(client: LLMClient, ctx: _Ctx, fork: ScenarioFork, kind: str, trace: Trace) -> tuple[ScenarioBranch, list[dict], int]:
    bank = _branch_bank(ctx, fork, kind)
    allowed = {ctx.src, PROFILE_SOURCE_ID, *(item_source_id(x) for x in ctx.related), *(h["id"] for h in bank)}
    feedback, draft, draft_issues, issues, retries = "", None, [], [], 0
    for attempt in range(STAGE_RETRIES + 1):
        p = render("scenario", branch=kind, branch_title=KIND_TITLES[kind], branch_hint=KIND_HINTS[kind],
                   fork=fork.model_dump(mode="json"), profile=ctx.profile.prompt_view(), position=ctx.profile.position_phrase(),
                   item=ctx.item.model_dump(mode="json", exclude={"dedup_cluster"}), item_source_id=ctx.src,
                   allowed_source_ids=sorted(allowed), related_news=_related_block(ctx.related), bank_chunks=_prompt_bank(bank),
                   base_card=_base_card_block(ctx.card), role_block=_role_block(ctx.role), feedback=feedback)
        try:
            res, _ = client.complete_json("scenario", p.system, p.user, BranchDraft, prompt_version=p.version, trace=trace,
                                          mock_context={"kind": kind, "profile": ctx.profile, "item_source_id": ctx.src, "fork": fork,
                                                        "features": ctx.card.why_it_matters.company_features_used})
            draft = normalize_branch(res, ctx, allowed)
            draft_issues = _validate_refs(draft, allowed) + branch_issues(draft, kind, ctx, bank, fork)
            issues = draft_issues
        except (ValidationError, ValueError, LLMError) as e:
            issues = _parse_issues(e)
        trace.log("scenario_branch", kind=kind, attempt=attempt, issues=issues)
        if not issues or attempt == STAGE_RETRIES:
            break
        retries += 1
        feedback = _feedback(issues)
    if draft is None:
        raise ScenarioError(f"не удалось получить ветку {kind}: {issues}")
    return ScenarioBranch(**draft.model_dump(), kind=kind, verifier_issues=draft_issues), bank, retries


def analyze(client: LLMClient, ctx: _Ctx, fork: ScenarioFork, branches: list[ScenarioBranch], bank: list[dict],
            trace: Trace) -> tuple[ScenarioAnalysis, list[str], int]:
    block = [b.model_dump(mode="json", exclude={"verifier_issues"}) for b in branches]
    feedback, result, result_issues, issues, retries = "", None, [], [], 0
    for attempt in range(STAGE_RETRIES + 1):
        p = render("scenario_analysis", fork=fork.model_dump(mode="json"), profile=ctx.profile.prompt_view(), branches=block,
                   item=ctx.item.model_dump(mode="json", include={"title", "summary", "facts"}),
                   bank_chunks=_prompt_bank(bank), role_block=_role_block(ctx.role), feedback=feedback)
        try:
            res, _ = client.complete_json("scenario_analysis", p.system, p.user, ScenarioAnalysis, prompt_version=p.version, trace=trace,
                                          mock_context={"fork": fork, "branches": block})
            result = normalize_analysis(res, branches)
            result_issues = analysis_issues(result, ctx, branches, fork, bank)
            issues = result_issues
        except (ValidationError, ValueError, LLMError) as e:
            issues = _parse_issues(e)
        trace.log("scenario_analysis", attempt=attempt, issues=issues,
                  decisions=[(d.type, d.text[:60]) for d in result.decisions] if result else None)
        if not issues or attempt == STAGE_RETRIES:
            break
        retries += 1
        feedback = _feedback(issues)
    if result is None:
        raise ScenarioError(f"не удалось получить разбор сценариев: {issues}")
    return result, result_issues, retries


def _source_refs(ctx: _Ctx, extra: dict[str, dict], used: set[str]) -> list[SourceRef]:
    by_id = {s.id: s for s in ctx.card.sources}
    out = []
    for sid in dict.fromkeys([ctx.src, *sorted(used)]):
        if sid == PROFILE_SOURCE_ID:
            continue
        if sid in by_id:
            out.append(by_id[sid])
        elif sid in extra:
            h = extra[sid]
            out.append(SourceRef(id=sid, kind=SourceKind.bank_analytics, title=h["title"], published_at=h["date"], doc_id=h["doc_id"],
                                 page=h["page"], quote=h["text"][:400], stale=h["stale"]))
    return out


# ---------------------------------------------------------------------------
# Вход
# ---------------------------------------------------------------------------


def run_scenarios(card: InsightCard, client: LLMClient | None = None, role: Role | None = None) -> ScenarioSet:
    if card.relevance.score < MIN_RELEVANCE:
        raise NotApplicable(f"сценарии не строятся: связь материала с компанией слабая (relevance {card.relevance.score:.2f} < {MIN_RELEVANCE})")
    client = client or get_client()
    trace = Trace("scenarios")
    ctx = build_context(card, role)
    trace.log("start", card_id=card.card_id, company_id=card.company_id, role=ctx.role.value, mock=client.mock)

    fork, fork_issues, retries = make_fork(client, ctx, trace)
    # Ветки независимы: условие каждой уже зафиксировано развилкой, поэтому пишутся параллельно.
    with ThreadPoolExecutor(max_workers=len(SCENARIO_KINDS)) as pool:
        results = list(pool.map(lambda k: write_branch(client, ctx, fork, k, trace), SCENARIO_KINDS))
    branches = [r[0] for r in results]
    banks: dict[str, dict] = {h["id"]: h for r in results for h in r[1]}
    retries += sum(r[2] for r in results)
    ctx.extra_bank = {i: h for i, h in banks.items() if i not in ctx.bank}

    analysis, analysis_fix, a_retries = analyze(client, ctx, fork, branches, list(banks.values()), trace)
    retries += a_retries
    trace.retries = retries

    used = {e for b in branches for e in b.referenced_source_ids()} | set(fork.evidence)
    issues = [*(f"fork: {i}" for i in fork_issues), *(f"{b.kind}: {i}" for b in branches for i in b.verifier_issues),
              *(f"analysis: {i}" for i in analysis_fix)]
    out = ScenarioSet(
        set_id=set_id_for(card), card_id=card.card_id, company_id=card.company_id, item_ids=card.item_ids, role=ctx.role.value,
        fork=fork, branches=branches, analysis=analysis, sources=_source_refs(ctx, ctx.extra_bank, used), trace_id=trace.trace_id,
        meta=ScenarioMeta(model=model_name(route("scenario").family), prompt_version=prompt_signature(),
                          generated_at=datetime.now(timezone.utc), card_generated_at=card.meta.generated_at, retries=retries,
                          cost_rub=round(trace.cost_rub, 4), verifier_issues=issues),
    )
    trace.log("done", set_id=out.set_id, issues=issues)
    trace.save()
    save_set(out)
    return out


def save_set(s: ScenarioSet) -> None:
    settings.paths.scenarios.mkdir(parents=True, exist_ok=True)
    (settings.paths.scenarios / f"{s.set_id}.json").write_text(s.model_dump_json(indent=1), encoding="utf-8")


def load_set(set_id: str) -> ScenarioSet:
    p = settings.paths.scenarios / f"{set_id}.json"
    if not p.exists():
        raise NotFound(f"сценарии {set_id} не найдены")
    return ScenarioSet.model_validate(json.loads(p.read_text(encoding="utf-8")))


def cached_scenarios(card: InsightCard) -> ScenarioSet | None:
    """Сохранённый набор, если он сделан на этой карточке текущими промптами и моделью; иначе None."""
    try:
        s = load_set(set_id_for(card))
    except NotFound:
        return None
    fresh = (s.meta.prompt_version == prompt_signature() and s.meta.model == model_name(route("scenario").family)
             and s.meta.card_generated_at == card.meta.generated_at)
    return s if fresh else None


__all__ = ["run_scenarios", "cached_scenarios", "load_set", "ScenarioError", "NotApplicable"]
