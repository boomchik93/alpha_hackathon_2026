"""Реестр инструментов агента: белый список, только чтение, журнал вызовов.

Разграничение доступа: инструменты видят только компанию текущего запуска.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ml.agent.trace import Trace
from ml.ingest.store import get_item
from ml.profile.build_profile import get_profile
from ml.ranking.ranker import score_item
from ml.retrieval.index import get_index
from ml.schemas import CompanyProfile, ParsedItem, SourceKind


class ToolDenied(PermissionError):
    pass


@dataclass
class RunContext:
    """Общий контекст запуска: шаги читают его, а не пересобирают данные."""

    company_id: str
    item_id: str | None = None
    profile: CompanyProfile | None = None
    item: ParsedItem | None = None
    exposure: list[tuple[float, str, str]] = field(default_factory=list)
    bank_hits: list[dict] = field(default_factory=list)
    extra_items: list[ParsedItem] = field(default_factory=list)  # соседние по времени материалы (search_news)
    network: dict | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "exposure": [r[1] for r in self.exposure],
            "network": self.network,
            "notes": self.notes,
        }


def _summarize(v: Any) -> Any:
    if isinstance(v, list):
        return f"list[{len(v)}]"
    if hasattr(v, "model_dump"):
        return type(v).__name__
    if isinstance(v, dict):
        return {k: _summarize(x) for k, x in list(v.items())[:6]}
    return v


class ToolRegistry:
    def __init__(self, ctx: RunContext, trace: Trace):
        self.ctx, self.trace = ctx, trace
        self._tools: dict[str, Callable[..., Any]] = {
            "get_company_profile": self._get_company_profile,
            "get_item": self._get_item,
            "search_bank_analytics": self._search_bank_analytics,
            "search_bank_analytics_multi": self._search_bank_analytics_multi,
            "search_news": self._search_news,
            "get_counterparty_network": self._get_counterparty_network,
            "compute_exposure": self._compute_exposure,
        }

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            self.trace.log("tool_denied", tool=name, args=kwargs)
            raise ToolDenied(f"инструмент {name} не в белом списке")
        cid = kwargs.get("company_id")
        if cid is not None and cid != self.ctx.company_id:
            self.trace.log("tool_denied", tool=name, args=kwargs, reason="чужая компания")
            raise ToolDenied("доступ только к компании текущего запуска")
        t0 = time.time()
        out = self._tools[name](**kwargs)
        self.trace.log("tool_call", tool=name, args=kwargs, result=_summarize(out), ms=round((time.time() - t0) * 1000))
        return out

    # -- инструменты ------------------------------------------------------

    def _get_company_profile(self, company_id: str) -> CompanyProfile:
        self.ctx.profile = get_profile(company_id)
        return self.ctx.profile

    def _get_item(self, item_id: str) -> ParsedItem:
        self.ctx.item = get_item(item_id)
        return self.ctx.item

    def _search_bank_analytics(self, query: str, k: int = 4, include_stale: bool = True, doc_ids: list[str] | None = None) -> list[dict]:
        hits = get_index().search(query, k=k, include_stale=include_stale, doc_ids=set(doc_ids) if doc_ids else None)
        return self._keep([(h, h.score) for h in hits])

    def _search_bank_analytics_multi(self, queries: list[str], k: int = 3, per_doc: int = 2, include_stale: bool = True) -> list[dict]:
        """Несколько запросов, слияние RRF, не больше per_doc фрагментов с одного документа.

        Разнообразие документов вместо соседних кусков одной страницы; скор фрагмента это лучший
        нормированный скор по запросам, чтобы пороги проверки работали как для одного запроса.
        """
        fused: dict[str, float] = {}
        best: dict[str, tuple] = {}
        for q in queries:
            for rank, h in enumerate(get_index().search(q, k=10, include_stale=include_stale)):
                cid = h.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1 / (60 + rank)
                if cid not in best or h.score > best[cid][1]:
                    best[cid] = (h, h.score)
        picked, per = [], {}
        for cid in sorted(fused, key=lambda c: -fused[c]):
            h, score = best[cid]
            if per.get(h.chunk.doc_id, 0) >= per_doc:
                continue
            per[h.chunk.doc_id] = per.get(h.chunk.doc_id, 0) + 1
            picked.append((h, score))
            if len(picked) == k:
                break
        return self._keep(picked)

    def _search_news(self, item_id: str, days: int = 3, limit: int = 2) -> list[dict]:
        """Другие материалы за ±days дней (BM25 по заголовку и фактам), без самого материала и его кластера дубликатов.
        Найденные запоминаются в контексте как дополнительные источники."""
        from datetime import timedelta

        from rank_bm25 import BM25Okapi

        from ml.ingest.store import load_items
        from ml.retrieval.index import tokenize

        it = self.ctx.item or get_item(item_id)
        own_cluster = it.dedup_cluster
        day = it.source.published_at
        pool = [x for x in load_items().values()
                if x.source.kind != SourceKind.bank_analytics and x.item_id != it.item_id
                and not (own_cluster and x.dedup_cluster == own_cluster)
                and abs((x.source.published_at - day).days) <= days]
        text = lambda x: f"{x.title} {x.summary or ''} {' '.join(f.text for f in x.facts)}"  # noqa: E731
        q = tokenize(text(it))
        docs = [tokenize(text(x)) for x in pool]
        if not pool or not q or not any(docs):
            return []
        raw = BM25Okapi(docs).get_scores(q)
        order = [k for k in sorted(range(len(pool)), key=lambda k: -raw[k]) if raw[k] > 0 and set(q) & set(docs[k])][:limit]
        found = [pool[k] for k in order]
        known = {x.item_id for x in self.ctx.extra_items}
        self.ctx.extra_items += [x for x in found if x.item_id not in known]
        return [{"id": f"src_{x.item_id}", "item_id": x.item_id, "title": x.title, "published_at": str(x.source.published_at)} for x in found]

    def _keep(self, hits: list[tuple]) -> list[dict]:
        found = [{
            "id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "title": h.chunk.title, "published_at": h.chunk.published_at,
            "page": h.chunk.page, "stale": h.chunk.stale, "score": round(score, 3), "topics": h.chunk.topics,
            # Числа, которые чтение PDF не смогло подтвердить, закрыты «[?]»: модель их не видит, цитата и проверка тоже.
            "text": h.chunk.trusted_text(),
        } for h, score in hits]
        known = {c["id"] for c in self.ctx.bank_hits}
        self.ctx.bank_hits += [c for c in found if c["id"] not in known]
        return found

    def _get_counterparty_network(self, company_id: str) -> dict:
        p = self.ctx.profile or get_profile(company_id)
        self.ctx.network = {
            "known_share": p.network.known_share,
            "counterparty_sectors": p.network.counterparty_sectors,
            "note": p.network.note,
        }
        return self.ctx.network

    def _compute_exposure(self, company_id: str, item_id: str) -> list[tuple[float, str, str]]:
        p = self.ctx.profile or get_profile(company_id)
        it = self.ctx.item or get_item(item_id)
        self.ctx.exposure = score_item(p, it).reasons
        return self.ctx.exposure
