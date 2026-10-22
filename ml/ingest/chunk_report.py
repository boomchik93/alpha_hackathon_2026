"""Отчёт о фрагментах аналитики по источнику чтения (REVIEW_PLAN.md, T9, шаг 2): vl, text_layer или ocr.

    python -m ml.ingest.chunk_report            # markdown в stdout

После `make vl` и `make chunks` показывает, сколько страниц прочитано vision-моделью или взято из текстового слоя,
какие остались на OCR и почему, и долю чисел, закрытых «[?]» (не подтверждены сверкой). Цель плана: не меньше 90% страниц
из `vl` или `text_layer`.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from ml.config import settings
from ml.ingest.analytics import Chunk, load_chunks

GOOD_SOURCES = ("vl", "text_layer")
TARGET_SHARE = 0.9


def page_source(chunks: list[Chunk]) -> dict[tuple[str, int], str]:
    """(документ, страница) -> источник; если у страницы разные источники, берётся худший (ocr < vl < text_layer)."""
    rank = {"ocr": 0, "vl": 1, "text_layer": 2}
    out: dict[tuple[str, int], str] = {}
    for c in chunks:
        k = (c.doc_id, c.page)
        if k not in out or rank.get(c.source, 0) < rank.get(out[k], 0):
            out[k] = c.source
    return out


def _vl_meta(vl_dir: Path, doc_stem: str, page: int) -> dict | None:
    d = next((x for x in vl_dir.iterdir() if unicodedata.normalize("NFC", x.name) == unicodedata.normalize("NFC", doc_stem)), None) \
        if vl_dir.exists() else None
    p = d / f"page_{page:03d}.json" if d else None
    return json.loads(p.read_text(encoding="utf-8")) if p and p.exists() else None


def summarize(chunks: list[Chunk], vl_dir: Path | None = None, doc_stems: dict[str, str] | None = None) -> dict:
    """doc_stems: doc_id -> имя PDF без расширения (каталог страниц VL); без него причина берётся как «нет данных»."""
    vl_dir = vl_dir or settings.paths.analytics_pages_vl
    pages = page_source(chunks)
    by_doc: dict[str, Counter] = defaultdict(Counter)
    for (doc, _), src in pages.items():
        by_doc[doc][src] += 1
    ocr_pages = []
    for (doc, page), src in sorted(pages.items()):
        if src != "ocr":
            continue
        meta = _vl_meta(vl_dir, (doc_stems or {}).get(doc, doc), page)
        reason = "страница не читалась VL" if meta is None else (f"режим {meta['mode']}, флаги: {', '.join(meta['flags']) or '—'}")
        ocr_pages.append({"doc": doc, "page": page, "reason": reason})
    total = sum(sum(c.values()) for c in by_doc.values())
    good = sum(c[s] for c in by_doc.values() for s in GOOD_SOURCES)
    n_chunks = Counter(c.source for c in chunks)
    uncertain = sum(len(c.uncertain_numbers) for c in chunks)
    return {"pages": total, "good_pages": good, "good_share": good / total if total else 0.0, "by_doc": {d: dict(c) for d, c in by_doc.items()},
            "chunks_by_source": dict(n_chunks), "ocr_pages": ocr_pages, "uncertain_numbers": uncertain,
            "chunks_with_uncertain": sum(bool(c.uncertain_numbers) for c in chunks), "chunks": len(chunks)}


def render(s: dict) -> str:
    L = ["# Фрагменты аналитики по источнику чтения", "",
         f"Страниц с фрагментами: {s['pages']}; из `vl` или `text_layer`: {s['good_pages']} ({s['good_share']:.0%}), цель ≥ {TARGET_SHARE:.0%}: "
         + ("достигнута." if s["good_share"] >= TARGET_SHARE else "не достигнута."),
         f"Фрагментов: {s['chunks']} ({', '.join(f'{k} {v}' for k, v in sorted(s['chunks_by_source'].items()))}); "
         f"чисел, закрытых «[?]»: {s['uncertain_numbers']} в {s['chunks_with_uncertain']} фрагментах.", "",
         "| Документ | страниц по источнику |", "|---|---|"]
    L += [f"| {d} | {', '.join(f'{k} {v}' for k, v in sorted(c.items()))} |" for d, c in sorted(s["by_doc"].items())]
    L += ["", "## Страницы, оставшиеся на OCR", ""]
    L += [f"- `{p['doc']}` стр. {p['page']}: {p['reason']}" for p in s["ocr_pages"]] or ["нет"]
    return "\n".join(L) + "\n"


def main() -> None:
    from ml.ingest.analytics import DOCS

    stems = {m.doc_id: stem for stem, m in DOCS.items()}
    print(render(summarize(load_chunks(), doc_stems=stems)))


if __name__ == "__main__":
    main()
