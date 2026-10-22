"""Отчёт о фрагментах по источнику чтения (REVIEW_PLAN.md, T9)."""

import json

from ml.ingest.analytics import Chunk
from ml.ingest.chunk_report import render, summarize


def _chunk(doc, page, source, unc=()):
    return Chunk(f"{doc}:p{page}:c0", doc, "t", "2026-09-01", page, "текст 1,5%", [], 1.0, False, source=source, uncertain_numbers=list(unc))


def test_summary_counts_worst_source_per_page_and_reasons(tmp_path):
    d = tmp_path / "docA"
    d.mkdir()
    (d / "page_002.json").write_text(json.dumps({"mode": "failed", "flags": ["loop"]}), encoding="utf-8")
    chunks = [_chunk("docA", 1, "vl"), _chunk("docA", 2, "ocr"), _chunk("docA", 3, "text_layer"), _chunk("docA", 3, "ocr"), _chunk("docB", 1, "vl", ["7,5"])]
    s = summarize(chunks, tmp_path, {"docA": "docA"})
    assert s["pages"] == 5 - 1 and s["good_pages"] == 2  # страница 3 считается по худшему источнику (ocr)
    assert [p["page"] for p in s["ocr_pages"]] == [2, 3]
    assert "failed" in s["ocr_pages"][0]["reason"] and "не читалась" in s["ocr_pages"][1]["reason"]
    assert s["uncertain_numbers"] == 1 and s["chunks_with_uncertain"] == 1
    assert "не достигнута" in render(s)
    assert "достигнута." in render(summarize([_chunk("d", i, "vl") for i in range(1, 5)], tmp_path))
