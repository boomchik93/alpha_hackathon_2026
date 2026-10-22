"""Чтение PDF vision-моделью: проверки страницы, нарезка, клиент, чанкер. Без сети и без tesseract."""

import io
import json
from collections import Counter

import httpx
import pytest

from ml.ingest import analytics, page_check as pc, pdf_vl
from ml.llm.client import LLMClient

# -- зацикливание -------------------------------------------------------------------


def test_degenerate_progression_of_axis_ticks():
    ticks = "\n".join(f"{-0.25 * i:.2f}".replace(".", ",") for i in range(60))
    assert pc.degenerate_reason("Заголовок\n" + ticks) == "арифметическая прогрессия"


def test_degenerate_empty_table_rows():
    rows = "\n".join("| " + " | ".join([""] * 6) + " |" for _ in range(9))
    assert pc.degenerate_reason("| а | б |\n|---|---|\n" + rows) == "пустые строки таблицы"


def test_degenerate_inline_run_of_invented_values():
    assert pc.degenerate_reason("Акцизные товары: " + "; ".join(["0,05"] * 20)) == "повтор числа внутри строки"


def test_degenerate_question_marks_and_repeated_line():
    assert pc.degenerate_reason("Тарифы: [?]\n" * 41)
    assert pc.degenerate_reason("текст\n" + "то же самое\n" * 12) == "повтор строки"


def test_normal_pages_are_not_degenerate():
    ticks = "\n".join(f"{1.4 - 0.1 * i:.2f}".replace(".", ",") for i in range(16))  # настоящие деления оси
    table = "| | 2025 | 2026П |\n|---|---|---|\n" + "\n".join(f"| Строка {i} | 0,0% | 0,0% |" for i in range(30))
    assert pc.degenerate_reason(ticks) is None
    assert pc.degenerate_reason(table) is None


def test_strip_degenerate_tail_keeps_head():
    head = "Недельная инфляция\nОсь Y: -0,25; 0,45\nЛегенда: Тарифы"
    text = head + "\n" + "\n".join(f"{-0.25 * i:.2f}".replace(".", ",") for i in range(300))
    out = pc.strip_degenerate_tail(text)
    assert out.startswith(head) and len(out) < len(text) / 3
    assert pc.degenerate_reason(out) is None


def test_scrub_chart_text_drops_unknown_and_invented_values():
    text = "Динамика месячной инфляции\nТарифы: [?]\nАкцизные товары: " + "; ".join(["0,05"] * 6) + "\nянв: 1,62\nмар: 0,60"
    out = pc.scrub_chart_text(text)
    assert "[?]" not in out and "0,05" not in out
    assert "1,62" in out and "0,60" in out


# -- форма таблиц и числа -----------------------------------------------------------

TABLE = """Табл.2
| | 2019 | 2020 | 2021 |
|---|---|---|---|
| Инфляция, % | 3,0% | 4,9% | 8,4% |
| Ставка, % | 6,3% | 4,3% |
| ВВП | 1,0% | 2,0% | 3,0% | 4,0% |
"""


def test_table_shape_issues_find_dropped_and_extra_cells():
    issues = pc.table_shape_issues(TABLE)
    assert [(i.row, i.got, i.expected) for i in issues] == [(2, 3, 4), (3, 5, 4)]
    assert not pc.table_shape_issues("| a | b |\n|---|---|\n| 1,0% | 2,0% |\n")


def test_significant_numbers_ignore_dates_isin_and_plain_integers():
    got = pc.significant_numbers("Купон 9,80% до 20.04.2028, RU000A106672, 12 выплат, объём 30, дюрация 0,6, было -2,7%")
    assert [r for r, _ in got] == ["9,80%", "0,6", "-2,7%"]
    assert pc.normalize_number("-2,70 %") == "2.7" == pc.normalize_number("2.7")


def test_cross_check_flags_number_that_ocr_does_not_confirm():
    # VL написала «12,6%» дважды, tesseract видит одно: вторая запись (подмена соседней ячейки) не подтверждена.
    md = "| Ставка | 12,0% | 12,6% |\n| Ставка 2 | 12,6% |"
    union = pc.number_counts("12,0% 12,6%")
    check = pc.cross_check(md, union, Counter())
    assert check.vl_only == ["12,6%"] and check.verified == 2


def test_cross_check_accepts_comma_lost_by_tesseract():
    check = pc.cross_check("| 2,2% | 16,0% |", Counter({"22": 1, "160": 1}), Counter({"22": 1, "160": 1}))
    assert check.vl_only == [] and check.ocr_missing == []


def test_cross_check_reports_numbers_vl_dropped():
    passes = [pc.number_counts("8,4% 19,2%"), pc.number_counts("8,4% 19,2%"), pc.number_counts("8,4%")]
    check = pc.cross_check("| 8,4% |", pc.union_counts(passes), pc.consensus_counts(passes))
    assert check.ocr_missing == ["19.2"]


def test_uncertain_numbers_include_numbers_of_broken_rows():
    issues = pc.table_shape_issues(TABLE)
    check = pc.CrossCheck(total=1, verified=0, vl_only=["12,6%"], ocr_missing=[])
    assert pc.uncertain_numbers(TABLE, check, issues) == sorted({"12,6%", "6,3%", "4,3%", "1,0%", "2,0%", "3,0%", "4,0%"})
    assert pc.uncertain_numbers(TABLE, None, []) == []


# -- нарезка страницы ----------------------------------------------------------------


def _page_png(h=1920, w=600, blocks=((100, 500), (800, 1200), (1400, 1850))) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    for a, b in blocks:
        d.rectangle((50, a, w - 50, b), fill="black")
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def test_segment_bands_overlap_cover_page_and_cut_in_blank_rows():
    from PIL import Image

    bands = [Image.open(io.BytesIO(b)) for b in pdf_vl.segment_bands(_page_png())]
    assert len(bands) == 3
    assert sum(b.height for b in bands) > 1920  # перекрытие
    assert all(b.width == 600 for b in bands)
    assert bands[0].height < 0.5 * 1920 and bands[-1].height < 0.5 * 1920
    # Внутренние границы падают на пустые строки: чёрные блоки не разрезаны.
    import numpy as np

    rows = [np.asarray(b.convert("L")) for b in bands]
    assert (rows[0][-1] > 200).all() and (rows[1][0] > 200).all() and (rows[1][-1] > 200).all() and (rows[2][0] > 200).all()


# -- клиент ---------------------------------------------------------------------------


class _Ledger:
    def check(self):
        pass

    def record(self, info):
        self.last = info


def _sse(deltas: list[str], usage: dict | None = None) -> bytes:
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': d}}]}, ensure_ascii=False)}\n\n" for d in deltas]
    if usage:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _client_with(body: bytes) -> LLMClient:
    c = LLMClient(mock=False, use_cache=False)
    c.ledger = _Ledger()
    c._http = httpx.Client(base_url="http://x/v1", transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)))
    return c


def test_complete_vision_streams_text_and_accounts_usage():
    c = _client_with(_sse(["Табл", "ица ", "1,0%"], {"prompt_tokens": 2800, "completion_tokens": 3, "cost": 0.12}))
    text, aborted, info = c.complete_vision("page_vl", "переписать", b"png")
    assert text == "Таблица 1,0%" and aborted is None
    assert info.cost_rub == pytest.approx(0.12) and info.prompt_tokens == 2800


def test_complete_vision_aborts_degenerate_stream_and_estimates_cost():
    deltas = [f"{-0.25 * i:.2f}".replace(".", ",") + "\n" for i in range(200)]
    c = _client_with(_sse(deltas))  # провайдер не прислал usage: обрыв
    text, aborted, info = c.complete_vision("chart_vl", "переписать", b"png", stop_check=pc.degenerate_reason)
    assert aborted == "арифметическая прогрессия"
    assert len(text.splitlines()) < 200
    assert info.cost_rub > 0 and info.completion_tokens < 200


def test_complete_vision_cache_key_depends_on_image(tmp_path):
    from ml.llm.cache import DiskCache

    c = LLMClient(mock=True, use_cache=False)
    c.cache = DiskCache(tmp_path)
    t1, _, i1 = c.complete_vision("page_vl", "p", b"A", mock_text="первый")
    t2, _, i2 = c.complete_vision("page_vl", "p", b"B", mock_text="второй")
    t3, _, i3 = c.complete_vision("page_vl", "p", b"A", mock_text="другой")
    assert (t1, t2, t3) == ("первый", "второй", "первый") and i3.cached and not i1.cached and not i2.cached


# -- чтение страницы ----------------------------------------------------------------


class _FakeVL:
    """page_vl зацикливается, полосы читаются нормально (с мусорной строкой, которую вычищает scrub)."""

    mock = False

    def __init__(self):
        self.calls = []

    def complete_vision(self, task, prompt, image, **kw):
        from ml.llm.client import CallInfo

        self.calls.append(task)
        info = CallInfo(task=task, model="m", cost_rub=0.1)
        if task == "page_vl":
            return "Заголовок\n", "арифметическая прогрессия", info
        return "Динамика месячной инфляции\nТарифы: [?]\nянв: 1,62", None, info


def test_read_with_vl_falls_back_to_crops_on_loop():
    vl = _FakeVL()
    text, mode, flags, cost = pdf_vl.read_with_vl(vl, _page_png())
    assert mode == "vl_crops" and vl.calls == ["page_vl", "chart_vl", "chart_vl", "chart_vl"]
    assert flags == ["fallback_crops:арифметическая прогрессия"]
    assert "1,62" in text and "[?]" not in text
    assert text.count("Динамика месячной инфляции") == 1  # строки перекрытия полос не дублируются
    assert cost == pytest.approx(0.4)


def test_read_page_uses_text_layer_without_calling_model(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_vl, "text_layer", lambda pdf, page: "Текст из PDF")
    vl = _FakeVL()
    r = pdf_vl.read_page(vl, tmp_path / "x.pdf", 1, None)
    assert (r.mode, r.text, r.uncertain_numbers) == ("text_layer", "Текст из PDF", []) and vl.calls == []


def test_check_page_marks_unconfirmed_numbers(monkeypatch):
    monkeypatch.setattr(pdf_vl, "ocr_number_passes", lambda png: [pc.number_counts("12,0% 3,0%")] * 3)
    flags, stats, unc = pdf_vl.check_page("| Ставка | 12,0% | 12,6% | 3,0% |\n|---|---|---|---|\n| X | 1,0% |", b"png")
    # 12,6% не подтверждена tesseract; 1,0% стоит в строке с неверным числом ячеек.
    assert unc == sorted({"12,6%", "1,0%"})
    assert stats["verified"] == 2


def test_write_and_skip_done_pages(tmp_path):
    r = pdf_vl.PageResult(3, "vl_full", "текст", ["row_shape:t0r1:3/4"], ["12,6%"], {"numbers": 5}, 0.12)
    pdf_vl.write_page(tmp_path, "doc", r)
    assert pdf_vl._done(tmp_path, 3) and not pdf_vl._done(tmp_path, 4)
    meta = json.loads((tmp_path / "page_003.json").read_text(encoding="utf-8"))
    assert meta["uncertain_numbers"] == ["12,6%"] and meta["mode"] == "vl_full"


def test_parse_only_normalizes_names():
    assert pdf_vl.parse_only(["ALFA-7M26:3,4", "Бонд_Бриф_14092026"]) == {"ALFA-7M26": [3, 4], "Бонд_Бриф_14092026": None}


# -- чанкер -------------------------------------------------------------------------

PAGE_MD = """Центр макроэкономического анализа

Табл.2: Основные макроэкономические показатели

| | 2019 | 2020 | 2021 |
|---|---|---|---|
| Инфляция, % | 3,0% | 4,9% | 8,4% |
| Ставка, % | 6,3% | 4,3% | 12,6% |

Ключевая ставка остаётся без изменения, банк ожидает паузу до конца года и снижения не раньше весны.
"""


def test_split_blocks_keeps_table_and_text_in_order():
    blocks = analytics.split_blocks(PAGE_MD)
    assert [k for k, _ in blocks] == ["text", "text", "table", "text"]
    assert blocks[2][1].splitlines()[1].startswith("|---")


def test_table_chunk_gets_title_and_split_repeats_header():
    rows = "\n".join(f"| Показатель {i} | {i},0% | {i},5% |" for i in range(40))
    table = "| | a | b |\n|---|---|---|\n" + rows
    parts = analytics.split_table(table, 400)
    assert len(parts) > 1
    assert all(p.startswith("| | a | b |\n|---|---|---|") for p in parts)
    assert sum(p.count("Показатель") for p in parts) == 40


def _mk(n, text, kind, source, uncertain):
    return analytics.Chunk(f"d:p1:c{n}", "d", "t", "2026-09-01", 1, text, [], 1.0, False, kind, source, uncertain)


def test_vl_page_chunks_table_has_title_and_uncertain_numbers():
    chunks = analytics._vl_page_chunks(PAGE_MD, {"mode": "vl_full", "uncertain_numbers": ["12,6%", "99,9%"]}, _mk)
    table = next(c for c in chunks if c.kind == "table")
    assert table.text.startswith("Табл.2: Основные макроэкономические показатели\n| |")
    assert table.uncertain_numbers == ["12,6%"] and table.source == "vl"
    assert "12,6%" not in table.trusted_text() and "6,3%" in table.trusted_text()
    assert any(c.kind == "text" and "паузу" in c.text for c in chunks)


def test_chart_mode_pages_are_tagged_chart_and_layer_pages_text_layer():
    md = "Динамика месячной инфляции, % м/м. Подписи данных: янв: 1,62; фев: 0,73; мар: 0,60; апр: 0,14"
    assert analytics._vl_page_chunks(md, {"mode": "vl_crops"}, _mk)[0].kind == "chart"
    assert analytics._vl_page_chunks(md, {"mode": "text_layer"}, _mk)[0].source == "text_layer"


def test_mask_numbers_respects_digit_boundaries():
    assert analytics.mask_numbers("16,0% и 16,05% и 116,0%", ["16,0%"]) == "[?] и 16,05% и 116,0%"
    assert analytics.mask_numbers("ставка 14,0", ["14,0"]) == "ставка [?]"


def test_chunk_doc_prefers_vl_page_and_falls_back_to_ocr(monkeypatch, tmp_path):
    stem = "ALFA-7M26"
    vl, ocr = tmp_path / "vl", tmp_path / "ocr"
    vl.mkdir()
    ocr.mkdir()
    (vl / "page_001.md").write_text(PAGE_MD, encoding="utf-8")
    (vl / "page_001.json").write_text(json.dumps({"mode": "vl_full", "uncertain_numbers": []}), encoding="utf-8")
    long_ocr = "Инфляция в августе замедлилась, потому что бензин подешевел, а продовольствие дорожало медленнее обычного."
    (ocr / "page_001.txt").write_text("СТАРЫЙ OCR " + long_ocr, encoding="utf-8")
    (ocr / "page_002.txt").write_text(long_ocr + " r/r", encoding="utf-8")
    monkeypatch.setattr(analytics, "_vl_dir", lambda s: vl)
    monkeypatch.setattr(analytics, "_pages_dir", lambda s: ocr)
    chunks = analytics.chunk_doc(stem)
    p1 = [c for c in chunks if c.page == 1]
    p2 = [c for c in chunks if c.page == 2]
    assert p1 and all(c.source == "vl" and "СТАРЫЙ" not in c.text for c in p1)
    assert p2 and p2[0].source == "ocr" and "г/г" in p2[0].text  # OCR_FIXES только для страниц tesseract


def test_old_chunk_file_without_new_fields_still_loads():
    c = analytics.Chunk(**{"chunk_id": "a", "doc_id": "d", "title": "t", "published_at": "2026-01-01", "page": 1, "text": "x",
                           "topics": [], "freshness": 1.0, "stale": False})
    assert (c.kind, c.source, c.uncertain_numbers) == ("text", "ocr", []) and c.trusted_text() == "x"
