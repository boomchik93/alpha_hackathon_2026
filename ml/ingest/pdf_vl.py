"""Чтение страниц PDF аналитики банка: текстовый слой, иначе vision-модель.

Делается один раз: `python -m ml.ingest.pdf_vl` (или `make vl`). Результат в
`data/analytics_pages_vl/<doc_stem>/page_NNN.md` и рядом `page_NNN.json` (режим чтения, флаги,
`uncertain_numbers`, стоимость). Промежуточные ответы модели кэширует клиент (`var/llm_cache/`),
поэтому повторный запуск после правки проверок ничего не стоит.

Порядок для каждой страницы:
1. Есть текстовый слой (`pdftotext`): берём его, модель не нужна. Так будет с обычными PDF другого банка.
2. Иначе страница целиком уходит в Qwen3-VL. Таблицы и обычный текст читаются точно.
3. Если ответ зациклился (плотные графики) или упёрся в лимит, страница режется на перекрывающиеся
   полосы, каждая читается отдельно по промпту для графиков. Выдуманные значения вычищаются.
4. Результат сверяется с tesseract (psm 3, 4, 11) и проверяется по форме таблиц: числа без
   подтверждения попадают в `uncertain_numbers`, bank_view не может на них опираться.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ml.config import settings
from ml.ingest import page_check as pc
from ml.llm.client import BudgetExceeded, LLMClient, get_client
from ml.prompts import render

VERSION = "vl2"
MIN_LAYER_LETTERS = 200   # меньше букв в текстовом слое страницы: считаем её сканом
RENDER_LONG_SIDE = 1920   # если в PDF нет одной картинки на страницу, растрируем до такой длинной стороны
OCR_PSMS = (3, 4, 11)
# Полосы для страниц с графиками: высота и шаг в долях страницы. Три полосы по 40% с шагом 30% целиком
# захватывают каждый из трёх рядов графиков типичной страницы; перекрытие дублирует только края.
BAND_HEIGHT, BAND_STRIDE = 0.40, 0.30
EST_RUB_PER_PAGE = 0.17   # по пилоту: 0,12-0,18 руб. за страницу целиком


@dataclass
class PageResult:
    page: int
    mode: str                      # text_layer | vl_full | vl_crops | failed
    text: str
    flags: list[str] = field(default_factory=list)
    uncertain_numbers: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    cost_rub: float = 0.0


# -- PDF -> картинки -------------------------------------------------------

def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def page_count(pdf: Path) -> int:
    out = _run(["pdfinfo", str(pdf)], text=True).stdout
    return int(re.search(r"^Pages:\s+(\d+)", out, re.M).group(1))


def text_layer(pdf: Path, page: int) -> str | None:
    """Текст страницы из PDF, если он есть и осмыслен; None для скана."""
    out = _run(["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(pdf), "-"], text=True).stdout
    letters = sum(ch.isalpha() for ch in out)
    return out.strip() if letters >= MIN_LAYER_LETTERS else None


def page_images(pdf: Path, pages: list[int], tmp: Path) -> dict[int, bytes]:
    """PNG страниц. Если страница это одна встроенная картинка, берём её как есть (без потери резкости),
    иначе растрируем `pdftoppm`."""
    out: dict[int, bytes] = {}
    embedded: dict[int, list[Path]] = {}
    try:
        _run(["pdfimages", "-png", "-p", str(pdf), str(tmp / "img")])
        for f in sorted(tmp.glob("img-*.png")):
            m = re.match(r"img-(\d+)-\d+\.png$", f.name)
            if m:
                embedded.setdefault(int(m.group(1)), []).append(f)
    except subprocess.CalledProcessError:
        pass
    from PIL import Image

    for p in pages:
        files = embedded.get(p, [])
        if len(files) == 1:
            with Image.open(files[0]) as im:
                if min(im.size) >= 800:
                    out[p] = files[0].read_bytes()
                    continue
        _run(["pdftoppm", "-f", str(p), "-l", str(p), "-scale-to", str(RENDER_LONG_SIDE), "-png", str(pdf), str(tmp / f"r{p}")])
        out[p] = next(tmp.glob(f"r{p}-*.png")).read_bytes()
    return out


def segment_bands(png: bytes) -> list[bytes]:
    """Перекрывающиеся горизонтальные полосы страницы. Границы сдвигаются к ближайшей пустой строке,
    чтобы не резать строки текста и подписи."""
    import numpy as np
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = im.size
    blank = ((np.asarray(im.convert("L")) < 235).sum(axis=1) <= max(2, int(0.002 * w)))
    snap_px = int(0.03 * h)

    def snap(y: int) -> int:
        for d in range(snap_px + 1):
            for yy in (y - d, y + d):
                if 0 < yy < h and blank[yy]:
                    return yy
        return y

    bands, top = [], 0.0
    while True:
        bottom = min(1.0, top + BAND_HEIGHT)
        a, b = (0 if top == 0 else snap(int(top * h))), (h if bottom >= 1.0 else snap(int(bottom * h)))
        buf = io.BytesIO()
        im.crop((0, a, w, b)).save(buf, "PNG")
        bands.append(buf.getvalue())
        if bottom >= 1.0:
            return bands
        top += BAND_STRIDE


# -- tesseract для сверки ---------------------------------------------------

def ocr_number_passes(png: bytes) -> list[Counter]:
    """Числа, найденные tesseract в каждом из режимов. Режимы дополняют друг друга: объединение даёт
    86-98% чисел страницы против 79-90% у одного psm 3 (замер пилота)."""
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    passes = []
    with tempfile.TemporaryDirectory() as d:
        img = Path(d) / "p.png"
        img.write_bytes(png)
        for psm in OCR_PSMS:
            text = _run(["tesseract", str(img), "-", "-l", "rus+eng", "--psm", str(psm)], text=True, env=env).stdout
            passes.append(pc.number_counts(text))
    return passes


# -- чтение страницы --------------------------------------------------------

def _prompt(name: str) -> tuple[str, str]:
    r = render(name)
    return f"{r.system}\n\n{r.user}", r.version


def _dedupe_lines(parts: list[str]) -> str:
    """Склейка полос: строки перекрытия, уже выведенные предыдущей полосой, пропускаем."""
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        for line in part.splitlines():
            key = line.strip()
            if len(key) > 8 and key in seen:
                continue
            seen.add(key)
            out.append(line)
        out.append("")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def read_with_vl(client: LLMClient, png: bytes) -> tuple[str, str, list[str], float]:
    """(текст, режим, флаги, стоимость). Целая страница, при зацикливании полосы."""
    flags: list[str] = []
    cost = 0.0
    prompt, ver = _prompt("page_vl")
    text, aborted, info = client.complete_vision("page_vl", prompt, png, stop_check=pc.degenerate_reason, prompt_version=ver)
    cost += info.cost_rub
    if not aborted and text.strip():
        return text.strip(), "vl_full", flags, cost
    flags.append(f"fallback_crops:{aborted or 'пустой ответ'}")

    prompt, ver = _prompt("chart_vl")
    parts = []
    for i, band in enumerate(segment_bands(png)):
        t, why, info = client.complete_vision("chart_vl", prompt, band, stop_check=pc.degenerate_reason, prompt_version=ver)
        cost += info.cost_rub
        if why:
            flags.append(f"band{i}:{why}")
            t = pc.strip_degenerate_tail(t)
        parts.append(pc.scrub_chart_text(t))
    text = _dedupe_lines(parts)
    return text, ("vl_crops" if text else "failed"), flags, cost


def check_page(text: str, png: bytes | None) -> tuple[list[str], dict, list[str]]:
    """(флаги, статистика, uncertain_numbers) по тексту страницы, прочитанному VL."""
    issues = pc.table_shape_issues(text)
    flags = [f"row_shape:t{i.table}r{i.row}:{i.got}/{i.expected}" for i in issues]
    stats: dict = {"row_issues": [i.text[:160] for i in issues]}
    check = None
    if png is not None:
        passes = ocr_number_passes(png)
        check = pc.cross_check(text, pc.union_counts(passes), pc.consensus_counts(passes))
        stats.update(numbers=check.total, verified=check.verified, vl_only=check.vl_only, ocr_missing=check.ocr_missing)
        if check.ocr_missing:
            flags.append(f"ocr_missing:{len(check.ocr_missing)}")
    return flags, stats, pc.uncertain_numbers(text, check, issues)


def read_page(client: LLMClient, pdf: Path, page: int, png: bytes | None) -> PageResult:
    layer = text_layer(pdf, page)
    if layer is not None:
        return PageResult(page, "text_layer", layer)
    assert png is not None
    text, mode, flags, cost = read_with_vl(client, png)
    cflags, stats, uncertain = check_page(text, png) if text else ([], {}, [])
    return PageResult(page, mode, text, flags + cflags, uncertain, stats, round(cost, 4))


# -- запись и обход ---------------------------------------------------------

def out_dir_for(stem: str) -> Path:
    return settings.paths.analytics_pages_vl / unicodedata.normalize("NFC", stem)


def _done(out: Path, page: int) -> bool:
    j = out / f"page_{page:03d}.json"
    return j.exists() and (out / f"page_{page:03d}.md").exists() and json.loads(j.read_text(encoding="utf-8")).get("version") == VERSION


def write_page(out: Path, doc: str, r: PageResult) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"page_{r.page:03d}.md").write_text(r.text + "\n", encoding="utf-8")
    meta = {"version": VERSION, "doc": doc, "page": r.page, "mode": r.mode, "flags": r.flags,
            "uncertain_numbers": r.uncertain_numbers, "stats": r.stats, "cost_rub": r.cost_rub}
    (out / f"page_{r.page:03d}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


class Spend:
    def __init__(self, limit: float | None):
        self.limit, self.total, self._lock = limit, 0.0, threading.Lock()

    def add(self, rub: float) -> None:
        with self._lock:
            self.total += rub

    def exhausted(self) -> bool:
        return self.limit is not None and self.total >= self.limit


def process_pdf(pdf: Path, pages: list[int] | None, client: LLMClient, spend: Spend, workers: int, dry_run: bool) -> dict[str, int]:
    out = out_dir_for(pdf.stem)
    todo = [p for p in (pages or range(1, page_count(pdf) + 1)) if not _done(out, p)]
    counts = {"skipped": (len(pages) if pages else page_count(pdf)) - len(todo), "text_layer": 0, "vl": 0, "failed": 0, "stopped": 0}
    if not todo:
        return counts
    with tempfile.TemporaryDirectory() as t:
        scans = [p for p in todo if text_layer(pdf, p) is None]
        if dry_run:
            counts["text_layer"], counts["vl"] = len(todo) - len(scans), len(scans)
            return counts
        images = page_images(pdf, scans, Path(t)) if scans else {}

        def work(p: int) -> None:
            if spend.exhausted():
                counts["stopped"] += 1
                return
            try:
                r = read_page(client, pdf, p, images.get(p))
            except BudgetExceeded as e:
                print(f"{pdf.name} стр. {p}: {e}", file=sys.stderr)
                spend.limit = 0.0
                counts["stopped"] += 1
                return
            spend.add(r.cost_rub)
            counts["text_layer" if r.mode == "text_layer" else "failed" if r.mode == "failed" else "vl"] += 1
            write_page(out, pdf.stem, r)
            print(f"{pdf.name} стр. {p}: {r.mode}, {r.cost_rub:.2f} руб., флаги: {', '.join(r.flags) or '-'}", file=sys.stderr)

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, todo))
    return counts


def parse_only(specs: list[str]) -> dict[str, list[int] | None]:
    """`--only ALFA-7M26:3,4` или `--only ALFA-7M26` (все страницы)."""
    out: dict[str, list[int] | None] = {}
    for s in specs:
        stem, _, pages = s.partition(":")
        out[unicodedata.normalize("NFC", stem)] = [int(x) for x in pages.split(",")] if pages else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", action="append", default=[], metavar="STEM[:p,p]", help="только этот документ (и страницы)")
    ap.add_argument("--max-rub", type=float, default=None, help="остановиться после такого расхода за этот запуск")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="только посчитать страницы и оценить цену")
    args = ap.parse_args()

    only = parse_only(args.only)
    client, spend = get_client(), Spend(args.max_rub)
    if not args.dry_run and client.mock:
        sys.exit("LLM_BASE_URL не задан: запуск пошёл бы в мок. Задайте .env или используйте --dry-run")
    total: Counter = Counter()
    for pdf in sorted(settings.paths.analytics_pdf.glob("*.pdf")):
        stem = unicodedata.normalize("NFC", pdf.stem)
        if only and stem not in only:
            continue
        c = process_pdf(pdf, only.get(stem) if only else None, client, spend, args.workers, args.dry_run)
        total.update(c)
        print(f"{pdf.name}: {dict(c)}", file=sys.stderr)
    est = total["vl"] * EST_RUB_PER_PAGE
    print(f"итого: {dict(total)}; потрачено {spend.total:.2f} руб." + (f"; оценка полного прогона {est:.1f} руб." if args.dry_run else ""),
          file=sys.stderr)


if __name__ == "__main__":
    main()
