"""Аналитика банка: разбиение страниц на фрагменты с метаданными.

Вход, по убыванию приоритета на каждой странице:
- `data/analytics_pages_vl/<doc>/page_NNN.md` и `.json`: чтение vision-моделью или текстовым слоем (`ml.ingest.pdf_vl`);
- `data/analytics_pages/<doc>/page_NNN.txt`: OCR tesseract (`ml.ingest.pdf_ocr`), чистка `OCR_FIXES`.
Выход: `var/analytics_chunks.jsonl`, одна строка на фрагмент.
`python -m ml.ingest.analytics`
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import lru_cache

from ml.config import settings
from ml.schemas import ItemSource, ParsedItem, SourceKind
from ml.topics import detect_topics


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    title: str
    published_at: date


# Метаданные по PLAN.md, раздел 1.2. Ключ: имя PDF без расширения.
DOCS: dict[str, DocMeta] = {
    "ALFA-Macro_Insights-Policy_Rate_11Sep": DocMeta("bank_rate_2026_09_11", "Ключевая ставка ЦБ РФ: ставка без изменения на 14,0%, риторика стала жестче", date(2026, 9, 11)),
    "Бонд_Бриф_14092026": DocMeta("bank_bond_brief_2026_09_14", "Бонд Бриф: ОФЗ после решения по ставке", date(2026, 9, 14)),
    "ALFA-7M26": DocMeta("bank_macro_7m26", "Макростатистика 7M26: рынок труда, розница, автомобили", date(2026, 9, 7)),
    "Inflation_tracker_02": DocMeta("bank_inflation_2026_09_02", "Инфляция: 25-31 августа", date(2026, 9, 2)),
    "Gold_digest_August26": DocMeta("bank_gold_2026_08", "Рынок золота: август 2026", date(2026, 8, 31)),
    "ALFA-consumption": DocMeta("bank_consumption_2026_08_05", "Потребление на пике", date(2026, 8, 5)),
    "ALFA-macro_stat1H26": DocMeta("bank_macro_1h26", "Макростатистика 1П26: ВВП", date(2026, 7, 31)),
    "FI_Strategy_3Q26_22072026": DocMeta("bank_fi_strategy_3q26", "Тактический взгляд на 3 кв. 2026: время собирать камни", date(2026, 7, 22)),
    "ALFA_Russian_Steels_1К21PreviewUpdate_15Apr2021": DocMeta("bank_steel_1q21", "Российские сталевары: превью 1К21", date(2021, 4, 15)),
}

# Типовые ошибки tesseract на этих документах.
OCR_FIXES = [
    (re.compile(r"\br/r\b"), "г/г"),
    (re.compile(r"\bm/m\b"), "м/м"),
    (re.compile(r"\bH/H\b"), "н/н"),
    (re.compile(r"(?<!\d)1126(?!\d)"), "1П26"),
    (re.compile(r"(?<!\d)1125(?!\d)"), "1П25"),
    # «25 б.п.» tesseract читает как «25 6. п.»: без правки проверка цифр не находит б.п. во фрагменте.
    (re.compile(r"(?<=\d) 6\.\s?п\."), " б.п."),
    (re.compile(r"[‘’]"), "'"),
    (re.compile(r"[“”„]"), '"'),
]

STALE_AFTER_DAYS = 180
FRESHNESS_HALF_LIFE_DAYS = 60
CHUNK_CHARS = 1200


def freshness(published_at: date, as_of: date | None = None) -> float:
    age = max(0, ((as_of or settings.as_of) - published_at).days)
    return round(0.5 ** (age / FRESHNESS_HALF_LIFE_DAYS), 4)


def is_stale(published_at: date, as_of: date | None = None) -> bool:
    return ((as_of or settings.as_of) - published_at).days > STALE_AFTER_DAYS


def _junk(line: str) -> bool:
    s = line.strip()
    if len(s) < 4:
        return True
    if sum(ch.isalnum() for ch in s) / len(s) < 0.5:
        return True
    # Мусор с графиков: нет ни одного нормального слова.
    return not re.search(r"[^\W\d_]{4,}", s)


def clean_text(text: str) -> str:
    for rx, repl in OCR_FIXES:
        text = rx.sub(repl, text)
    lines = [l.rstrip() for l in text.splitlines()]
    lines = ["" if _junk(l) else l for l in lines]
    text = "\n".join(lines)
    # Склейка переносов: «слово-\nпродолжение».
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


BOILERPLATE = re.compile(r"исключительной собственностью|несанкционированное копирование|не является (индивидуальной )?инвестиционной рекомендаци|настоящий (отч[её]т|материал) (и|не)", re.I)


def _paragraphs(text: str) -> list[str]:
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in text.split("\n\n")]
    return [p for p in paras if len(p) >= 40 and not BOILERPLATE.search(p)]


MASK = "[?]"


def mask_numbers(text: str, numbers: list[str]) -> str:
    """Закрывает «[?]» записи чисел, которые чтение PDF не подтвердило (граница по цифрам, чтобы 16,0 не задел 16,05)."""
    for raw in sorted(set(numbers), key=len, reverse=True):
        text = re.sub(rf"(?<![\d,.]){re.escape(raw)}(?!\d)", MASK, text)
    return text


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    published_at: str
    page: int
    text: str
    topics: list[str]
    freshness: float
    stale: bool
    kind: str = "text"                 # text | table | chart
    source: str = "ocr"                # ocr | vl | text_layer
    uncertain_numbers: list[str] = field(default_factory=list)

    def trusted_text(self) -> str:
        """Текст для модели и проверки: неподтверждённые числа закрыты."""
        return mask_numbers(self.text, self.uncertain_numbers) if self.uncertain_numbers else self.text


def _dir_for(root: Path, stem: str) -> Path | None:
    # macOS хранит кириллицу в именах файлов в NFD: сравниваем в NFC.
    if not root.exists():
        return None
    want = unicodedata.normalize("NFC", stem)
    return next((d for d in root.iterdir() if unicodedata.normalize("NFC", d.name) == want), None)


def _pages_dir(stem: str) -> Path | None:
    return _dir_for(settings.paths.analytics_pages, stem)


def _vl_dir(stem: str) -> Path | None:
    return _dir_for(settings.paths.analytics_pages_vl, stem)


# -- страницы, прочитанные vision-моделью ------------------------------------

TITLE_MAX = 120


def split_blocks(md: str) -> list[tuple[str, str]]:
    """Страница в порядке чтения: ('table', строки таблицы) и ('text', абзац)."""
    out: list[tuple[str, str]] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            out.append(("text", re.sub(r"\s*\n\s*", " ", "\n".join(para)).strip()))
            para.clear()

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|"):
            flush_para()
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            # Берём исходные строки блока, включая разделитель шапки: он нужен, чтобы таблица осталась таблицей.
            out.append(("table", "\n".join(lines[i:j])))
            i = j
        elif not lines[i].strip():
            flush_para()
            i += 1
        else:
            para.append(lines[i])
            i += 1
    flush_para()
    return out


def split_table(table: str, limit: int) -> list[str]:
    """Длинную таблицу режем по строкам, повторяя шапку (первая строка и разделитель)."""
    if len(table) <= limit:
        return [table]
    lines = table.splitlines()
    head = lines[:2] if len(lines) > 1 and re.match(r"^\s*\|?\s*:?-{2,}", lines[1]) else lines[:1]
    out, cur = [], list(head)
    for row in lines[len(head):]:
        if len("\n".join(cur)) + len(row) > limit and len(cur) > len(head):
            out.append("\n".join(cur))
            cur = list(head)
        cur.append(row)
    if len(cur) > len(head):
        out.append("\n".join(cur))
    return out


def _vl_page_chunks(md: str, meta_json: dict, mk) -> list[Chunk]:
    """Фрагменты страницы VL: абзацы копятся до CHUNK_CHARS, таблица идёт отдельным фрагментом с заголовком."""
    mode = meta_json.get("mode", "vl_full")
    text_kind = "chart" if mode == "vl_crops" else "text"
    source = "text_layer" if mode == "text_layer" else "vl"
    uncertain = meta_json.get("uncertain_numbers", [])
    out: list[Chunk] = []
    buf: list[str] = []
    last_short = ""

    def emit(text: str, kind: str) -> None:
        mine = [n for n in uncertain if n in text]
        out.append(mk(len(out), text, kind, source, mine))

    def flush() -> None:
        nonlocal buf
        if buf:
            emit(" ".join(buf), text_kind)
            buf = []

    for kind, content in split_blocks(md):
        if kind == "table":
            flush()
            title = last_short
            for part in split_table(content, CHUNK_CHARS):
                emit(f"{title}\n{part}" if title else part, "table")
            last_short = ""
            continue
        if BOILERPLATE.search(content):
            continue
        last_short = content if len(content) <= TITLE_MAX else ""
        if len(content) < 40:
            continue
        if buf and sum(map(len, buf)) + len(content) > CHUNK_CHARS:
            flush()
        buf.append(content)
    flush()
    return out


def _ocr_page_chunks(text: str, mk) -> list[Chunk]:
    out: list[Chunk] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            out.append(mk(len(out), " ".join(buf), "text", "ocr", []))
            buf = []

    for para in _paragraphs(clean_text(text)):
        if buf and sum(map(len, buf)) + len(para) > CHUNK_CHARS:
            flush()
        buf.append(para)
    flush()
    return out


def chunk_doc(stem: str) -> list[Chunk]:
    meta = DOCS[stem]
    ocr_dir, vl_dir = _pages_dir(stem), _vl_dir(stem)
    pages: dict[int, Path] = {}
    if ocr_dir:
        pages.update({int(p.stem.split("_")[1]): p for p in ocr_dir.glob("page_*.txt")})
    vl_pages = {int(p.stem.split("_")[1]): p for p in vl_dir.glob("page_*.md")} if vl_dir else {}
    fr, stale = freshness(meta.published_at), is_stale(meta.published_at)
    out: list[Chunk] = []
    for page_no in sorted(set(pages) | set(vl_pages)):
        def mk(n: int, text: str, kind: str, source: str, uncertain: list[str], page_no: int = page_no) -> Chunk:
            return Chunk(f"{meta.doc_id}:p{page_no}:c{n}", meta.doc_id, meta.title, meta.published_at.isoformat(),
                         page_no, text, detect_topics(text, min_hits=1), fr, stale, kind, source, uncertain)

        if page_no in vl_pages:
            j = vl_pages[page_no].with_suffix(".json")
            info = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {}
            out.extend(_vl_page_chunks(vl_pages[page_no].read_text(encoding="utf-8"), info, mk))
        else:
            out.extend(_ocr_page_chunks(pages[page_no].read_text(encoding="utf-8"), mk))
    return out


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for stem in DOCS:
        if _pages_dir(stem) is None and _vl_dir(stem) is None:
            print(f"нет текста страниц для {stem} (make ocr или make vl), пропуск", file=sys.stderr)
            continue
        chunks.extend(chunk_doc(stem))
    return chunks


def main() -> None:
    chunks = build_chunks()
    out = settings.paths.chunks
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"chunks={len(chunks)} -> {out}", file=sys.stderr)


@lru_cache(maxsize=1)
def load_chunks() -> list[Chunk]:
    p = settings.paths.chunks
    if not p.exists():
        return []
    return [Chunk(**json.loads(l)) for l in p.read_text(encoding="utf-8").splitlines()]


def chunk_by_id(chunk_id: str) -> Chunk | None:
    return next((c for c in load_chunks() if c.chunk_id == chunk_id), None)


def doc_items(importance_client=None) -> list[ParsedItem]:
    """Каждый документ банка как материал ленты. Темы по ключевым словам, без LLM.
    С `importance_client` дополнительно оценивается общая важность документа (по заголовку и первым фрагментам)."""
    by_doc: dict[str, list[Chunk]] = {}
    for c in load_chunks():
        by_doc.setdefault(c.doc_id, []).append(c)
    items = []
    for stem, meta in DOCS.items():
        cs = by_doc.get(meta.doc_id)
        if not cs:
            continue
        text = " ".join(c.text for c in cs)
        item = ParsedItem(
            item_id=meta.doc_id,
            source=ItemSource(kind=SourceKind.bank_analytics, title=meta.title, publisher="Альфа-Банк, Центр макроэкономического анализа",
                              published_at=meta.published_at, doc_id=meta.doc_id, page=1),
            title=meta.title,
            summary=cs[0].text[:400],
            event_type="other",
            topics=detect_topics(text)[:4],
            freshness=freshness(meta.published_at),
        )
        if importance_client is not None:
            from ml.ingest.importance import with_importance

            item = with_importance(item, " ".join(c.text for c in cs[:3]), importance_client)
        items.append(item)
    return items


if __name__ == "__main__":
    main()
