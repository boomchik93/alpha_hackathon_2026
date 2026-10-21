"""Проверки страницы, прочитанной vision-моделью: зацикливание, форма таблиц, сверка чисел с tesseract.

VL-модель читает таблицы намного точнее tesseract, но ошибается тихо: подставляет число из соседней
строки, роняет ячейку и сдвигает строку, на плотных графиках зацикливается. Здесь собраны дешёвые
проверки без LLM, которые превращают такие ошибки в явные флаги и список `uncertain_numbers`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# Значимое число: десятичное или с процентом. Именно такие числа проверяет ml.agent.verifier (_NUM);
# целые без знака процента (годы, номера страниц, частота купона) для сверки шум.
_SIG = re.compile(r"(?<![\w.,])[-−–]?\d+(?:[.,]\d+)?\s?%|(?<![\w.,])[-−–]?\d+[.,]\d+(?![\w])")
_DATE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")
# Подписи оси дат на графиках («02.03», «31.08»): десятичные числа в отчётах пишутся с запятой.
_SHORT_DATE = re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\b(?![.,]?\d)")
_ISIN = re.compile(r"\bRU\w{10}\b")


def normalize_number(raw: str) -> str:
    """«-2,70 %» и «2.7» это одно число: знак и хвостовые нули не различаем (tesseract теряет минус)."""
    n = re.sub(r"[-−–%\s]", "", raw).replace(",", ".")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n


def significant_numbers(text: str) -> list[tuple[str, str]]:
    """(запись как в тексте, нормализованное значение) в порядке появления."""
    text = _ISIN.sub(" ", _SHORT_DATE.sub(" ", _DATE.sub(" ", text)))
    return [(m.group(0).strip(), normalize_number(m.group(0))) for m in _SIG.finditer(text)]


def number_counts(text: str) -> Counter:
    return Counter(n for _, n in significant_numbers(text))


# -- зацикливание ----------------------------------------------------------

_EMPTY_ROW = re.compile(r"^\|(\s*\|)+\s*$")
_ONLY_NUM = re.compile(r"^[-−]?\d+(?:[.,]\d+)?$")
# «0,05; 0,05; 0,05; ...»: модель выдумывает значения столбцов и повторяет одно число внутри строки.
_INLINE_RUN = re.compile(r"(?<![\d,.])(-?\d+[.,]\d+)(?:\s*[;,|]?\s*\1(?![\d])){11,}")


def degenerate_reason(text: str) -> str | None:
    """Причина, по которой ответ модели выродился, либо None. Вызывается на растущем тексте при потоковом ответе."""
    lines = [l.strip() for l in text.splitlines()]
    if sum(1 for l in lines if _EMPTY_ROW.match(l)) >= 8:
        return "пустые строки таблицы"
    if text.count("[?]") >= 40:
        return "много [?]"
    if _INLINE_RUN.search(text[-3000:]):
        return "повтор числа внутри строки"
    tail = [l for l in lines[-120:] if l]
    run = 1
    for a, b in zip(tail, tail[1:]):
        run = run + 1 if a == b else 1
        if run >= 10:
            return "повтор строки"
    nums: list[float] = []
    for l in reversed(tail):
        if not _ONLY_NUM.match(l):
            break
        nums.append(float(l.replace(",", ".").replace("−", "-")))
    if len(nums) >= 30:
        steps = {round(a - b, 6) for a, b in zip(nums, nums[1:])}
        if len(steps) == 1:
            return "арифметическая прогрессия"
    if len(text) > 2000 and text.count(text[-200:]) >= 3:
        return "повтор блока"
    return None


def strip_degenerate_tail(text: str) -> str:
    """Обрезает выродившийся хвост: оставляет текст до того места, где начался повтор."""
    lines = text.splitlines()
    while lines and degenerate_reason("\n".join(lines)):
        # Снимаем строки пачками: хвост из сотен одинаковых строк по одной снимать долго.
        cut = max(1, len(lines) // 20)
        lines = lines[:-cut]
        if len(lines) < 5:
            break
    return "\n".join(lines).rstrip()


_UNKNOWN_VALUE = re.compile(r":\s*(\[\?\][\s;,]*)+$")
_RUN_IN_LINE = re.compile(r"(?<![\d,.])(-?\d+[.,]\d+)(?:\s*[;,|]?\s*\1(?![\d])){3,}")


def scrub_chart_text(text: str) -> str:
    """Убирает из текста графика строки, где модель не прочитала значение («Тарифы: [?]») или повторяет одно число.
    Такие строки либо пустые, либо выдуманные: напечатанные подписи данных к ним не относятся."""
    kept = [l for l in text.splitlines() if not _UNKNOWN_VALUE.search(l.strip()) and not _RUN_IN_LINE.search(l)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


# -- форма таблиц ----------------------------------------------------------

_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def split_cells(line: str) -> list[str]:
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") else s
    return [c.strip() for c in s.split("|")]


def table_blocks(md: str) -> list[list[str]]:
    """Подряд идущие строки Markdown-таблицы. Разделитель шапки (|---|) не включается."""
    blocks: list[list[str]] = []
    cur: list[str] = []
    for line in md.splitlines():
        if _ROW.match(line):
            if not _SEP.match(line):
                cur.append(line)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


@dataclass(frozen=True)
class RowIssue:
    table: int
    row: int
    expected: int
    got: int
    text: str


def table_shape_issues(md: str) -> list[RowIssue]:
    """Строки, в которых ячеек не столько, сколько в шапке: модель уронила или добавила ячейку."""
    out = []
    for t, block in enumerate(table_blocks(md)):
        width = len(split_cells(block[0]))
        for r, line in enumerate(block[1:], start=1):
            got = len(split_cells(line))
            if got != width:
                out.append(RowIssue(t, r, width, got, line.strip()))
    return out


# -- сверка с tesseract ----------------------------------------------------

def union_counts(passes: list[Counter]) -> Counter:
    """Число считается найденным tesseract, если его нашёл любой из проходов (psm 3, 4, 11)."""
    out: Counter = Counter()
    for c in passes:
        out |= c
    return out


def consensus_counts(passes: list[Counter]) -> Counter:
    """Числа, которые нашли минимум два прохода: по ним можно судить, что VL что-то пропустила."""
    out: Counter = Counter()
    for k in set().union(*passes) if passes else ():
        counts = sorted((p[k] for p in passes), reverse=True)
        if len(counts) >= 2 and counts[1] > 0:
            out[k] = counts[1]
    return out


@dataclass
class CrossCheck:
    total: int
    verified: int
    vl_only: list[str]      # записи VL, которых tesseract не подтвердил (либо VL ошиблась, либо tesseract пропустил)
    ocr_missing: list[str]  # числа, которые нашли минимум два прохода tesseract, а VL нет: возможно, потеряна ячейка


def _digits(raw: str) -> str:
    """«16,0%» -> «160»: так tesseract читает число, потеряв запятую."""
    return re.sub(r"\D", "", raw)


def cross_check(vl_text: str, union: Counter, consensus: Counter) -> CrossCheck:
    """tesseract теряет запятую («2,2%» становится «22%»): такое чтение тоже подтверждает число."""
    avail = Counter(union)
    verified, vl_only = 0, []
    nums = significant_numbers(vl_text)
    for raw, norm in nums:
        if avail[norm] > 0:
            avail[norm] -= 1
            verified += 1
        elif re.search(r"[.,]", raw) and avail[_digits(raw)] > 0:
            avail[_digits(raw)] -= 1
            verified += 1
        else:
            vl_only.append(raw)
    have = Counter(n for _, n in nums) + Counter(_digits(r) for r, _ in nums if re.search(r"[.,]", r))
    missing = consensus - have
    return CrossCheck(len(nums), verified, vl_only, sorted(missing.elements()))


def uncertain_numbers(md: str, check: CrossCheck | None, issues: list[RowIssue]) -> list[str]:
    """Записи чисел, на которые нельзя опираться в bank_view: не подтверждены сверкой или стоят в кривой строке."""
    bad = set(check.vl_only) if check else set()
    for i in issues:
        bad.update(raw for raw, _ in significant_numbers(i.text))
    return sorted(bad)
