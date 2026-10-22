"""Загрузка версионированных промптов.

Формат файла: первая строка `<!-- version: vN -->`, затем секции `## system` и
`## user`. Переменные подставляются как `{{name}}`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DIR = Path(__file__).parent
_VERSION = re.compile(r"<!--\s*version:\s*(\S+)\s*-->")
_VAR = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class Rendered:
    name: str
    version: str
    system: str
    user: str


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, indent=1, default=str)


def _version(name: str, raw: str) -> str:
    m = _VERSION.search(raw)
    return f"{name}@{m.group(1) if m else 'v0'}"


def prompt_version(name: str) -> str:
    """Версия промпта в том же виде, что `Rendered.version` и `CardMeta.prompt_version`, без подстановки переменных."""
    return _version(name, (DIR / f"{name}.md").read_text(encoding="utf-8"))


def render(name: str, **vars: Any) -> Rendered:
    raw = (DIR / f"{name}.md").read_text(encoding="utf-8")
    version = _version(name, raw)
    parts = re.split(r"^## (system|user)\s*$", raw, flags=re.M)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}

    def sub(text: str) -> str:
        def repl(mm: re.Match) -> str:
            key = mm.group(1)
            if key not in vars:
                raise KeyError(f"в промпте {name} не передана переменная {key}")
            return _fmt(vars[key])
        return _VAR.sub(repl, text)

    return Rendered(name, version, sub(sections["system"]), sub(sections["user"]))
