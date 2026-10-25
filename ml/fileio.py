"""Запись файлов данных, которые читает работающий ML-сервис.

Боевой конвейер (`ml/live.py`) переписывает `var/items.jsonl` и соседние файлы, пока сервис отвечает на запросы.
Запись через временный файл и `os.replace`: читатель видит либо старую, либо новую версию, но не половину файла.
"""

from __future__ import annotations

import os
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
