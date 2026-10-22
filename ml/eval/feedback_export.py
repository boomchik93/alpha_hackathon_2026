"""Выгрузка обратной связи в разметку и калибровку.

    python -m ml.eval.feedback_export [--out var/feedback_export]

- «not_relevant»: кандидаты в разметку релевантности (компания x материал), которые стоит проверить и при необходимости
  добавить в `ml/eval/labels.jsonl` скриптами из PROTOCOL.md (вручную файл разметки не правится);
- «wrong_fact»: очередь ручной калибровки судьи — карточки, на которые пользователь указал фактическую ошибку.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.config import settings
from ml.feedback import load_rows


def export(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    down = [r for r in rows if r["rating"] == "down"]
    cand = [{"company_id": r["company_id"], "item_id": r["item_ids"][0], "card_id": r["card_id"], "role": r.get("role"),
             "comment": r.get("comment"), "ts": r["ts"]} for r in down if r.get("reason") == "not_relevant"]
    queue = [{"card_id": r["card_id"], "trace_id": r["trace_id"], "prompt_version": r["prompt_version"], "model": r["model"],
              "comment": r.get("comment"), "ts": r["ts"]} for r in down if r.get("reason") == "wrong_fact"]
    return cand, queue


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(settings.paths.var / "feedback_export"))
    out = Path(ap.parse_args(argv).out)
    out.mkdir(parents=True, exist_ok=True)
    cand, queue = export(load_rows())
    for name, rows in (("relevance_candidates.jsonl", cand), ("judge_calibration_queue.jsonl", queue)):
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"кандидатов в разметку релевантности: {len(cand)}, в очереди калибровки судьи: {len(queue)} -> {out}")


if __name__ == "__main__":
    main()
