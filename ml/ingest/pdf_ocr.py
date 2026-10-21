"""Постраничный OCR сканов аналитики банка.

Делается один раз: `python -m ml.ingest.pdf_ocr`. Результат лежит в
`data/analytics_pages/<doc_stem>/page_NNN.txt`, чтобы ссылки на первоисточник
содержали номер страницы.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from ml.config import settings


def ocr_pdf(pdf: Path, out_dir: Path, dpi: int = 300, lang: str = "rus+eng") -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "p"
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(prefix)], check=True)
        images = sorted(Path(tmp).glob("p-*.png"))
        for i, img in enumerate(images, start=1):
            target = out_dir / f"page_{i:03d}.txt"
            if target.exists():
                continue
            text = subprocess.run(
                ["tesseract", str(img), "-", "-l", lang, "--psm", "3"],
                check=True, capture_output=True, text=True,
            ).stdout
            target.write_text(text, encoding="utf-8")
    return len(images)


def main() -> None:
    src = settings.paths.analytics_pdf
    dst = settings.paths.analytics_pages
    for pdf in sorted(src.glob("*.pdf")):
        n = ocr_pdf(pdf, dst / pdf.stem)
        print(f"{pdf.name}: {n} pages", file=sys.stderr)


if __name__ == "__main__":
    main()
