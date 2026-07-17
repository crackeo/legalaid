"""Extract and clean text from legal PDFs, preserving page numbers.

Output unit: a list of pages, each a list of cleaned paragraphs. Page
numbers are kept because chunks must cite the page they came from.
"""

import re
from pathlib import Path

import fitz  # PyMuPDF

# OCR fallback for scanned PDFs. Optional: needs `pip install pytesseract
# pillow` and the tesseract binary (apt install tesseract-ocr). Without it,
# scanned pages come back empty and the build step flags the document.
try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = bool(pytesseract.get_tesseract_version())
except Exception:  # not installed / binary missing
    OCR_AVAILABLE = False

_MIN_TEXT_CHARS = 20  # fewer than this on a page => treat as scanned
_OCR_DPI = 300


def _ocr_page(page: "fitz.Page") -> str:
    import io

    pix = page.get_pixmap(dpi=_OCR_DPI, colorspace=fitz.csGRAY)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image, lang="eng")


def extract_pages(pdf_path: Path, ocr: bool = True) -> list[str]:
    """Raw text per page; scanned pages are OCRed when tesseract is available.

    OCR runs per page, not per document — many OAG PDFs mix born-digital
    pages with scanned annexes.
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if ocr and OCR_AVAILABLE and len(text.strip()) < _MIN_TEXT_CHARS:
                try:
                    text = _ocr_page(page)
                except Exception:
                    pass  # keep whatever the text layer had
            pages.append(text)
    return pages


_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.I)


def _strip_repeated_lines(pages: list[list[str]]) -> list[list[str]]:
    """Drop headers/footers: first/last lines repeated on >=40% of pages."""
    if len(pages) < 4:
        return pages
    from collections import Counter
    edge_counts: Counter = Counter()
    for lines in pages:
        for ln in {*lines[:2], *lines[-2:]}:
            key = ln.strip().lower()
            if key:
                edge_counts[key] += 1
    threshold = max(3, int(len(pages) * 0.4))
    repeated = {ln for ln, n in edge_counts.items() if n >= threshold}

    cleaned = []
    for lines in pages:
        keep = []
        for i, ln in enumerate(lines):
            at_edge = i < 2 or i >= len(lines) - 2
            if at_edge and ln.strip().lower() in repeated:
                continue
            keep.append(ln)
        cleaned.append(keep)
    return cleaned


def clean_pages(raw_pages: list[str]) -> list[str]:
    """Cleaned text per page: page numbers and repeated headers/footers
    removed, hyphenation across line breaks repaired, whitespace normalised.
    """
    pages_lines: list[list[str]] = []
    for text in raw_pages:
        lines = [ln.rstrip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln.strip() and not _PAGE_NUMBER_RE.match(ln)]
        pages_lines.append(lines)

    pages_lines = _strip_repeated_lines(pages_lines)

    cleaned: list[str] = []
    for lines in pages_lines:
        text = "\n".join(lines)
        # join words hyphenated across line breaks: "provi-\nsion" -> "provision"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        cleaned.append(text.strip())
    return cleaned


def pdf_to_clean_pages(pdf_path: Path) -> list[str]:
    return clean_pages(extract_pages(pdf_path))
