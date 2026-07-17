"""Extract and clean text from legal PDFs, preserving page numbers.

Output unit: a list of pages, each a list of cleaned paragraphs. Page
numbers are kept because chunks must cite the page they came from.
"""

import re
from pathlib import Path

import fitz  # PyMuPDF


def extract_pages(pdf_path: Path) -> list[str]:
    """Raw text per page. Empty string for pages with no text layer."""
    with fitz.open(pdf_path) as doc:
        return [page.get_text("text") for page in doc]


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
