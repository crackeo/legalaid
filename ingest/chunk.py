"""Structure-aware chunking of Bhutanese legal text.

Bhutanese Acts follow a consistent drafting style:

    CHAPTER 3
    OFFENCES AGAINST THE PERSON
    Murder
    92. A defendant shall be guilty of the offence of murder if ...
    93. The offence of murder shall be a felony of the first degree ...

The Constitution uses "Article 7" with numbered clauses. We split on those
natural units: one chunk per numbered section (or per Article), carrying
the current chapter heading and page number as metadata. Oversized
sections are split at sub-clause boundaries with context repeated.
"""

import re
from dataclasses import dataclass, field, asdict

PAGE_BREAK = "\f"  # sentinel inserted between pages before chunking

MAX_CHUNK_CHARS = 4000   # ~1000 tokens; split beyond this
_SECTION_RE = re.compile(r"^(\d{1,4})\.\s+(\S.*)")          # "92. A defendant ..."
_ARTICLE_RE = re.compile(r"^Article\s+(\d{1,3})\b[.:]?\s*(.*)", re.I)
_CHAPTER_RE = re.compile(r"^(?:CHAPTER|PART)\s+([IVXLC\d]+)\b.*", re.I)
_SUBCLAUSE_RE = re.compile(r"^\(\w{1,4}\)\s+")               # "(1) ", "(a) "


@dataclass
class Chunk:
    doc_title: str
    section_number: str      # "92" or "Article 7"
    section_heading: str     # chapter heading + optional marginal note
    text: str
    page: int                # 1-based page where the section starts
    part: int = 0            # >0 when an oversized section was split
    source_url: str = ""
    doc_type: str = ""
    language: str = "en"

    def to_dict(self) -> dict:
        return asdict(self)


def _pages_to_lines(clean_pages: list[str]) -> list[tuple[int, str]]:
    """Flatten cleaned pages into (page_number, line) pairs."""
    out = []
    for pageno, text in enumerate(clean_pages, start=1):
        for ln in text.splitlines():
            if ln.strip():
                out.append((pageno, ln.strip()))
    return out


def _split_oversized(chunk: Chunk) -> list[Chunk]:
    if len(chunk.text) <= MAX_CHUNK_CHARS:
        return [chunk]
    lines = chunk.text.splitlines()
    pieces: list[list[str]] = [[]]
    size = 0
    for ln in lines:
        boundary = _SUBCLAUSE_RE.match(ln) and size > MAX_CHUNK_CHARS // 2
        if boundary or size + len(ln) > MAX_CHUNK_CHARS:
            pieces.append([])
            size = 0
        pieces[-1].append(ln)
        size += len(ln) + 1
    header = f"[{chunk.section_number} (continued)] "
    out = []
    for i, piece in enumerate([p for p in pieces if p]):
        text = "\n".join(piece)
        if i > 0:
            text = header + text
        out.append(Chunk(**{**chunk.to_dict(), "text": text, "part": i + 1}))
    return out


def chunk_document(clean_pages: list[str], doc_title: str, *,
                   source_url: str = "", doc_type: str = "",
                   language: str = "en") -> list[Chunk]:
    """Split a cleaned document into one chunk per section/Article."""
    lines = _pages_to_lines(clean_pages)
    chunks: list[Chunk] = []

    chapter = ""
    pending_heading: list[str] = []   # marginal notes like "Murder" before "92."
    current: Chunk | None = None

    def _is_heading_like(ln: str) -> bool:
        return (len(ln) < 80 and not ln.rstrip().endswith((".", ";", ":", ","))
                and not _SUBCLAUSE_RE.match(ln))

    def flush(reclaim_heading: bool = False):
        """Close the current chunk. When the next line starts a new section,
        trailing heading-like lines belong to that section (marginal notes
        printed between sections), so pull them back into pending_heading."""
        nonlocal current, pending_heading
        if current is not None:
            if reclaim_heading:
                body = current.text.splitlines()
                while len(body) > 1 and _is_heading_like(body[-1]):
                    pending_heading.append(body.pop())
                pending_heading = pending_heading[-2:]
                current.text = "\n".join(body)
            if current.text.strip():
                chunks.extend(_split_oversized(current))
        current = None

    for page, ln in lines:
        m_ch = _CHAPTER_RE.match(ln)
        if m_ch:
            flush()
            chapter = ln
            pending_heading = []
            continue

        m = _SECTION_RE.match(ln) or _ARTICLE_RE.match(ln)
        if m:
            flush(reclaim_heading=True)
            if _ARTICLE_RE.match(ln):
                number = f"Article {m.group(1)}"
            else:
                number = m.group(1)
            heading_bits = [chapter] + pending_heading
            heading = " — ".join(b for b in heading_bits if b)
            body = m.group(2).strip()
            current = Chunk(
                doc_title=doc_title, section_number=number,
                section_heading=heading, text=body, page=page,
                source_url=source_url, doc_type=doc_type, language=language,
            )
            pending_heading = []
            continue

        if current is None:
            # Short standalone line before a section = marginal note/heading.
            if len(ln) < 80:
                pending_heading = (pending_heading + [ln])[-2:]
            continue

        current.text += "\n" + ln

    flush()
    return chunks
