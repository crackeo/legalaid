"""Registry of official sources for Bhutan's laws.

Each source is an index page that links (directly or one hop away) to PDF
documents. The crawler scrapes these pages for .pdf links.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    key: str                 # short id used in paths and the manifest
    name: str                # human-readable name
    index_urls: tuple        # index pages to scrape for PDF links
    doc_type: str            # "act" | "rule" | "constitution" | "other"
    language: str = "en"     # dominant language of documents
    same_host_only: bool = True  # only follow PDF links on the same host


SOURCES: list[Source] = [
    Source(
        key="oag-acts",
        name="Office of the Attorney General — Acts",
        index_urls=("https://oag.gov.bt/language/en/resources/acts-2/",),
        doc_type="act",
    ),
    Source(
        key="oag-rules",
        name="Office of the Attorney General — Rules & Regulations",
        index_urls=("https://oag.gov.bt/language/en/resources/rules-regulations/",),
        doc_type="rule",
    ),
    Source(
        key="nab-acts",
        name="National Assembly of Bhutan — Acts",
        index_urls=("https://www.nab.gov.bt/business/acts",),
        doc_type="act",
    ),
]


def get_source(key: str) -> Source:
    for s in SOURCES:
        if s.key == key:
            return s
    raise KeyError(f"unknown source '{key}' (known: {[s.key for s in SOURCES]})")
