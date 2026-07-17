# Bhutan Legal Aid AI

An AI-powered legal assistance system for Bhutan. It answers questions about
Bhutanese law — the Constitution, Acts, the Penal Code, rules and regulations —
through a chat interface with optional voice input/output, always citing the
exact legal provision the answer comes from.

## Why

Ordinary citizens often cannot afford or access legal counsel. Bhutan's laws
are publicly available as PDFs scattered across government websites, but they
are hard to search and harder to interpret. This project makes them
conversationally accessible.

## How it works (high level)

1. **Ingest** — download every Act, code, rule and regulation from official
   sources (Office of the Attorney General, National Assembly, Judiciary).
2. **Index** — extract text from PDFs, split into provision-level chunks
   (section/article granularity), embed them into a vector database.
3. **Answer** — a RAG (Retrieval-Augmented Generation) pipeline: the user's
   question retrieves the most relevant provisions, and an LLM composes an
   answer grounded in — and citing — those provisions only.
4. **Voice** — speech-to-text on the way in, text-to-speech on the way out,
   wrapped around the same chat pipeline.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design,
technology choices, data sources, and build roadmap.

## Important disclaimer

This system provides **legal information, not legal advice**. Every answer
must carry a disclaimer and cite its sources. For real disputes, users should
be directed to the Bhutan National Legal Institute's Legal Aid Center and
licensed Jabmi (legal counsel).

## Status

**Phase 1 (ingestion pipeline) built and tested.** The `ingest/` package
crawls the official sources, downloads every PDF (rate-limited, cached,
checksummed), extracts and cleans the text, and chunks it into
section-level JSONL with citations metadata.

### Running Phase 1

```bash
pip install -r requirements.txt

# 1. Download every Act / Rule PDF from the registered official sources
#    (needs open internet access to oag.gov.bt / nab.gov.bt)
python -m ingest.pipeline crawl

# 2. Build the corpus: extract, clean, chunk -> corpus/chunks.jsonl
python -m ingest.pipeline build

# tests
python -m pytest tests/
```

Each line of `corpus/chunks.jsonl` is one legal section:

```json
{"doc_title": "Penal Code of Bhutan 2004", "section_number": "92",
 "section_heading": "CHAPTER 3 — OFFENCES AGAINST THE PERSON — Murder",
 "text": "A defendant shall be guilty of the offence of murder if ...",
 "page": 2, "source_url": "https://oag.gov.bt/...pdf", "doc_type": "act",
 "language": "en", "part": 0}
```

Next: Phase 2 — index the corpus into Postgres/pgvector and build the
retrieval + LLM answering core (see the architecture doc).
