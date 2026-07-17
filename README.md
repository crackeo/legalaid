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

Pre-development. Architecture and data-source research complete
(see `docs/`). Next step: build the document ingestion pipeline (Phase 1 in
the architecture doc).
