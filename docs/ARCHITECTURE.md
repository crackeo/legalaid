# Bhutan Legal Aid AI — System Architecture

> **Status: this was the design document; the system is now built.** Where
> implementation diverged, the "As built" notes below are authoritative —
> the main differences are that the vector store is **SQLite (FTS5 + numpy
> cosine)** instead of Postgres/pgvector, reranking is **reciprocal rank
> fusion** without a cross-encoder stage, and the roadmap grew from five
> phases to eleven (WhatsApp, admin dashboard, LLM-graded eval, security
> and correctness audits). See the README for the as-built run guide per
> phase and `CLAUDE.md` for the repo map.

This document answers three questions:

1. **Where do Bhutan's laws live online, and how do we get them?**
2. **How do we turn thousands of pages of PDFs into an AI that answers
   accurately with citations?**
3. **What is the full system — chat, voice, backend — and in what order do
   we build it?**

---

## 1. Data sources: where the laws are

All of Bhutan's primary legislation is published as PDFs on official
government sites. These are the ingestion targets, in priority order:

| Source | URL | What it has |
|---|---|---|
| Office of the Attorney General — Acts | https://oag.gov.bt/language/en/resources/acts-2/ | The most complete collection of Acts, English + Dzongkha PDFs |
| Office of the Attorney General — Rules & Regulations | https://oag.gov.bt/language/en/resources/rules-regulations/ | Subsidiary legislation |
| National Assembly of Bhutan | https://www.nab.gov.bt/business/acts | Acts as passed by Parliament |
| Judiciary of Bhutan | https://www.judiciary.gov.bt/ | Court structure, forms, some judgments |
| AsianLII — Bhutan | https://www.asianlii.org/bt/ | HTML versions of selected laws (e.g. Penal Code 2004) — useful because HTML is cleaner than PDF |
| Bhutan Media Foundation | http://www.bmf.bt/resources/acts-of-bhutan/ | Mirror of Acts, useful fallback |
| Library of Congress guide | https://guides.loc.gov/law-bhutan | Meta-index of all official sources |

Core documents that must be in the first corpus:

- **The Constitution of the Kingdom of Bhutan (2008)**
- **Penal Code of Bhutan 2004** (+ Amendment Act 2011)
- **Civil and Criminal Procedure Code 2001** (+ amendments)
- **Evidence Act 2005**
- Marriage Act, Land Act 2007, Labour and Employment Act 2007,
  Consumer Protection Act 2012, Domestic Violence Prevention Act 2013,
  Child Care and Protection Act 2011, Contract Act 2013, and the rest of the
  OAG Acts catalogue (~150+ Acts).

### Ingestion pipeline (Phase 1)

```
crawler → downloader → text extraction → cleaning → chunking → metadata → store
```

1. **Crawler/downloader** (Python: `httpx` + `BeautifulSoup`): scrape the OAG
   Acts and Rules index pages, collect every PDF link, download with
   rate-limiting (be polite: 1 req/sec, identify with a User-Agent, cache
   everything so we never re-download). Store raw PDFs in object storage or
   a `corpus/raw/` directory, keyed by source URL + SHA-256 hash so
   re-crawls detect updates and amendments.
2. **Text extraction** (`PyMuPDF` first; `pdfplumber` fallback): most OAG
   PDFs have a text layer. For scanned ones, OCR with `Tesseract` (English).
   Dzongkha OCR is a known hard problem — start English-only and treat
   Dzongkha text extraction as a later phase.
3. **Cleaning**: strip headers/footers/page numbers, fix hyphenation,
   normalise whitespace. Keep the original page number per paragraph — we
   need it for citations.
4. **Structure-aware chunking** — this is the single most important quality
   decision. Do NOT chunk by fixed token count. Legal text has natural
   units: split by **Section / Article**, using regex over the numbering
   conventions (`Section 12.`, `Article 7`, `12(1)(a)`). Each chunk =
   one section (or one article clause), typically 100–800 tokens. If a
   section exceeds ~1,000 tokens, split at subsection boundaries with
   the section heading repeated in each piece.
5. **Metadata per chunk** (stored alongside the text):
   `{act_title, year, act_status (in-force/amended/repealed), section_number,
   section_heading, page, source_url, language, checksum}`.
   Status matters: the Penal Code 2004 is amended by the 2011 Amendment Act —
   the system must know which version controls.
6. **Store**: chunks as JSONL in the repo/object storage (the canonical
   corpus), then indexed into the vector DB (below). The JSONL corpus is the
   source of truth; the index is rebuildable from it.

---

## 2. The AI core: RAG, not fine-tuning

**Do not fine-tune a model on the laws.** Fine-tuning bakes text into weights
where it can be misremembered, can't be updated when a law is amended, and
can't cite sources. **Retrieval-Augmented Generation (RAG)** is the correct
architecture for legal Q&A:

```
question → [rewrite/expand] → hybrid retrieval → rerank → LLM with retrieved
sections in context → answer with citations → guardrails → user
```

### Components

- **Embeddings**: a multilingual embedding model (e.g. `voyage-3` or
  `text-embedding-3-large`, or open-source `bge-m3` if self-hosting).
  Multilingual matters because users will ask in Dzongkha-inflected English
  and, later, Dzongkha itself.
- **Vector DB**: start with **pgvector on Postgres** (one database for app
  data + vectors, simplest ops) or **Qdrant** if you want a dedicated store.
  The corpus is small by vector-DB standards (~150 Acts ≈ tens of thousands
  of chunks) — anything works; choose for operational simplicity.
  *As built:* even simpler — a single SQLite file (`rag/store.py`) holding
  chunks, an FTS5 index for BM25, and embedding vectors scored with numpy
  cosine. Zero infrastructure; swap for pgvector behind the same `Store`
  interface if the corpus ever outgrows it.
- **Hybrid retrieval**: combine vector similarity with **BM25 keyword
  search**. Legal queries are full of exact terms of art ("nangi zhib",
  "felony of the fourth degree", "Section 410") where keyword search beats
  embeddings. Fuse with reciprocal-rank fusion, retrieve top ~20, then
  **rerank** (e.g. Cohere Rerank or a cross-encoder) down to the ~6 best
  sections that actually go into the prompt.
  *As built:* fusion is reciprocal rank fusion only (`rag/retrieve.py`);
  a cross-encoder rerank stage remains a future upgrade if eval shows
  retrieval as the bottleneck.
- **LLM**: Claude (Anthropic API) is a strong fit for legal reasoning and
  long context; the system prompt must instruct it to:
  1. answer **only** from the provided sections,
  2. cite Act + section number for every claim,
  3. say "I don't find this in the laws I have" rather than guess,
  4. append the legal-information-not-legal-advice disclaimer,
  5. for serious matters (criminal charges, domestic violence), also point
     to real help: BNLI Legal Aid Center, police, court registry.
- **Conversation memory**: store chat history per session; condense the
  running conversation into the retrieval query (question rewriting) so
  follow-ups like "what's the penalty for that?" retrieve correctly.

### Quality & safety layer (non-negotiable for legal domain)

- **Citation verification**: after generation, programmatically check every
  cited section number exists in the retrieved set; if not, regenerate or
  refuse.
- **Evaluation set**: build 100–200 question/answer pairs reviewed by someone
  with legal knowledge (BNLI would be the ideal partner). Run on every
  pipeline change. Measure: retrieval hit-rate, citation accuracy, refusal
  correctness.
- **Out-of-scope detection**: questions asking for advice on *committing*
  offences, or for predictions of case outcomes, get a safe template
  response.
- **Amendment awareness**: retrieval must prefer in-force text; answers about
  amended sections must mention the amendment.

---

## 3. Full system architecture

```
┌─────────────────────────────────────────────────────┐
│  Clients: Web app (chat + mic)  ·  later: mobile,   │
│  Telegram/WhatsApp bot (huge reach for low cost)    │
└──────────────────────┬──────────────────────────────┘
                       │ HTTPS / WebSocket
┌──────────────────────▼──────────────────────────────┐
│  API backend (FastAPI, Python)                      │
│  · /chat (SSE streaming)  · /voice (audio in/out)   │
│  · sessions, rate limiting, feedback endpoint       │
├───────────────┬─────────────────┬───────────────────┤
│ RAG service   │ STT service     │ TTS service       │
│ retrieve →    │ Whisper API /   │ ElevenLabs /      │
│ rerank → LLM  │ faster-whisper  │ Azure / Piper     │
├───────────────┴─────────────────┴───────────────────┤
│  Postgres (+pgvector): chunks, vectors, sessions,   │
│  feedback, eval results                             │
│  Object storage: raw PDFs (corpus/raw)              │
└─────────────────────────────────────────────────────┘
        ▲
        │ scheduled re-crawl (weekly) — detects new/amended Acts
┌───────┴───────────────┐
│ Ingestion pipeline    │  (Section 1 above)
└───────────────────────┘
```

### Voice: the honest assessment

Voice = STT (speech → text) + the same chat pipeline + TTS (text → speech).

- **English voice: easy.** OpenAI Whisper (API or self-hosted
  `faster-whisper`) for STT handles accented English well. TTS via
  ElevenLabs, Azure Speech, or open-source Piper.
- **Dzongkha voice: the hard problem.** No mainstream STT/TTS supports
  Dzongkha well today. Whisper has no Dzongkha; it sometimes misroutes it to
  Tibetan with poor results. Realistic path:
  - **Phase A**: launch voice in English only, be explicit about it.
  - **Phase B**: investigate NLP work from Bhutanese institutions
    (Dzongkha Development Commission, college research groups) and
    fine-tuning Whisper on Dzongkha data — a genuine research/collection
    effort (needs hundreds of hours of transcribed audio).
  - Text chat in Dzongkha is nearer-term: modern LLMs have some Dzongkha
    ability and translation-assisted flows (Dzongkha ↔ English around an
    English RAG core) can work — but must be evaluated with native speakers
    before shipping.
- **Interface pattern**: push-to-talk in the browser
  (`MediaRecorder` → send audio → STT → chat → TTS → play). WebSocket or
  simple request/response both work; don't build full duplex streaming
  voice until the basics are solid.

### Tech stack summary (recommended)

| Layer | Choice | Why |
|---|---|---|
| Ingestion | Python, httpx, BeautifulSoup, PyMuPDF, Tesseract | standard, boring, reliable |
| Corpus store | JSONL in object storage / repo | rebuildable, diffable, auditable |
| Vector + app DB | Postgres + pgvector *(as built: SQLite — FTS5 + numpy)* | one database to operate |
| Keyword search | Postgres full-text or Elasticsearch/Meilisearch | hybrid retrieval |
| Backend | FastAPI | async, SSE streaming, Python ML ecosystem |
| LLM | Claude API | legal reasoning, long context, citations |
| Embeddings | Voyage / OpenAI / bge-m3 | multilingual |
| STT | Whisper (API or faster-whisper) | best accented-English STT |
| TTS | ElevenLabs / Azure / Piper | quality vs cost vs self-host |
| Frontend | Next.js/React chat UI with mic button | streaming UX |
| Deploy | Docker; single VM is fine to start | corpus is small, traffic modest |

---

## 4. Build roadmap

**Phase 1 — Corpus (1–2 weeks)**
Crawler + downloader for OAG Acts/Rules pages → raw PDF archive → extraction
→ section-level chunker → JSONL corpus with metadata. *Deliverable: every
in-force Act of Bhutan as clean, cited, section-level chunks.*

**Phase 2 — RAG core (1–2 weeks)**
Index corpus into pgvector + BM25; retrieval + rerank; Claude prompt with
citation and refusal rules; CLI/notebook for testing; first 50-question
eval set. *Deliverable: accurate cited answers in a terminal.*

**Phase 3 — Chat product (2–3 weeks)**
FastAPI backend, streaming chat endpoint, sessions, web UI, disclaimers,
feedback thumbs, logging. *Deliverable: usable web chatbot.*

**Phase 4 — Voice (1–2 weeks for English)**
Push-to-talk mic in UI → Whisper STT → pipeline → TTS playback.
*Deliverable: voice chat in English.*

**Phase 5 — Hardening & reach**
Weekly re-crawl for amendments, expanded eval set with legal review,
Telegram/WhatsApp channel, Dzongkha text support behind evaluation,
Dzongkha voice research track.

---

## 5. Legal & ethical requirements

- Every answer: *"This is legal information, not legal advice."* + citations.
- Direct users with real disputes to the **Bhutan National Legal Institute
  Legal Aid Center** (established 2022) and licensed Jabmi.
- Respect source sites: rate-limited crawling, attribution, and ideally a
  conversation with OAG/BNLI about official cooperation — a legal-aid tool
  endorsed by the institutions that publish the law is far stronger than an
  unofficial scrape.
- Privacy: legal questions are sensitive. Don't require accounts for basic
  use; minimise logging of personal details; state a retention policy.
- Bias/accuracy: publish known limitations; keep the human-reviewed eval set
  growing; add a visible "report a wrong answer" button.
