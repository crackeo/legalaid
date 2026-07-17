# Bhutan Legal Aid AI — repo guide

AI legal-information chatbot for Bhutan: crawls the official law PDFs,
answers questions with **enforced citations**, over web chat (streaming),
voice, Telegram, and WhatsApp. Legal information, not legal advice — every
answer carries a disclaimer and cites Act + section.

## Map

| Path | What it is |
|---|---|
| `ingest/` | Phase 1: crawler (SHA-256 manifest, retries), PDF text extraction (+ per-page tesseract OCR), section-level chunker, `update` command |
| `rag/` | Phase 2/9: SQLite index (FTS5 BM25 + embedding vectors), hybrid retrieval (RRF), Claude answering with citation verification, LLM-graded eval, feedback export |
| `app/` | Phases 3–8/10: FastAPI backend (SSE chat, sessions, feedback, voice, WhatsApp webhook, admin API), Telegram bot, rate limiter |
| `web/` | Self-contained UI pages: `index.html` (chat + mic), `admin.html` (dashboard) — no build step |
| `eval/` | Eval question set (retrieval hit-rate + graded answers) |
| `tests/` | 74 tests; phases 5+ are grouped in `test_phaseN*.py` |
| `docs/ARCHITECTURE.md` | Original design doc with "as built" annotations |
| `docs/RUNBOOK.md` | Go-live checklist and operations |

## Commands

```bash
python -m pytest tests/ -q                 # full suite, no network/keys needed
python -m ingest.pipeline crawl|build|update   # corpus (network required)
python -m rag.cli index|ask|eval|export-feedback
uvicorn app.main:app                       # web app
python -m app.telegram                     # telegram bot
```

## Conventions that matter

- **Citation enforcement is the core invariant.** Answers must cite `[n]`
  markers verified against the provisions actually supplied
  (`rag/answer.py:verify_citations`). Never weaken this path; anything that
  changes answering must keep the verify → retry → safe-refusal chain.
- **Every external dependency is injectable and faked in tests**: Claude
  (`FakeLLM` / `FakeStreamingLLM` yield queued responses through the
  `.messages.stream(...)` interface), STT/TTS (`FakeSTT`/`FakeTTS`),
  Telegram (`FakeAPI`), WhatsApp (`FakeWAClient`). Tests must run offline
  with no API keys; `os.environ["LEGALAID_AUTOINIT"] = "0"` before
  importing `app.main`.
- **Claude usage**: model `claude-opus-4-8`, adaptive thinking, streaming,
  system prompt carries `cache_control` — see `rag/answer.py`. The grader
  (`rag/grade.py`) uses a fresh context on purpose.
- The corpus JSONL (`corpus/chunks.jsonl`) is the source of truth; the
  SQLite index is always rebuildable from it. The index records which
  embedder built it — mismatches fail loudly (`check_embedder`).
- Fixture PDFs are generated with PyMuPDF in the Bhutanese drafting style
  (`tests/conftest.py`); the chunker splits on `^\d+\.` sections,
  `Article N`, and `CHAPTER`/`PART` headings.
- `corpus/` is gitignored (PDFs, index, sessions) — never commit it.

## Env vars

| Var | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | answers via Claude (preferred backend) |
| `GEMINI_API_KEY` (+ optional `GEMINI_MODEL`) | answers via Google Gemini when no Anthropic key is set (`rag/llm.py`) |
| `OPENAI_API_KEY` | voice: Whisper STT + TTS (optional) |
| `VOYAGE_API_KEY` | semantic embeddings; else offline hash embedder |
| `LEGALAID_ADMIN_TOKEN` | enables `/admin` (absent ⇒ 404) |
| `LEGALAID_RATE_LIMIT` | requests/min per client IP (default 20) |
| `LEGALAID_TRUST_PROXY=1` | honour X-Forwarded-For (only behind a proxy) |
| `LEGALAID_INDEX_DB`, `LEGALAID_SESSIONS_DB` | data paths |
| `TELEGRAM_BOT_TOKEN` | telegram bot |
| `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_APP_SECRET` | WhatsApp webhook |
