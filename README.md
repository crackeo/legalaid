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

### Running Phase 2 (RAG core)

```bash
# 1. Build the search index from the Phase 1 corpus (SQLite: BM25 + vectors)
#    Uses Voyage AI embeddings if VOYAGE_API_KEY is set; otherwise an
#    offline hashing embedder (BM25 then carries most of the weight).
python -m rag.cli index

# 2. Check retrieval quality — no API key needed; run on every change
python -m rag.cli eval

# 3. Ask a question (needs ANTHROPIC_API_KEY for Claude)
export ANTHROPIC_API_KEY=sk-ant-...
python -m rag.cli ask "What is the punishment for defamation in Bhutan?"
```

How answering works: the question runs through hybrid retrieval (BM25 +
vector similarity, fused with reciprocal rank fusion), the top provisions go
to Claude (`claude-opus-4-8`) with strict grounding rules, and every citation
in the answer is programmatically verified against the supplied provisions —
an answer with invented or missing citations is retried once, then replaced
by a safe refusal. Every answer carries the legal-information-not-legal-advice
disclaimer.

`eval/questions.jsonl` is the starter evaluation set (retrieval hit-rate);
grow it with legally reviewed Q&A pairs as the corpus fills in.

### Running Phase 3 (web chatbot)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

The web app streams answers token-by-token over Server-Sent Events, shows
the cited sources (linked to the official PDFs) under each answer, keeps
per-session conversation history so follow-up questions work, and collects
👍/👎 feedback per answer. If citation verification fails on the full text,
the client is told to replace the streamed draft with a safe response — an
unverified answer is never final.

Privacy: no accounts, no tracking; sessions live in a local SQLite file and
`SessionStore.purge_older_than(days)` implements retention. Config via env:
`LEGALAID_INDEX_DB` (default `corpus/index.db`), `LEGALAID_SESSIONS_DB`
(default `corpus/sessions.db`).

### Running Phase 4 (voice)

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # answers (Claude)
export OPENAI_API_KEY=sk-...          # voice (Whisper STT + TTS)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

With `OPENAI_API_KEY` set, the chat UI shows a 🎤 button: click to record,
click again to stop — the audio is transcribed by Whisper, the question runs
through the normal cited-answer pipeline, and the answer is read aloud
(citation markers and the markdown footer are stripped for speech, replaced
by one short spoken disclaimer). Every answer also gets a 🔊 replay button,
and a "read answers aloud" toggle controls auto-playback. Without the key,
voice endpoints return 503 and the UI simply hides the mic — text chat is
unaffected.

Voice is **English-only for now**. Dzongkha speech is a research track (no
mainstream model supports it) — the STT/TTS backends in `app/voice.py` are
single-method interfaces precisely so a fine-tuned Dzongkha model can be
dropped in later. Note: browsers require HTTPS (or localhost) for
microphone access — put the app behind TLS in production.

### Running Phase 5 (hardening & reach)

**Keeping the corpus current** — one command re-crawls the official sources,
and only if a new or amended PDF appeared does it rebuild the corpus and
re-index (unchanged documents are never re-downloaded thanks to the SHA-256
manifest):

```bash
python -m ingest.pipeline update
# weekly cron:
# 0 3 * * 1  cd /srv/legalaid && python -m ingest.pipeline update >> update.log 2>&1
```

**Telegram bot** — the same cited-answer pipeline over Telegram (voice notes
included when `OPENAI_API_KEY` is set):

```bash
export TELEGRAM_BOT_TOKEN=123:abc   # from @BotFather
python -m app.telegram
```

**Feedback → evaluation loop** — export every 👎-voted answer with its
question for legal review; confirmed problems become new entries in
`eval/questions.jsonl`:

```bash
python -m rag.cli export-feedback   # -> eval/review_queue.jsonl
```

**Dzongkha text (experimental)** — questions written in Dzongkha script are
answered in Dzongkha (Act names and section numbers stay in English, same
citation rules), and every such answer carries an explicit experimental
warning. This stays experimental until evaluated by native speakers;
Dzongkha *speech* remains a research track needing transcribed audio data.

### Running Phase 6 (production readiness)

**OCR for scanned PDFs** — pages without a text layer are now OCRed
automatically during extraction (per page, since many OAG PDFs mix
born-digital pages with scanned annexes). Needs the tesseract binary:

```bash
sudo apt install tesseract-ocr && pip install pytesseract pillow
python -m ingest.pipeline update --force   # rebuild to pick up scanned Acts
```

**Rate limiting** — `/api/chat` and `/api/voice/*` are capped per client IP
(default 20 requests/minute, `LEGALAID_RATE_LIMIT` to change) so a single
client can't burn the Claude/Whisper budget. Honours `X-Forwarded-For`
behind a reverse proxy; returns 429 when exceeded.

**Health endpoint** — `GET /api/health` reports corpus size, embedder,
model, and voice availability; wired into the Docker healthcheck.

**Docker deployment (single VM):**

```bash
export ANTHROPIC_API_KEY=sk-ant-...             # + OPENAI/VOYAGE/TELEGRAM as wanted
docker compose run --rm web python -m ingest.pipeline update   # fill the corpus
docker compose up -d                                           # web on :8000
docker compose --profile telegram up -d                        # + telegram bot
# weekly cron on the host:
# 0 3 * * 1  cd /srv/legalaid && docker compose run --rm web python -m ingest.pipeline update
```

Put TLS in front (Caddy/nginx — browsers require HTTPS for microphone
access). The `corpus/` volume holds everything stateful: raw PDFs, the
chunk corpus, the search index, and chat sessions.

### Running Phase 7 (operations & quality dashboard)

```bash
export LEGALAID_ADMIN_TOKEN=$(openssl rand -hex 24)   # enables /admin
uvicorn app.main:app --port 8000
# open http://localhost:8000/admin and paste the token
```

The dashboard shows sessions/questions/answers, the **verified-citation
rate** (the system's core quality number), average answer latency,
👍/👎 totals, a questions-per-day chart, and the **feedback review queue** —
every vote with its question, answer, comment, and whether the citation
check passed. Admin is token-gated (constant-time comparison) and entirely
absent (404) when `LEGALAID_ADMIN_TOKEN` is unset. Answer latency is now
recorded per message (existing session databases migrate automatically),
and every Q&A emits a structured log line with duration, verification
result, and citation count.

### Running Phase 8 (WhatsApp channel)

The same cited-answer pipeline over WhatsApp, via the Meta Cloud API
(webhook-based — the web app must be reachable over public HTTPS):

1. On [developers.facebook.com](https://developers.facebook.com): create a
   Business app → add the WhatsApp product → get a permanent access token
   and the phone number ID.
2. Configure the webhook to `https://<your-host>/api/whatsapp/webhook`,
   subscribed to `messages`, with a verify token you choose.
3. Run the web app with:

```bash
export WHATSAPP_TOKEN=EAAG...            # Graph API token
export WHATSAPP_PHONE_NUMBER_ID=1234...
export WHATSAPP_VERIFY_TOKEN=<same as portal>
export WHATSAPP_APP_SECRET=...           # recommended: enables signature checks
```

Incoming texts get cited answers with the disclaimer; voice notes are
transcribed when `OPENAI_API_KEY` is set; greetings get a welcome message.
The webhook acknowledges instantly and answers in a background task (Meta
retries slow webhooks), deduplicates re-delivered messages, and — with the
app secret set — rejects any request whose `X-Hub-Signature-256` HMAC
doesn't verify. Unconfigured deployments don't expose the endpoint at all.

### Running Phase 9 (quality assurance & CI)

**LLM-graded answer evaluation** — beyond retrieval hit-rate, grade the
actual answers: a fresh Claude context acts as a strict examiner, checking
each answer for groundedness, citation correctness, completeness and
clarity against the very provisions it cited:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m rag.cli eval --grade        # -> eval/graded_report.jsonl
```

Failed grades are a triage signal for the human legal review queue, not a
replacement for it — review `fail` answers first. Run before every deploy
and after every corpus update.

**CI** — `.github/workflows/ci.yml` runs the full test suite (58 tests,
including real-tesseract OCR) on Python 3.11 and 3.12 for every push and
pull request.

**Crawler hardening** — downloads now retry transient failures (network
errors, 5xx) three times with exponential backoff; 4xx responses are not
retried. A flaky government server no longer costs the corpus a document.

### Phase 10 (security audit & fixes)

A full audit of the attack surface (web API, both bot channels, crawler,
browser UI) found and fixed seven issues:

| Finding | Fix |
|---|---|
| XSS: source titles/URLs (from crawled PDF link text) reached the DOM via `innerHTML` | Sources rendered with DOM nodes only; links restricted to http(s) URLs |
| Rate-limit bypass: `X-Forwarded-For` trusted unconditionally (spoofable without a proxy) | Header honoured only with `LEGALAID_TRUST_PROXY=1` — set it when behind nginx/Caddy |
| Feedback abuse: unlimited, unvalidated writes | Rate-limited; `message_id` must exist; vote must be ±1 |
| Memory DoS: unbounded per-chat histories in Telegram/WhatsApp bots | LRU cap (5000 chats) |
| Unbounded bot questions | Capped at 4000 chars (matches web API) |
| No browser hardening headers | CSP, `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy` on every response |
| WhatsApp signature check silently optional | Loud startup warning when `WHATSAPP_APP_SECRET` is unset |

Notes that remain true by design: session IDs are unguessable UUIDs and the
only access key to a conversation; prompt-injection attempts via questions
are constrained by the grounding rules plus programmatic citation
verification; PDFs are parsed from official government sources only.

Still open (needs partners, not code): legal review of the eval set with
BNLI/OAG, native-speaker Dzongkha evaluation, and the Dzongkha speech
data-collection track.
