# Go-live runbook

The checklist for taking the Bhutan Legal Aid AI from this repo to a
public deployment, and the operations that follow. Assumes a single VM
with Docker (see `docker-compose.yml`).

## 1. Fill the corpus (first machine with open internet)

```bash
pip install -r requirements.txt
python -m ingest.pipeline crawl          # downloads every Act/Rule PDF
python -m ingest.pipeline build          # -> corpus/chunks.jsonl
python -m rag.cli index                  # -> corpus/index.db
```

Watch for:
- `NO-TEXT (needs OCR)` lines → install tesseract (`apt install
  tesseract-ocr`) and re-run `build`; still-empty documents need manual
  attention.
- `FAIL <url>` lines → the crawler already retried 3×; check the URL by
  hand, it may be a dead link on the source site.
- Skim `corpus/chunks.jsonl` for a few Acts you know: are section numbers
  and headings sensible? The chunker was built against the standard
  drafting style; oddly formatted Acts may need a new pattern in
  `ingest/chunk.py`.

## 2. Validate quality BEFORE exposing it

```bash
python -m rag.cli eval                        # retrieval hit-rate (free)
export ANTHROPIC_API_KEY=sk-ant-...
python -m rag.cli eval --grade                # LLM-graded answers
```

Gate: retrieval hit-rate should be near 100% on the starter set; read
every `fail` in `eval/graded_report.jsonl`. Fix retrieval/chunking before
launch — do not ship on a failing eval.

## 3a. Deploy on Render (easiest)

1. Render dashboard → **New → Blueprint** → connect the GitHub repo (uses
   `render.yaml`; TLS and the domain are automatic).
2. In the service's **Environment** tab set `ANTHROPIC_API_KEY` (plus
   `OPENAI_API_KEY` for voice, `WHATSAPP_*` for WhatsApp).
   `LEGALAID_ADMIN_TOKEN` is auto-generated — copy it for `/admin`.
3. First boot serves an **empty index** — fill the corpus once from the
   service's **Shell** tab (steps 1–2 of this runbook: `crawl`, `build`,
   `rag.cli index`, `rag.cli eval`). The disk at `/srv/legalaid/corpus`
   persists across deploys.
4. Verify `https://<service>.onrender.com/api/health` shows the chunk
   count, then ask a test question at the root URL.
5. Weekly: run `python -m ingest.pipeline update` in the Shell tab (the
   persistent disk is attached to this one service, so run updates there
   rather than from a separate cron service).

Telegram: just set `TELEGRAM_BOT_TOKEN` in the Environment tab — the
container's start script runs the bot alongside the web app in the same
service, sharing the corpus disk.

## 3b. Deploy on a VPS (docker compose)

```bash
# .env: ANTHROPIC_API_KEY (+ OPENAI_API_KEY for voice, VOYAGE_API_KEY,
#       LEGALAID_ADMIN_TOKEN, LEGALAID_TRUST_PROXY=1, WHATSAPP_*/TELEGRAM_*)
docker compose build
docker compose run --rm web python -m rag.cli eval   # sanity inside the image
docker compose up -d
docker compose --profile telegram up -d              # if bot token set
```

- **TLS is mandatory**: put Caddy/nginx in front (mic + WhatsApp webhook
  both require HTTPS). Set `LEGALAID_TRUST_PROXY=1` so rate limiting sees
  real client IPs — and only then.
- Verify: `curl https://<host>/api/health` → status ok, `chunks` in the
  tens of thousands; open `/` and ask a question; open `/admin`.

## 4. Recurring operations

| Cadence | Action |
|---|---|
| weekly (cron) | `docker compose run --rm web python -m ingest.pipeline update` — pulls new/amended Acts |
| after every update | `python -m rag.cli eval` (add `--grade` monthly) |
| weekly | `/admin` → review feedback queue; `rag.cli export-feedback` for the legal reviewer |
| daily (cron) | back up the `corpus/` volume (index + sessions are both in it) |
| per retention policy | `SessionStore.purge_older_than(days)` — wire into cron when a policy is agreed |

## 5. Incidents

| Symptom | First move |
|---|---|
| `/api/health` down | `docker compose logs web`; container restarts automatically (`unless-stopped`) |
| Answers refuse everything ("couldn't produce a reliably cited answer") | Check `chunks` in `/api/health` (empty index?), then `RuntimeError: index was built with embedder ...` in logs — re-run `rag.cli index` |
| 429s for legitimate users | Raise `LEGALAID_RATE_LIMIT`; confirm `LEGALAID_TRUST_PROXY=1` is set behind the proxy (otherwise all users share one bucket) |
| Anthropic/OpenAI outage | Chat degrades to error events, voice returns 502; no action beyond status pages — the app self-recovers |
| Bad answer reported | It's in the admin feedback queue; export, review, and add the corrected expectation to `eval/questions.jsonl` |
| Corpus update broke retrieval | The old index is in the last backup of `corpus/`; restore it, then diff `corpus/raw/manifest.json` to find the offending document |

## 6. Partners (the non-code launch work)

- **BNLI / OAG**: corpus blessing + legal review of `eval/questions.jsonl`
  and the graded reports. A tool endorsed by the source institutions is a
  different product from an unofficial scrape.
- **Native Dzongkha speakers**: evaluate Dzongkha answers before removing
  the "experimental" note (`rag/answer.py:DZONGKHA_NOTE`).
- **Meta Business verification**: required for the WhatsApp number.
