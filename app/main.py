"""FastAPI backend for the Bhutan Legal Aid chatbot.

Endpoints:
    POST /api/chat        {"message": ..., "session_id": optional} -> SSE stream
    GET  /api/session/{id}                                         -> history
    POST /api/feedback    {"message_id": ..., "vote": 1|-1, "comment": ""}
    GET  /                                                         -> chat UI

Run:  uvicorn app.main:app --reload      (needs ANTHROPIC_API_KEY and a
built index at corpus/index.db — see README).

The chat endpoint streams Server-Sent Events, one JSON object per event:
sources -> delta* -> [replace] -> done. `replace` means the client must
swap everything shown so far for the given text (failed citation check or
safety refusal). `done` carries the final verified text and message_id
for feedback.
"""

import json
import logging
import os
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag.answer import LegalAidRAG
from rag.embed import embedder_from_env
from rag.store import Store

from fastapi import BackgroundTasks

from .ratelimit import RateLimiter
from .sessions import SessionStore
from .whatsapp import WhatsAppBot, bot_from_env, verify_signature
from .voice import MAX_AUDIO_BYTES, MAX_SPEAK_CHARS, speech_text, stt_from_env, tts_from_env

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_QUESTION_CHARS = 4000

log = logging.getLogger("legalaid")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = None


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_SPEAK_CHARS * 2)


class FeedbackRequest(BaseModel):
    message_id: int
    vote: int = Field(ge=-1, le=1)
    comment: str = Field(default="", max_length=2000)


def create_app(rag: LegalAidRAG | None = None,
               sessions: SessionStore | None = None,
               stt=None, tts=None,
               limiter: RateLimiter | None = None,
               whatsapp: WhatsAppBot | None = None) -> FastAPI:
    app = FastAPI(title="Bhutan Legal Aid AI")

    if limiter is None:
        limiter = RateLimiter(
            per_minute=int(os.environ.get("LEGALAID_RATE_LIMIT", "20")))

    # X-Forwarded-For is client-controlled unless a reverse proxy sets it;
    # trusting it unconditionally would let anyone rotate fake IPs past the
    # rate limiter. Only honour it when the operator says a proxy is in front.
    trust_proxy = os.environ.get("LEGALAID_TRUST_PROXY") == "1"

    def check_rate(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        if trust_proxy:
            fwd = request.headers.get("x-forwarded-for", "")
            client = fwd.split(",")[0].strip() or client
        if not limiter.allow(client):
            raise HTTPException(
                429, "Too many requests — please wait a minute and try again.")

    if stt is None:
        stt = stt_from_env()
    if tts is None:
        tts = tts_from_env()
    if rag is None:
        index_db = os.environ.get("LEGALAID_INDEX_DB", "corpus/index.db")
        rag = LegalAidRAG(Store(index_db), embedder_from_env())
    if sessions is None:
        sessions_db = os.environ.get("LEGALAID_SESSIONS_DB", "corpus/sessions.db")
        sessions = SessionStore(sessions_db)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "media-src 'self' blob:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    def sse(event: dict) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    @app.post("/api/chat")
    def chat(req: ChatRequest, request: Request):
        check_rate(request)
        sid = req.session_id
        if sid and not sessions.session_exists(sid):
            raise HTTPException(404, "unknown session")
        if not sid:
            sid = sessions.create_session()
        history = sessions.history(sid)
        sessions.add_message(sid, "user", req.message)

        def event_stream():
            yield sse({"type": "session", "session_id": sid})
            final = None
            sources = []
            started = time.monotonic()
            try:
                for event in rag.ask_stream(req.message, history=history):
                    if event["type"] == "sources":
                        sources = event["sources"]
                    if event["type"] == "done":
                        final = event
                    yield sse(event)
            except Exception:
                log.exception("chat: answering failed (session %s)", sid)
                yield sse({"type": "error",
                           "text": "Something went wrong answering this "
                                   "question. Please try again."})
                return
            if final is not None:
                duration_ms = int((time.monotonic() - started) * 1000)
                mid = sessions.add_message(sid, "assistant", final["text"],
                                           sources=sources,
                                           verified=final["verified"],
                                           duration_ms=duration_ms)
                log.info("chat: answered in %dms verified=%s cited=%d "
                         "sources=%d session=%s", duration_ms,
                         final["verified"], len(final.get("cited", [])),
                         len(sources), sid)
                yield sse({"type": "saved", "message_id": mid})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/session/{sid}")
    def get_session(sid: str):
        if not sessions.session_exists(sid):
            raise HTTPException(404, "unknown session")
        return {"session_id": sid, "messages": sessions.messages(sid)}

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest, request: Request):
        check_rate(request)
        if req.vote not in (-1, 1):
            raise HTTPException(422, "vote must be 1 or -1")
        if not sessions.message_exists(req.message_id):
            raise HTTPException(404, "unknown message")
        sessions.add_feedback(req.message_id, req.vote, req.comment)
        return {"ok": True}

    @app.get("/api/health")
    def health():
        """Liveness + corpus visibility, for Docker healthchecks and ops.

        `chunks: 0` means the index is empty — the app answers "I don't find
        this in the laws available to me" to everything until the corpus is
        built (see docs/RUNBOOK.md).
        """
        # Report the backend actually answering, not the Claude default:
        # GeminiLLM carries its own .model, so a Gemini deployment no longer
        # reports a Claude model name.
        backend = "gemini" if type(rag.llm).__name__ == "GeminiLLM" else "claude"
        return {
            "status": "ok",
            "chunks": rag.store.count(),
            "corpus_ready": rag.store.count() > 0,
            "embedder": rag.embedder.name,
            "backend": backend,
            "model": getattr(rag.llm, "model", None) or rag.model,
            "voice": {"stt": stt is not None, "tts": tts is not None},
        }

    @app.get("/api/voice/config")
    def voice_config():
        return {"stt": stt is not None, "tts": tts is not None}

    @app.post("/api/voice/transcribe")
    async def transcribe(audio: UploadFile, request: Request):
        check_rate(request)
        if stt is None:
            raise HTTPException(503, "voice is not configured (set OPENAI_API_KEY)")
        data = await audio.read()
        if not data:
            raise HTTPException(422, "empty audio")
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "audio too large")
        try:
            text = stt.transcribe(data, mime=audio.content_type or "audio/webm")
        except Exception:
            raise HTTPException(502, "transcription failed, please try again")
        if not text:
            raise HTTPException(422, "could not hear a question in the audio")
        return {"text": text}

    @app.post("/api/voice/speak")
    def speak(req: SpeakRequest, request: Request):
        check_rate(request)
        if tts is None:
            raise HTTPException(503, "voice is not configured (set OPENAI_API_KEY)")
        try:
            audio = tts.synthesize(speech_text(req.text))
        except Exception:
            raise HTTPException(502, "speech synthesis failed, please try again")
        return Response(content=audio, media_type="audio/mpeg")

    # -- WhatsApp webhook (enabled when WHATSAPP_* env or bot injected) -----

    if whatsapp is None:
        whatsapp = bot_from_env(rag, stt=stt)
    wa_verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
    wa_app_secret = os.environ.get("WHATSAPP_APP_SECRET", "")

    if whatsapp is not None:
        if not wa_app_secret:
            log.warning(
                "WhatsApp webhook is enabled WITHOUT signature verification — "
                "set WHATSAPP_APP_SECRET so forged webhook posts are rejected.")

        @app.get("/api/whatsapp/webhook")
        def whatsapp_verify(request: Request):
            """Meta's one-time webhook verification handshake."""
            params = request.query_params
            if (params.get("hub.mode") == "subscribe"
                    and wa_verify_token
                    and secrets.compare_digest(
                        params.get("hub.verify_token", ""), wa_verify_token)):
                return Response(params.get("hub.challenge", ""),
                                media_type="text/plain")
            raise HTTPException(403, "verification failed")

        @app.post("/api/whatsapp/webhook")
        async def whatsapp_webhook(request: Request,
                                   background: BackgroundTasks):
            raw = await request.body()
            if wa_app_secret and not verify_signature(
                    wa_app_secret, raw,
                    request.headers.get("x-hub-signature-256")):
                raise HTTPException(403, "bad signature")
            try:
                payload = json.loads(raw)
            except ValueError:
                raise HTTPException(422, "invalid JSON")
            # Ack immediately (Meta retries slow webhooks); answer in the
            # background — replies go out via the Graph API, not this response.
            background.add_task(whatsapp.handle_payload, payload)
            return {"ok": True}

    # -- admin (enabled only when LEGALAID_ADMIN_TOKEN is set) -------------

    admin_token = os.environ.get("LEGALAID_ADMIN_TOKEN", "")

    def check_admin(request: Request) -> None:
        if not admin_token:
            raise HTTPException(404)  # admin disabled: don't reveal it exists
        supplied = request.headers.get("authorization", "")
        if not supplied.startswith("Bearer ") or not secrets.compare_digest(
                supplied[7:], admin_token):
            raise HTTPException(401, "invalid admin token")

    @app.get("/api/admin/stats")
    def admin_stats(request: Request):
        check_admin(request)
        return sessions.stats()

    @app.get("/api/admin/feedback")
    def admin_feedback(request: Request, limit: int = 50):
        check_admin(request)
        return {"feedback": sessions.recent_feedback(limit=min(limit, 200))}

    @app.get("/admin")
    def admin_page():
        if not admin_token:
            raise HTTPException(404)
        return FileResponse(WEB_DIR / "admin.html")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app() if os.environ.get("LEGALAID_AUTOINIT", "1") == "1" else None
