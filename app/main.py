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

from .ratelimit import RateLimiter
from .sessions import SessionStore
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
               limiter: RateLimiter | None = None) -> FastAPI:
    app = FastAPI(title="Bhutan Legal Aid AI")

    if limiter is None:
        limiter = RateLimiter(
            per_minute=int(os.environ.get("LEGALAID_RATE_LIMIT", "20")))

    def check_rate(request: Request) -> None:
        # Honour the first X-Forwarded-For hop when behind a reverse proxy.
        fwd = request.headers.get("x-forwarded-for", "")
        client = fwd.split(",")[0].strip() or (
            request.client.host if request.client else "unknown")
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
    def feedback(req: FeedbackRequest):
        sessions.add_feedback(req.message_id, req.vote, req.comment)
        return {"ok": True}

    @app.get("/api/health")
    def health():
        """Liveness + corpus visibility, for Docker healthchecks and ops."""
        return {
            "status": "ok",
            "chunks": rag.store.count(),
            "embedder": rag.embedder.name,
            "model": rag.model,
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
