import hashlib
import hmac
import json
import os

import pytest

os.environ["LEGALAID_AUTOINIT"] = "0"

from fastapi.testclient import TestClient

from app.main import create_app
from app.sessions import SessionStore
from app.telegram import WELCOME
from app.whatsapp import WhatsAppBot, verify_signature
from rag.answer import LegalAidRAG
from rag.embed import HashEmbedder
from rag.store import Store, index_corpus
from tests.test_rag import CORPUS, FakeLLM
from tests.test_voice import FakeSTT


class FakeWAClient:
    def __init__(self):
        self.sent = []          # (to, text)
        self.media = {}         # media_id -> (bytes, mime)

    def send_text(self, to, text):
        self.sent.append((to, text))

    def download_media(self, media_id):
        return self.media[media_id]


def _rag(tmp_path, llm):
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    return LegalAidRAG(Store(tmp_path / "index.db"), HashEmbedder(), llm=llm)


def _payload(message, mid="wamid.1"):
    message = {"id": mid, **message}
    return {"entry": [{"changes": [{"value": {"messages": [message]}}]}]}


# -- bot logic ----------------------------------------------------------------

def test_text_message_answered_with_citations(tmp_path):
    wa = FakeWAClient()
    bot = WhatsAppBot(_rag(tmp_path, FakeLLM(["Murder is a felony [2]."])), wa)
    n = bot.handle_payload(_payload(
        {"from": "97517000000", "type": "text",
         "text": {"body": "What is the punishment for murder?"}}))
    assert n == 1
    to, text = wa.sent[-1]
    assert to == "97517000000"
    assert "felony" in text and "Sources:" in text
    assert "legal information, not legal advice" in text
    assert len(bot._histories["97517000000"]) == 2


def test_greeting_and_duplicate_delivery(tmp_path):
    wa = FakeWAClient()
    bot = WhatsAppBot(_rag(tmp_path, FakeLLM([])), wa)
    payload = _payload({"from": "1", "type": "text", "text": {"body": "hello"}},
                       mid="wamid.dup")
    bot.handle_payload(payload)
    bot.handle_payload(payload)  # Meta re-delivery: must not send twice
    assert [t for _, t in wa.sent] == [WELCOME]


def test_voice_note_transcribed(tmp_path):
    wa = FakeWAClient()
    wa.media["M1"] = (b"ogg-bytes", "audio/ogg")
    bot = WhatsAppBot(_rag(tmp_path, FakeLLM(["Assault is an offence [1]."])),
                      wa, stt=FakeSTT("is assault a crime?"))
    bot.handle_payload(_payload({"from": "2", "type": "audio",
                                 "audio": {"id": "M1", "voice": True}}))
    assert "Assault is an offence" in wa.sent[-1][1]


def test_voice_without_stt_and_ignored_types(tmp_path):
    wa = FakeWAClient()
    bot = WhatsAppBot(_rag(tmp_path, FakeLLM([])), wa)
    bot.handle_payload(_payload({"from": "3", "type": "audio",
                                 "audio": {"id": "M9"}}))
    assert "type your question" in wa.sent[-1][1]
    bot.handle_payload(_payload({"from": "3", "type": "sticker"}, mid="wamid.s"))
    assert len(wa.sent) == 1  # sticker ignored

    # status-only payloads (delivered/read receipts) carry no messages
    assert bot.handle_payload(
        {"entry": [{"changes": [{"value": {"statuses": [{"status": "read"}]}}]}]}
    ) == 0


# -- webhook endpoints ----------------------------------------------------------

SECRET = "app-secret"


@pytest.fixture()
def wa_client(tmp_path, monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "vtok")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", SECRET)
    wa = FakeWAClient()
    bot = WhatsAppBot(_rag(tmp_path, FakeLLM(["Murder is a felony [1]."] * 3)), wa)
    client = TestClient(create_app(
        rag=bot.rag, sessions=SessionStore(tmp_path / "s.db"), whatsapp=bot))
    return client, wa


def _signed(body: bytes) -> dict:
    sig = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    return {"X-Hub-Signature-256": f"sha256={sig}",
            "Content-Type": "application/json"}


def test_webhook_verification_handshake(wa_client):
    client, _ = wa_client
    ok = client.get("/api/whatsapp/webhook", params={
        "hub.mode": "subscribe", "hub.verify_token": "vtok",
        "hub.challenge": "12345"})
    assert ok.status_code == 200 and ok.text == "12345"
    bad = client.get("/api/whatsapp/webhook", params={
        "hub.mode": "subscribe", "hub.verify_token": "wrong",
        "hub.challenge": "12345"})
    assert bad.status_code == 403


def test_webhook_requires_valid_signature(wa_client):
    client, wa = wa_client
    body = json.dumps(_payload({"from": "4", "type": "text",
                                "text": {"body": "what is murder?"}})).encode()
    assert client.post("/api/whatsapp/webhook", content=body,
                       headers={"Content-Type": "application/json"}).status_code == 403
    assert client.post(
        "/api/whatsapp/webhook", content=body,
        headers={**_signed(body), "X-Hub-Signature-256": "sha256=" + "0" * 64},
    ).status_code == 403

    resp = client.post("/api/whatsapp/webhook", content=body, headers=_signed(body))
    assert resp.status_code == 200 and resp.json() == {"ok": True}
    assert wa.sent and "felony" in wa.sent[-1][1]  # background task ran


def test_webhook_absent_when_unconfigured(tmp_path, monkeypatch):
    for var in ("WHATSAPP_TOKEN", "WHATSAPP_PHONE_NUMBER_ID",
                "WHATSAPP_VERIFY_TOKEN", "WHATSAPP_APP_SECRET"):
        monkeypatch.delenv(var, raising=False)
    client = TestClient(create_app(
        rag=_rag(tmp_path, FakeLLM([])), sessions=SessionStore(tmp_path / "s.db")))
    assert client.post("/api/whatsapp/webhook", json={}).status_code == 404


# -- signature helper -----------------------------------------------------------

def test_verify_signature():
    body = b'{"x":1}'
    sig = "sha256=" + hmac.new(b"s", body, hashlib.sha256).hexdigest()
    assert verify_signature("s", body, sig)
    assert not verify_signature("s", body, "sha256=deadbeef")
    assert not verify_signature("s", body, None)
    assert not verify_signature("s", b"tampered", sig)
