import json
import os

import pytest

os.environ["LEGALAID_AUTOINIT"] = "0"

from fastapi.testclient import TestClient

from app.main import create_app
from app.ratelimit import RateLimiter
from app.sessions import SessionStore
from app.telegram import ChatHistories
from rag.answer import LegalAidRAG
from rag.embed import HashEmbedder
from rag.store import Store, index_corpus
from tests.test_app import FakeStreamingLLM, GOOD_ANSWER, read_events
from tests.test_rag import CORPUS


def _client(tmp_path, monkeypatch=None, trust_proxy=False, burst=2):
    if monkeypatch and trust_proxy:
        monkeypatch.setenv("LEGALAID_TRUST_PROXY", "1")
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    rag = LegalAidRAG(Store(tmp_path / "index.db"), HashEmbedder(),
                      llm=FakeStreamingLLM([GOOD_ANSWER] * 20))
    sessions = SessionStore(tmp_path / "s.db")
    return TestClient(create_app(
        rag=rag, sessions=sessions,
        limiter=RateLimiter(per_minute=60, burst=burst))), sessions


def test_spoofed_forwarded_for_cannot_bypass_rate_limit(tmp_path):
    client, _ = _client(tmp_path, burst=2)
    for i in range(2):
        assert client.post("/api/chat", json={"message": "q"},
                           headers={"X-Forwarded-For": f"1.2.3.{i}"}).status_code == 200
    # third request rotates the spoofed header again — must still be blocked
    assert client.post("/api/chat", json={"message": "q"},
                       headers={"X-Forwarded-For": "9.9.9.9"}).status_code == 429


def test_forwarded_for_honoured_when_proxy_trusted(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, trust_proxy=True, burst=2)
    for i in range(3):
        resp = client.post("/api/chat", json={"message": "q"},
                           headers={"X-Forwarded-For": f"10.0.0.{i}"})
        assert resp.status_code == 200  # distinct real clients, separate buckets


def test_feedback_validates_message_and_vote(tmp_path):
    client, _ = _client(tmp_path, burst=50)
    resp = client.post("/api/chat", json={"message": "what is murder?"})
    mid = next(e for e in read_events(resp) if e["type"] == "saved")["message_id"]

    assert client.post("/api/feedback",
                       json={"message_id": mid, "vote": 1}).status_code == 200
    assert client.post("/api/feedback",
                       json={"message_id": 999999, "vote": 1}).status_code == 404
    assert client.post("/api/feedback",
                       json={"message_id": mid, "vote": 0}).status_code == 422


def test_feedback_rate_limited(tmp_path):
    client, _ = _client(tmp_path, burst=2)
    resp = client.post("/api/chat", json={"message": "q"})  # spends 1 token
    mid = next(e for e in read_events(resp) if e["type"] == "saved")["message_id"]
    assert client.post("/api/feedback",
                       json={"message_id": mid, "vote": 1}).status_code == 200
    assert client.post("/api/feedback",
                       json={"message_id": mid, "vote": 1}).status_code == 429


def test_security_headers_present(tmp_path):
    client, _ = _client(tmp_path)
    resp = client.get("/")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "no-referrer"
    csp = resp.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp


def test_ui_renders_sources_without_innerhtml_interpolation():
    html = open("web/index.html").read()
    render = html.split("function renderSources")[1].split("function ")[0]
    assert "innerHTML" not in render          # XSS fix: DOM nodes only
    assert "textContent" in render
    assert "u.protocol" in render             # only http(s) URLs become links


def test_chat_histories_lru_cap():
    h = ChatHistories(maxsize=3)
    for cid in range(5):
        h.set_history(cid, [{"role": "user", "content": str(cid)}])
    assert len(h) == 3
    assert 0 not in h and 1 not in h and 4 in h
    # touching an old chat keeps it alive
    h.get_history(2)
    h.set_history(5, [])
    assert 2 in h and 3 not in h


def test_bot_question_truncated(tmp_path):
    from app.whatsapp import WhatsAppBot
    from tests.test_phase8 import FakeWAClient, _payload
    from tests.test_rag import FakeLLM

    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    llm = FakeLLM(["ok [1]."])
    bot = WhatsAppBot(LegalAidRAG(Store(tmp_path / "index.db"), HashEmbedder(),
                                  llm=llm), FakeWAClient())
    bot.handle_payload(_payload({"from": "1", "type": "text",
                                 "text": {"body": "murder " * 3000}}))
    sent_question = llm.requests[0]["messages"][-1]["content"]
    assert "QUESTION:" in sent_question
    q = sent_question.split("QUESTION:")[1]
    assert len(q) < 4200  # capped near MAX_QUESTION_CHARS, not ~21000
