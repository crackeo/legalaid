import json
import os

import pytest

os.environ["LEGALAID_AUTOINIT"] = "0"

from fastapi.testclient import TestClient

from app.main import create_app
from app.sessions import SessionStore
from rag.answer import LegalAidRAG
from rag.embed import HashEmbedder
from rag.store import Store, index_corpus
from tests.test_app import FakeStreamingLLM, GOOD_ANSWER, read_events
from tests.test_rag import CORPUS


@pytest.fixture()
def admin_client(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGALAID_ADMIN_TOKEN", "sekrit")
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    rag = LegalAidRAG(Store(tmp_path / "index.db"), HashEmbedder(),
                      llm=FakeStreamingLLM([GOOD_ANSWER] * 5))
    sessions = SessionStore(tmp_path / "s.db")
    return TestClient(create_app(rag=rag, sessions=sessions)), sessions


AUTH = {"Authorization": "Bearer sekrit"}


def test_admin_requires_token(admin_client):
    client, _ = admin_client
    assert client.get("/api/admin/stats").status_code == 401
    assert client.get("/api/admin/stats",
                      headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/admin/stats", headers=AUTH).status_code == 200


def test_admin_disabled_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("LEGALAID_ADMIN_TOKEN", raising=False)
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    rag = LegalAidRAG(Store(tmp_path / "index.db"), HashEmbedder(),
                      llm=FakeStreamingLLM([]))
    client = TestClient(create_app(rag=rag, sessions=SessionStore(tmp_path / "s.db")))
    assert client.get("/api/admin/stats", headers=AUTH).status_code == 404
    assert client.get("/admin").status_code == 404


def test_stats_reflect_chat_activity(admin_client):
    client, _ = admin_client
    resp = client.post("/api/chat", json={"message": "What is murder?"})
    saved = next(e for e in read_events(resp) if e["type"] == "saved")
    client.post("/api/feedback", json={"message_id": saved["message_id"],
                                       "vote": -1, "comment": "check this"})

    stats = client.get("/api/admin/stats", headers=AUTH).json()
    assert stats["sessions"] == 1
    assert stats["questions"] == 1 and stats["answers"] == 1
    assert stats["verified_rate"] == 1.0
    assert stats["avg_answer_ms"] is not None and stats["avg_answer_ms"] >= 0
    assert stats["feedback_down"] == 1 and stats["feedback_up"] == 0
    assert stats["questions_per_day"][-1]["count"] == 1

    fb = client.get("/api/admin/feedback", headers=AUTH).json()["feedback"]
    assert len(fb) == 1
    assert fb[0]["question"] == "What is murder?"
    assert fb[0]["comment"] == "check this"
    assert fb[0]["vote"] == -1 and fb[0]["verified"] is True


def test_admin_page_served_when_enabled(admin_client):
    client, _ = admin_client
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert "Admin access" in resp.text


def test_duration_recorded_on_messages(admin_client):
    client, sessions = admin_client
    client.post("/api/chat", json={"message": "What is murder?"})
    row = sessions.db.execute(
        "SELECT duration_ms FROM messages WHERE role='assistant'").fetchone()
    assert row[0] is not None and row[0] >= 0


def test_migration_adds_duration_column(tmp_path):
    import sqlite3
    db_path = tmp_path / "old.db"
    db = sqlite3.connect(db_path)  # pre-Phase-7 schema, no duration_ms
    db.executescript("""
        CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at REAL);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT,
            role TEXT, content TEXT, sources TEXT, verified INTEGER,
            created_at REAL);
        CREATE TABLE feedback (id INTEGER PRIMARY KEY, message_id INTEGER,
            vote INTEGER, comment TEXT, created_at REAL);
    """)
    db.commit(); db.close()
    store = SessionStore(db_path)  # migration runs here
    sid = store.create_session()
    assert store.add_message(sid, "assistant", "x", duration_ms=42)
    assert store.stats()["avg_answer_ms"] == 42
