import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

os.environ["LEGALAID_AUTOINIT"] = "0"  # don't build the real app at import

from fastapi.testclient import TestClient

from app.main import create_app
from app.sessions import SessionStore
from rag.answer import LegalAidRAG
from rag.embed import HashEmbedder
from rag.store import Store, index_corpus
from tests.test_rag import CORPUS


class FakeStreamingLLM:
    """Fake anthropic client with token streaming."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.messages = SimpleNamespace(stream=self._stream)

    @contextmanager
    def _stream(self, **kwargs):
        text = self._responses.pop(0)
        tokens = [text[i:i + 12] for i in range(0, len(text), 12)]
        final = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=text)],
        )
        yield SimpleNamespace(text_stream=iter(tokens),
                              get_final_message=lambda: final)


GOOD_ANSWER = ("Under the Penal Code of Bhutan 2004, Section 92, murder is "
               "premeditated homicide [1] and a felony of the first degree [2].")


@pytest.fixture()
def client(tmp_path):
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    rag = LegalAidRAG(Store(tmp_path / "index.db"), HashEmbedder(),
                      llm=FakeStreamingLLM([GOOD_ANSWER, GOOD_ANSWER]))
    sessions = SessionStore(tmp_path / "sessions.db")
    return TestClient(create_app(rag=rag, sessions=sessions))


def read_events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.text.splitlines()
            if line.startswith("data: ")]


def test_chat_streams_and_persists(client):
    resp = client.post("/api/chat", json={"message": "What is murder?"})
    assert resp.status_code == 200
    events = read_events(resp)
    types = [e["type"] for e in events]
    assert types[0] == "session" and types[1] == "sources"
    assert "delta" in types and types[-2] == "done" and types[-1] == "saved"
    done = next(e for e in events if e["type"] == "done")
    assert done["verified"] and done["cited"] == [1, 2]
    assert "legal information, not legal advice" in done["text"].lower()

    # follow-up in the same session sees history; session endpoint returns it
    sid = events[0]["session_id"]
    resp2 = client.post("/api/chat", json={"message": "and assault?", "session_id": sid})
    assert resp2.status_code == 200
    hist = client.get(f"/api/session/{sid}").json()
    roles = [m["role"] for m in hist["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert hist["messages"][1]["sources"]


def test_feedback_roundtrip(client):
    resp = client.post("/api/chat", json={"message": "What is murder?"})
    saved = next(e for e in read_events(resp) if e["type"] == "saved")
    fb = client.post("/api/feedback",
                     json={"message_id": saved["message_id"], "vote": -1,
                           "comment": "wrong section"})
    assert fb.status_code == 200 and fb.json() == {"ok": True}


def test_unknown_session_404(client):
    assert client.post("/api/chat", json={"message": "hi", "session_id": "nope"}).status_code == 404
    assert client.get("/api/session/nope").status_code == 404


def test_validation(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 5000}).status_code == 422


def test_index_page_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Bhutan Legal Aid AI" in resp.text


def test_session_purge(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    sid = store.create_session()
    store.add_message(sid, "user", "hello")
    assert store.purge_older_than(days=-1) == 1  # everything is "older"
    assert not store.session_exists(sid)
