import json
import os

import pytest

os.environ["LEGALAID_AUTOINIT"] = "0"

from fastapi.testclient import TestClient

from app.main import create_app
from app.sessions import SessionStore
from app.voice import speech_text
from rag.answer import DISCLAIMER, LegalAidRAG
from rag.embed import HashEmbedder
from rag.store import Store, index_corpus
from tests.test_app import FakeStreamingLLM, GOOD_ANSWER
from tests.test_rag import CORPUS


class FakeSTT:
    def __init__(self, text="What is the punishment for murder?"):
        self.text = text
        self.calls = []

    def transcribe(self, audio: bytes, mime: str = "audio/webm", language="en") -> str:
        self.calls.append((len(audio), mime))
        return self.text


class FakeTTS:
    def __init__(self):
        self.spoken = []

    def synthesize(self, text: str) -> bytes:
        self.spoken.append(text)
        return b"ID3fake-mp3-bytes"


@pytest.fixture()
def voice_client(tmp_path):
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    rag = LegalAidRAG(Store(tmp_path / "index.db"), HashEmbedder(),
                      llm=FakeStreamingLLM([GOOD_ANSWER]))
    stt, tts = FakeSTT(), FakeTTS()
    client = TestClient(create_app(
        rag=rag, sessions=SessionStore(tmp_path / "s.db"), stt=stt, tts=tts))
    return client, stt, tts


def test_voice_config(voice_client):
    client, _, _ = voice_client
    assert client.get("/api/voice/config").json() == {"stt": True, "tts": True}


def test_transcribe(voice_client):
    client, stt, _ = voice_client
    resp = client.post("/api/voice/transcribe",
                       files={"audio": ("q.webm", b"\x1aE\xdf\xa3fake-webm", "audio/webm")})
    assert resp.status_code == 200
    assert resp.json() == {"text": "What is the punishment for murder?"}
    assert stt.calls[0][1] == "audio/webm"


def test_transcribe_rejects_empty(voice_client):
    client, _, _ = voice_client
    resp = client.post("/api/voice/transcribe",
                       files={"audio": ("q.webm", b"", "audio/webm")})
    assert resp.status_code == 422


def test_speak_returns_mp3_of_cleaned_text(voice_client):
    client, _, tts = voice_client
    answer = ("Murder is a felony of the first degree [1][2]." + DISCLAIMER)
    resp = client.post("/api/voice/speak", json={"text": answer})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content.startswith(b"ID3")
    spoken = tts.spoken[0]
    assert "[1]" not in spoken and "---" not in spoken and "*" not in spoken
    assert "Murder is a felony of the first degree." in spoken
    assert "legal information, not legal advice" in spoken


def test_voice_unconfigured_returns_503(tmp_path):
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    rag = LegalAidRAG(Store(tmp_path / "index.db"), HashEmbedder(),
                      llm=FakeStreamingLLM([]))

    class _Unset:  # sentinel meaning "not configured" without env fallback
        pass

    app = create_app(rag=rag, sessions=SessionStore(tmp_path / "s.db"))
    client = TestClient(app)
    if os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY present in environment")
    assert client.get("/api/voice/config").json() == {"stt": False, "tts": False}
    resp = client.post("/api/voice/transcribe",
                       files={"audio": ("q.webm", b"xx", "audio/webm")})
    assert resp.status_code == 503
    assert client.post("/api/voice/speak", json={"text": "hi"}).status_code == 503


def test_speech_text_cleaning():
    text = ("Under the Penal Code, Section 92 [1], murder is premeditated "
            "homicide [2][3].\n\n---\n*This is legal information, not legal "
            "advice. Contact BNLI.*")
    spoken = speech_text(text)
    assert "[" not in spoken and "---" not in spoken and "*" not in spoken
    assert spoken.startswith("Under the Penal Code, Section 92, murder")
    assert spoken.endswith("contact the Legal Aid Center.")
