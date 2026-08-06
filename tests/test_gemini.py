import json
import os

import httpx
import pytest

os.environ["LEGALAID_AUTOINIT"] = "0"

from rag.answer import LegalAidRAG
from rag.embed import HashEmbedder
from rag.llm import GeminiLLM, llm_from_env
from rag.store import Store, index_corpus
from tests.test_rag import CORPUS


def gemini_transport(reply_text=None, finish="STOP", block_reason=None,
                     captured=None):
    """MockTransport standing in for the Gemini REST API."""

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(json.loads(request.content))
            captured.append(dict(request.headers))
        body = {"candidates": [{
            "content": {"parts": [{"text": reply_text or ""}]},
            "finishReason": finish,
        }]}
        if block_reason:
            body["promptFeedback"] = {"blockReason": block_reason}
            body["candidates"] = [{}]
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def test_request_shape_and_response_parsing():
    captured = []
    llm = GeminiLLM(api_key="g-key", transport=gemini_transport(
        reply_text="Murder is a felony [1].", captured=captured))
    with llm.messages.stream(
        model="claude-opus-4-8",              # Claude-specific args are ignored
        thinking={"type": "adaptive"},
        max_tokens=1234,
        system=[{"type": "text", "text": "You are a legal assistant.",
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "q1"},
                  {"role": "assistant", "content": "a1"},
                  {"role": "user", "content": "q2"}],
    ) as stream:
        tokens = list(stream.text_stream)
        final = stream.get_final_message()

    assert "".join(tokens) == "Murder is a felony [1]."
    assert final.stop_reason == "end_turn"
    assert final.content[0].text == "Murder is a felony [1]."

    body, headers = captured
    assert body["system_instruction"]["parts"][0]["text"] == "You are a legal assistant."
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["generationConfig"]["maxOutputTokens"] == 1234
    assert headers["x-goog-api-key"] == "g-key"


def test_safety_block_maps_to_refusal():
    llm = GeminiLLM(api_key="k", transport=gemini_transport(
        reply_text="", finish="SAFETY"))
    with llm.messages.stream(messages=[{"role": "user", "content": "x"}]) as s:
        assert s.get_final_message().stop_reason == "refusal"

    llm2 = GeminiLLM(api_key="k", transport=gemini_transport(
        block_reason="PROHIBITED_CONTENT"))
    with llm2.messages.stream(messages=[{"role": "user", "content": "x"}]) as s:
        assert s.get_final_message().stop_reason == "refusal"


def test_full_pipeline_with_gemini_backend(tmp_path):
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    llm = GeminiLLM(api_key="k", transport=gemini_transport(
        reply_text="Under the Penal Code of Bhutan 2004, murder is a felony "
                   "of the first degree [1][2]."))
    rag = LegalAidRAG(Store(tmp_path / "index.db"), HashEmbedder(), llm=llm)

    ans = rag.ask("What is the punishment for murder?")
    assert ans.verified and ans.cited == [1, 2]
    assert "legal information, not legal advice" in ans.text.lower()

    events = list(rag.ask_stream("What is the punishment for murder?"))
    types = [e["type"] for e in events]
    assert types[0] == "sources" and "delta" in types and types[-1] == "done"
    assert events[-1]["verified"]


def test_llm_from_env_selection(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm = llm_from_env()
    assert isinstance(llm, GeminiLLM) and llm.model == "gemini-2.5-flash"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    import anthropic
    assert isinstance(llm_from_env(), anthropic.Anthropic)  # Claude wins


def test_health_reports_gemini_backend(tmp_path):
    """A Gemini deployment must not report a Claude model name — that field
    is the ops signal for which backend is actually answering."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.sessions import SessionStore

    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    llm = GeminiLLM(api_key="k", model="gemini-flash-latest",
                    transport=gemini_transport(reply_text="x [1]."))
    client = TestClient(create_app(
        rag=LegalAidRAG(Store(tmp_path / "index.db"), HashEmbedder(), llm=llm),
        sessions=SessionStore(tmp_path / "s.db")))

    body = client.get("/api/health").json()
    assert body["backend"] == "gemini"
    assert body["model"] == "gemini-flash-latest"
    assert body["corpus_ready"] is True


def test_health_flags_empty_corpus(tmp_path):
    """Fresh deploy before the crawl: healthy, but corpus_ready False."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.sessions import SessionStore

    llm = GeminiLLM(api_key="k", transport=gemini_transport(reply_text=""))
    client = TestClient(create_app(
        rag=LegalAidRAG(Store(tmp_path / "empty.db"), HashEmbedder(), llm=llm),
        sessions=SessionStore(tmp_path / "s.db")))

    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == 0 and body["corpus_ready"] is False
