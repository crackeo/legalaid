import json
import os

import pytest

os.environ["LEGALAID_AUTOINIT"] = "0"

from fastapi.testclient import TestClient

from app.main import create_app
from app.ratelimit import RateLimiter
from app.sessions import SessionStore
from rag.answer import LegalAidRAG
from rag.embed import HashEmbedder
from rag.store import Store, index_corpus
from tests.test_app import FakeStreamingLLM, GOOD_ANSWER
from tests.test_rag import CORPUS


# -- OCR fallback for scanned PDFs -------------------------------------------

def _scanned_pdf(tmp_path):
    """A PDF whose pages are images only (no text layer), like a scanned Act."""
    import fitz

    src = fitz.open()
    page = src.new_page()
    page.insert_text((72, 72),
                     "CHAPTER 1\nOffences\n1. Arson is a felony of the "
                     "third degree.", fontsize=14)
    pix = src[0].get_pixmap(dpi=200)
    src.close()

    scanned = fitz.open()
    page = scanned.new_page()
    page.insert_image(page.rect, pixmap=pix)
    path = tmp_path / "scanned.pdf"
    scanned.save(path)
    scanned.close()
    return path


def test_ocr_extracts_scanned_pages(tmp_path):
    from ingest.extract import OCR_AVAILABLE, extract_pages

    path = _scanned_pdf(tmp_path)
    assert extract_pages(path, ocr=False)[0].strip() == ""   # truly scanned
    if not OCR_AVAILABLE:
        pytest.skip("tesseract not installed")
    text = extract_pages(path, ocr=True)[0]
    assert "Arson" in text and "felony" in text


def test_scanned_pdf_flows_through_chunker(tmp_path):
    from ingest.chunk import chunk_document
    from ingest.extract import OCR_AVAILABLE, pdf_to_clean_pages

    if not OCR_AVAILABLE:
        pytest.skip("tesseract not installed")
    pages = pdf_to_clean_pages(_scanned_pdf(tmp_path))
    chunks = chunk_document(pages, doc_title="Scanned Act (fixture)")
    assert any(c.section_number == "1" and "Arson" in c.text for c in chunks)


# -- rate limiting -------------------------------------------------------------

def test_rate_limiter_bucket():
    rl = RateLimiter(per_minute=60, burst=3)
    assert [rl.allow("a") for _ in range(4)] == [True, True, True, False]
    assert rl.allow("b")  # other clients unaffected


@pytest.fixture()
def limited_client(tmp_path):
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    rag = LegalAidRAG(Store(tmp_path / "index.db"), HashEmbedder(),
                      llm=FakeStreamingLLM([GOOD_ANSWER] * 10))
    return TestClient(create_app(
        rag=rag, sessions=SessionStore(tmp_path / "s.db"),
        limiter=RateLimiter(per_minute=60, burst=2)))


def test_chat_rate_limited(limited_client):
    ok1 = limited_client.post("/api/chat", json={"message": "q1"})
    ok2 = limited_client.post("/api/chat", json={"message": "q2"})
    blocked = limited_client.post("/api/chat", json={"message": "q3"})
    assert ok1.status_code == ok2.status_code == 200
    assert blocked.status_code == 429
    assert "wait a minute" in blocked.json()["detail"]


def test_health_not_rate_limited(limited_client):
    for _ in range(5):
        resp = limited_client.get("/api/health")
        assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks"] == len(CORPUS)
    assert body["corpus_ready"] is True
    assert body["embedder"] == "hash-v1"
    assert body["backend"] == "claude"          # FakeStreamingLLM stands in for Claude
    assert body["voice"] == {"stt": False, "tts": False}
