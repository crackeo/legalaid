import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from rag.answer import DISCLAIMER, LegalAidRAG, build_context, verify_citations
from rag.embed import HashEmbedder
from rag.retrieve import retrieve
from rag.store import Store, index_corpus

CORPUS = [
    {"doc_title": "Penal Code of Bhutan 2004", "section_number": "92",
     "section_heading": "CHAPTER 3 — OFFENCES AGAINST THE PERSON — Murder",
     "text": "A defendant shall be guilty of the offence of murder if the "
             "defendant commits premeditated homicide.",
     "page": 20, "source_url": "https://oag.gov.bt/pc.pdf", "doc_type": "act"},
    {"doc_title": "Penal Code of Bhutan 2004", "section_number": "93",
     "section_heading": "CHAPTER 3 — Murder",
     "text": "The offence of murder shall be a felony of the first degree.",
     "page": 20, "source_url": "https://oag.gov.bt/pc.pdf", "doc_type": "act"},
    {"doc_title": "Constitution of the Kingdom of Bhutan", "section_number": "Article 7",
     "section_heading": "Fundamental Rights",
     "text": "All persons shall have the right to life, liberty and security "
             "of person and shall not be deprived of such rights.",
     "page": 14, "source_url": "https://oag.gov.bt/const.pdf", "doc_type": "constitution"},
    {"doc_title": "Labour and Employment Act of Bhutan 2007", "section_number": "170",
     "section_heading": "Working hours",
     "text": "An employee shall not be required to work more than eight hours "
             "a day or forty-eight hours a week.",
     "page": 60, "source_url": "https://oag.gov.bt/labour.pdf", "doc_type": "act"},
]


@pytest.fixture()
def store(tmp_path):
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    db = tmp_path / "index.db"
    n = index_corpus(corpus_file, db, HashEmbedder())
    assert n == len(CORPUS)
    return Store(db)


def test_bm25_finds_terms_of_art(store):
    hits = store.search_bm25("felony first degree murder")
    assert hits and hits[0].section_number == "93"


def test_hybrid_retrieval_ranks_relevant_law_first(store):
    hits = retrieve(store, HashEmbedder(), "what is the punishment for murder?", k=2)
    assert {h.section_number for h in hits} <= {"92", "93"}

    hits = retrieve(store, HashEmbedder(), "maximum working hours per week", k=1)
    assert hits[0].doc_title.startswith("Labour")


def test_build_context_numbers_sources(store):
    hits = retrieve(store, HashEmbedder(), "murder", k=2)
    ctx = build_context(hits)
    assert ctx.startswith("PROVISIONS:")
    assert "[1] Penal Code of Bhutan 2004 — Section" in ctx


def test_verify_citations():
    ok, cited = verify_citations("Murder is a felony [1][2].", 3)
    assert ok and cited == [1, 2]
    ok, _ = verify_citations("Murder is a felony [7].", 3)      # invented source
    assert not ok
    ok, _ = verify_citations("Murder is bad.", 3)               # no citation
    assert not ok
    ok, _ = verify_citations("I don't find this in the laws available to me.", 3)
    assert ok                                                    # refusal needs none


class FakeLLM:
    """Stands in for anthropic.Anthropic(); returns queued responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = SimpleNamespace(stream=self._stream)

    @contextmanager
    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        text = self._responses.pop(0)
        final = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=text)],
        )
        yield SimpleNamespace(get_final_message=lambda: final)


def test_ask_returns_verified_cited_answer(store):
    llm = FakeLLM(["Under the Penal Code of Bhutan 2004, Section 92, murder is "
                   "premeditated homicide [1], a felony of the first degree [2]."])
    rag = LegalAidRAG(store, HashEmbedder(), llm=llm)
    ans = rag.ask("What is the punishment for murder?")
    assert ans.verified and ans.cited == [1, 2]
    assert ans.text.endswith(DISCLAIMER)
    assert len(ans.sources) >= 2
    # prompt hygiene: system rules + provisions actually sent
    req = llm.requests[0]
    assert "ONLY the numbered legal provisions" in req["system"][0]["text"]
    assert "PROVISIONS:" in req["messages"][-1]["content"]


def test_ask_retries_then_refuses_on_bad_citations(store):
    llm = FakeLLM(["Murder carries the death penalty [9].",   # invented citation
                   "Trust me, it's definitely life imprisonment."])  # no citation
    rag = LegalAidRAG(store, HashEmbedder(), llm=llm)
    ans = rag.ask("What is the punishment for murder?")
    assert not ans.verified
    assert "couldn't produce a reliably cited answer" in ans.text
    assert len(llm.requests) == 2  # one retry, then safe refusal


def test_ask_recovers_on_retry(store):
    llm = FakeLLM(["It is a felony [9].",
                   "Murder is a felony of the first degree [2]."])
    rag = LegalAidRAG(store, HashEmbedder(), llm=llm)
    ans = rag.ask("What is the punishment for murder?")
    assert ans.verified and ans.cited == [2]
