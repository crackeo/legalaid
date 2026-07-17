import json
import os

import httpx
import pytest

os.environ["LEGALAID_AUTOINIT"] = "0"

from rag.cli import main as rag_cli
from rag.embed import HashEmbedder
from rag.grade import Grade, grade_answer
from rag.store import Hit, index_corpus
from tests.test_rag import CORPUS, FakeLLM

SOURCES = [Hit(chunk_id=1, doc_title="Penal Code of Bhutan 2004",
               section_number="92", section_heading="Murder",
               text="A defendant shall be guilty of murder if ...",
               page=20, source_url="", score=1.0)]


# -- grader ---------------------------------------------------------------------

def test_grade_answer_parses_verdict():
    llm = FakeLLM([json.dumps({
        "grounded": True, "citations_correct": True, "complete": True,
        "clear": True, "verdict": "pass", "issues": []})])
    grade = grade_answer(llm, "What is murder?", "Murder is ... [1]", SOURCES)
    assert grade.verdict == "pass" and grade.grounded and not grade.issues
    # grader prompt contains the provisions and the answer under review
    sent = llm.requests[0]["messages"][0]["content"]
    assert "PROVISIONS:" in sent and "CHATBOT ANSWER TO GRADE" in sent
    assert "strict legal examiner" in llm.requests[0]["system"][0]["text"]


def test_grade_answer_fail_and_prose_wrapped_json():
    llm = FakeLLM(['Here is my assessment:\n{"grounded": false, '
                   '"citations_correct": false, "complete": true, "clear": true,'
                   ' "verdict": "fail", "issues": ["invented penalty"]}'])
    grade = grade_answer(llm, "q", "a [1]", SOURCES)
    assert grade.verdict == "fail"
    assert grade.issues == ["invented penalty"]


def test_grade_answer_handles_garbage():
    assert grade_answer(FakeLLM(["not json at all"]), "q", "a", SOURCES).verdict == "error"
    assert grade_answer(FakeLLM(["{broken json"]), "q", "a", SOURCES).verdict == "error"


# -- graded eval end-to-end -------------------------------------------------------

def test_eval_grade_writes_report(tmp_path, monkeypatch, capsys):
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    questions = tmp_path / "q.jsonl"
    questions.write_text(json.dumps(
        {"question": "What is the punishment for murder?", "expect_doc": "Penal Code"}) + "\n")

    responses = [
        "Murder is a felony of the first degree [1].",          # the answer
        json.dumps({"grounded": True, "citations_correct": True,  # the grade
                    "complete": True, "clear": True, "verdict": "pass",
                    "issues": []}),
    ]
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda: FakeLLM(responses))

    report = tmp_path / "report.jsonl"
    rc = rag_cli(["eval", "--questions", str(questions), "--db",
                  str(tmp_path / "index.db"), "--grade", "--report", str(report)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "answer grades: 1/1 pass" in out
    row = json.loads(report.read_text().splitlines()[0])
    assert row["retrieval_hit"] and row["grade"] == "pass" and row["verified"]


# -- crawler retries ---------------------------------------------------------------

class FlakyClient:
    """Fails n times, then succeeds."""

    def __init__(self, failures, exc=None, status=None):
        self.failures = failures
        self.exc = exc
        self.status = status
        self.calls = 0

    def get(self, url, **kw):
        self.calls += 1
        if self.calls <= self.failures:
            if self.exc:
                raise self.exc
            request = httpx.Request("GET", url)
            return httpx.Response(self.status, request=request)
        return httpx.Response(200, request=httpx.Request("GET", url),
                              content=b"ok")


def test_crawler_retries_transient_errors(monkeypatch):
    from ingest import crawler
    monkeypatch.setattr(crawler.time, "sleep", lambda s: None)

    flaky = FlakyClient(2, exc=httpx.ConnectError("boom"))
    assert crawler._get(flaky, "https://oag.gov.bt/x").status_code == 200
    assert flaky.calls == 3

    flaky500 = FlakyClient(1, status=503)
    assert crawler._get(flaky500, "https://oag.gov.bt/x").status_code == 200
    assert flaky500.calls == 2


def test_crawler_does_not_retry_4xx_and_gives_up(monkeypatch):
    from ingest import crawler
    monkeypatch.setattr(crawler.time, "sleep", lambda s: None)

    notfound = FlakyClient(99, status=404)
    with pytest.raises(httpx.HTTPStatusError):
        crawler._get(notfound, "https://oag.gov.bt/gone")
    assert notfound.calls == 1  # 404 not retried

    dead = FlakyClient(99, exc=httpx.ConnectError("down"))
    with pytest.raises(httpx.ConnectError):
        crawler._get(dead, "https://oag.gov.bt/x")
    assert dead.calls == crawler.RETRIES + 1
