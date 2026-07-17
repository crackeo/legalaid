import json
import os

import pytest

os.environ["LEGALAID_AUTOINIT"] = "0"

from app.sessions import SessionStore
from rag.answer import DISCLAIMER, DZONGKHA_NOTE, LegalAidRAG
from rag.cli import main as rag_cli
from rag.embed import HashEmbedder, VoyageEmbedder
from rag.store import Store, check_embedder, index_corpus
from tests.test_rag import CORPUS, FakeLLM


def _index(tmp_path):
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())
    return tmp_path / "index.db"


# -- fix: embedder mismatch detected loudly -----------------------------------

def test_embedder_mismatch_raises(tmp_path):
    db = _index(tmp_path)  # built with hash-v1
    voyage = VoyageEmbedder.__new__(VoyageEmbedder)  # no API key needed
    voyage.name = "voyage-3.5"
    with pytest.raises(RuntimeError, match="hash-v1.*voyage-3.5"):
        check_embedder(Store(db), voyage)
    with pytest.raises(RuntimeError):
        LegalAidRAG(Store(db), voyage, llm=FakeLLM([]))
    # matching embedder passes
    LegalAidRAG(Store(db), HashEmbedder(), llm=FakeLLM([]))


def test_legacy_index_without_meta_is_accepted(tmp_path):
    store = Store(tmp_path / "bare.db")  # no meta row written
    check_embedder(store, HashEmbedder())  # must not raise


# -- fix: re-index removes WAL sidecars ----------------------------------------

def test_index_removes_stale_wal_sidecars(tmp_path):
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    db = tmp_path / "index.db"
    for suffix in ("", "-wal", "-shm"):
        (tmp_path / f"index.db{suffix}").write_bytes(b"stale")
    rc = rag_cli(["index", "--corpus", str(corpus_file), "--db", str(db)])
    assert rc == 0
    assert Store(db).count() == len(CORPUS)
    # stale sidecars are gone (sqlite removes its own live ones on close)
    assert (tmp_path / "index.db-wal").read_bytes() != b"stale" \
        if (tmp_path / "index.db-wal").exists() else True


# -- fix: history replay strips the footer -------------------------------------

def test_history_strips_disclaimer_footer(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    sid = store.create_session()
    store.add_message(sid, "user", "what is murder?")
    store.add_message(sid, "assistant", "Murder is a felony [1]." + DISCLAIMER)
    hist = store.history(sid)
    assert hist[1]["content"] == "Murder is a felony [1]."
    # but the UI-facing messages() keeps the footer for display
    assert DISCLAIMER.strip("\n") in store.messages(sid)[1]["content"]


def test_history_strips_dzongkha_note_too(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    sid = store.create_session()
    store.add_message(sid, "assistant", "ལན། [1]" + DISCLAIMER + DZONGKHA_NOTE)
    assert store.history(sid)[0]["content"] == "ལན། [1]"


# -- fix: cross-turn citation instruction + no-hits footer consistency ---------

def test_system_prompt_warns_about_stale_markers(tmp_path):
    db = _index(tmp_path)
    llm = FakeLLM(["Fresh answer [1]."])
    rag = LegalAidRAG(Store(db), HashEmbedder(), llm=llm)
    rag.ask("and the penalty?", history=[
        {"role": "user", "content": "what is murder?"},
        {"role": "assistant", "content": "Murder ... [3]"},
    ])
    system = llm.requests[0]["system"][0]["text"]
    assert "earlier turns" in system and "re-derive every citation" in system


def test_no_hits_paths_carry_dzongkha_note(tmp_path):
    db = _index(tmp_path)
    rag = LegalAidRAG(Store(db), HashEmbedder(), llm=FakeLLM([]))
    # force empty retrieval
    rag.store.db.execute("DELETE FROM chunks")
    rag.store.db.execute("DELETE FROM chunks_fts")
    rag.store.db.execute("DELETE FROM embeddings")
    rag.store.db.commit()
    rag.store._matrix = None

    ans = rag.ask("བསད་པའི་ཉེས་ཆད་ག་ཅི་སྨོ?")
    assert DZONGKHA_NOTE in ans.text
    events = list(rag.ask_stream("བསད་པའི་ཉེས་ཆད་ག་ཅི་སྨོ?"))
    assert DZONGKHA_NOTE in events[-1]["text"]


# -- fix: feedback buttons attach to the answer bubble --------------------------

def test_feedback_attaches_to_bubble_not_last_element():
    html = open("web/index.html").read()
    assert "function renderFeedback(messageId, bubbleEl)" in html
    assert "renderFeedback(ev.message_id, bot)" in html
    assert "chat.lastElementChild.querySelector" not in html
