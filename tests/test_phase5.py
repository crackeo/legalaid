import json
import os

import pytest

os.environ["LEGALAID_AUTOINIT"] = "0"

from app.sessions import SessionStore
from app.telegram import TelegramBot, WELCOME, format_reply
from rag.answer import DZONGKHA_NOTE, LegalAidRAG, is_dzongkha
from rag.cli import main as rag_cli
from rag.embed import HashEmbedder
from rag.store import Store, index_corpus
from tests.test_rag import CORPUS, FakeLLM
from tests.test_voice import FakeSTT


@pytest.fixture()
def rag(tmp_path):
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text("\n".join(json.dumps(c) for c in CORPUS))
    index_corpus(corpus_file, tmp_path / "index.db", HashEmbedder())

    def make(llm):
        return LegalAidRAG(Store(tmp_path / "index.db"), HashEmbedder(), llm=llm)
    return make


# -- Dzongkha (experimental) -------------------------------------------------

def test_is_dzongkha():
    assert is_dzongkha("བསད་པའི་ཉེས་ཆད་ག་ཅི་སྨོ?")
    assert not is_dzongkha("What is the penalty for murder?")


def test_dzongkha_question_gets_instruction_and_note(rag):
    llm = FakeLLM(["Answer in Dzongkha [1]."])
    r = rag(llm)
    ans = r.ask("བསད་པའི་ཉེས་ཆད་ག་ཅི་སྨོ?")
    assert ans.verified
    assert DZONGKHA_NOTE in ans.text
    sent = llm.requests[0]["messages"][-1]["content"]
    assert "The user asked in Dzongkha" in sent

    llm2 = FakeLLM(["English answer [1]."])
    ans2 = rag(llm2).ask("What is murder?")
    assert DZONGKHA_NOTE not in ans2.text
    assert "Dzongkha" not in llm2.requests[0]["messages"][-1]["content"]


# -- Telegram bot -------------------------------------------------------------

class FakeAPI:
    def __init__(self):
        self.calls = []

    def __call__(self, method, **params):
        self.calls.append((method, params))
        return []

    def sent(self):
        return [p["text"] for m, p in self.calls if m == "sendMessage"]


def _update(text=None, voice=None, chat_id=7, uid=1):
    msg = {"chat": {"id": chat_id}}
    if text is not None:
        msg["text"] = text
    if voice is not None:
        msg["voice"] = voice
    return {"update_id": uid, "message": msg}


def test_bot_start_and_question(rag):
    llm = FakeLLM(["Murder is a felony of the first degree [2]."])
    api = FakeAPI()
    bot = TelegramBot("tok", rag(llm), api=api)

    bot.handle_update(_update(text="/start"))
    assert api.sent() == [WELCOME]

    bot.handle_update(_update(text="What is the punishment for murder?"))
    reply = api.sent()[-1]
    assert "felony of the first degree" in reply
    assert "Sources:" in reply
    assert "Penal Code of Bhutan 2004, Section 9" in reply  # the cited section
    assert "legal information, not legal advice" in reply
    assert "*" not in reply and "---" not in reply
    # history retained for follow-ups
    assert len(bot._histories[7]) == 2


def test_bot_voice_note_transcribed(rag):
    llm = FakeLLM(["Murder is premeditated homicide [1]."])
    api = FakeAPI()
    bot = TelegramBot("tok", rag(llm), stt=FakeSTT("what is murder?"),
                      api=api, download=lambda file_id: b"ogg-bytes")
    bot.handle_update(_update(voice={"file_id": "F1"}))
    assert "premeditated homicide" in api.sent()[-1]


def test_bot_voice_without_stt_prompts_typing(rag):
    api = FakeAPI()
    bot = TelegramBot("tok", rag(FakeLLM([])), api=api)
    bot.handle_update(_update(voice={"file_id": "F1"}))
    assert "type your question" in api.sent()[-1]


def test_bot_survives_rag_errors(rag):
    class BoomLLM:
        class messages:
            @staticmethod
            def stream(**kw):
                raise RuntimeError("api down")
    api = FakeAPI()
    bot = TelegramBot("tok", rag(BoomLLM()), api=api)
    bot.handle_update(_update(text="what is murder?"))
    assert "Something went wrong" in api.sent()[-1]


# -- feedback export -----------------------------------------------------------

def test_export_feedback(tmp_path, capsys):
    store = SessionStore(tmp_path / "s.db")
    sid = store.create_session()
    store.add_message(sid, "user", "What is the penalty for arson?")
    mid = store.add_message(sid, "assistant", "Arson is ... [1]", verified=True)
    store.add_feedback(mid, -1, "wrong section cited")
    good = store.add_message(sid, "assistant", "fine answer", verified=True)
    store.add_feedback(good, 1)

    out = tmp_path / "queue.jsonl"
    rc = rag_cli(["export-feedback", "--sessions-db", str(tmp_path / "s.db"),
                  "--out", str(out)])
    assert rc == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1  # only the downvote
    assert rows[0]["question"] == "What is the penalty for arson?"
    assert rows[0]["comment"] == "wrong section cited"
