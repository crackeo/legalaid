"""Telegram bot channel — the cheapest way to reach people without a laptop.

Long-polling (no public webhook URL needed, works from any server):

    export TELEGRAM_BOT_TOKEN=123:abc   # from @BotFather
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m app.telegram

Text messages run through the same cited-answer pipeline as the web app.
Voice notes are transcribed first when OPENAI_API_KEY is set. Conversation
history is kept in memory per chat (restarting the bot clears it — fine for
follow-up questions, and nothing sensitive is written to disk).
"""

import os
import re
import time

import httpx

from rag.answer import Answer, LegalAidRAG

WELCOME = (
    "Kuzu zangpo la! I answer questions about the laws of Bhutan — the "
    "Constitution, Acts, the Penal Code, rules and regulations — always "
    "citing the exact provision.\n\n"
    "Ask in plain language, e.g. \"What is the penalty for defamation?\" "
    "You can also send a voice note.\n\n"
    "I give legal information, not legal advice. For your specific case, "
    "contact the Bhutan National Legal Institute's Legal Aid Center."
)

MAX_HISTORY_TURNS = 6
_MD_FOOTER_RE = re.compile(r"\n*---\n\*(.*?)\*\s*$", re.S)


def format_reply(answer: Answer) -> str:
    """Plain-text reply: answer, cited sources, one-line disclaimer."""
    text = _MD_FOOTER_RE.sub(lambda m: "\n\n" + m.group(1), answer.text)
    text = text.replace("*", "")
    cited = [h for i, h in enumerate(answer.sources, start=1)
             if not answer.cited or i in answer.cited]
    if cited:
        lines = [f"• {h.doc_title}, Section {h.section_number} (p.{h.page})"
                 for h in cited]
        text += "\n\nSources:\n" + "\n".join(dict.fromkeys(lines))
    return text[:4096]  # Telegram message limit


class TelegramBot:
    def __init__(self, token: str, rag: LegalAidRAG, stt=None,
                 api=None, download=None):
        self.token = token
        self.rag = rag
        self.stt = stt
        self._api = api or self._http_api
        self._download = download or self._http_download
        self._histories: dict[int, list[dict]] = {}

    # -- Telegram HTTP layer (injected in tests) --------------------------

    def _http_api(self, method: str, **params):
        resp = httpx.post(f"https://api.telegram.org/bot{self.token}/{method}",
                          json=params, timeout=90)
        resp.raise_for_status()
        return resp.json()["result"]

    def _http_download(self, file_id: str) -> bytes:
        info = self._api("getFile", file_id=file_id)
        resp = httpx.get(
            f"https://api.telegram.org/file/bot{self.token}/{info['file_path']}",
            timeout=120)
        resp.raise_for_status()
        return resp.content

    # -- update handling ---------------------------------------------------

    def _question_from(self, message: dict) -> str | None:
        if "text" in message:
            return message["text"].strip()
        if "voice" in message and self.stt is not None:
            audio = self._download(message["voice"]["file_id"])
            return self.stt.transcribe(audio, mime="audio/ogg")
        if "voice" in message:
            self._send(message["chat"]["id"],
                       "Voice notes aren't enabled on this bot yet — "
                       "please type your question.")
        return None

    def _send(self, chat_id: int, text: str) -> None:
        self._api("sendMessage", chat_id=chat_id, text=text)

    def handle_update(self, update: dict) -> None:
        message = update.get("message")
        if not message or "chat" not in message:
            return
        chat_id = message["chat"]["id"]
        try:
            question = self._question_from(message)
        except Exception:
            self._send(chat_id, "I couldn't process that voice note — "
                                "please try again or type your question.")
            return
        if not question:
            return
        if question.startswith("/start") or question.startswith("/help"):
            self._send(chat_id, WELCOME)
            return

        self._api("sendChatAction", chat_id=chat_id, action="typing")
        history = self._histories.get(chat_id, [])
        try:
            answer = self.rag.ask(question, history=history)
        except Exception:
            self._send(chat_id, "Something went wrong answering that — "
                                "please try again in a moment.")
            return
        self._send(chat_id, format_reply(answer))
        history = history + [{"role": "user", "content": question},
                             {"role": "assistant", "content": answer.text}]
        self._histories[chat_id] = history[-MAX_HISTORY_TURNS * 2:]

    # -- polling loop --------------------------------------------------------

    def run_polling(self) -> None:
        print("bot: polling for updates (Ctrl-C to stop)")
        offset = None
        while True:
            try:
                updates = self._api("getUpdates", timeout=50, offset=offset,
                                    allowed_updates=["message"])
            except (httpx.HTTPError, KeyError):
                time.sleep(5)
                continue
            for update in updates:
                offset = update["update_id"] + 1
                try:
                    self.handle_update(update)
                except Exception as exc:  # never let one chat kill the loop
                    print(f"bot: error handling update: {exc!r}")


def main() -> None:
    from rag.embed import embedder_from_env
    from rag.store import Store

    from .voice import stt_from_env

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    index_db = os.environ.get("LEGALAID_INDEX_DB", "corpus/index.db")
    rag = LegalAidRAG(Store(index_db), embedder_from_env())
    TelegramBot(token, rag, stt=stt_from_env()).run_polling()


if __name__ == "__main__":
    main()
