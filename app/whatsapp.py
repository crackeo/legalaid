"""WhatsApp channel via the Meta Cloud API — webhook-based.

Setup (all on the Meta developer portal, developers.facebook.com):
1. Create a Business app, add the WhatsApp product, get a permanent token
   and the phone number ID.
2. Configure the webhook to point at  https://<your-host>/api/whatsapp/webhook
   with your chosen verify token, subscribed to `messages`.
3. Run the web app with:
       WHATSAPP_TOKEN=EAAG...            # Graph API token
       WHATSAPP_PHONE_NUMBER_ID=1234...  # sender phone number id
       WHATSAPP_VERIFY_TOKEN=anything    # must match the portal webhook config
       WHATSAPP_APP_SECRET=...           # optional but recommended: enables
                                         # X-Hub-Signature-256 verification

Same cited-answer pipeline as web/Telegram; voice notes are transcribed
when OPENAI_API_KEY is set. Answering happens in a background task because
Meta requires the webhook to acknowledge within seconds.
"""

import hashlib
import hmac
import os
from collections import OrderedDict

import httpx

from rag.answer import LegalAidRAG

from .telegram import (MAX_QUESTION_CHARS, WELCOME, ChatHistories,
                       format_reply)

GRAPH = "https://graph.facebook.com/v21.0"
_SEEN_MAX = 2048  # Meta re-delivers on slow/failed acks; dedupe by message id


class WhatsAppClient:
    """Thin Graph API wrapper (replaced by a fake in tests)."""

    def __init__(self, token: str, phone_number_id: str):
        self.token = token
        self.phone_number_id = phone_number_id

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def send_text(self, to: str, text: str) -> None:
        resp = httpx.post(
            f"{GRAPH}/{self.phone_number_id}/messages",
            headers=self._headers(),
            json={"messaging_product": "whatsapp", "to": to,
                  "type": "text", "text": {"body": text[:4096]}},
            timeout=60,
        )
        resp.raise_for_status()

    def download_media(self, media_id: str) -> tuple[bytes, str]:
        meta = httpx.get(f"{GRAPH}/{media_id}", headers=self._headers(),
                         timeout=60)
        meta.raise_for_status()
        info = meta.json()
        blob = httpx.get(info["url"], headers=self._headers(), timeout=120)
        blob.raise_for_status()
        return blob.content, info.get("mime_type", "audio/ogg")


def verify_signature(app_secret: str, raw_body: bytes, header: str | None) -> bool:
    """Check Meta's X-Hub-Signature-256 header (sha256=<hmac hexdigest>)."""
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(header[7:], expected)


class WhatsAppBot:
    def __init__(self, rag: LegalAidRAG, client: WhatsAppClient, stt=None):
        self.rag = rag
        self.client = client
        self.stt = stt
        self._histories = ChatHistories()
        self._seen: OrderedDict[str, None] = OrderedDict()

    def _dedupe(self, message_id: str) -> bool:
        """True if this message was already handled."""
        if message_id in self._seen:
            return True
        self._seen[message_id] = None
        while len(self._seen) > _SEEN_MAX:
            self._seen.popitem(last=False)
        return False

    def _question_from(self, message: dict) -> str | None:
        kind = message.get("type")
        if kind == "text":
            return message["text"]["body"].strip()
        if kind == "audio":  # voice notes arrive as type=audio with voice=true
            if self.stt is None:
                self.client.send_text(
                    message["from"],
                    "Voice notes aren't enabled on this bot yet — "
                    "please type your question.")
                return None
            audio, mime = self.client.download_media(message["audio"]["id"])
            return self.stt.transcribe(audio, mime=mime)
        return None  # stickers, images, reactions, ... — ignore

    def handle_message(self, message: dict) -> None:
        sender = message.get("from")
        if not sender or self._dedupe(message.get("id", "")):
            return
        try:
            question = self._question_from(message)
        except Exception:
            self.client.send_text(sender, "I couldn't process that voice note — "
                                          "please try again or type your question.")
            return
        if not question:
            return
        question = question[:MAX_QUESTION_CHARS]
        if question.lower() in ("hi", "hello", "start", "help", "kuzu zangpo"):
            self.client.send_text(sender, WELCOME)
            return
        history = self._histories.get_history(sender)
        try:
            answer = self.rag.ask(question, history=history)
        except Exception:
            self.client.send_text(sender, "Something went wrong answering that — "
                                          "please try again in a moment.")
            return
        self.client.send_text(sender, format_reply(answer))
        self._histories.set_history(
            sender, history + [{"role": "user", "content": question},
                               {"role": "assistant", "content": answer.text}])

    def handle_payload(self, payload: dict) -> int:
        """Process a full webhook payload; returns messages handled."""
        handled = 0
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                # status callbacks (sent/delivered/read) carry no messages
                for message in change.get("value", {}).get("messages", []):
                    self.handle_message(message)
                    handled += 1
        return handled


def bot_from_env(rag: LegalAidRAG, stt=None) -> WhatsAppBot | None:
    token = os.environ.get("WHATSAPP_TOKEN")
    phone_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
    if not (token and phone_id):
        return None
    return WhatsAppBot(rag, WhatsAppClient(token, phone_id), stt=stt)
