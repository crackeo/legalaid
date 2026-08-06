"""Voice backends: speech-to-text and text-to-speech, English first.

Pluggable so providers can be swapped (the Dzongkha track will need it):

- STT: OpenAI Whisper API (`whisper-1`) — best accented-English recognition
  available over an API. Self-hosted faster-whisper can implement the same
  one-method interface later.
- TTS: OpenAI TTS (`tts-1`). ElevenLabs or local Piper are drop-in
  replacements — anything with synthesize(text) -> mp3 bytes.

Both need OPENAI_API_KEY; without it the /api/voice endpoints return 503
and the UI hides the mic (text chat keeps working).
"""

import io
import os
import re

import httpx

OPENAI_BASE = "https://api.openai.com/v1"
MAX_AUDIO_BYTES = 15 * 1024 * 1024  # ~15 MB, plenty for a spoken question
MAX_SPEAK_CHARS = 4000


class OpenAIWhisperSTT:
    """Speech -> text via the OpenAI Whisper API."""

    def __init__(self, api_key: str | None = None, model: str = "whisper-1"):
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.model = model

    def transcribe(self, audio: bytes, mime: str = "audio/webm",
                   language: str = "en") -> str:
        ext = (mime.split("/")[-1].split(";")[0] or "webm")
        resp = httpx.post(
            f"{OPENAI_BASE}/audio/transcriptions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": (f"question.{ext}", io.BytesIO(audio), mime)},
            data={"model": self.model, "language": language},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["text"].strip()


class OpenAITTS:
    """Text -> speech (mp3) via the OpenAI TTS API."""

    def __init__(self, api_key: str | None = None, model: str = "tts-1",
                 voice: str = "alloy"):
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.model = model
        self.voice = voice

    def synthesize(self, text: str) -> bytes:
        resp = httpx.post(
            f"{OPENAI_BASE}/audio/speech",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "voice": self.voice,
                  "input": text[:MAX_SPEAK_CHARS], "response_format": "mp3"},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.content


def stt_from_env():
    return OpenAIWhisperSTT() if os.environ.get("OPENAI_API_KEY") else None


def tts_from_env():
    return OpenAITTS() if os.environ.get("OPENAI_API_KEY") else None


_MARKER_RE = re.compile(r"\s*\[\d{1,2}\]")
_SPOKEN_DISCLAIMER = (" This is legal information, not legal advice — for your "
                      "specific situation contact the Legal Aid Center.")


def speech_text(answer_text: str) -> str:
    """Turn a written answer into something worth listening to.

    Drops [n] citation markers and the markdown disclaimer footer, and
    replaces the footer with one short spoken sentence.
    """
    text = answer_text.split("\n\n---\n")[0]
    text = _MARKER_RE.sub("", text)
    text = text.replace("*", "").replace("#", "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text + _SPOKEN_DISCLAIMER
