"""LLM backends.

Claude (Anthropic) is the primary, tested backend. GeminiLLM is an adapter
for users who have a Google AI Studio key instead: it speaks the same
narrow interface the pipeline uses —

    with llm.messages.stream(system=..., messages=..., ...) as stream:
        for token in stream.text_stream: ...
        response = stream.get_final_message()   # .stop_reason, .content

so retrieval, citation verification, grading and every channel work
unchanged. Selection is automatic from the environment (Anthropic key wins
when both are set). Note: prompts here are tuned against Claude; the
citation verifier is the safety net either way, but expect to re-run
`rag.cli eval --grade` when switching backends.
"""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import httpx

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
# Floating alias -> always the current model; concrete names like
# "gemini-2.5-flash" get retired for new accounts. Flash-tier because
# free-tier keys have little/no pro-model quota (pro 429s immediately).
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

# Gemini finish reasons that mean "declined", mapped to Claude's "refusal".
_GEMINI_REFUSALS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}


class GeminiLLM:
    """Google Gemini via the REST generateContent API.

    The pipeline's streaming is emulated client-side (the full response is
    fetched, then chunked into text_stream) — answers appear in one burst
    rather than token-by-token, but every feature works.
    """

    def __init__(self, api_key: str | None = None,
                 model: str = DEFAULT_GEMINI_MODEL, transport=None):
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.model = model
        self._client = httpx.Client(transport=transport, timeout=300)
        self.messages = SimpleNamespace(stream=self._stream)

    @contextmanager
    def _stream(self, *, system=None, messages=(), max_tokens: int = 8192,
                **_claude_specific_ignored):
        if isinstance(system, str):
            system_text = system
        else:  # Claude-style list of text blocks
            system_text = "\n".join(b.get("text", "") for b in (system or []))

        contents = [{
            "role": "model" if m["role"] == "assistant" else "user",
            "parts": [{"text": m["content"] if isinstance(m["content"], str)
                       else str(m["content"])}],
        } for m in messages]

        body = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if system_text:
            body["system_instruction"] = {"parts": [{"text": system_text}]}

        resp = self._client.post(
            f"{GEMINI_BASE}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates") or [{}]
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        finish = candidates[0].get("finishReason", "STOP")
        blocked = bool(data.get("promptFeedback", {}).get("blockReason"))
        stop_reason = ("refusal" if finish in _GEMINI_REFUSALS
                       or (blocked and not text) else "end_turn")

        final = SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
        )

        def chunks(step: int = 80):
            for i in range(0, len(text), step):
                yield text[i:i + step]

        yield SimpleNamespace(text_stream=chunks(),
                              get_final_message=lambda: final)


def llm_from_env():
    """Pick the LLM backend from the environment.

    ANTHROPIC_API_KEY -> Claude (preferred). Otherwise GEMINI_API_KEY (or
    GOOGLE_API_KEY) -> Gemini. With neither set, fall back to the Anthropic
    client, which also resolves `ant auth login` profiles and auth tokens.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return anthropic.Anthropic()
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        return GeminiLLM(api_key=gemini_key,
                         model=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    import anthropic
    return anthropic.Anthropic()
