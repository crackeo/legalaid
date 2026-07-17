"""The answering layer: retrieved provisions + Claude, with citation enforcement.

Every answer must cite the retrieved sections it relies on using [n] markers.
After generation we verify programmatically that every cited marker refers to
a section that was actually provided; an answer with invented or missing
citations is replaced by a safe refusal. Legal information, not legal advice.
"""

import re
from dataclasses import dataclass, field

from .retrieve import retrieve
from .store import Hit, Store

MODEL = "claude-opus-4-8"

DISCLAIMER = (
    "\n\n---\n*This is legal information, not legal advice. For advice on "
    "your specific situation, contact the Bhutan National Legal Institute's "
    "Legal Aid Center or a licensed Jabmi (legal counsel).*"
)

SYSTEM_PROMPT = """\
You are a legal information assistant for the Kingdom of Bhutan. You answer
questions using ONLY the numbered legal provisions supplied in each request —
never from your general knowledge or memory of other legal systems.

Rules, in priority order:
1. Ground every legal claim in the supplied provisions and cite them with
   bracketed markers like [1] or [2][3] immediately after the claim. Also
   name the law in prose (e.g. "Under the Penal Code of Bhutan 2004,
   Section 92 [1] ..."). Never cite a number that was not supplied.
2. If the supplied provisions do not answer the question, say plainly:
   "I don't find this in the laws available to me" and suggest which Act
   might cover it, clearly labelled as a suggestion. Do not guess.
3. Never invent section numbers, penalties, or legal thresholds.
4. Do not predict the outcome of any specific case or advise anyone on how
   to commit, conceal, or evade liability for an offence. Explaining what
   the law says — including penalties — is always fine.
5. If the question suggests immediate danger (violence, abuse), include the
   relevant law AND advise contacting the Royal Bhutan Police, and for
   domestic violence also RENEW and the NCWC helpline.
6. Answer in clear, plain language a non-lawyer can follow. Quote the exact
   statutory words for the load-bearing part. Be concise.
"""

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")

# Dzongkha is written in Tibetan script (U+0F00–U+0FFF).
_DZONGKHA_RE = re.compile(r"[ༀ-࿿]")

DZONGKHA_NOTE = (
    "\n\n*Dzongkha support is experimental (རྫོང་ཁའི་རྒྱབ་སྐྱོར་འདི་ཚོད་ལྟའི་གནས་"
    "རིམ་ལུ་ཡོད) — the underlying laws are indexed in English and the "
    "translation has not been reviewed by a native speaker. Verify against "
    "the cited English provisions.*"
)

DZONGKHA_INSTRUCTION = (
    "\n\nThe user asked in Dzongkha. Reply in Dzongkha, but keep Act titles "
    "and section numbers in English and follow the same [n] citation rules."
)


def is_dzongkha(text: str) -> bool:
    return bool(_DZONGKHA_RE.search(text))


@dataclass
class Answer:
    text: str
    sources: list[Hit]
    cited: list[int] = field(default_factory=list)   # 1-based indices into sources
    verified: bool = False


def build_context(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        header = f"[{i}] {h.doc_title} — Section {h.section_number}"
        if h.section_heading:
            header += f" ({h.section_heading})"
        parts.append(f"{header}, page {h.page}\n{h.text}")
    return "PROVISIONS:\n\n" + "\n\n".join(parts)


def verify_citations(text: str, n_sources: int) -> tuple[bool, list[int]]:
    """Every cited [n] must exist; a substantive answer must cite something."""
    cited = sorted({int(m) for m in _CITATION_RE.findall(text)})
    if any(c < 1 or c > n_sources for c in cited):
        return False, cited
    is_refusal = "don't find this in the laws" in text.lower()
    if not cited and not is_refusal:
        return False, cited
    return True, cited


class LegalAidRAG:
    """retrieve -> prompt -> Claude -> verify -> answer with disclaimer."""

    def __init__(self, store: Store, embedder, llm=None, model: str = MODEL):
        self.store = store
        self.embedder = embedder
        self.model = model
        if llm is None:
            import anthropic
            llm = anthropic.Anthropic()
        self.llm = llm  # anything with .messages.stream(...) (fake in tests)

    def _call_llm(self, question: str, context: str,
                  history: list[dict] | None) -> str:
        messages = list(history or [])
        messages.append({
            "role": "user",
            "content": f"{context}\n\nQUESTION: {question}"
                       + (DZONGKHA_INSTRUCTION if is_dzongkha(question) else ""),
        })
        with self.llm.messages.stream(
            model=self.model,
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "adaptive"},
            messages=messages,
        ) as stream:
            response = stream.get_final_message()
        if response.stop_reason == "refusal":
            return ("I can't help with that request. For legal assistance, "
                    "contact the Bhutan National Legal Institute's Legal Aid Center.")
        return "".join(b.text for b in response.content if b.type == "text")

    def ask_stream(self, question: str, history: list[dict] | None = None,
                   k: int = 6):
        """Generator of events for a streaming UI.

        Yields dicts: {"type": "sources", ...} once, then {"type": "delta",
        "text": ...} as tokens arrive, and finally {"type": "done", ...}
        with the verified full answer. Citation verification runs on the
        complete text; if it fails (after one non-streamed retry) a
        {"type": "replace", ...} event tells the client to swap the shown
        text for the safe version — so nothing unverified is ever final.
        """
        hits = retrieve(self.store, self.embedder, question, k=k)
        yield {"type": "sources", "sources": [
            {"n": i, "doc_title": h.doc_title, "section": h.section_number,
             "page": h.page, "source_url": h.source_url}
            for i, h in enumerate(hits, start=1)
        ]}
        if not hits:
            text = "I don't find this in the laws available to me." + DISCLAIMER
            yield {"type": "replace", "text": text}
            yield {"type": "done", "text": text, "verified": True, "cited": []}
            return

        footer = DISCLAIMER + (DZONGKHA_NOTE if is_dzongkha(question) else "")
        context = build_context(hits)
        messages = list(history or [])
        messages.append({"role": "user",
                         "content": f"{context}\n\nQUESTION: {question}"
                                    + (DZONGKHA_INSTRUCTION if is_dzongkha(question) else "")})
        parts = []
        with self.llm.messages.stream(
            model=self.model,
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "adaptive"},
            messages=messages,
        ) as stream:
            for token in stream.text_stream:
                parts.append(token)
                yield {"type": "delta", "text": token}
            response = stream.get_final_message()

        if response.stop_reason == "refusal":
            text = ("I can't help with that request. For legal assistance, "
                    "contact the Bhutan National Legal Institute's Legal Aid "
                    "Center.") + DISCLAIMER
            yield {"type": "replace", "text": text}
            yield {"type": "done", "text": text, "verified": True, "cited": []}
            return

        text = "".join(parts)
        ok, cited = verify_citations(text, len(hits))
        if not ok:
            # Retry unstreamed; replace what the client has shown so far.
            text = self._call_llm(
                question + "\n\n(Your previous draft cited provisions that were "
                "not supplied or made claims without citations. Answer again, "
                "citing only the numbered provisions above.)",
                context, history,
            )
            ok, cited = verify_citations(text, len(hits))
            if not ok:
                text = ("I couldn't produce a reliably cited answer to this "
                        "question. Please rephrase, or consult the Bhutan "
                        "National Legal Institute's Legal Aid Center.")
            yield {"type": "replace", "text": text + footer}
        yield {"type": "done", "text": text + footer,
               "verified": ok, "cited": cited}

    def ask(self, question: str, history: list[dict] | None = None,
            k: int = 6) -> Answer:
        footer = DISCLAIMER + (DZONGKHA_NOTE if is_dzongkha(question) else "")
        hits = retrieve(self.store, self.embedder, question, k=k)
        if not hits:
            return Answer(
                text="I don't find this in the laws available to me." + DISCLAIMER,
                sources=[], verified=True,
            )
        text = self._call_llm(question, build_context(hits), history)
        ok, cited = verify_citations(text, len(hits))
        if not ok:
            # One retry with an explicit correction, then refuse rather than
            # ship an answer whose citations don't check out.
            text = self._call_llm(
                question + "\n\n(Your previous draft cited provisions that were "
                "not supplied or made claims without citations. Answer again, "
                "citing only the numbered provisions above.)",
                build_context(hits), history,
            )
            ok, cited = verify_citations(text, len(hits))
            if not ok:
                return Answer(
                    text="I couldn't produce a reliably cited answer to this "
                         "question. Please rephrase, or consult the Bhutan "
                         "National Legal Institute's Legal Aid Center." + DISCLAIMER,
                    sources=hits, verified=False,
                )
        return Answer(text=text + footer, sources=hits,
                      cited=cited, verified=True)
