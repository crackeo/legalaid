"""LLM-graded answer evaluation.

Retrieval hit-rate (rag.cli eval) says whether the right law was found;
this grades whether the final *answer* is any good. A separate Claude
call — with fresh context, acting as a strict examiner — scores each
answer against the provisions it cited. Judging with a fresh context
catches errors the answering model can't see in its own output.

Grades are a triage signal for the human legal review queue, not a
substitute for it: answers graded 'fail' get reviewed first.
"""

import json
import re
from dataclasses import dataclass

from .answer import MODEL, build_context
from .store import Hit

GRADER_SYSTEM = """\
You are a strict legal examiner reviewing answers produced by a Bhutan
legal-information chatbot. You are given the legal provisions the chatbot
retrieved and the answer it gave. Judge ONLY against those provisions.

Return a single JSON object, nothing else:
{
  "grounded": true|false,      // every legal claim is supported by the cited provisions
  "citations_correct": true|false, // each [n] marker points at a provision that says what the answer claims
  "complete": true|false,      // the question is actually answered (or honestly declined)
  "clear": true|false,         // a non-lawyer could follow it
  "verdict": "pass"|"fail",    // fail if grounded or citations_correct is false
  "issues": ["short description of each problem found", ...]
}
Be adversarial: hunt for invented thresholds, penalties or section numbers,
claims stretching beyond the quoted text, and misattributed citations.
"""

_JSON_RE = re.compile(r"\{.*\}", re.S)


@dataclass
class Grade:
    verdict: str            # "pass" | "fail" | "error"
    grounded: bool | None = None
    citations_correct: bool | None = None
    complete: bool | None = None
    clear: bool | None = None
    issues: list[str] | None = None


def grade_answer(llm, question: str, answer_text: str, sources: list[Hit],
                 model: str = MODEL) -> Grade:
    prompt = (f"{build_context(sources)}\n\n"
              f"QUESTION ASKED: {question}\n\n"
              f"CHATBOT ANSWER TO GRADE:\n{answer_text}")
    with llm.messages.stream(
        model=model,
        max_tokens=2000,
        system=[{"type": "text", "text": GRADER_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()
    text = "".join(b.text for b in response.content if b.type == "text")
    match = _JSON_RE.search(text)
    if not match:
        return Grade(verdict="error", issues=[f"grader returned no JSON: {text[:200]}"])
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return Grade(verdict="error", issues=["grader returned invalid JSON"])
    return Grade(
        verdict=data.get("verdict", "error"),
        grounded=data.get("grounded"),
        citations_correct=data.get("citations_correct"),
        complete=data.get("complete"),
        clear=data.get("clear"),
        issues=data.get("issues", []),
    )
