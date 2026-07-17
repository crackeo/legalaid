"""Phase 2 CLI.

    python -m rag.cli index [--corpus corpus/chunks.jsonl] [--db corpus/index.db]
    python -m rag.cli ask "Is defamation a crime in Bhutan?" [--db corpus/index.db]
    python -m rag.cli eval [--questions eval/questions.jsonl] [--db corpus/index.db]

`index` builds the SQLite index from the Phase 1 corpus. `ask` answers one
question with citations (needs ANTHROPIC_API_KEY). `eval` measures retrieval
hit-rate against the question set — no API key needed, so run it on every
pipeline change.
"""

import argparse
import json
import sys
from pathlib import Path

from .answer import LegalAidRAG
from .embed import embedder_from_env
from .retrieve import retrieve
from .store import Store, index_corpus


def cmd_index(args) -> int:
    embedder = embedder_from_env()
    if Path(args.db).exists():
        Path(args.db).unlink()
    n = index_corpus(args.corpus, args.db, embedder)
    print(f"indexed {n} chunks into {args.db} (embedder: {embedder.name})")
    return 0


def cmd_ask(args) -> int:
    rag = LegalAidRAG(Store(args.db), embedder_from_env())
    answer = rag.ask(args.question)
    print(answer.text)
    print("\nSources:")
    for i, h in enumerate(answer.sources, start=1):
        marker = "*" if i in answer.cited else " "
        print(f" {marker}[{i}] {h.doc_title} — Section {h.section_number}, p.{h.page}")
    return 0


def cmd_eval(args) -> int:
    """Retrieval hit-rate: for each question, is a chunk from the expected
    document (and section, if given) in the top-k?"""
    store, embedder = Store(args.db), embedder_from_env()
    questions = [json.loads(l) for l in Path(args.questions).read_text().splitlines()
                 if l.strip() and not l.startswith("//")]
    hits_at_k = 0
    for q in questions:
        results = retrieve(store, embedder, q["question"], k=args.k)
        found = any(
            q["expect_doc"].lower() in h.doc_title.lower()
            and (not q.get("expect_section") or q["expect_section"] == h.section_number)
            for h in results
        )
        hits_at_k += found
        print(f"{'PASS' if found else 'MISS'}  {q['question'][:70]}")
    total = len(questions)
    print(f"\nretrieval hit-rate@{args.k}: {hits_at_k}/{total} "
          f"({100 * hits_at_k / max(total, 1):.0f}%)")
    return 0 if hits_at_k == total else 1


def cmd_export_feedback(args) -> int:
    """Export downvoted answers (with their questions) as review candidates.

    Each line is a question/answer pair a legal reviewer can check; confirmed
    problems become new entries in eval/questions.jsonl. This closes the loop
    from the 👎 button to the evaluation set.
    """
    import sqlite3
    db = sqlite3.connect(args.sessions_db)
    rows = db.execute("""
        SELECT f.created_at, f.vote, f.comment, m.id, m.content, m.verified,
               (SELECT content FROM messages u
                WHERE u.session_id = m.session_id AND u.role = 'user'
                  AND u.id < m.id ORDER BY u.id DESC LIMIT 1)
        FROM feedback f JOIN messages m ON m.id = f.message_id
        WHERE f.vote < 0 ORDER BY f.created_at
    """).fetchall()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for created_at, vote, comment, mid, answer, verified, question in rows:
            out.write(json.dumps({
                "question": question, "answer": answer,
                "verified": bool(verified) if verified is not None else None,
                "comment": comment, "message_id": mid, "at": created_at,
            }, ensure_ascii=False) + "\n")
    print(f"exported {len(rows)} downvoted answers to {out_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rag.cli", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="build the index from the Phase 1 corpus")
    p.add_argument("--corpus", default="corpus/chunks.jsonl")
    p.add_argument("--db", default="corpus/index.db")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="answer one question (needs ANTHROPIC_API_KEY)")
    p.add_argument("question")
    p.add_argument("--db", default="corpus/index.db")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="measure retrieval hit-rate (no API key needed)")
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--db", default="corpus/index.db")
    p.add_argument("--k", type=int, default=6)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("export-feedback",
                       help="dump downvoted answers for legal review")
    p.add_argument("--sessions-db", default="corpus/sessions.db")
    p.add_argument("--out", default="eval/review_queue.jsonl")
    p.set_defaults(func=cmd_export_feedback)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
