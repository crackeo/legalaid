"""Chat session persistence (SQLite).

Privacy: legal questions are sensitive. We store only what the product
needs — the conversation text per session (so follow-up questions work and
users can reload a session) and feedback votes. No accounts, no IPs, no
tracking. purge_older_than() implements the retention policy.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path


class SessionStore:
    def __init__(self, db_path: str | Path):
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, created_at REAL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY,
            session_id TEXT REFERENCES sessions(id),
            role TEXT, content TEXT, sources TEXT, verified INTEGER,
            created_at REAL
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            message_id INTEGER REFERENCES messages(id),
            vote INTEGER,          -- +1 / -1
            comment TEXT, created_at REAL
        );
        """)
        # Lightweight migration for databases created before Phase 7.
        try:
            self.db.execute("ALTER TABLE messages ADD COLUMN duration_ms INTEGER")
        except sqlite3.OperationalError:
            pass  # column already exists
        self.db.commit()

    def create_session(self) -> str:
        sid = uuid.uuid4().hex
        self.db.execute("INSERT INTO sessions VALUES (?, ?)", (sid, time.time()))
        self.db.commit()
        return sid

    def session_exists(self, sid: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (sid,)
        ).fetchone() is not None

    def add_message(self, sid: str, role: str, content: str,
                    sources: list | None = None, verified: bool | None = None,
                    duration_ms: int | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO messages (session_id, role, content, sources, verified,"
            " created_at, duration_ms) VALUES (?,?,?,?,?,?,?)",
            (sid, role, content, json.dumps(sources or []),
             None if verified is None else int(verified), time.time(),
             duration_ms),
        )
        self.db.commit()
        return cur.lastrowid

    def history(self, sid: str, max_turns: int = 10) -> list[dict]:
        """Recent turns in Claude messages format (role/content only)."""
        rows = self.db.execute(
            "SELECT role, content FROM messages WHERE session_id = ?"
            " ORDER BY id DESC LIMIT ?", (sid, max_turns * 2),
        ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def messages(self, sid: str) -> list[dict]:
        """Full messages with metadata, for rendering a reloaded session."""
        rows = self.db.execute(
            "SELECT id, role, content, sources, verified FROM messages"
            " WHERE session_id = ? ORDER BY id", (sid,),
        ).fetchall()
        return [{"id": i, "role": r, "content": c,
                 "sources": json.loads(s or "[]"),
                 "verified": None if v is None else bool(v)}
                for i, r, c, s, v in rows]

    def add_feedback(self, message_id: int, vote: int, comment: str = "") -> None:
        self.db.execute(
            "INSERT INTO feedback (message_id, vote, comment, created_at)"
            " VALUES (?,?,?,?)", (message_id, vote, comment, time.time()),
        )
        self.db.commit()

    def stats(self, days: int = 14) -> dict:
        """Operational overview for the admin dashboard."""
        one = lambda q, *p: self.db.execute(q, p).fetchone()[0]
        answers = one("SELECT COUNT(*) FROM messages WHERE role='assistant'")
        verified = one("SELECT COUNT(*) FROM messages WHERE role='assistant'"
                       " AND verified=1")
        cutoff = time.time() - days * 86400
        per_day = self.db.execute(
            "SELECT date(created_at, 'unixepoch') d, COUNT(*)"
            " FROM messages WHERE role='user' AND created_at >= ?"
            " GROUP BY d ORDER BY d", (cutoff,)).fetchall()
        return {
            "sessions": one("SELECT COUNT(*) FROM sessions"),
            "questions": one("SELECT COUNT(*) FROM messages WHERE role='user'"),
            "answers": answers,
            "verified_rate": round(verified / answers, 3) if answers else None,
            "avg_answer_ms": one("SELECT CAST(AVG(duration_ms) AS INTEGER)"
                                 " FROM messages WHERE duration_ms IS NOT NULL"),
            "feedback_up": one("SELECT COUNT(*) FROM feedback WHERE vote > 0"),
            "feedback_down": one("SELECT COUNT(*) FROM feedback WHERE vote < 0"),
            "questions_per_day": [{"day": d, "count": c} for d, c in per_day],
        }

    def recent_feedback(self, limit: int = 50) -> list[dict]:
        """Latest feedback with question + answer, newest first (review queue)."""
        rows = self.db.execute("""
            SELECT f.created_at, f.vote, f.comment, m.id, m.content, m.verified,
                   (SELECT content FROM messages u
                    WHERE u.session_id = m.session_id AND u.role = 'user'
                      AND u.id < m.id ORDER BY u.id DESC LIMIT 1)
            FROM feedback f JOIN messages m ON m.id = f.message_id
            ORDER BY f.created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [{"at": at, "vote": vote, "comment": comment, "message_id": mid,
                 "answer": answer,
                 "verified": None if v is None else bool(v),
                 "question": question}
                for at, vote, comment, mid, answer, v, question in rows]

    def purge_older_than(self, days: float) -> int:
        cutoff = time.time() - days * 86400
        cur = self.db.execute(
            "DELETE FROM messages WHERE session_id IN"
            " (SELECT id FROM sessions WHERE created_at < ?)", (cutoff,))
        self.db.execute("DELETE FROM sessions WHERE created_at < ?", (cutoff,))
        self.db.commit()
        return cur.rowcount
