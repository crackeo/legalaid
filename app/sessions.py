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
                    sources: list | None = None, verified: bool | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO messages (session_id, role, content, sources, verified,"
            " created_at) VALUES (?,?,?,?,?,?)",
            (sid, role, content, json.dumps(sources or []),
             None if verified is None else int(verified), time.time()),
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

    def purge_older_than(self, days: float) -> int:
        cutoff = time.time() - days * 86400
        cur = self.db.execute(
            "DELETE FROM messages WHERE session_id IN"
            " (SELECT id FROM sessions WHERE created_at < ?)", (cutoff,))
        self.db.execute("DELETE FROM sessions WHERE created_at < ?", (cutoff,))
        self.db.commit()
        return cur.rowcount
