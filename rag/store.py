"""SQLite index over the corpus: chunks + FTS5 (BM25) + embedding vectors.

One file, zero infrastructure. The corpus (~150 Acts, tens of thousands of
chunks) is small enough that brute-force cosine over an in-memory matrix is
fast; swap for pgvector/Qdrant later without touching callers — the public
surface is index_corpus() and Store.search_*().
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Hit:
    chunk_id: int
    doc_title: str
    section_number: str
    section_heading: str
    text: str
    page: int
    source_url: str
    score: float


class Store:
    def __init__(self, db_path: str | Path):
        # check_same_thread=False: the web app serves queries from worker
        # threads; access is read-only after indexing, so this is safe.
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._create()
        self._matrix: np.ndarray | None = None  # lazily-loaded embedding matrix
        self._matrix_ids: list[int] = []

    def _create(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_title TEXT, section_number TEXT, section_heading TEXT,
            text TEXT, page INTEGER, part INTEGER,
            source_url TEXT, doc_type TEXT, language TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            doc_title, section_heading, text, content='chunks', content_rowid='id'
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id),
            model TEXT, vector BLOB
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)
        self.db.commit()

    # -- indexing ---------------------------------------------------------

    def add_chunks(self, chunks: list[dict]) -> list[int]:
        ids = []
        cur = self.db.cursor()
        for c in chunks:
            cur.execute(
                "INSERT INTO chunks (doc_title, section_number, section_heading,"
                " text, page, part, source_url, doc_type, language)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (c["doc_title"], c["section_number"], c["section_heading"],
                 c["text"], c["page"], c.get("part", 0), c.get("source_url", ""),
                 c.get("doc_type", ""), c.get("language", "en")),
            )
            rowid = cur.lastrowid
            cur.execute(
                "INSERT INTO chunks_fts (rowid, doc_title, section_heading, text)"
                " VALUES (?,?,?,?)",
                (rowid, c["doc_title"], c["section_heading"], c["text"]),
            )
            ids.append(rowid)
        self.db.commit()
        return ids

    def add_embeddings(self, chunk_ids: list[int], vectors: np.ndarray, model: str) -> None:
        cur = self.db.cursor()
        for cid, vec in zip(chunk_ids, vectors):
            cur.execute(
                "INSERT OR REPLACE INTO embeddings (chunk_id, model, vector) VALUES (?,?,?)",
                (cid, model, vec.astype(np.float32).tobytes()),
            )
        self.db.commit()
        self._matrix = None

    # -- search -----------------------------------------------------------

    def _hit(self, row, score: float) -> Hit:
        return Hit(chunk_id=row[0], doc_title=row[1], section_number=row[2],
                   section_heading=row[3], text=row[4], page=row[5],
                   source_url=row[6], score=score)

    _COLS = ("chunks.id, chunks.doc_title, chunks.section_number,"
             " chunks.section_heading, chunks.text, chunks.page, chunks.source_url")

    def search_bm25(self, query: str, k: int = 20) -> list[Hit]:
        # FTS5 query syntax chokes on punctuation; use quoted terms.
        terms = [t for t in "".join(ch if ch.isalnum() else " " for ch in query).split() if t]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms)
        rows = self.db.execute(
            f"SELECT {self._COLS}, bm25(chunks_fts) FROM chunks_fts"
            f" JOIN chunks ON chunks.id = chunks_fts.rowid"
            f" WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (fts_query, k),
        ).fetchall()
        # bm25() is a rank (lower = better); negate so higher = better.
        return [self._hit(r, -r[-1]) for r in rows]

    def _load_matrix(self) -> None:
        rows = self.db.execute("SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id").fetchall()
        if not rows:
            self._matrix, self._matrix_ids = np.zeros((0, 1), np.float32), []
            return
        self._matrix_ids = [r[0] for r in rows]
        self._matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])

    def search_vector(self, query_vec: np.ndarray, k: int = 20) -> list[Hit]:
        if self._matrix is None:
            self._load_matrix()
        if not self._matrix_ids:
            return []
        sims = self._matrix @ query_vec.astype(np.float32)
        top = np.argsort(-sims)[:k]
        hits = []
        for idx in top:
            cid = self._matrix_ids[int(idx)]
            row = self.db.execute(
                f"SELECT {self._COLS} FROM chunks WHERE id = ?", (cid,)
            ).fetchone()
            hits.append(self._hit(row, float(sims[idx])))
        return hits

    def get_chunk(self, chunk_id: int) -> Hit | None:
        row = self.db.execute(
            f"SELECT {self._COLS} FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return self._hit(row, 0.0) if row else None

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


    def embedder_name(self) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
        return row[0] if row else None


def check_embedder(store: Store, embedder) -> None:
    """Fail loudly when the query embedder differs from the index embedder.

    Happens in practice when VOYAGE_API_KEY appears (or disappears) after the
    index was built: query vectors would live in a different space — and a
    different dimension — than the indexed ones, crashing or silently
    returning garbage.
    """
    indexed = store.embedder_name()
    if indexed is not None and indexed != embedder.name:
        raise RuntimeError(
            f"index was built with embedder '{indexed}' but queries would use "
            f"'{embedder.name}'. Re-run `python -m rag.cli index` (or restore "
            f"the matching *_API_KEY environment).")


def index_corpus(jsonl_path: str | Path, db_path: str | Path, embedder) -> int:
    """Load corpus/chunks.jsonl into a fresh index with embeddings."""
    chunks = [json.loads(line) for line in Path(jsonl_path).read_text().splitlines() if line.strip()]
    store = Store(db_path)
    ids = store.add_chunks(chunks)
    # Embed heading + text so "murder" matches the marginal note too.
    texts = [f"{c['doc_title']} — {c['section_heading']}\n{c['text']}" for c in chunks]
    for start in range(0, len(texts), 512):
        batch_ids = ids[start:start + 512]
        vecs = embedder.embed(texts[start:start + 512])
        store.add_embeddings(batch_ids, vecs, embedder.name)
    store.db.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (embedder.name,))
    store.db.commit()
    return len(ids)
