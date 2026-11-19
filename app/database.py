"""
database.py — SQLite metadata store for document chunks.
Stores chunk id, source filename, page number, and chunk text
so retrieved FAISS results can be traced back to their origin.
"""

import sqlite3
import os

DB_PATH = os.environ.get("DB_PATH", "data/chunks.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the chunks table if it doesn't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      TEXT NOT NULL,
                filename    TEXT NOT NULL,
                page        INTEGER,
                chunk_index INTEGER,
                text        TEXT NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id)")
        conn.commit()


def insert_chunks(doc_id: str, filename: str, chunks: list[dict]) -> list[int]:
    """Insert a list of {page, chunk_index, text} dicts, return their row ids."""
    with get_connection() as conn:
        ids = []
        for c in chunks:
            cur = conn.execute(
                "INSERT INTO chunks (doc_id, filename, page, chunk_index, text) VALUES (?,?,?,?,?)",
                (doc_id, filename, c["page"], c["chunk_index"], c["text"]),
            )
            ids.append(cur.lastrowid)
        conn.commit()
    return ids


def get_chunks_by_ids(row_ids: list[int]) -> list[dict]:
    """Fetch chunk rows by their SQLite row ids."""
    if not row_ids:
        return []
    placeholders = ",".join("?" * len(row_ids))
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", row_ids
        ).fetchall()
    return [dict(r) for r in rows]


def delete_doc(doc_id: str):
    """Remove all chunks for a document (used when re-ingesting)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.commit()


def list_documents() -> list[dict]:
    """Return one row per unique document ingested."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT doc_id, filename, COUNT(*) as chunk_count, MAX(created_at) as ingested_at "
            "FROM chunks GROUP BY doc_id ORDER BY ingested_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
