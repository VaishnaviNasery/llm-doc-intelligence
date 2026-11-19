"""
evaluator.py — Evaluation metrics for RAG pipeline quality.

Metrics tracked per query:
- latency_ms:      end-to-end response time in milliseconds
- precision_at_k:  fraction of retrieved chunks that contain query keywords
                   (lightweight proxy when no ground truth labels exist)
- faithfulness:    fraction of answer sentences that can be grounded in
                   at least one retrieved chunk (keyword overlap heuristic)
- chunk_count:     number of chunks retrieved
- token_usage:     prompt + completion tokens from LLM

All results are persisted to SQLite for trend analysis.
"""

import re
import os
import sqlite3
import time
from datetime import datetime

EVAL_DB_PATH = os.environ.get("EVAL_DB_PATH", "data/eval_log.db")


# ── DB setup ──────────────────────────────────────────────────────────────────

def _get_eval_conn():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_eval_db():
    with _get_eval_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                query            TEXT,
                latency_ms       REAL,
                precision_at_k   REAL,
                faithfulness     REAL,
                chunk_count      INTEGER,
                prompt_tokens    INTEGER,
                completion_tokens INTEGER,
                created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


# ── Metric helpers ────────────────────────────────────────────────────────────

def _keywords(text: str) -> set[str]:
    """Extract meaningful lowercase words (length > 3) as a keyword set."""
    stopwords = {"that", "this", "with", "have", "from", "they", "will",
                 "been", "were", "what", "when", "which", "their", "there"}
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    return {w for w in words if w not in stopwords}


def compute_precision_at_k(query: str, chunks: list[dict]) -> float:
    """
    Precision@k: fraction of retrieved chunks relevant to the query.
    A chunk is 'relevant' if it shares ≥2 keywords with the query.
    """
    if not chunks:
        return 0.0
    q_kw = _keywords(query)
    if not q_kw:
        return 0.0
    hits = sum(1 for c in chunks if len(_keywords(c["text"]) & q_kw) >= 2)
    return round(hits / len(chunks), 4)


def compute_faithfulness(answer: str, chunks: list[dict]) -> float:
    """
    Faithfulness: fraction of answer sentences grounded in retrieved chunks.
    A sentence is grounded if it shares ≥3 keywords with any chunk.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
    if not sentences:
        return 0.0
    chunk_kw_sets = [_keywords(c["text"]) for c in chunks]
    grounded = 0
    for sent in sentences:
        s_kw = _keywords(sent)
        if any(len(s_kw & ckw) >= 3 for ckw in chunk_kw_sets):
            grounded += 1
    return round(grounded / len(sentences), 4)


# ── Timer context manager ─────────────────────────────────────────────────────

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = round((time.perf_counter() - self.start) * 1000, 2)


# ── Log result ────────────────────────────────────────────────────────────────

def log_eval(query: str, chunks: list[dict], answer: str,
             latency_ms: float, usage: dict) -> dict:
    """Compute all metrics, persist to DB, and return the metrics dict."""
    p_at_k = compute_precision_at_k(query, chunks)
    faith  = compute_faithfulness(answer, chunks)

    metrics = {
        "latency_ms":         latency_ms,
        "precision_at_k":     p_at_k,
        "faithfulness":       faith,
        "chunk_count":        len(chunks),
        "prompt_tokens":      usage.get("prompt_tokens", 0),
        "completion_tokens":  usage.get("completion_tokens", 0),
    }

    with _get_eval_conn() as conn:
        conn.execute("""
            INSERT INTO eval_log
              (query, latency_ms, precision_at_k, faithfulness, chunk_count,
               prompt_tokens, completion_tokens)
            VALUES (?,?,?,?,?,?,?)
        """, (query, latency_ms, p_at_k, faith, len(chunks),
              metrics["prompt_tokens"], metrics["completion_tokens"]))
        conn.commit()

    return metrics


def get_eval_history(limit: int = 50) -> list[dict]:
    """Return the most recent eval log entries."""
    with _get_eval_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_log ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_avg_metrics() -> dict:
    """Return aggregate averages across all logged queries."""
    with _get_eval_conn() as conn:
        row = conn.execute("""
            SELECT
                ROUND(AVG(latency_ms), 2)     AS avg_latency_ms,
                ROUND(AVG(precision_at_k), 4) AS avg_precision_at_k,
                ROUND(AVG(faithfulness), 4)   AS avg_faithfulness,
                COUNT(*)                      AS total_queries
            FROM eval_log
        """).fetchone()
    return dict(row) if row else {}
