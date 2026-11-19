"""
ingestor.py — PDF parsing and semantic chunking.

Strategy: sentence-boundary chunking with overlap.
- Split by sentences (spaCy-style via simple regex fallback)
- Group sentences into chunks of ~TARGET_TOKENS tokens
- Overlap by OVERLAP_SENTENCES sentences between chunks
- This is more semantically coherent than fixed-character splitting
"""

import re
import uuid
import hashlib
from pathlib import Path

import pypdf

TARGET_TOKENS = 300       # approximate tokens per chunk
OVERLAP_SENTENCES = 2     # sentence overlap between adjacent chunks
AVG_CHARS_PER_TOKEN = 4   # rough estimate


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter — no heavy NLP dependency."""
    # Split on .  !  ? followed by whitespace and uppercase or end
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _chunk_sentences(sentences: list[str], page: int, chunk_offset: int) -> list[dict]:
    """Group sentences into token-approximate chunks with overlap."""
    chunks = []
    i = 0
    chunk_idx = chunk_offset
    while i < len(sentences):
        current_chars = 0
        window = []
        j = i
        while j < len(sentences):
            window.append(sentences[j])
            current_chars += len(sentences[j])
            j += 1
            if current_chars >= TARGET_TOKENS * AVG_CHARS_PER_TOKEN:
                break
        text = " ".join(window)
        if text:
            chunks.append({"page": page, "chunk_index": chunk_idx, "text": text})
            chunk_idx += 1
        # move forward, keeping OVERLAP_SENTENCES for next chunk
        i = max(i + 1, j - OVERLAP_SENTENCES)
    return chunks, chunk_idx


def ingest_pdf(file_path: str) -> tuple[str, str, list[dict]]:
    """
    Parse a PDF and return (doc_id, filename, chunks).
    doc_id is a stable hash of the file content.
    """
    path = Path(file_path)
    raw_bytes = path.read_bytes()
    doc_id = hashlib.sha256(raw_bytes).hexdigest()[:16]
    filename = path.name

    reader = pypdf.PdfReader(file_path)
    all_chunks = []
    chunk_offset = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue
        sentences = _split_sentences(text)
        page_chunks, chunk_offset = _chunk_sentences(sentences, page_num, chunk_offset)
        all_chunks.extend(page_chunks)

    return doc_id, filename, all_chunks
