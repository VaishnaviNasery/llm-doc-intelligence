"""
embedder.py — Embedding + FAISS vector index management.

Uses sentence-transformers (all-MiniLM-L6-v2) for local, free embeddings.
FAISS index is persisted to disk so ingested documents survive restarts.
Each vector's position in the FAISS index maps to a SQLite row id,
enabling full chunk metadata retrieval after search.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "data/faiss.index")
FAISS_MAP_PATH   = os.environ.get("FAISS_MAP_PATH",   "data/faiss_map.json")
MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_id_map: list[int] = []   # position i → SQLite row id


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _load_index():
    global _index, _id_map
    os.makedirs("data", exist_ok=True)
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAP_PATH):
        _index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_MAP_PATH) as f:
            _id_map = json.load(f)
    else:
        dim = _get_model().get_sentence_embedding_dimension()
        _index = faiss.IndexFlatL2(dim)
        _id_map = []


def _save_index():
    faiss.write_index(_index, FAISS_INDEX_PATH)
    with open(FAISS_MAP_PATH, "w") as f:
        json.dump(_id_map, f)


def embed_and_index(texts: list[str], row_ids: list[int]):
    """Embed texts and add them to the FAISS index, mapping to SQLite row ids."""
    global _index, _id_map
    if _index is None:
        _load_index()

    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    vectors = np.array(vectors, dtype="float32")

    _index.add(vectors)
    _id_map.extend(row_ids)
    _save_index()


def search(query: str, top_k: int = 5) -> list[tuple[int, float]]:
    """
    Search the FAISS index for the top-k most similar chunks.
    Returns list of (sqlite_row_id, distance) tuples.
    """
    global _index, _id_map
    if _index is None:
        _load_index()
    if _index.ntotal == 0:
        return []

    model = _get_model()
    q_vec = model.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")

    k = min(top_k, _index.ntotal)
    distances, indices = _index.search(q_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_id_map):
            continue
        results.append((_id_map[idx], float(dist)))
    return results


def get_index_size() -> int:
    """Return number of vectors in the index (0 if no index exists yet)."""
    global _index
    if _index is None:
        # Don't force model download just to return 0 when no index exists
        if not os.path.exists(FAISS_INDEX_PATH):
            return 0
        _load_index()
    return _index.ntotal


def reset_index():
    """Clear the entire FAISS index and map (used in tests)."""
    global _index, _id_map
    dim = _get_model().get_sentence_embedding_dimension()
    _index = faiss.IndexFlatL2(dim)
    _id_map = []
    _save_index()
