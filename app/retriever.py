"""
retriever.py — Top-k retrieval with optional MMR-style reranking.

Pulls FAISS search results, fetches chunk metadata from SQLite,
then applies Maximal Marginal Relevance (MMR) to balance relevance
vs. diversity so we don't return 5 copies of the same sentence.
"""

import numpy as np
from app.embedder import search, _get_model
from app.database import get_chunks_by_ids


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def retrieve(query: str, top_k: int = 5, fetch_k: int = 20, lambda_mmr: float = 0.6) -> list[dict]:
    """
    Retrieve top_k relevant, diverse chunks for a query.

    Steps:
    1. FAISS search for fetch_k candidates
    2. Fetch metadata from SQLite
    3. MMR reranking: iteratively pick chunk that maximises
       λ·relevance − (1−λ)·max_similarity_to_already_selected
    Returns list of chunk dicts with added 'relevance_score' field.
    """
    # Step 1: FAISS search
    raw_results = search(query, top_k=fetch_k)
    if not raw_results:
        return []

    row_ids = [r[0] for r in raw_results]
    dist_map = {r[0]: r[1] for r in raw_results}

    # Step 2: fetch metadata
    chunks = get_chunks_by_ids(row_ids)
    if not chunks:
        return []

    # Convert L2 distance → similarity score (lower distance = higher score)
    max_dist = max(dist_map.values()) or 1.0
    for c in chunks:
        c["relevance_score"] = round(1.0 - dist_map[c["id"]] / max_dist, 4)

    if len(chunks) <= top_k:
        return sorted(chunks, key=lambda x: x["relevance_score"], reverse=True)

    # Step 3: MMR reranking
    model = _get_model()
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    chunk_vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

    selected_indices = []
    remaining = list(range(len(chunks)))

    for _ in range(top_k):
        if not remaining:
            break
        scores = []
        for i in remaining:
            relevance = _cosine_sim(query_vec, chunk_vecs[i])
            if selected_indices:
                redundancy = max(_cosine_sim(chunk_vecs[i], chunk_vecs[j]) for j in selected_indices)
            else:
                redundancy = 0.0
            mmr_score = lambda_mmr * relevance - (1 - lambda_mmr) * redundancy
            scores.append((i, mmr_score))
        best = max(scores, key=lambda x: x[1])[0]
        selected_indices.append(best)
        remaining.remove(best)

    return [chunks[i] for i in selected_indices]


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {c['filename']}, page {c['page']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)
