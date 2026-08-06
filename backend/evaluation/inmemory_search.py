"""
Brute-force in-memory retrieval used by ablation Stages 1, 2, 3, 5, 6, 7.

Why in-memory instead of hitting live Pinecone for every ablation variant:
each variant (parser backend, chunk strategy, embedding model, alpha value...)
would otherwise mean re-upserting a throwaway batch of vectors into the
production index just to score it once. For an 18-page corpus, brute-force
cosine/dot-product over a few hundred chunk vectors is instant and exact —
no ANN approximation error, no index pollution, no extra Pinecone cost.
Stage 4 (vectordb) is where the real Pinecone/Chroma/FAISS clients get
exercised, since that stage's question is about the DB engines themselves.

The Stage 6 hybrid math here (dense*alpha + sparse*(1-alpha) dot product) is
the same formula Pinecone's server-side hybrid query applies — see
EVALUATION_METHODOLOGY.md Part C — so scoring it in-memory is not an
approximation of what Pinecone does, it's the same computation.
"""
from __future__ import annotations

import numpy as np


def dense_topk(
    query_vector: list[float],
    chunk_vectors: dict[str, list[float]],
    k: int,
) -> list[str]:
    """Rank chunk_ids by cosine similarity to query_vector, descending."""
    if not chunk_vectors:
        return []
    ids = list(chunk_vectors.keys())
    mat = np.array([chunk_vectors[i] for i in ids], dtype=np.float64)
    q = np.array(query_vector, dtype=np.float64)
    denom = np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) + 1e-12) + 1e-12
    sims = (mat @ q) / denom
    order = np.argsort(-sims)[:k]
    return [ids[i] for i in order]


def sparse_topk(
    query_sparse: dict[str, float],  # {index (as str): weight}
    chunk_sparse: dict[str, dict[str, float]],  # chunk_id -> {index: weight}
    k: int,
) -> list[str]:
    """Rank chunk_ids by BM25 sparse dot product against query_sparse."""
    if not chunk_sparse or not query_sparse:
        return []
    scored = []
    for chunk_id, weights in chunk_sparse.items():
        score = sum(w * query_sparse.get(idx, 0.0) for idx, w in weights.items())
        scored.append((chunk_id, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [chunk_id for chunk_id, _ in scored[:k]]


def hybrid_alpha_topk(
    query_dense: list[float],
    query_sparse: dict[str, float],
    chunk_dense: dict[str, list[float]],
    chunk_sparse: dict[str, dict[str, float]],
    alpha: float,
    k: int,
) -> list[str]:
    """Pinecone-native hybrid: ONE combined score per chunk from alpha-scaled
    dense + (1-alpha)-scaled sparse vectors, ranked in a single pass — not
    two separate rankings merged afterward. See EVALUATION_METHODOLOGY.md
    Part C (Stage 6) for why this differs from RRF."""
    ids = set(chunk_dense) | set(chunk_sparse)
    if not ids:
        return []

    dense_mat = np.array([chunk_dense.get(i, []) for i in ids if chunk_dense.get(i)], dtype=np.float64)
    dense_ids = [i for i in ids if chunk_dense.get(i)]
    q_dense = np.array(query_dense, dtype=np.float64) * alpha
    dense_scores = {}
    if dense_ids:
        raw = dense_mat @ q_dense
        dense_scores = dict(zip(dense_ids, raw))

    sparse_scores = {}
    if query_sparse:
        scaled_query_sparse = {idx: w * (1 - alpha) for idx, w in query_sparse.items()}
        for chunk_id, weights in chunk_sparse.items():
            sparse_scores[chunk_id] = sum(
                w * scaled_query_sparse.get(idx, 0.0) for idx, w in weights.items()
            )

    combined = {
        chunk_id: dense_scores.get(chunk_id, 0.0) + sparse_scores.get(chunk_id, 0.0)
        for chunk_id in ids
    }
    ranked = sorted(combined.items(), key=lambda pair: pair[1], reverse=True)
    return [chunk_id for chunk_id, _ in ranked[:k]]


def rrf_merge(
    dense_ranked: list[str],
    sparse_ranked: list[str],
    k: int,
    rrf_k: int = 60,
) -> list[str]:
    """Reciprocal Rank Fusion of two independently-ranked lists:
    score(id) = sum over lists containing id of 1/(rrf_k + rank).
    rrf_k=60 is the standard default from the original RRF paper — it damps
    the influence of very top ranks so a single list's #1 doesn't dominate."""
    scores: dict[str, float] = {}
    for ranked_list in (dense_ranked, sparse_ranked):
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [chunk_id for chunk_id, _ in ranked[:k]]
