"""
FAISS counterpart to pinecone_upsert.py — used only by
evaluation/ablation/stage4_vectordb.py to compare vector DB engines on
identical vectors. Not wired into the production ingestion pipeline.

FAISS is a pure ANN library: it stores int64 ids and float vectors only, no
metadata and no native string ids. This module wraps an
IndexIDMap2(IndexFlatIP) with a sidecar dict mapping a sequential int64 id
<-> chunk_id/metadata, so callers still work in terms of chunk_id strings
like the Pinecone/Chroma modules do. Everything here is in-memory — there's
no persistence step, since Stage 4 rebuilds the index fresh per ablation run
(the corpus is ~40 chunks; rebuilding is instant).

Vectors are L2-normalized before indexing and before querying so
IndexFlatIP's inner product is cosine similarity — matching the cosine
ranking used by evaluation/inmemory_search.py in Stages 1-3, keeping
Stage 4's quality comparison apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import faiss
import numpy as np

from ingestion.embeddings import EmbeddedChunk
from ingestion.parsers.parser import Chunk


def _chunk_metadata(chunk: Chunk) -> dict:
    raw = {f.name: getattr(chunk, f.name) for f in fields(chunk)}
    return {k: v for k, v in raw.items() if v is not None}


@dataclass
class FaissStore:
    index: "faiss.IndexIDMap2"
    id_to_chunk_id: dict[int, str]
    metadata_by_chunk_id: dict[str, dict]
    dim: int


def build_index(dim: int) -> FaissStore:
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    return FaissStore(index=index, id_to_chunk_id={}, metadata_by_chunk_id={}, dim=dim)


def upsert_embedded_chunks(embedded_chunks: list[EmbeddedChunk], store: FaissStore) -> int:
    """Add vectors to the index. NOT idempotent like the Pinecone/Chroma
    versions — IndexFlatIP has no update-by-id, so re-running on the same
    chunks would duplicate them. Fine for a single ablation run; callers
    doing incremental upserts would need to rebuild the index instead."""
    if not embedded_chunks:
        return 0

    vectors = np.array([ec.values for ec in embedded_chunks], dtype="float32")
    faiss.normalize_L2(vectors)

    start_id = len(store.id_to_chunk_id)
    int_ids = np.arange(start_id, start_id + len(embedded_chunks), dtype="int64")
    store.index.add_with_ids(vectors, int_ids)

    for int_id, ec in zip(int_ids, embedded_chunks):
        store.id_to_chunk_id[int(int_id)] = ec.chunk.chunk_id
        store.metadata_by_chunk_id[ec.chunk.chunk_id] = _chunk_metadata(ec.chunk)

    return len(embedded_chunks)


def query_dense(store: FaissStore, query_vector: list[float], top_k: int) -> list[str]:
    """Return chunk_ids ranked by cosine similarity, descending."""
    q = np.array([query_vector], dtype="float32")
    faiss.normalize_L2(q)
    _distances, indices = store.index.search(q, top_k)
    return [store.id_to_chunk_id[i] for i in indices[0] if i != -1]
