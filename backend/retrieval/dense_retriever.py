"""
Dense (cosine, Pinecone-native) retrieval at query time.

Dense-only, no sparse/hybrid/reranking — per evaluation/ablation Stages 5-7,
sparse (BM25) never improved retrieval on this corpus at any hybrid_alpha
weight, and reranking measurably hurt. See EVALUATION_METHODOLOGY.md Part C.
This queries the same index ingestion.pinecone_upsert wrote to, unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass

from api.core.config import Settings, get_settings
from ingestion.pinecone_upsert import get_index


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    source_doc: str
    law: str
    section: str
    page_num: int | None
    chunk_index: int | None


def _to_retrieved_chunk(match) -> RetrievedChunk:
    """Pinecone's metadata store round-trips numbers as floats (e.g.
    page_num=3 comes back as 3.0) — cast back to int for display/schema
    consistency with Chunk.page_num's actual type."""
    metadata = match.metadata or {}
    page_num = metadata.get("page_num")
    chunk_index = metadata.get("chunk_index")
    return RetrievedChunk(
        chunk_id=match.id,
        score=match.score,
        text=metadata.get("text", ""),
        source_doc=metadata.get("source_doc", ""),
        law=metadata.get("law", ""),
        section=metadata.get("section", ""),
        page_num=int(page_num) if page_num is not None else None,
        chunk_index=int(chunk_index) if chunk_index is not None else None,
    )


def retrieve_dense(
    query_vector: list[float],
    top_k: int | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Query the Pinecone index for the top_k nearest chunks to query_vector,
    ranked by cosine similarity (the index's configured metric). Returns an
    empty list if the index has no vectors yet, never raises for "no
    results" — only for a real connection/config failure."""
    settings = settings or get_settings()
    top_k = top_k or settings.rag_top_k

    _pc, index = get_index(settings)
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=settings.pinecone_namespace,
        include_values=False,
        include_metadata=True,
    )
    return [_to_retrieved_chunk(match) for match in response.matches]
