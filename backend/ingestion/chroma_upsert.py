"""
Chroma counterpart to pinecone_upsert.py — used only by
evaluation/ablation/stage4_vectordb.py to compare vector DB engines on
identical vectors. Not wired into the production ingestion pipeline
(ingest_embed.py still targets Pinecone only).

Chroma has no native sparse/hybrid vector support the way Pinecone does, so
EmbeddedChunk.sparse_values is intentionally dropped here — this module is
dense-only. The collection is created with hnsw:space="cosine" so ranking
matches the cosine similarity used by evaluation/inmemory_search.py in
Stages 1-3, keeping Stage 4's quality comparison apples-to-apples.

Storage is local (PersistentClient writes to disk at `persist_directory`),
so running this does not touch any shared/external service — unlike the
live Pinecone index pinecone_upsert.py targets.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import chromadb

from ingestion.embeddings import EmbeddedChunk
from ingestion.parsers.parser import Chunk

_DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[1] / "evaluation" / ".chroma_scratch"


def _chunk_metadata(chunk: Chunk) -> dict:
    """Same non-None-fields convention as pinecone_upsert.py's
    _chunk_metadata, minus `text` (Chroma stores that separately as the
    collection's `documents`, not in metadata)."""
    raw = {f.name: getattr(chunk, f.name) for f in fields(chunk) if f.name != "text"}
    return {k: v for k, v in raw.items() if v is not None}


def get_collection(
    collection_name: str = "rag-finance-ablation",
    persist_directory: Path | str | None = _DEFAULT_PERSIST_DIR,
):
    """persist_directory=None uses an in-memory EphemeralClient instead of
    PersistentClient — no disk writes, nothing to clean up. Preferred for
    ablation/scoring runs; pass a real path only if you want the collection
    to survive past the current process."""
    if persist_directory is None:
        client = chromadb.EphemeralClient()
    else:
        client = chromadb.PersistentClient(path=str(persist_directory))
    return client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )


def upsert_embedded_chunks(embedded_chunks: list[EmbeddedChunk], collection) -> int:
    """Upsert vectors, batched by chunk_id — upsert-by-id makes this
    idempotent the same way pinecone_upsert.upsert_embedded_chunks() is."""
    if not embedded_chunks:
        return 0

    ids = [ec.chunk.chunk_id for ec in embedded_chunks]
    embeddings = [ec.values for ec in embedded_chunks]
    documents = [ec.chunk.text for ec in embedded_chunks]
    metadatas = [_chunk_metadata(ec.chunk) for ec in embedded_chunks]

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(embedded_chunks)


def query_dense(collection, query_vector: list[float], top_k: int) -> list[str]:
    """Return chunk_ids ranked by cosine similarity, descending."""
    result = collection.query(query_embeddings=[query_vector], n_results=top_k, include=[])
    return list(result["ids"][0])
