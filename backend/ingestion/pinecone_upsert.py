"""
Stage 4 (vector upsert) — pushes embedded chunks into the pre-existing
Pinecone index (created externally for text-embedding-3-large @
Settings.embedding_dimensions). This module never creates, deletes, or
reconfigures the index — it only connects, validates, and upserts.

Hybrid search support
---------------------
When EmbeddedChunk.sparse_values is populated, to_pinecone_vectors() includes
it in the upsert record as the `sparse_values` key. Pinecone requires the index
metric to be `dotproduct` for hybrid queries — verify_index_dimensions() now
also checks this and raises IndexDimensionMismatch if the metric is wrong.
"""
from __future__ import annotations

from dataclasses import fields
from functools import lru_cache

from pinecone import Pinecone

from api.core.config import Settings, get_settings
from ingestion.embeddings import EmbeddedChunk
from ingestion.parsers.parser import Chunk


class IndexDimensionMismatch(RuntimeError):
    """The configured embedding_dimensions doesn't match the live index,
    or the index metric is not dotproduct (required for hybrid search)."""


@lru_cache(maxsize=2)
def _get_client(api_key: str) -> Pinecone:
    if not api_key:
        raise RuntimeError("Settings.pinecone_api_key is not set.")
    return Pinecone(api_key=api_key)


def get_index(settings: Settings | None = None):
    """Return (Pinecone client, Index handle) for Settings.pinecone_index_name."""
    settings = settings or get_settings()
    pc = _get_client(settings.pinecone_api_key)
    return pc, pc.Index(settings.pinecone_index_name)


def verify_index_dimensions(pc: Pinecone, settings: Settings) -> None:
    """Fail fast — before spending any OpenAI calls — if the live index's
    dimension doesn't match what we're configured to embed at, or if the
    index metric is not dotproduct (required for hybrid queries)."""
    info = pc.describe_index(settings.pinecone_index_name)
    if info.dimension != settings.embedding_dimensions:
        raise IndexDimensionMismatch(
            f"Pinecone index {settings.pinecone_index_name!r} has dimension="
            f"{info.dimension}, but Settings.embedding_dimensions="
            f"{settings.embedding_dimensions}."
        )
    metric = getattr(info, "metric", None)
    if metric and metric.lower() != "dotproduct":
        raise IndexDimensionMismatch(
            f"Pinecone index {settings.pinecone_index_name!r} uses metric={metric!r}. "
            "Hybrid search requires metric='dotproduct'. "
            "Create a new index with dotproduct metric and update PINECONE_INDEX_NAME."
        )


def _chunk_metadata(chunk: Chunk) -> dict:
    """All non-None Chunk fields, Pinecone-safe (no null values allowed).

    Includes `text` itself — there's no separate blob store, so
    retrieval/context_builder.py reads chunk text back out of the metadata on
    retrieved matches. Also includes `chunk_id`, duplicating the vector ID;
    harmless, and simpler than special-casing it out.
    """
    raw = {f.name: getattr(chunk, f.name) for f in fields(chunk)}
    return {k: v for k, v in raw.items() if v is not None}


def to_pinecone_vectors(
    embedded_chunks: list[EmbeddedChunk],
) -> list[dict]:
    """Build the list of upsert records Pinecone's SDK accepts.

    Each record is a dict with:
      - id: the chunk_id (content hash)
      - values: dense float vector
      - sparse_values: BM25 sparse vector dict (only when present)
      - metadata: all non-None Chunk fields including the raw text

    Using dicts (instead of the old 3-tuple form) lets us conditionally include
    sparse_values without breaking the upsert call when it's absent.
    """
    records = []
    for ec in embedded_chunks:
        record: dict = {
            "id": ec.chunk.chunk_id,
            "values": ec.values,
            "metadata": _chunk_metadata(ec.chunk),
        }
        if ec.sparse_values is not None:
            record["sparse_values"] = ec.sparse_values.to_pinecone_dict()
        records.append(record)
    return records


def existing_chunk_ids(
    index,
    chunk_ids: list[str],
    namespace: str | None = None,
    batch_size: int = 100,
) -> set[str]:
    """Which of these chunk_ids already have a vector stored.

    chunk_id is itself a content hash (see parser.py), so "ID exists" is
    equivalent to "content unchanged" — this is what lets run_ingestion skip
    re-embedding unchanged chunks when force=False.
    """
    found: set[str] = set()
    for i in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[i : i + batch_size]
        resp = index.fetch(ids=batch, namespace=namespace)
        found.update(resp.vectors.keys())
    return found


def upsert_embedded_chunks(
    embedded_chunks: list[EmbeddedChunk],
    settings: Settings | None = None,
) -> int:
    """Upsert vectors, batched. Returns the number of vectors sent.

    Upsert is by chunk_id, so re-running with the same chunks overwrites
    rather than duplicates — this is the idempotency guarantee, independent
    of whether the caller skipped unchanged chunks upstream.

    When EmbeddedChunk.sparse_values is set, each record is upserted with
    both dense and sparse vectors, enabling hybrid queries on the index.
    """
    if not embedded_chunks:
        return 0

    settings = settings or get_settings()
    pc, index = get_index(settings)
    verify_index_dimensions(pc, settings)

    records = to_pinecone_vectors(embedded_chunks)
    batch_size = settings.pinecone_upsert_batch_size

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        index.upsert(vectors=batch, namespace=settings.pinecone_namespace)

    return len(records)
