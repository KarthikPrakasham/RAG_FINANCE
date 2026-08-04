"""
Ingestion orchestrator: parse -> chunk -> embed (dense + sparse) -> upsert.

Wired into POST /ingest via api/routers/ingest.py. The FRED CSV never routes
through here — it's loaded separately via structured_data/fred_loader.py.

Hybrid search support
---------------------
The pipeline now calls embed_chunks_with_sparse() instead of embed_chunks().
This produces both dense (OpenAI) and sparse (BM25) vectors in one pass and
saves the fitted BM25 encoder to backend/ingestion/bm25_encoder.json so the
retrieval side can reload it at query time via sparse_index.load_bm25().
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from api.core.config import Settings, get_settings
from ingestion.embeddings import embed_chunks_with_sparse
from ingestion.parsers.parser import Chunk, parse_and_chunk
from ingestion.pinecone_upsert import (
    existing_chunk_ids,
    get_index,
    upsert_embedded_chunks,
    verify_index_dimensions,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    status: str  # "success" | "error"
    documents_processed: int
    chunks_upserted: int
    message: str


async def run_ingestion(
    force: bool,
    settings: Settings | None = None,
) -> IngestionResult:
    """Parse the 4 source PDFs, chunk them (default strategy), embed with both
    dense and sparse vectors, and upsert into Pinecone.

    force=False (default): skip re-embedding chunks whose chunk_id (a content
        hash — see parser.py) already exists in the index, since that means
        the content hasn't changed. No OpenAI calls for unchanged chunks.

        Note: even when chunks are skipped for dense embedding, the BM25
        encoder is always re-fit on the *full* corpus so its vocabulary stays
        consistent with the full index. Only the Pinecone upsert is skipped.

    force=True: embed and upsert every chunk regardless. Still idempotent —
        Pinecone upsert is by ID, so re-sending an unchanged chunk overwrites
        rather than duplicates it.
    """
    settings = settings or get_settings()
    try:
        chunks: list[Chunk] = await asyncio.to_thread(
            parse_and_chunk,
            parser_backend="pypdf",
            chunk_strategy="sentence",
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            return IngestionResult(
                "error", 0, 0, "parse_and_chunk() returned no chunks — check backend/data/*.pdf"
            )

        documents_processed = len({c.source_doc for c in chunks})

        pc, index = await asyncio.to_thread(get_index, settings)
        # Fail fast, before any OpenAI call, if the index doesn't match our
        # configured embedding dimensions or metric.
        await asyncio.to_thread(verify_index_dimensions, pc, settings)

        to_embed = chunks
        skipped = 0
        if not force:
            present = await asyncio.to_thread(
                existing_chunk_ids, index, [c.chunk_id for c in chunks]
            )
            to_embed = [c for c in chunks if c.chunk_id not in present]
            skipped = len(chunks) - len(to_embed)
            logger.info(
                "force=False: %d/%d chunks unchanged, skipping re-embed", skipped, len(chunks)
            )

        # Dense + sparse embedding in one pass.
        # BM25 is always fit on the full corpus (all chunks, not just to_embed)
        # so the saved encoder vocabulary is stable regardless of what's skipped.
        embedded = await asyncio.to_thread(
            embed_chunks_with_sparse, to_embed, settings
        )
        upserted = await asyncio.to_thread(upsert_embedded_chunks, embedded, settings)

        return IngestionResult(
            status="success",
            documents_processed=documents_processed,
            chunks_upserted=upserted,
            message=(
                f"Parsed {len(chunks)} chunks from {documents_processed} documents; "
                f"{upserted} embedded+upserted (dense+sparse), {skipped} skipped as unchanged."
            ),
        )
    except Exception as exc:
        logger.exception("Ingestion failed")
        return IngestionResult("error", 0, 0, f"Ingestion failed: {exc}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run_ingestion(force=True))
    print(result)
