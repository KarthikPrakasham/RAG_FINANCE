"""
Stage 3 (embedding) — turns parsed/chunked text into vectors.

Default production model is OpenAI text-embedding-3-large, truncated to
Settings.embedding_dimensions (1024) via the API's native `dimensions` param,
matching the pre-existing Pinecone index this feeds into (see
pinecone_upsert.py). Not to be confused with parsers/parser.py's own
_get_embed_model, which loads a separate, independently-configured embedding
model only for Stage 2's optional "semantic" chunk-boundary detection.

Hybrid search support
---------------------
EmbeddedChunk now carries an optional `sparse_values` field (a SparseVector).
When populated, pinecone_upsert.py includes it in the upsert record so the
index can serve hybrid (dense + sparse) queries. Use embed_chunks_with_sparse()
to produce both dense and sparse vectors in one pass, or call embed_chunks()
for dense-only (sparse_values will be None).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from api.core.config import Settings, get_settings
from ingestion.parsers.parser import Chunk
from ingestion.sparse_index import SparseVector


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    values: list[float]
    sparse_values: SparseVector | None = field(default=None)


@lru_cache(maxsize=4)
def _get_embed_model(model_name: str, dimensions: int, api_key: str, batch_size: int):
    """Lazily construct (and cache) the OpenAI embedding client.

    Imported locally so importing this module doesn't require the openai
    client unless embed_chunks() is actually called.
    """
    if not api_key:
        raise RuntimeError(
            "Settings.openai_api_key is not set — required to embed chunks "
            f"with {model_name!r}."
        )
    from llama_index.embeddings.openai import OpenAIEmbedding

    return OpenAIEmbedding(
        model=model_name,
        dimensions=dimensions,
        api_key=api_key,
        embed_batch_size=batch_size,
    )


def embed_chunks(
    chunks: list[Chunk],
    settings: Settings | None = None,
) -> list[EmbeddedChunk]:
    """Embed each chunk's text, in the same order as the input list.

    Empty input returns an empty list without making any API call.
    """
    if not chunks:
        return []

    settings = settings or get_settings()
    embed_model = _get_embed_model(
        settings.embedding_model,
        settings.embedding_dimensions,
        settings.openai_api_key,
        settings.embedding_batch_size,
    )

    vectors = embed_model.get_text_embedding_batch(
        [chunk.text for chunk in chunks], show_progress=True
    )
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} vectors for {len(chunks)} chunks"
        )
    return [EmbeddedChunk(chunk=c, values=v) for c, v in zip(chunks, vectors)]


def embed_chunks_with_sparse(
    chunks: list[Chunk],
    settings: Settings | None = None,
    bm25_encoder_path: "Path | None" = None,  # noqa: F821 — Path imported lazily
) -> list[EmbeddedChunk]:
    """Embed chunks with *both* dense (OpenAI) and sparse (BM25) vectors.

    This is the function ingest_embed.py should call instead of embed_chunks()
    when hybrid search is enabled. It:

    1. Produces dense vectors exactly as embed_chunks() does.
    2. Fits (or loads) a BM25 encoder on the chunk texts of this batch.
    3. Encodes each chunk's text into a SparseVector and attaches it.

    The BM25 encoder is saved to disk (backend/ingestion/bm25_encoder.json by
    default) so that retrieval/sparse_retriever.py can reload it for query-time
    sparse encoding without re-fitting.

    Parameters
    ----------
    chunks:
        The list of Chunk objects to embed — same as embed_chunks().
    settings:
        App settings; defaults to get_settings().
    bm25_encoder_path:
        Override the default save/load path for the BM25 encoder JSON.
        Useful in tests or multi-environment setups.

    Returns
    -------
    List of EmbeddedChunk with both `values` (dense) and `sparse_values` set.
    """
    if not chunks:
        return []

    from pathlib import Path as _Path

    from ingestion.sparse_index import fit_and_save_bm25, encode_sparse

    # --- Dense pass (unchanged from embed_chunks) ---
    dense_results = embed_chunks(chunks, settings)

    # --- Sparse pass ---
    # Always re-fit on the full batch so the encoder vocabulary reflects the
    # current corpus. If you're doing incremental upserts (force=False), the
    # new chunks will extend the vocabulary — fitting only on new chunks would
    # produce an incompatible encoder. Re-fitting is fast (pure Python, no GPU).
    encoder_path = _Path(bm25_encoder_path) if bm25_encoder_path else None
    corpus_texts = [c.text for c in chunks]
    kwargs = {"save_path": encoder_path} if encoder_path else {}
    encoder = fit_and_save_bm25(corpus_texts, **kwargs)
    sparse_vectors = encode_sparse(corpus_texts, encoder)

    # --- Merge ---
    return [
        EmbeddedChunk(chunk=ec.chunk, values=ec.values, sparse_values=sv)
        for ec, sv in zip(dense_results, sparse_vectors)
    ]
