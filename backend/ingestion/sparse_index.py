"""
Sparse embedding via BM25 — used alongside dense vectors for hybrid search.

The BM25Encoder (from pinecone-text) must be *fit* on your full corpus once
during ingestion so that the term→index mapping is stable. The fitted encoder
is serialised to disk so the retrieval side can reload it at query time without
re-fitting.

Typical ingestion flow
----------------------
1. Collect all chunk texts after chunking.
2. Call fit_and_save_bm25(texts) once — writes bm25_encoder.json next to this
   file (or to the path you pass in).
3. Call encode_sparse(texts, encoder) to get a SparseVector for every chunk.
4. Attach each SparseVector to its EmbeddedChunk before upserting.

Typical retrieval flow
----------------------
1. Call load_bm25() to get the saved encoder.
2. Call encode_sparse([query_text], encoder)[0] for the query's sparse vector.
3. Pass both the dense query vector and the sparse query vector to Pinecone's
   hybrid query (alpha controls the dense/sparse blend — see Settings.hybrid_alpha).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default save location: backend/ingestion/bm25_encoder.json
_DEFAULT_ENCODER_PATH = Path(__file__).resolve().parent / "bm25_encoder.json"


@dataclass
class SparseVector:
    """Pinecone-compatible sparse representation."""
    indices: list[int]
    values: list[float]

    def to_pinecone_dict(self) -> dict:
        """Return the dict Pinecone's SDK expects for sparse_values."""
        return {"indices": self.indices, "values": self.values}


def _get_encoder():
    """Import BM25Encoder lazily so the module loads without pinecone-text
    installed unless sparse encoding is actually needed."""
    try:
        from pinecone_text.sparse import BM25Encoder
    except ImportError as exc:
        raise ImportError(
            "pinecone-text is required for sparse (BM25) embeddings. "
            "Install it with: pip install pinecone-text"
        ) from exc
    return BM25Encoder


def fit_and_save_bm25(
    corpus: list[str],
    save_path: Path = _DEFAULT_ENCODER_PATH,
) -> "BM25Encoder":  # type: ignore[name-defined]  # noqa: F821
    """Fit a BM25 encoder on *corpus* and persist it to *save_path*.

    Must be called once (per corpus change) during ingestion. The saved file
    is a JSON that the retrieval side loads via load_bm25().

    Parameters
    ----------
    corpus:
        All chunk texts that will be indexed — the vocabulary is built from
        this set. Passing a subset will cause out-of-vocabulary terms at query
        time to silently score zero; passing the full corpus avoids that.
    save_path:
        Where to write the serialised encoder. Defaults to
        backend/ingestion/bm25_encoder.json.

    Returns
    -------
    The fitted BM25Encoder instance (also available to the caller without a
    separate load_bm25() call).
    """
    BM25Encoder = _get_encoder()
    encoder = BM25Encoder.default()   # uses default tokeniser + BM25 params
    logger.info("Fitting BM25 encoder on %d documents …", len(corpus))
    encoder.fit(corpus)
    encoder.dump(str(save_path))
    logger.info("BM25 encoder saved to %s", save_path)
    return encoder


def load_bm25(
    save_path: Path = _DEFAULT_ENCODER_PATH,
) -> "BM25Encoder":  # type: ignore[name-defined]  # noqa: F821
    """Load a previously fitted BM25 encoder from *save_path*.

    Raises FileNotFoundError if the encoder has never been fitted/saved.
    """
    if not save_path.exists():
        raise FileNotFoundError(
            f"BM25 encoder not found at {save_path}. "
            "Run fit_and_save_bm25() during ingestion first."
        )
    BM25Encoder = _get_encoder()
    encoder = BM25Encoder.load(str(save_path))
    logger.info("BM25 encoder loaded from %s", save_path)
    return encoder


def encode_sparse(
    texts: list[str],
    encoder,  # BM25Encoder — typed loosely to avoid the import at class-def time
) -> list[SparseVector]:
    """Encode *texts* into BM25 sparse vectors using a fitted *encoder*.

    Returns one SparseVector per text, in the same order as the input.
    Zero-length texts produce an empty sparse vector (indices=[], values=[]).

    Parameters
    ----------
    texts:
        The raw strings to encode — chunk texts during ingestion, query text
        at retrieval time.
    encoder:
        A fitted BM25Encoder loaded via load_bm25() or returned by
        fit_and_save_bm25().
    """
    raw: list[dict] = encoder.encode_documents(texts)
    results: list[SparseVector] = []
    for item in raw:
        # BM25Encoder returns dicts with "indices" and "values" keys
        indices = item.get("indices", [])
        values = item.get("values", [])
        results.append(SparseVector(indices=indices, values=values))
    return results


def encode_sparse_queries(
    texts: list[str],
    encoder,
) -> list[SparseVector]:
    """Like encode_sparse but uses encode_queries() — BM25Encoder applies
    slightly different tokenisation for queries vs documents (no IDF
    normalisation on the query side).

    Use this at retrieval time; use encode_sparse() during ingestion.
    """
    raw: list[dict] = encoder.encode_queries(texts)
    results: list[SparseVector] = []
    for item in raw:
        indices = item.get("indices", [])
        values = item.get("values", [])
        results.append(SparseVector(indices=indices, values=values))
    return results
