"""
Query-side embedding — turns a user's question into the same vector space
the corpus was embedded into.

Reuses ingestion.embeddings._get_embed_model (the same cached OpenAI client
ingestion uses) rather than building a second client, so the query is
guaranteed to be embedded with the identical model/dimensions the index was
built with (text-embedding-3-large @ Settings.embedding_dimensions) —
mismatched query/corpus embedding spaces would silently return garbage
nearest-neighbors, not an error.
"""
from __future__ import annotations

from api.core.config import Settings, get_settings
from ingestion.embeddings import _get_embed_model


def embed_query(query: str, settings: Settings | None = None) -> list[float]:
    """Embed a single query string. Raises if query is empty/whitespace —
    embedding an empty string is a caller bug, not a valid retrieval."""
    if not query or not query.strip():
        raise ValueError("embed_query() requires a non-empty query string")

    settings = settings or get_settings()
    embed_model = _get_embed_model(
        settings.embedding_model,
        settings.embedding_dimensions,
        settings.openai_api_key,
        settings.embedding_batch_size,
    )
    return embed_model.get_text_embedding(query)
