"""Sparse-style retrieval wrapper for the same local document index."""
from __future__ import annotations

from typing import Any

from retrieval.dense_retriever import SimpleDenseRetriever


class SparseRetriever(SimpleDenseRetriever):
    """Alias the lightweight retriever as a sparse-style backend for compatibility."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return super().retrieve(query=query, top_k=top_k)
