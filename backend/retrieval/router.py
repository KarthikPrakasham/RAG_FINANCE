"""Query routing and retrieval orchestration for the chat endpoint."""
from __future__ import annotations

from typing import Any

from api.core.config import Settings, get_settings
from retrieval.context_builder import build_context_text
from retrieval.dense_retriever import SimpleDenseRetriever
from retrieval.sparse_retriever import SparseRetriever


class RetrievalService:
    """Route a query and retrieve grounded context.

    Current implementation limitation:
        ``SimpleDenseRetriever`` re-parses PDFs from ``backend/data`` and uses
        a local TF-IDF-like keyword scorer. It does not query the Pinecone
        dense+sparse vectors created by ``ingestion.ingest_embed``. Replace
        this temporary path with Pinecone/BM25 hybrid retrieval so query-time
        search uses the index produced during ingestion.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dense_retriever = SimpleDenseRetriever(settings=self.settings)
        self.sparse_retriever = SparseRetriever(settings=self.settings)

    def route_query(self, query: str) -> str:
        lowered = query.lower()
        if any(term in lowered for term in ["trend", "increase", "decrease", "up or down", "last year", "credit went"]):
            return "trend"
        if any(term in lowered for term in ["denied", "discrimination", "fair lending", "regulation b", "housing", "mortgage", "race", "age", "sex", "income"]):
            return "legal"
        if any(term in lowered for term in ["what is", "can you", "help me", "policy"]):
            return "legal"
        return "out_of_scope"

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        route = self.route_query(query)
        if route == "out_of_scope":
            return []
        top_k = top_k or self.settings.rag_top_k
        # Use the dense retriever as the primary source and the sparse retriever
        # as a compatibility fallback for the same corpus.
        chunks = self.dense_retriever.retrieve(query=query, top_k=top_k)
        if chunks:
            return chunks
        return self.sparse_retriever.retrieve(query=query, top_k=top_k)

    def build_context(self, query: str, top_k: int | None = None) -> tuple[str, list[dict[str, Any]], str]:
        chunks = self.retrieve(query=query, top_k=top_k)
        return build_context_text(chunks), chunks, self.route_query(query)
