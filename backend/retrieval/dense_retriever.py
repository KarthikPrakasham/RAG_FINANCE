"""Lightweight retrieval implementation over the bundled policy PDFs."""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from api.core.config import Settings, get_settings


class SimpleDenseRetriever:
    """Temporary local keyword retriever that works without vector-DB setup.

    This class is not a true dense retriever: it re-parses the source PDFs on
    demand and ranks chunks with a TF-IDF-like keyword score. It deliberately
    bypasses the Pinecone dense+sparse vectors and fitted BM25 encoder produced
    by ``ingestion.ingest_embed``. Replace this implementation when wiring
    query-time Pinecone/BM25 hybrid retrieval.
    """

    def __init__(self, data_dir: str | Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / "data"

    def _load_chunks(self) -> list[dict[str, Any]]:
        if not self.data_dir.exists():
            return []

        chunks: list[dict[str, Any]] = []
        for pdf_path in sorted(self.data_dir.glob("*.pdf")):
            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = re.sub(r"\s+", " ", text).strip()
                if not text:
                    continue

                for chunk_idx, chunk_text in enumerate(self._chunk_text(text)):
                    law = self._law_for_path(pdf_path.name)
                    chunks.append(
                        {
                            "source_doc": pdf_path.name,
                            "law": law,
                            "section": f"page {page_number}",
                            "chunk_id": f"{pdf_path.stem}#chunk{chunk_idx}",
                            "content": chunk_text,
                        }
                    )
        return chunks

    def _chunk_text(self, text: str, chunk_size: int = 500) -> list[str]:
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks: list[str] = []
        for start in range(0, len(words), chunk_size):
            chunk = " ".join(words[start : start + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def _law_for_path(self, filename: str) -> str:
        name = filename.lower()
        if "ecoa" in name:
            return "ECOA/RegB"
        if "fair_housing" in name:
            return "FairHousingAct"
        if "overview" in name:
            return "Overview"
        return "HandbookIntro"

    @lru_cache(maxsize=1)
    def _cached_index(self) -> list[dict[str, Any]]:
        return self._load_chunks()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.settings.rag_top_k
        if not query.strip():
            return []

        chunks = self._cached_index()
        if not chunks:
            return []

        query_terms = set(self._tokenize(query))
        if not query_terms:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        idf = self._build_idf(chunks)
        for chunk in chunks:
            tokens = self._tokenize(chunk["content"])
            tf = Counter(tokens)
            score = sum(idf.get(term, 1.0) * tf.get(term, 0) for term in query_terms)
            if query_terms & set(tokens):
                score += 0.5
            scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k] if chunk.get("content")]

    def _build_idf(self, chunks: list[dict[str, Any]]) -> dict[str, float]:
        token_to_docs: dict[str, int] = Counter()
        for chunk in chunks:
            tokens = set(self._tokenize(chunk["content"]))
            for token in tokens:
                token_to_docs[token] += 1

        total_docs = max(1, len(chunks))
        return {token: max(1.0, (total_docs / count)) for token, count in token_to_docs.items()}

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[a-zA-Z]{2,}", text.lower()) if token not in {"the", "and", "for", "that", "with", "from", "this", "your"}]
