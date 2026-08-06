"""Formatting helpers for retrieved chunks."""
from __future__ import annotations

from typing import Any


def build_context_text(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Return the retrieved chunks as a readable text block for prompt assembly."""
    if not retrieved_chunks:
        return "- no retrieved context available"

    return "\n".join(
        f"- {chunk.get('source_doc', 'unknown')} | {chunk.get('law', 'unknown')} | {chunk.get('content', '')}"
        for chunk in retrieved_chunks
    )
