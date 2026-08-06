"""
Turns ranked RetrievedChunks into (a) a prompt-ready context string and
(b) citation records — the two things the (not-yet-built) orchestration
layer needs to assemble a grounded LLM prompt and populate
ChatResponse.citations.

Citation is a plain dataclass here, not api.schemas.chat.Citation — this
module deliberately doesn't depend on the API presentation layer. Field
names match that schema exactly (source_doc, law, section, chunk_id) so
converting one to the other later is a trivial 1:1 mapping.

Grounding threshold: Settings.grounding_threshold (0.65 default) decides
whether what was retrieved is good enough to answer from at all — per
REQUIREMENT.md's guardrail ("never invent a legal protection not present in
these documents"), a caller should treat is_grounded=False as a signal to
decline or hedge rather than pass thin/irrelevant context to the LLM as if
it were solid ground.
"""
from __future__ import annotations

from dataclasses import dataclass

from api.core.config import Settings, get_settings
from retrieval.dense_retriever import RetrievedChunk


@dataclass
class Citation:
    source_doc: str
    law: str
    section: str
    chunk_id: str


@dataclass
class ContextResult:
    context_text: str
    citations: list[Citation]
    is_grounded: bool


def build_context(
    chunks: list[RetrievedChunk],
    settings: Settings | None = None,
) -> ContextResult:
    """Filter chunks below grounding_threshold, then format the survivors
    into a numbered, citation-labeled context block for the LLM prompt.

    is_grounded is False when every candidate scored below threshold (or
    none were retrieved at all) — the caller should not fabricate an answer
    in that case."""
    settings = settings or get_settings()
    grounded_chunks = [c for c in chunks if c.score >= settings.grounding_threshold]

    if not grounded_chunks:
        return ContextResult(context_text="", citations=[], is_grounded=False)

    blocks = []
    citations = []
    for i, chunk in enumerate(grounded_chunks, start=1):
        label = f"[{i}] ({chunk.law}, {chunk.source_doc}, page {chunk.page_num})"
        blocks.append(f"{label}\n{chunk.text}")
        citations.append(
            Citation(
                source_doc=chunk.source_doc,
                law=chunk.law,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
            )
        )

    return ContextResult(
        context_text="\n\n".join(blocks),
        citations=citations,
        is_grounded=True,
    )
