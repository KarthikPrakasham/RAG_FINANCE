"""Prompt assembly and orchestration helpers for the grounded RAG chat flow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from api.core.config import Settings, get_settings
from orchestration.llm_client import LLMClient

if TYPE_CHECKING:
    from retrieval.context_builder import ContextResult


DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a finance-focused assistant for consumer credit and economic trends. "
    "Use the supplied context and conversation history when available. "
    "If the context is incomplete, say so clearly and avoid inventing facts."
)
UNGROUNDED_RESPONSE = (
    "I couldn't find sufficiently relevant information in the available sources "
    "to answer that reliably. Please rephrase your question or provide more detail."
)


@dataclass(frozen=True)
class GroundedAnswer:
    """The answer plus the retrieval result that supports it."""

    answer: str
    token_usage: dict[str, int]
    prompt: str
    context: ContextResult


def _normalize_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Normalize memory records into role/content prompt entries."""
    normalized: list[dict[str, str]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role") or item.get("type") or "user")
        content = str(item.get("content") or item.get("message") or "").strip()
        if role in {"user", "assistant", "system"} and content:
            normalized.append({"role": role, "content": content})

    return normalized


def _normalize_context(context: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Normalize legacy dictionary context into source/content prompt entries."""
    normalized: list[dict[str, str]] = []
    for item in context or []:
        if not isinstance(item, dict):
            continue
        source_doc = str(item.get("source_doc") or item.get("source") or "unknown")
        content = str(item.get("content") or item.get("text") or "").strip()
        if content:
            normalized.append({"source_doc": source_doc, "content": content})
    return normalized


def build_prompt(
    question: str,
    history: list[dict[str, Any]] | None = None,
    context: list[dict[str, Any]] | None = None,
    context_text: str | None = None,
    system_instruction: str | None = None,
    user_intent: dict[str, Any] | None = None,
    summary: str | None = None,
) -> str:
    """Assemble the final prompt sent to the LLM.

    ``context_text`` accepts the citation-labelled text produced by the current
    retrieval layer. ``context`` remains supported for callers using the older
    dictionary contract.

    ``user_intent`` injects the user's persisted profile (long-term memory)
    so the LLM has context about their situation across sessions.

    ``summary`` injects the rolling conversation summary (short-term memory
    compressed by the token budget manager) for continuity.
    """
    instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
    history_rows = _normalize_history(history)
    history_block = "\n".join(
        f"- {row['role']}: {row['content']}" for row in history_rows
    ) or "- none"

    if context_text and context_text.strip():
        context_block = context_text.strip()
    else:
        context_rows = _normalize_context(context)
        context_block = "\n".join(
            f"- {row['source_doc']}: {row['content']}" for row in context_rows
        ) or "- no retrieved context available"

    # Build user intent block (long-term memory)
    intent_block = ""
    if user_intent:
        import json
        intent_block = f"\nUser profile (persisted facts about this user):\n{json.dumps(user_intent, indent=2)}\n"

    # Build summary block (compressed short-term memory)
    summary_block = ""
    if summary and summary.strip():
        summary_block = f"\nConversation summary (older turns):\n{summary.strip()}\n"

    return (
        f"System instruction: {instruction}\n"
        f"{intent_block}"
        f"{summary_block}\n"
        f"Recent conversation:\n{history_block}\n\n"
        f"Context:\n{context_block}\n\n"
        f"User question:\n{question}"
    )


class OrchestrationService:
    """Retrieve grounded context, build the prompt, and invoke the configured LLM."""

    def __init__(self, settings: Settings | None = None, llm_client: LLMClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient()

    def retrieve_context(self, question: str, top_k: int | None = None) -> ContextResult:
        """Embed the question, retrieve ranked chunks, and enforce grounding."""
        from retrieval.context_builder import build_context
        from retrieval.dense_retriever import retrieve_dense
        from retrieval.embed_query import embed_query

        query_vector = embed_query(question, settings=self.settings)
        chunks = retrieve_dense(query_vector, top_k=top_k, settings=self.settings)
        return build_context(chunks, settings=self.settings)

    async def generate_grounded_answer(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
        top_k: int | None = None,
        system_instruction: str | None = None,
        user_intent: dict[str, Any] | None = None,
        summary: str | None = None,
    ) -> GroundedAnswer:
        """Answer only when the retrieval layer returns grounded context."""
        context_result = self.retrieve_context(question=question, top_k=top_k)
        prompt = build_prompt(
            question=question,
            history=history,
            context_text=context_result.context_text,
            system_instruction=system_instruction,
            user_intent=user_intent,
            summary=summary,
        )
        if not context_result.is_grounded:
            return GroundedAnswer(
                answer=UNGROUNDED_RESPONSE,
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                prompt=prompt,
                context=context_result,
            )

        answer, token_usage = await self.llm_client.generate(prompt=prompt, settings=self.settings)
        return GroundedAnswer(
            answer=answer,
            token_usage=token_usage,
            prompt=prompt,
            context=context_result,
        )
