"""Prompt assembly and orchestration helpers for the RAG chat flow."""
from __future__ import annotations

from typing import Any

from api.core.config import Settings, get_settings
from orchestration.llm_client import LLMClient

DEFAULT_SYSTEM_INSTRUCTION = (
	"You are a finance-focused assistant for consumer credit and economic trends. "
	"Use the supplied context and conversation history when available. "
	"If the context is incomplete, say so clearly and avoid inventing facts."
)


def _normalize_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
	"""Normalize history records from the memory layer into a simple list of role/content entries."""
	normalized: list[dict[str, str]] = []
	for item in history or []:
		if not isinstance(item, dict):
			continue

		role = str(item.get("role") or item.get("type") or "user")
		content = str(item.get("content") or item.get("message") or "").strip()
		if role in {"user", "assistant", "system"} and content:
			normalized.append({"role": role, "content": content})

	return normalized


def _normalize_context(context: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
	"""Normalize retrieval context items into a lightweight list of source/content dicts."""
	normalized: list[dict[str, Any]] = []
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
	system_instruction: str | None = None,
) -> str:
	"""Assemble the final prompt sent to the LLM."""
	instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
	history_rows = _normalize_history(history)
	context_rows = _normalize_context(context)

	history_block = "\n".join(
		f"- {row['role']}: {row['content']}" for row in history_rows
	) or "- none"

	context_block = "\n".join(
		f"- {row['source_doc']}: {row['content']}" for row in context_rows
	) or "- no retrieved context available"

	return (
		f"System instruction: {instruction}\n\n"
		f"History:\n{history_block}\n\n"
		f"Context:\n{context_block}\n\n"
		f"User question:\n{question}"
	)


class OrchestrationService:
	"""Small orchestration wrapper that builds prompts and invokes the configured LLM client."""

	def __init__(self, settings: Settings | None = None, llm_client: LLMClient | None = None) -> None:
		self.settings = settings or get_settings()
		self.llm_client = llm_client or LLMClient()

	async def generate_answer(
		self,
		question: str,
		history: list[dict[str, Any]] | None = None,
		context: list[dict[str, Any]] | None = None,
		system_instruction: str | None = None,
	) -> tuple[str, dict[str, int], str]:
		prompt = build_prompt(
			question=question,
			history=history,
			context=context,
			system_instruction=system_instruction,
		)
		answer, token_usage = await self.llm_client.generate(prompt=prompt, settings=self.settings)
		return answer, token_usage, prompt
