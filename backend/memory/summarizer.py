"""
Conversation summariser for token-budget management.

Called by token_budget.py when cumulative unsummarised tokens exceed threshold T.
Summarises older turns into a concise paragraph that preserves key facts,
legal context, and user intent — so the LLM prompt stays informed even after
raw messages are pruned from the context window.

Uses the project's configured LLM (Gemini by default) via google-generativeai.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.core.config import Settings

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """\
You are a summarisation assistant for a consumer credit rights chatbot.

Summarise the following conversation between a user and an AI assistant.
Preserve:
- Key legal questions the user asked
- Which laws/regulations were discussed (ECOA/Regulation B, Fair Housing Act, Fair Lending Overview, Handbook Intro)
- Any facts about the user's situation (denied credit, protected basis mentioned, type of loan, etc.)
- The assistant's key conclusions or advice given
- Any complaint-filing suggestions made

Keep the summary under 200 words. Write in third person ("The user asked...").

Conversation:
{conversation}

Summary:"""


def summarize_messages(messages: list[dict], settings: "Settings") -> str:
    """
    Summarise a list of conversation turns into a concise paragraph.

    Args:
        messages: List of dicts with keys 'role' and 'message'.
                  e.g. [{"role": "user", "message": "..."}, {"role": "assistant", "message": "..."}]
        settings: App settings (provides gemini_api_key and llm config).

    Returns:
        A summary string (typically 100–200 words).

    Raises:
        RuntimeError: If no LLM API key is configured.
    """
    if not messages:
        return ""

    # Format the conversation for the prompt
    formatted = "\n".join(
        f"{m['role'].capitalize()}: {m['message']}" for m in messages
    )
    prompt = _SUMMARY_PROMPT.format(conversation=formatted)

    # Try Gemini first (project default), fall back to OpenAI
    if settings.gemini_api_key:
        return _summarize_gemini(prompt, settings)
    elif settings.openai_api_key:
        return _summarize_openai(prompt, settings)
    else:
        raise RuntimeError(
            "No LLM API key configured for summarisation. "
            "Set GEMINI_API_KEY or OPENAI_API_KEY in .env"
        )


def _summarize_gemini(prompt: str, settings: "Settings") -> str:
    """Call Gemini via google-generativeai SDK."""
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.summarizer_model)

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=settings.summarizer_max_tokens,
        ),
    )

    summary = response.text.strip()
    logger.info("Summarised %d chars of conversation into %d chars via Gemini (%s)",
                len(prompt), len(summary), settings.summarizer_model)
    return summary


def _summarize_openai(prompt: str, settings: "Settings") -> str:
    """Call OpenAI via the openai SDK (fallback if Gemini key not set)."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.summarizer_openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=settings.summarizer_max_tokens,
    )

    summary = response.choices[0].message.content.strip()
    logger.info("Summarised %d chars of conversation into %d chars via OpenAI (%s)",
                len(prompt), len(summary), settings.summarizer_openai_model)
    return summary
